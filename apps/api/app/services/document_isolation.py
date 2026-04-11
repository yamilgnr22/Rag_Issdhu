import json
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

import httpx

from app.services.document_classification import LLMProviderConfig
from app.services.extractors import ExtractionBlock, ExtractionResult


SUMARIO_PATTERN = re.compile(r"^\s*#{0,6}\s*SUMARIO\b", re.IGNORECASE | re.MULTILINE)
HEADING_PATTERN = re.compile(r"(?m)^(#{1,6})\s+(.+?)\s*$")
ARTICLE_PATTERN = re.compile(
    r"(?mi)^(?:#{0,6}\s*)?(?:art[íi]culo|art\.)\s*\.?\s*(\d+)\b"
)

INSTRUMENT_HEADING_PATTERNS = [
    re.compile(r"^\s*ley\s*n", re.IGNORECASE),
    re.compile(r"^\s*decreto\s*n", re.IGNORECASE),
    re.compile(r"^\s*acuerdo\s+presidencial", re.IGNORECASE),
    re.compile(r"^\s*resoluci[oó]n", re.IGNORECASE),
    re.compile(r"^\s*reglamento\b", re.IGNORECASE),
    re.compile(r"^\s*c[oó]digo\b", re.IGNORECASE),
]

CONTINUATION_HEADING_PATTERNS = [
    re.compile(r"^\s*t[íi]tulo\b", re.IGNORECASE),
    re.compile(r"^\s*cap[íi]tulo\b", re.IGNORECASE),
    re.compile(r"^\s*secci[oó]n\b", re.IGNORECASE),
    re.compile(r"^\s*art[íi]culo\b", re.IGNORECASE),
    re.compile(r"^\s*art\.", re.IGNORECASE),
    re.compile(r"^\s*anexo\b", re.IGNORECASE),
]

LEGAL_PREFACE_PATTERNS = [
    re.compile(r"\bel presidente de la rep[úu]blica\b", re.IGNORECASE),
    re.compile(r"\bla asamblea nacional\b", re.IGNORECASE),
    re.compile(r"\bconsiderando\b", re.IGNORECASE),
    re.compile(r"\bpor tanto\b", re.IGNORECASE),
    re.compile(r"\bha dictado\b", re.IGNORECASE),
    re.compile(r"\bacuerda\b", re.IGNORECASE),
    re.compile(r"\ben uso de sus facultades\b", re.IGNORECASE),
]


@dataclass
class BoundaryCandidate:
    start: int
    heading: str
    candidate_kind: str
    score: float
    reasons: list[str]
    preceding_article_count: int
    max_article_before: int
    first_article_after: int | None
    preface_markers_after: int
    before_snippet: str
    after_snippet: str


@dataclass
class BoundarySelection:
    candidate_index: int
    confidence: float
    reason_short: str
    source: str


class DocumentIsolationService:
    def __init__(self, settings) -> None:
        self.settings = settings

    def isolate(self, *, filename: str, extraction: ExtractionResult) -> ExtractionResult:
        if not extraction.blocks:
            return extraction
        if len(extraction.blocks) != 1:
            return extraction

        block = extraction.blocks[0]
        if block.block_type != "docling_markdown":
            return extraction

        target = self._infer_target_from_filename(filename)
        if target is None:
            return extraction

        text = block.text
        if not self._looks_like_multi_document_normative_source(text):
            return extraction

        isolated_text = self._isolate_target_text(filename=filename, text=text, target=target)
        if not isolated_text or isolated_text == text:
            return extraction

        isolated_block = ExtractionBlock(
            block_type=block.block_type,
            text=isolated_text,
            page=block.page,
            section_path=block.section_path,
        )
        return replace(extraction, blocks=[isolated_block])

    def _looks_like_multi_document_normative_source(self, text: str) -> bool:
        if not text or len(text) < 1500:
            return False
        if not SUMARIO_PATTERN.search(text):
            return False
        heading_hits = sum(1 for _ in HEADING_PATTERN.finditer(text))
        return heading_hits >= 4

    def _infer_target_from_filename(self, filename: str) -> dict[str, str] | None:
        stem = Path(filename).stem
        normalized = self._normalize_for_matching(stem)

        law_match = re.search(r"\bley\b.*?\b(\d{2,5})\b", normalized)
        if law_match:
            return {"kind": "LEY", "number": law_match.group(1)}

        decree_match = re.search(r"\bdecreto\b.*?\b(\d{1,4}(?:-\d{2,4})?)\b", normalized)
        if decree_match:
            return {"kind": "DECRETO", "number": decree_match.group(1)}

        resolution_match = re.search(r"\bresolucion\b.*?\b(\d{1,5}(?:-\d{2,4})?)\b", normalized)
        if resolution_match:
            return {"kind": "RESOLUCION", "number": resolution_match.group(1)}

        return None

    def _isolate_target_text(self, *, filename: str, text: str, target: dict[str, str]) -> str | None:
        start_match = self._find_target_start(text, target)
        if not start_match:
            return None

        start = start_match.start()
        tail = text[start:].strip()
        candidates = self._detect_boundary_candidates(tail)
        if not candidates:
            return tail if len(tail) >= 200 else None

        selection = self._select_boundary(filename=filename, target=target, tail=tail, candidates=candidates)
        if selection is None:
            return tail if len(tail) >= 200 else None

        candidate = candidates[selection.candidate_index]
        cut_start = self._adjust_cut_start_for_continuity(tail, candidate)
        isolated = tail[:cut_start].strip()
        if len(isolated) < 200:
            return tail if len(tail) >= 200 else None
        return isolated

    def _detect_boundary_candidates(self, text: str) -> list[BoundaryCandidate]:
        if len(text) < 2000:
            return []

        headings = list(HEADING_PATTERN.finditer(text))
        if not headings:
            return []

        articles = [(match.start(), int(match.group(1))) for match in ARTICLE_PATTERN.finditer(text)]
        candidates: list[BoundaryCandidate] = []

        for idx, match in enumerate(headings):
            start = match.start()
            if start < 1500:
                continue

            heading_text = match.group(2).strip()
            normalized_heading = self._normalize_for_matching(heading_text)
            score = 0.0
            reasons: list[str] = []

            if self._matches_any(normalized_heading, CONTINUATION_HEADING_PATTERNS):
                continue

            if self._matches_any(normalized_heading, INSTRUMENT_HEADING_PATTERNS):
                score += 4.0
                reasons.append("instrument_heading")

            if self._looks_like_high_salience_heading(heading_text):
                score += 1.5
                reasons.append("high_salience_heading")

            before_window = text[max(0, start - 500) : start].strip()
            after_window = text[start : start + 1200].strip()
            preface_markers_after = self._preface_hits(after_window)
            if preface_markers_after >= 2:
                score += 3.5
                reasons.append("legal_preface_after")
            elif preface_markers_after == 1:
                score += 1.0
                reasons.append("single_preface_marker_after")

            preceding_articles = [number for pos, number in articles if pos < start]
            following_articles = [number for pos, number in articles if start <= pos < start + 1600]
            max_article_before = max(preceding_articles) if preceding_articles else 0
            first_article_after = following_articles[0] if following_articles else None

            if max_article_before >= 10 and first_article_after is not None and first_article_after <= 5:
                score += 4.0
                reasons.append("article_number_restart")
            elif max_article_before >= 25 and first_article_after is not None and first_article_after < max_article_before / 2:
                score += 1.5
                reasons.append("article_discontinuity")

            if self._looks_like_table_transition(before_window, after_window):
                score += 1.0
                reasons.append("structural_transition")

            next_heading_gap = None
            if idx + 1 < len(headings):
                next_heading_gap = headings[idx + 1].start() - start
            if next_heading_gap is not None and next_heading_gap <= 220:
                score += 1.5
                reasons.append("heading_cluster_after")

            if "high_salience_heading" in reasons and "structural_transition" in reasons:
                score += 1.5
                reasons.append("heading_after_table_wall")

            if score < 3.0:
                continue

            candidate_kind = "hard_boundary" if "instrument_heading" in reasons else "continuity_boundary"

            candidates.append(
                BoundaryCandidate(
                    start=start,
                    heading=heading_text,
                    candidate_kind=candidate_kind,
                    score=round(score, 2),
                    reasons=reasons,
                    preceding_article_count=len(preceding_articles),
                    max_article_before=max_article_before,
                    first_article_after=first_article_after,
                    preface_markers_after=preface_markers_after,
                    before_snippet=before_window[-350:],
                    after_snippet=after_window[:500],
                )
            )

        candidates = self._inject_preboundary_candidates(text, candidates, articles, headings)
        candidates = self._dedupe_candidates(candidates)
        if not candidates:
            return []

        ordered = sorted(candidates, key=lambda item: item.start)
        early = ordered[:12]
        highest = sorted(candidates, key=lambda item: (-item.score, item.start))[:6]

        protected = [candidate for candidate in candidates if "preboundary_heading_cluster" in candidate.reasons]

        combined: list[BoundaryCandidate] = []
        seen: set[tuple[int, str]] = set()
        for candidate in protected + early + highest:
            key = (candidate.start, candidate.heading)
            if key in seen:
                continue
            seen.add(key)
            combined.append(candidate)

        return sorted(combined, key=lambda item: item.start)

    def _inject_preboundary_candidates(
        self,
        text: str,
        candidates: list[BoundaryCandidate],
        articles: list[tuple[int, int]],
        headings: list[re.Match[str]],
    ) -> list[BoundaryCandidate]:
        expanded = list(candidates)
        hard_boundaries = [candidate for candidate in candidates if candidate.candidate_kind == "hard_boundary"]
        for boundary in hard_boundaries:
            for heading in headings:
                start = heading.start()
                if start >= boundary.start or boundary.start - start > 250:
                    continue
                heading_text = heading.group(2).strip()
                normalized_heading = self._normalize_for_matching(heading_text)
                if self._matches_any(normalized_heading, CONTINUATION_HEADING_PATTERNS):
                    continue
                if not self._looks_like_high_salience_heading(heading_text):
                    continue

                before_window = text[max(0, start - 500) : start].strip()
                after_window = text[start : start + 1200].strip()
                preface_markers_after = self._preface_hits(after_window)
                preceding_articles = [number for pos, number in articles if pos < start]
                following_articles = [number for pos, number in articles if start <= pos < start + 1600]
                max_article_before = max(preceding_articles) if preceding_articles else 0
                first_article_after = following_articles[0] if following_articles else None

                expanded.append(
                    BoundaryCandidate(
                        start=start,
                        heading=heading_text,
                        candidate_kind="continuity_boundary",
                        score=5.0,
                        reasons=["preboundary_heading_cluster", "high_salience_heading"],
                        preceding_article_count=len(preceding_articles),
                        max_article_before=max_article_before,
                        first_article_after=first_article_after,
                        preface_markers_after=preface_markers_after,
                        before_snippet=before_window[-350:],
                        after_snippet=after_window[:500],
                    )
                )
        return expanded

    def _select_boundary(
        self,
        *,
        filename: str,
        target: dict[str, str],
        tail: str,
        candidates: list[BoundaryCandidate],
    ) -> BoundarySelection | None:
        llm_selection = self._select_boundary_with_llm(
            filename=filename,
            target=target,
            tail=tail,
            candidates=candidates,
        )
        if llm_selection is not None and llm_selection.confidence >= 0.7:
            return self._prefer_continuity_boundary(candidates, llm_selection)

        for idx, candidate in enumerate(candidates):
            if candidate.score >= 6.0:
                fallback = BoundarySelection(
                    candidate_index=idx,
                    confidence=0.74,
                    reason_short="Structural discontinuity indicates a new instrument boundary.",
                    source="structural_fallback",
                )
                return self._prefer_continuity_boundary(candidates, fallback)

        return None

    def _select_boundary_with_llm(
        self,
        *,
        filename: str,
        target: dict[str, str],
        tail: str,
        candidates: list[BoundaryCandidate],
    ) -> BoundarySelection | None:
        providers = self._provider_sequence()
        if not providers:
            return None

        anchor = tail[:1200]
        candidate_lines = []
        for idx, candidate in enumerate(candidates):
            candidate_lines.append(
                "\n".join(
                    [
                        f"Candidate {idx}",
                        f"heading: {candidate.heading}",
                        f"candidate_kind: {candidate.candidate_kind}",
                        f"score: {candidate.score}",
                        f"signals: {', '.join(candidate.reasons) or 'none'}",
                        f"preceding_article_count: {candidate.preceding_article_count}",
                        f"max_article_before: {candidate.max_article_before}",
                        f"first_article_after: {candidate.first_article_after}",
                        f"preface_markers_after: {candidate.preface_markers_after}",
                        f"before_snippet: {candidate.before_snippet}",
                        f"after_snippet: {candidate.after_snippet}",
                    ]
                )
            )

        system_prompt = (
            "You detect where a target legal instrument ends inside a larger official publication. "
            "Choose only among the provided candidates. Return strict JSON only."
        )
        user_prompt = (
            "El archivo parece contener varios instrumentos jurídicos. "
            "Debes seleccionar el primer candidato a partir del cual el texto deja de pertenecer al instrumento objetivo.\n"
            "Si existe una ruptura estructural o editorial inmediatamente antes del encabezado formal del siguiente instrumento, "
            "prefiere esa ruptura más temprana.\n\n"
            f"filename: {filename}\n"
            f"target_kind: {target['kind']}\n"
            f"target_number: {target['number']}\n"
            f"anchor_excerpt:\n{anchor}\n\n"
            "Candidates:\n"
            f"{chr(10).join(candidate_lines)}\n\n"
            "Responde solo JSON con:\n"
            "{\n"
            '  "end_candidate_index": integer or null,\n'
            '  "confidence": number,\n'
            '  "reason_short": string\n'
            "}\n"
            "Si ningún candidato marca claramente el inicio de otro instrumento, usa null."
        )

        for provider in providers:
            try:
                with httpx.Client(timeout=provider.timeout) as client:
                    headers = {"Content-Type": "application/json"}
                    if provider.api_key:
                        headers["Authorization"] = f"Bearer {provider.api_key}"
                    response = client.post(
                        f"{provider.base_url.rstrip('/')}/chat/completions",
                        headers=headers,
                        json={
                            "model": provider.model,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                        },
                    )
                    response.raise_for_status()
                    payload = response.json()
            except Exception:
                continue

            try:
                content = payload["choices"][0]["message"]["content"]
                match = re.search(r"\{.*\}", content, flags=re.DOTALL)
                data = json.loads(match.group(0) if match else content)
                candidate_index = data.get("end_candidate_index")
                confidence = float(data.get("confidence", 0.0))
                if candidate_index is None:
                    return None
                if not isinstance(candidate_index, int):
                    return None
                if candidate_index < 0 or candidate_index >= len(candidates):
                    return None
                return BoundarySelection(
                    candidate_index=candidate_index,
                    confidence=round(confidence, 2),
                    reason_short=str(data.get("reason_short", "LLM boundary selection."))[:180],
                    source=f"llm_{provider.provider_name}",
                )
            except Exception:
                continue

        return None

    def _prefer_continuity_boundary(
        self,
        candidates: list[BoundaryCandidate],
        selection: BoundarySelection,
    ) -> BoundarySelection:
        chosen = candidates[selection.candidate_index]
        if chosen.candidate_kind != "hard_boundary":
            return selection

        best_index = selection.candidate_index
        best_candidate = chosen
        for idx, candidate in enumerate(candidates):
            if candidate.start >= chosen.start:
                break
            if candidate.candidate_kind != "continuity_boundary":
                continue
            if chosen.start - candidate.start > 250:
                continue
            if candidate.score < 4.0:
                continue
            best_index = idx
            best_candidate = candidate

        if best_index == selection.candidate_index:
            return selection

        return BoundarySelection(
            candidate_index=best_index,
            confidence=selection.confidence,
            reason_short=(
                "A continuity break immediately precedes the next formal instrument heading; "
                f"cutting at '{best_candidate.heading}' avoids trailing editorial residue."
            )[:180],
            source=selection.source,
        )

    def _adjust_cut_start_for_continuity(self, text: str, chosen: BoundaryCandidate) -> int:
        if chosen.candidate_kind != "hard_boundary":
            return chosen.start

        window_start = max(0, chosen.start - 250)
        local = text[window_start : chosen.start]
        local_headings = list(HEADING_PATTERN.finditer(local))
        if not local_headings:
            return chosen.start

        adjusted = chosen.start
        for heading in local_headings:
            heading_text = heading.group(2).strip()
            normalized_heading = self._normalize_for_matching(heading_text)
            if self._matches_any(normalized_heading, CONTINUATION_HEADING_PATTERNS):
                continue
            if not self._looks_like_high_salience_heading(heading_text):
                continue
            adjusted = min(adjusted, window_start + heading.start())

        return adjusted

    def _provider_sequence(self) -> list[LLMProviderConfig]:
        mode = self.settings.classifier_mode
        if mode == "heuristic":
            return []
        if mode == "local_llm":
            return [provider for provider in [self._provider_config("local")] if provider]
        if mode == "cloud_llm":
            return [provider for provider in [self._provider_config("cloud")] if provider]

        providers = []
        for provider_name in ("local", "cloud"):
            provider = self._provider_config(provider_name)
            if provider is not None:
                providers.append(provider)
        return providers

    def _provider_config(self, provider_name: str) -> LLMProviderConfig | None:
        if provider_name == "cloud":
            base_url = self.settings.classifier_cloud_base_url or self.settings.classifier_llm_base_url
            api_key = self.settings.classifier_cloud_api_key or self.settings.classifier_llm_api_key
            model = self.settings.classifier_cloud_model or self.settings.classifier_llm_model
            timeout = self.settings.classifier_cloud_timeout or self.settings.classifier_llm_timeout
        else:
            base_url = self.settings.classifier_local_base_url
            api_key = self.settings.classifier_local_api_key
            model = self.settings.classifier_local_model
            timeout = self.settings.classifier_local_timeout or self.settings.classifier_llm_timeout

        if not base_url or not model:
            return None
        return LLMProviderConfig(
            provider_name=provider_name,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
        )

    def _find_target_start(self, text: str, target: dict[str, str]) -> re.Match[str] | None:
        number = re.escape(target["number"])
        if target["kind"] == "LEY":
            patterns = [
                re.compile(
                    rf"(?:^|\n)\s*(?:#{0,6}\s*)?LEY\s*N\s*[Oº°0]?\s*[\.\-:]?\s*{number}\b",
                    re.IGNORECASE,
                ),
                re.compile(
                    rf"(?:^|\n)\s*(?:#{0,6}\s*)?LEYNO\s*[\.\-:]?\s*{number}\b",
                    re.IGNORECASE,
                ),
            ]
        elif target["kind"] == "DECRETO":
            patterns = [
                re.compile(
                    rf"(?:^|\n)\s*(?:#{0,6}\s*)?DECRETO\s*N\s*[Oº°0]?\s*[\.\-:]?\s*{number}\b",
                    re.IGNORECASE,
                ),
            ]
        else:
            patterns = [
                re.compile(
                    rf"(?:^|\n)\s*(?:#{0,6}\s*)?RESOLUCI[ÓO]N\s*N\s*[Oº°0]?\s*[\.\-:]?\s*{number}\b",
                    re.IGNORECASE,
                ),
            ]

        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match
        return None

    def _looks_like_high_salience_heading(self, heading: str) -> bool:
        stripped = heading.strip()
        if len(stripped) < 8 or len(stripped) > 120:
            return False
        letters = [char for char in stripped if char.isalpha()]
        if not letters:
            return False
        uppercase_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
        return uppercase_ratio >= 0.7

    def _preface_hits(self, text: str) -> int:
        return sum(1 for pattern in LEGAL_PREFACE_PATTERNS if pattern.search(text))

    def _looks_like_table_transition(self, before_window: str, after_window: str) -> bool:
        before_pipes = before_window.count("|")
        after_pipes = after_window.count("|")
        return before_pipes >= 10 or after_pipes >= 10

    def _dedupe_candidates(self, candidates: list[BoundaryCandidate]) -> list[BoundaryCandidate]:
        if not candidates:
            return []

        deduped: list[BoundaryCandidate] = [candidates[0]]
        for candidate in sorted(candidates, key=lambda item: item.start):
            previous = deduped[-1]
            if candidate.start - previous.start < 250:
                if candidate.score > previous.score:
                    deduped[-1] = candidate
                continue
            deduped.append(candidate)
        return deduped

    def _matches_any(self, text: str, patterns: list[re.Pattern[str]]) -> bool:
        return any(pattern.search(text) for pattern in patterns)

    def _normalize_for_matching(self, value: str) -> str:
        decomposed = unicodedata.normalize("NFKD", value)
        ascii_text = "".join(char for char in decomposed if not unicodedata.combining(char))
        return ascii_text.lower()
