export type ACL = {
  users: string[];
  groups: string[];
};

export type DocumentContract = {
  document_id: string;
  source: string;
  source_authority: "primary" | "secondary" | "derived";
  mime_type: string;
  language: string;
  acl: ACL;
  created_at: string;
};

export type VersionContract = {
  version_id: string;
  document_id: string;
  checksum: string;
  version_status: "active" | "superseded" | "draft" | "deprecated";
  created_at: string;
};

export type BlockContract = {
  block_id: string;
  document_id: string;
  version_id: string;
  block_type: string;
  text: string;
  page: number;
  section_path: string[];
};

export type ChunkContract = {
  chunk_id: string;
  document_id: string;
  version_id: string;
  block_refs: string[];
  text: string;
  metadata: Record<string, unknown>;
  acl: ACL;
  index_status: "pending" | "indexed" | "error";
};

export type CitationContract = {
  label: string;
  document_id: string;
  version_id: string;
  chunk_id: string;
};

export type IngestionResponse = {
  document: DocumentContract;
  version: VersionContract;
  storage_uri: string;
  artifact_uri: string | null;
  extraction_status: "extracted" | "manual_review" | "failed";
  review_required: boolean;
  review_reason: string | null;
  blocks_count: number;
  text_length: number;
};

export type QueryResponse = {
  answer: string;
  citations: CitationContract[];
  confidence: number;
  applied_filters: Record<string, unknown>;
  rewrites_applied: string[];
  decision_trace: string[];
  fallback_reason: string | null;
  source_conflict: boolean;
};
