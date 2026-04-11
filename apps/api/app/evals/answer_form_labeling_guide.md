## Answer Form Labeling Guide

These seed files are a lightweight gold-label template for the next answer layer:

- `direct_unit`: the correct answer is mainly the identity of one canonical unit.
- `unit_set`: the correct answer is a set of canonical units.
- `rule_summary`: the correct answer is a substantive rule or obligation, usually grounded in one main unit.
- `global_summary`: the correct answer is a high-level overview across a broader block.

### How to review

1. Confirm or edit `suggested_answer_form`.
2. Confirm or edit `suggested_primary_unit` when one unit is clearly central.
3. Confirm or edit `suggested_expected_units` when the answer should enumerate multiple canonical units.
4. Leave `review_status` as `needs_review` until you are happy with the suggestion.

### Mapping from current strategy fixtures

- `strategy = focal` usually maps to `rule_summary`.
- `strategy = multi_branch` usually maps to `unit_set`.
- `strategy = global` usually maps to `global_summary`.

Exceptions are expected:

- some focal questions are really `direct_unit`
- some multi-branch questions may collapse to one dominant section plus subunits

### Review priority

Review these first because they are the most useful for the next implementation step:

- direct-unit cases added from recent manual tests
- legal `unit_set` cases around indemnizacion laboral
- NIIF cases whose correct answer depends on section or sub-section identity
