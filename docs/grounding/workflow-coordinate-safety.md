# Grounding workflow and coordinate safety

These changes improve tool handling and human review. They do not recognize
objects, prove a model's evidence declarations, or establish an accuracy gain.

## Fewer repair calls

- A clean crop needs `region` and `reason`; `zoom` is optional and automatically
  fits a new crop. Recalling an unchanged `viewId` retains its display scale.
- `region` changes the viewport, not the target. For compatibility, a
  `bbox` + `zoom` call without `region` is interpreted as **crop-only**, with a
  returned normalization notice. Use both `region` and `bbox` when both a crop
  and a new hypothesis are intended. Bare `bbox` remains a source hypothesis.
- A save call may include `contract` directly. Existing `state.contract` remains
  reusable. Omitting `originalQuery` uses the exact loaded query; a supplied
  mismatch is rejected. Put query excerpts in `requirements[].queryText`, not
  on the interpretation. The final expanded contract is still bounded to 32 KiB,
  with a 12,000-character original query and 600-character query excerpts.
- Notebook updates remain partial: omitted facts, selection and contract stay
  intact. Unsupported claims remain unresolved; no extra crop or color call is
  required to submit an uncertain result for human review.

## Coordinates and approval

- For a questionable or tiny proposal, either save tool accepts optional
  `previewOnly: true`: it returns the proposed overlay plus clean contextual
  pixels to the model using the same coordinate mapping as a real save. It
  opens no review dialog and does not persist evidence, save or advance. Inspect
  those pixels, then submit without the flag. This is optional and does not
  automatically recognize the object or certify the box
- Prefer an actual returned visible `viewId` with `view_pixels` or
  `view_normalized`. Comparison coordinates refer to the whole canvas inside
  the selected panel. Stale-ID errors list current IDs for recovery.
- Save and visible color sampling reject infrared/depth view IDs, even when
  image dimensions match. There is no supported registered sensor transform.
  Infrared/depth views remain available as evidence in their own coordinates;
  their hypotheses cannot overwrite the visible review box.
- Legacy `last_crop` saves require the latest crop to be visible. Otherwise use
  an earlier visible view ID or remeasure in visible-source coordinates.
- Small/thin boxes receive clean visible-source pixels with context **before**
  approval. The live outline follows edits; hiding it exposes underlying pixels.
  The crop stays fixed and warns if edits extend beyond it. The overview remains
  available. This preview is for human inspection, not automatic verification.
- Check all four edges and the complete requested silhouette, including faint
  outlines and protruding parts. Every result still requires human approval;
  rejection never advances the record. Waiting for review is not inference time.

## Human lesson limits

Advice remains explicitly human-authored and advisory. It never overrides the
current query, images or safety rules and does not inject old boxes.

- Mixed Han/digit/Latin queries retain reusable Han bigrams and word tokens
- Default retrieval returns at most six deduplicated lessons: up to two global
  slots are reserved when available, with the rest available to relevant similar
  lessons. Globals may fill otherwise empty slots; the configurable maximum is 12
- Persisted query context is capped at 1,000 characters; advice is 8–1,200
  characters. This retrieval limit does not truncate the current task's query
- Appends preserve existing bytes and separate valid or damaged unterminated
  tails. The existing same-process, same-path write queue remains; this is not a
  cross-process transaction or protection against a concurrent external editor
- Editing/removing lessons affects future retrieval, not content already present
  in a model context. Learning-write failure does not block the annotation

Regression cases accompany the changes. They were added and source-reviewed,
but no tests, lint, typecheck, build, app, model or dataset execution was performed
for this change.
