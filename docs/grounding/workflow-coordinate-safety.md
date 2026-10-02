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

## Unresolved referents and real pauses

- When no useful visual check remains and the query's referent is unclear,
  `grounding_evidence` accepts `clarification` with one concrete question. It
  displays the question once and ends the current run on the same unsaved key.
  No bbox or full contract is required, and no result is approved or advanced.
- The wait is persisted across restart. Repeated tools, internal follow-ups,
  and output-limit/settle recovery cannot bypass it. Actual interactive/RPC user
  input, including a short answer or image-only correction, releases the wait;
  that response never approves a box or loads the next record by itself.
- The completed-turn boundary also aborts the pending run. This is needed
  because the SDK honors tool-level `terminate` only when every result in a
  batch requests it; an earlier normal result or invalid sibling must not
  restart the search. Tool results and the question remain persisted. Provider
  cancellation uses the SDK abort signal; zero provider-adapter invocations
  after that signal have not been verified by execution.
- A useful best-supported proposal can still go directly to ordinary
  `unresolved` human review. Too few observed candidates does not make the last
  visible candidate the missing ordinal. Further inspection needs a concrete
  observed ambiguity, rather than a hypothetical hidden member.
- This adds an explicit exit, not an object recognizer or automatic semantic
  information-gain detector. It does not impose a crop-count limit or block
  useful refinements before the model requests clarification.
- A case-only `requirements[].queryText` error suggests the exact source span
  to copy into that field and says no new image call is needed. Validation
  remains case-sensitive; neither the query nor the supplied contract is
  silently rewritten. Noncontiguous or unrelated excerpts still fail.

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

Version 2 stores only newly human-authored **general procedures**: category,
applicability (4–240 characters), error (4–400), method (8–800), check (4–400),
and an explicit human confirmation that the text is sample-independent.

- No original or paraphrased sample query, image/path, sample ID, concrete
  answer, bbox or ground truth belongs in a procedure. There are no fields for
  these, nor automatic record ID, query, modality, review outcome or timestamp
- The optional form starts blank and disabled. Editing a procedure clears the
  sample-independence confirmation. Nothing is extracted or automatically
  “sanitized” from the current sample or review reason
- The server rejects extra/source fields and obvious URLs, paths, sample IDs,
  image payloads or coordinates. **Lexical checks cannot prove semantic
  independence of free text.** Human review remains necessary, including for
  paraphrases and concrete answers that lexical rules cannot recognize
- Query-overlap indexing and retrieval are removed. Retrieval receives no query
  or image context: it selects recent distinct general procedures with category
  diversity, default six and maximum twelve. These procedures are advisory,
  never answers or evidence about the current record
- The default store is now `~/.pi/agent/grounding-methods-v2.jsonl`. The old
  default file is left untouched and is not loaded. If a custom
  `PI_WEB_GROUNDING_LESSONS_PATH` contains mixed versions, every v1 row is
  excluded. Even a v2 row is excluded if it has unexpected source fields or
  lacks the human confirmation. No automatic migration or deletion occurs
- Appends preserve bytes and separate unterminated tails. The same-process,
  same-path write queue is retained; this is not a cross-process transaction or
  protection against concurrent external edits
- Changes affect future retrieval, not old contents already in a conversation.
  Historical lesson files and transcripts are not automatically erased.
  A procedure-write failure still does not block saving the human annotation

Regression cases accompany the changes. They were added and source-reviewed,
but no tests, lint, typecheck, build, app, model or dataset execution was performed
for this change.
