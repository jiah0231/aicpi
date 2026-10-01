# Grounding tools and review workflow

Every prediction still requires explicit browser approval before it is saved.
Rejection keeps the same record. Cropping and color analysis remain optional,
with no per-record crop quota. No detector, OCR model or extra vision model is used.

## Preserve the requested target

The original query and the user's task requirements remain the target throughout
observation, candidate changes and review. A new object, a convenient color mask
or another tool result must not silently replace the requested object or drop a
required attribute, relation or order. Keep those requirements in the evidence
state's `target`; changing candidate interpretations belong in `hypotheses`.

Before submitting a box, reconcile the proposal with the requirements actually
present in the request and explain the supporting evidence. Measuring a region
only establishes pixel bounds, not that it satisfies the query. If a required
condition is unestablished or contradicted, preserve that uncertainty, explain
what is missing and submit the best-supported candidate as `unresolved` with
low confidence. Do not reinterpret the query to fit the easiest candidate.
This is a reasoning obligation, not a fixed list of extra tool calls; a clear
case can proceed directly to human review.

View/crop, comparison, color-analysis and evidence responses repeat `originalQuery`
from the loaded source record with a short `taskReminder`. This stays separate
from the model's editable working state, so a changed hypothesis cannot rewrite
the reminder. The unsaved-record nudge also repeats the source query and allows
an unresolved submission. These reminders reinforce the task; they do not
automatically prove candidate identity or query satisfaction.

## Job scope and continuation

Start with `grounding_next_batch` and provide `queryPath`, a dedicated `outputDir`,
and `targetCount` for the number of additional records the user requested.
`limit` remains 1; it is not the job size. The job target, initial approved
count, pending key, source-space working box, evidence state, and revision flag
are persisted in the session's version 2 `grounding:job` custom entry. Version 1
entries remain readable. Restoring a wrapper resumes the pending key before any
other unfinished key rather than treating previous approvals as new work.
Wrapper-local view IDs are deliberately not persisted; obtain fresh views after
a restart, or submit an already established box in `source` coordinates.

Use `grounding_save_and_next` between records. The last record or a revision
can use `grounding_save_result`. An intermediate `save_result` in a known job
returns `nextAction: "grounding_next_batch"` and does not terminate the batch.
Both save paths stop at the requested count. A later request such as “接下来只处理
1 条” while a record is loaded preserves that record and changes the remaining
scope. A revision never automatically advances to another record.

`grounding_status` returns the current key, last approved key, revision key,
review state, target count and approved count for that job. Inspecting another
dataset or output directory does not switch the active job. Its approved boxes
are this run's predictions, not reference annotations.

## Clean views and comparisons

`grounding_view` accepts an optional working `bbox`. A target does not need to
be guessed before its region can be inspected. Set `decorations` to `none`,
`grid`, `hypothesis`, or `all` (default). `none` removes the grid, frame and labels.
Crops accept a model-selected `region` and `zoom`; a large zoom is fitted to the
display limit while retaining all requested source pixels. Requesting a first
crop of another modality also supplies its full overview in that call.

Every view and comparison panel can include `sourceReuse`. It classifies exact,
contained, and near-duplicate source coverage, with covered fraction, new source
pixels, IoU, magnification, and display-scale ratio. A full-image overview does
not make the first useful local crop redundant. `decisionCheckpoint` asks the
agent to name the unresolved question before it repeats substantially the same
source pixels. This feedback is advisory and does not block a needed view.

`grounding_compare` accepts 1–4 labeled source-normalized regions per call and
returns one image containing the full overview and A/B/C/D detail panels.
The limit bounds one payload, not total comparisons. Labels are outside the
image content. Different panel display scales must not be interpreted as
different object sizes; use the overview and source regions for those relations.

Example tool arguments (coordinates are illustrative, not predictions):

```json
{
  "regions": [
    { "label": "left candidate", "region": [0.2, 0.3, 0.4, 0.9] },
    { "label": "center candidate", "region": [0.5, 0.3, 0.75, 0.9] }
  ],
  "reason": "Compare candidate bodies and heads to resolve target identity."
}
```

### Ordinal targets and cross-modal evidence

For queries such as "third from the left", first establish which candidates
match the requested object description, then sort their positions along the
specified axis and direction in a common source-image frame. Recompute spatial
order whenever a candidate is added, removed or reidentified. Discovery order
and comparison panel labels are not rank. For example, if three verified objects
have horizontal centers at 0.24, 0.49 and 0.74, their left-to-right ranks are
1, 2 and 3 even if the middle object was discovered last. Do not invent or promote
an uncertain object just to satisfy the requested numeral.

Visible, infrared and depth images with the same dimensions are not necessarily
spatially registered. Establish object correspondence and alignment before
transferring an infrared/depth box into the visible image used for review.
`viewId` coordinate mapping converts a display location to its own modality's
source frame; it does not align sensors. Conflicting positions or structure are
unresolved evidence, not confirmation that both views show the same target.

When identity, the requested rank or cross-modal correspondence remains
unresolved, submit the best-supported candidate as `unresolved` with low
confidence and explain what is missing. A measured box does not justify `ok`.
For ordinal targets, the review reason should explain the supported candidate
count and spatial order so the user can check the interpretation.

## Stable coordinates

Each returned view has an ID valid for its loaded record. Save tools accept:

| coordinateSpace | Meaning |
|---|---|
| `source` | Full source-image normalized pixel edges, default |
| `view_pixels` | Pixel edges on the display canvas identified by `viewId` |
| `view_normalized` | Normalized edges on that same entire display canvas |
| `last_crop` | Legacy normalized coordinates of the latest ordinary crop |

For a comparison, use the chosen panel's `id` as `viewId`; coordinates refer to
the entire comparison canvas, not a locally numbered panel. The box must lie
inside that panel's `displayRect`. Labels/padding and IDs from another record
are rejected. Later crops do not change an older view's mapping. The resulting
source box goes to the ordinary human review, which saves the approved edges.

Pass an earlier `viewId` to `grounding_view` to render its source region again.
It returns a new ID describing the new display. EXIF orientation is removed
before full-image resizing so full images, crops, color measurements and saved
coordinates consistently use raw source pixel axes. Pixel-edge roundoff is
snapped before crop extraction, preventing tiny recalled crops from expanding.

## Working evidence and color measurements

`grounding_evidence` lists view IDs and accepts `pin`, `unpin`, `archive`,
`restore`, and a concise `state` with `target`, `facts`, `hypotheses`,
`openQuestions`, and `ruledOut`. `facts` contain direct visible observations.
Object and body-part interpretations remain in `hypotheses` until visible
structure supports them; a color match cannot promote one to a fact. The model
decides which evidence is superseded; there is no automatic
last-N deletion. Only explicitly archived image blocks leave later provider
input. Original full-record evidence, user corrections, signed thinking and
tool-call/result pairs remain. A multi-panel image is omitted only when all
of its view IDs are archived; pinning takes precedence. The on-disk transcript
is not rewritten. View IDs and active evidence selections are wrapper-local;
the concise evidence state and current box survive a wrapper restart.

Changing display scale does not produce more source detail. First inspect the
target's shape, structure and surrounding context. If identity remains uncertain,
use the existing evidence or `grounding_view` / `grounding_compare` as needed;
color matching cannot resolve object or body-part identity on its own.

`grounding_color_region` is optional, not a stage to run for every record, colored
target, correction, or difficult example. Consider it only when an identified
target or part has reliable local color contrast and measuring its matching
pixels would help answer a remaining boundary question. If the box is already
clear or color is not discriminative, skip color analysis and use the visual
evidence to submit the box for review. Unresolved alternatives and low confidence
are valid; no color-derived box needs to be forced.

The tool always samples the original **visible** image. Infrared/depth colors may
be display palettes, not the target's visible color. A `viewId` maps coordinates
only; it neither switches the sampled modality nor proves cross-modal alignment.
Lighting, shadows, reflections, similarly colored neighbors, low resolution and
occlusion can merge, fragment or hide the relevant pixels. A mask measures only
the pixels that match; its bounds are not necessarily the complete target's
bounds and cannot recover hidden boundaries.

For example, a visually identified red label against a clearly different local
background may benefit from color measurement if its edges are hard to place.
A dark bird's beak beside a shadow, several similar-colored objects, or a target
identified mainly through an infrared signature should not trigger color analysis
by default. Inspect structure and the relevant modality instead.

When color measurement is useful, choose a small ROI with enough context to check
the boundary. Inspect the clean image, mask, point sample and selection assessment
before using any measured box. `largest` selects the largest matching component,
not the most likely target; black pixels can be background or shadow. `point`
confirms pixel membership, not identity, and a no-match point does not prove
target absence. Use `all` only when visible evidence supports grouping the
separated matching parts. A mask filling most of the ROI or reaching several
edges calls for checking background/clipping, not increased confidence. Avoid
repeated threshold changes merely to obtain a match, and discard a misleading
mask rather than treating it as stronger evidence than the image.

`grounding_color_region` accepts `source`, `view_pixels`, and `view_normalized`
coordinates for both the ROI and optional point. It returns `pointSample` with
source coordinates, RGBA, hex, HSV, match status, and retained component ID.
`selectionAssessment` explicitly marks the result as a pixel measurement that
does not establish object identity, and reports whether the ROI is clipped, the
component is tiny, or the selection was measured. `touchesRoiEdges` is distinct
from `touchesSourceEdges`; `clippedRoiEdges` contains only artificial ROI edges
that can be expanded. Source edges have no further pixels.

## Review, drift, and exact prediction edges

Every save call requires a short reason based on visible structural evidence.
The review compares the proposed box with the previous working box. A material
change reports IoU, center movement in source pixels, edge movement, and area
ratio, and the review overlay shows the previous box in cyan and the proposal in
orange. The agent should explain the new evidence before submitting a large
move, especially when identity was already established.

The reviewed box is the prediction. It is saved exactly as approved, including
small boxes; no fixed padding or calibration expands it. A magnified verification
crop may include context around a small approved box, but that preview never
changes the prediction coordinates.

## Verification

The grounding test suite covers mandatory review/cancellation, changed job
counts, persistence and job isolation, EXIF consistency, exact crop replay,
view/panel coordinate mapping, color membership and edges, and safe evidence
archival. Real-model speed and localization quality need a separate comparison
with human-reviewed outcomes; synthetic tests do not establish those gains.

No click-to-identify UI or categorized rejection controls were added.
