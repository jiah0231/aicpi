# Grounding tools and review workflow

Every prediction still requires explicit browser approval before it is saved.
Rejection keeps the same record. Cropping and color analysis remain optional,
with no per-record crop quota. No detector, OCR model or extra vision model is used.

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

Changing display scale does not produce more source detail. For identity uncertainty, compare candidates;
for a known part's uncertain boundary, measure pixels. Avoid repeated threshold
changes merely to obtain a match. A no-match point selection does not prove
absence, and the largest matching component can be background.

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
