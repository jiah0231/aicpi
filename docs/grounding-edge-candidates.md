# Optional edge-assisted candidate boxes

`grounding_refine_box` takes a full visible-source normalized `region` containing
`coarseBox`, a concrete boundary `reason`, optional `point`, and optional
`lowThreshold`/`highThreshold` (defaults 20/50; 1 <= low < high <= 255).
This first version accepts source coordinates only. It rejects viewId and
coordinateSpace inputs rather than mistaking display or sensor coordinates for
source positions.

It applies fixed mild 3x3 binomial blur, signed Sobel gradients, directional
nonmaximum suppression and hysteresis, then bounds connected edge components.
These are edge candidates, not necessarily closed contours, filled segmentation
or complete object bounds. An optional point only filters component bounding
rectangles; it is not proven foreground membership. No pretrained model,
additional dependency, remote image upload or file mutation is involved.

Up to three candidates are ranked by agreement with the coarse box (IoU), with
a penalty for touching the analysis boundary. `rankScore` is a geometric ordering
heuristic, never confidence. No largest-component choice, selected identity,
merged detached part, changed lock or saved annotation is produced.

The sheet contains clean original pixels, derived edge pixels, and a labeled
candidate overlay (E1 orange, E2 blue, E3 purple; dashed white coarse box).
Every panel has its own view ID and full-canvas displayRect mapped to the exact
source ROI. Derived panels are measurement_only and cannot establish identity;
recalling them via grounding_view returns original source pixels. All proposals
still go through ordinary human review. Review waits, clarification pauses,
record ownership and locked-target checks remain active.

## Bounds and caveats

- Source: one still image, <=64 MB /100 million pixels; ROI <=1 million pixels,
  at least 7x7. Analysis never silently downsamples; only previews are resized
- Up to 4096 components and 3 retained candidates. A component-limit hit is
  explicitly incomplete and scanning-order biased
- CPU loops yield to the event loop and check AbortSignal. Typed arrays and
  fixed-size pixel queues avoid per-pixel objects; native image operations are
  bounded by source/ROI limits but are not forcibly interrupted mid-operation
- Three ROI-border pixels are excluded. Border contact means a wider ROI may
  be needed. Alpha-affected neighborhoods are excluded; display flattens on white
- Blur or weak contrast can remove fine tails; texture/shadows/contact can join
  neighboring structures. Open boundaries and disconnected protrusions remain
  unresolved. No morphology or automatic bridging invents a complete silhouette
- Verify all candidates against the original, especially tails, labels and
  protrusions. Larger context may be needed to notice detached parts. Geometry
  cannot establish target identity or recover occluded boundaries

Regression sources were added but not executed. No tests, lint, typecheck,
build, app, model or dataset run was performed; no accuracy or speed claim is
established by source inspection.
