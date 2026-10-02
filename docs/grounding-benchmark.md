# Grounding performance and visual audit

There is no ground truth for this workspace. Historical `bbox`, `expected.json`,
approved outputs and agreement with another model are **not** accuracy labels.
Evaluate correctness by inspecting the source image, query and proposed box.
Separate visible evidence from an uncertain interpretation; do not publish a
numerical accuracy rate until the individual visual audits support it.

## Run a small live batch

Reuse the running Web on port 30141. This command makes real model requests
using its current default model, without changing provider/model settings:

```powershell
node e2e/grounding-benchmark.mjs --dataset D:\Desktop\aicgrounding\datasets\source\hard_cases_originals\queries.json --keys 001772_006,019071_001,015541_005 --cwd D:\Desktop\aicgrounding --fixture-root D:\Desktop\aicgrounding\datasets\fixtures\benchmark\NEW-RUN --out D:\Desktop\aicgrounding\outputs\NEW-RUN --variant baseline --timeout-ms 180000 --expected-model linuxdo/mimo-v2.5
```

Choose explicit keys (at most 12), never the entire official dataset. Both run
directories must be new. Each selected record gets an isolated, annotation-free
fixture and session. Only `query` and absolute source image paths enter the
fixture; the benchmark ignores annotation fields even in its own evaluator.

The harness connects SSE before sending a prompt, captures the first valid
review proposal, then **aborts its own session without approving it**. It checks
that the wrapper has shut down and no progress/prediction files were saved.
Errors, timeouts and missing proposals remain in the results. It must never
approve or reject a candidate on the human's behalf.

Benchmark prompts also opt into the grounding extension's request-control
diagnostic. Each case's `result.json` records only allow-listed model identity
and thinking controls, model output/context limits, and serialized output and
thinking budgets observed at the provider-request hook. Later extensions or an
upstream proxy can still change the request. It does
not record request messages, URLs, headers or credentials. Compare these
`requestControls` with returned `thinkingBlocks`; a UI thinking level alone does
not prove that a provider honored the serialized control. `off` is a requested
level, not verified upstream behavior. Missing fields mean that no recognized
control was observed, not that reasoning is disabled or generation is unlimited.
The diagnostics remain opt-in per marked turn; they never change model settings
or generation budgets. Token counts must be integers in 0–1,000,000,000 (output
and context limits must be positive); this is a validation bound, not a runtime cap.

`focused` is an experimental extra-prompt variant, not a recommended default.
Compare the same model, cases, deadline and runtime revision. Sequential runs
avoid competing requests, but provider/network variation still needs repeats;
a single faster run does not establish a general improvement.

## Inspect the proposals

```powershell
node e2e/grounding-benchmark-review.mjs D:\Desktop\aicgrounding\outputs\NEW-RUN
```

Inspect `proposal-full.png` and `proposal-crop-clean.png` before judging the
yellow candidate in `proposal-crop.png`. Magnifying pixels adds no new detail.
Audit these separately:

- Object identity and the required owner/part.
- Required rank, direction and spatial relation in the full scene.
- Complete visible target boundaries, excluding adjacent objects/background.
- Visible/thermal/depth correspondence; equal dimensions do not prove alignment.
- What remains uncertain and whether the proposed confidence is justified.

Record the finding as an **assistant visual audit**, not a human approval or
ground truth. Suggested outcomes: supported, boundary-needs-adjustment,
wrong-target, uncertain, or no-proposal. Retain query ambiguity rather than
inventing a definite target to pass the test.

## Interpret performance honestly

`elapsedMs` measures prompt admission to first proposal/terminal condition;
`setupMs` measures session creation separately. Tool elapsed time is measured
from observed tool events. `nonToolElapsedMs` includes model work, networking
and scheduling—it is not pure inference latency. Token totals include only
completed responses observed before stopping; a timed-out request may have
unreported usage. OpenAI-compatible `completion_tokens` includes reasoning tokens;
output usage is therefore not necessarily visible answer length. A zero provider-reported cost is not proof of a free request.

The initial `benchmark-baseline-20261002-001` artifact predates the no-ground-truth
clarification. Its reference/IoU fields are withdrawn and must not be used;
only its original images, proposals and measured runtime events remain useful.
