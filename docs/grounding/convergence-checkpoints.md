# Grounding decision checkpoints

The runtime keeps the exact loaded query and actual image evidence as the authority for the task. Its condition contract is a **model declaration**, not an object detector or proof that the image matches the query.

## Per-record progress

An ephemeral decision checkpoint accompanies the current-record context anchor and inspection/evidence receipts. It reports unresolved query excerpts, structural contract issues, declared candidate-set order, and inspections since the last declared condition change. Rewording facts, reasons, or evidence prose does not reset this counter. Candidate identity basis/status, requirement status, and declared ordering changes do. These counters do not measure semantic correctness and never increase confidence.

The model can decide directly from sufficiently clear images. View, compare, color, processing, and edge helpers are optional aids for a condition the model finds uncertain, not a mandatory pipeline. A reasonable trial can fail, return no match, expose contradictory evidence, or add nothing useful. None of these outcomes alone means the model used the tool incorrectly. A successful tool call or newly rendered image is likewise not proof of progress or correctness.

After a trial, assess what the actual observation changes about the candidate and remaining condition. Try another hypothesis, method, or tool when a concrete observable distinction justifies it. If the evidence supports a proposal, request human review. When no useful further observation is available, keep uncertainty in the best-supported proposal for review, or request clarification without inventing a box. There is no tool quota, ban on cross-tool exploration, or demand for a confident guess. An invisible or occluded body part cannot, by itself, support a requested pose.

### Optional trial feedback

`state.lastTrial` can carry a concise assessment of the preceding result:

- `question`: the uncertainty tested
- `outcome`: `useful`, `inconclusive`, `contradictory`, or `failed`
- `observation`: what was actually learned (or what failed)
- `remainingUnknown`: what still prevents a supported decision
- `nextObservation`: a useful next observable distinction, or `null` when none is available

Piggyback it on the next view, compare, color, processing, edge-refinement, or evidence call. It is optional; no extra evidence call, long pre-tool justification, or separate post-tool summary is required. A final assessment may remain in the ordinary proposal explanation. Updating this feedback does not itself resolve requirements or change candidate support: carry any justified changes in the existing contract/selection fields.

Receipts and checkpoints label the assessment as model-declared, not runtime verification. Inspection counts and unchanged declarations are diagnostic history, not a measure of usefulness. An explicit report of no useful next observation offers review or clarification; it does not impose a hard stop.

## Retained evidence reuse

A bounded history of 12 recent view/compare source actions detects immediate repeats and action cycles. Rephrased reasons, renamed comparison labels, and display zoom changes do not make those source actions novel. A repeat returns the prior view IDs without generating another image **only if the exact tool-produced image blocks and their ownership mapping are still present in the current projected context**. Missing, partial, deduplicated, corrupted, or explicitly archived evidence remains recoverable.

A distinct source region/sensor or changed overlay geometry remains available. A concrete boundary/part measurement, new visible counterevidence, or unavailable-evidence recovery can explicitly declare `inspectionIntent`. A different concrete observable for the same unresolved condition may justify a revisit; exact repeats still reuse retained pixels. The runtime cannot recognize semantic rephrasing, classify the truth of an explanation, or promise to prevent every model reasoning loop. Differently transformed views can aid inspection without creating new source evidence.

## Proposals are not verification

An unsaved preview returns `verification: "not_verified"`, an effective status/confidence, the original model proposal separately, and the current constraint assessment. Missing or contradicted conditions force the effective preview status to `unresolved` and cap its confidence at 0.49, matching the existing review boundary. Rendering a box successfully does not verify identity, rank, pose, or silhouette. Even a structurally supported contract is not verified visual truth.

No checkpoint approves, saves, advances, silently resolves a condition, changes source pixels, or persists task queries into learning. Human review remains mandatory. Arbitrary model narration is not rewritten or censored; the authoritative tool state remains explicit about uncertainty. Live model quality and latency still require independent measurement, rather than being inferred from these synthetic regressions.
