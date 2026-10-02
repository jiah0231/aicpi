# Grounding decision checkpoints

The runtime keeps the exact loaded query and actual image evidence as the authority for the task. Its condition contract is a **model declaration**, not an object detector or proof that the image matches the query.

## Per-record progress

An ephemeral decision checkpoint accompanies the current-record context anchor and inspection/evidence receipts. It reports unresolved query excerpts, structural contract issues, declared candidate-set order, and inspections since the last declared condition change. Rewording facts, reasons, or evidence prose does not reset this counter. Candidate geometry, identity basis/status, requirement status, and declared ordering changes do. These counters do not measure semantic correctness and never increase confidence.

The checkpoint asks for a concise decision: identify the remaining condition and an observable distinction that a useful next inspection could resolve, submit an unresolved low-confidence proposal for human review, or ask clarification and pause. It does not impose a view count or force a candidate. An invisible or occluded body part cannot, by itself, support a requested pose.

## Retained evidence reuse

A bounded history of 12 recent view/compare source actions detects immediate repeats and action cycles. Rephrased reasons, renamed comparison labels, and display zoom changes do not make those source actions novel. A repeat returns the prior view IDs without generating another image **only if the exact tool-produced image blocks and their ownership mapping are still present in the current projected context**. Missing, partial, deduplicated, corrupted, or explicitly archived evidence remains recoverable.

A distinct source region/sensor or changed overlay geometry remains available. A concrete boundary/part measurement, new visible counterevidence, or unavailable-evidence recovery can explicitly declare `inspectionIntent`. The runtime does not classify the truth of that explanation or promise to prevent every model reasoning loop. Differently transformed views can aid inspection without creating new source evidence.

## Proposals are not verification

An unsaved preview returns `verification: "not_verified"`, an effective status/confidence, the original model proposal separately, and the current constraint assessment. Missing or contradicted conditions force the effective preview status to `unresolved` and cap its confidence at 0.49, matching the existing review boundary. Rendering a box successfully does not verify identity, rank, pose, or silhouette. Even a structurally supported contract is not verified visual truth.

No checkpoint approves, saves, advances, silently resolves a condition, changes source pixels, or persists task queries into learning. Human review remains mandatory. Arbitrary model narration is not rewritten or censored; the authoritative tool state remains explicit about uncertainty. Live model quality and latency still require independent measurement, rather than being inferred from these synthetic regressions.
