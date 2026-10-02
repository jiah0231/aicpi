# Explicit grounding generation budget

Grounding has no default reduced output budget. The selected model and UI reasoning level stay unchanged, including for difficult identity/pose cases.

To request a per-generation output limit for one user turn, add exactly one marker to the request:

```text
Process the next grounding record. [grounding-generation:max-output-tokens=8192]
```

`8192` is an example, not a recommendation or default. This is a **per-generation** provider output-token request, not a total task budget or a wall-clock timeout. Some providers count reasoning in this budget; provider behavior varies. A gateway can ignore the setting. A visible notice says the budget was requested, not that provider enforcement has been verified. Existing request-control diagnostics can record the actual modified payload's supported fields.

The marker applies only to an active grounding batch. It is read only from the current user request, stripped before grounding prompt classification, never restored from history, and cleared on a new user turn or session load. An ordinary chat is unaffected. The same marker must be included again to use the budget on the next user turn.

## Supported requests and failure behavior

The runtime only lowers positive existing `max_tokens`, `max_completion_tokens`, or `max_output_tokens` fields. It does not invent a field for an unknown provider, increase an existing lower limit, change the selected model, lower reasoning, or change a thinking budget. Requests with no recognized limit or a conflicting explicit thinking budget are visibly blocked. Increase/remove the marker or configure a supported provider; an unsupported budget is not silently ignored.

When the provider ends a budgeted generation with `stopReason=length`:

- The run is aborted immediately and a visible `Grounding generation blocked` message is added
- SDK 0.87.1 rejects every tool call from that truncated response, including arguments which its salvage parser could make look valid
- Automatic settle continuation, retry, and automatic compaction are suppressed for the blocked run; manual compaction remains available
- A partial answer is not converted into an approved result or a successful grounding decision
- The current record stays pending unless an earlier, fully completed tool call already saved it

Send an explicit new request to continue, with a larger budget or without the marker for the full existing model/reasoning behavior. Existing completed progress is preserved. Removing the marker removes the budget; it is not a hidden model switch. Review and evidence requirements are unchanged.

## Verification boundaries

Unit tests cover parsing, supported/unsupported payloads, preserving reasoning, stale-run isolation, cancellation and repeat/resume behavior. Installed SDK tests verify that truncated tools never execute, post-run retries are skipped, and the OpenAI-compatible transport makes no fetch after a pre-request abort. These are local no-network tests, not measured provider latency or proof that a remote gateway enforces the budget. Live latency and grounding accuracy must be evaluated separately with authorized provider calls and reviewed outcomes.
