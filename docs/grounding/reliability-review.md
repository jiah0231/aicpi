# Grounding evidence and restart review

## What is checked

`grounding_evidence.state.contract` records the exact original query, all plausible readings, requirements quoted from that query, and model-declared candidate identities with source-normalized boxes. The runtime never rewrites the query to fit a newly discovered object.

Contracts are bounded to 32 KiB serialized UTF-8, and requirement excerpts to 600 characters. The exact original query is preserved separately; oversized input is rejected rather than truncated. Paginated status summaries do not repeat full saved contracts.

For an ordinal reading, declare `spatialOrder` with `axis`, `direction`, `ordinal`, `candidateIds`, `selectedCandidateId`, and evidence for `candidateSet`. Ascending x means left-to-right; descending x means right-to-left. The runtime sorts candidate centers in the common visible-source frame, independent of discovery order or comparison-panel order. It reports supported/possible counts, ties and uncertain ranks. Candidate membership, semantic direction, interpretation completeness and object identity remain model declarations, not detector results. The existing narrow, unambiguous ordinal hint also catches an omitted or changed ordinal declaration; it does not supply a general language parser.

A supported lock requires a consistent contract. Missing contracts, uncertain identities, conflicting requirements, ambiguous readings or rank mismatches defer a requested lock to `reconsidering`. This includes legacy notebook-only states, which remain loadable. Boundary refinement can retain the chosen candidate box while uncertainty stays explicit.

If the contract is incomplete, a best-supported tentative box can still reach mandatory human review. Review defaults become `unresolved`, confidence at most `0.49`, and the panel displays the unresolved checks, proposed confidence, candidate order and alternative readings. Raising status or confidence requires the reviewer to explicitly acknowledge resolving the listed issues. Approval still happens only through the human review response; opening the panel, rendering a crop or restoring a session cannot save a result. The saved progress entry retains the model contract and assessment separately from any human resolution.

Repeated views, magnification and color-only pixel measurements do not establish identity. Color measurement remains optional and is intended for an already visually identified target's boundary. There is no requirement to invoke it on each record. These checks cannot detect a model falsely claiming structural visual evidence or guarantee that an apparently supported box encloses the correct object.

### Proposal geometry advisory

The existing assessment, optional preview and human review expose `proposalGeometry`: how much of the declared selected rectangle the proposal covers, how much of the proposal lies inside it, and overlaps with other declared candidate rectangles. The review panel recomputes these numbers when the human edits the box. These are rectangle-area ratios, not foreground coverage or detected object counts. They never block legitimate occlusion, part refinement or approval, and introduce no required call or contract field. A wrong declared candidate box can match the proposal at 100%; geometry cannot detect a fence mislabeled as a bird or an omitted candidate.

Ordering must preserve the counting set in the original wording, separately from qualifiers describing the selected target. An explicitly filtered counting set is still respected. Use the retained overview to audit count/rank and identity, then check that the proposal encloses the requested silhouette without unintended neighbors or empty ground. Existing optional preview supports this audit; it is not a mandatory extra round.

## View coordinates

- Recall the same region: `viewId` without `region`
- Crop inside a displayed view: `viewId`, `region`, and explicit `coordinateSpace: view_normalized` or `view_pixels`
- New source-image crop: `region` and `zoom`, optionally `coordinateSpace: source`
- With both `viewId` and `region`, omitting the coordinate frame returns an actionable correction rather than guessing
- `bbox` on `grounding_view` always remains in source-normalized coordinates

Old view IDs remain invalid after a wrapper restart. Reload the pending record for new image views; direct source-coordinate submission is still allowed and conservatively reviewed.

## Durable progress

Session restoration, status inspection and direct save reconcile job counters from the persisted approved records belonging to the current dataset. A pending entry is not approval. A unique new human-approval ID also disambiguates a committed revision from an older saved box. A crash after committing progress but before clearing its pending entry cannot cause the same result to be saved again. A completed job no longer injects a “not started” continuation. Restoration and status check derived JSON/ZIP consistency without writing; if incomplete, they report a recovery warning. Calling `grounding_next_batch` regenerates only those owned artifacts from already approved rows, without another review or approval count.

Explicit `reviewSource: model` or `runtime_auto` legacy entries cannot count as human approvals. Resumption returns a clear error and leaves those files unchanged; no migration or relabeling is performed automatically. Older full-schema `reviewed: true` records with no `reviewSource` keep their existing compatibility, so that historical provenance is not independently reverified.

## User validation checklist

These changes were reviewed statically. Tests, lint, type checking, builds, the app, and dataset/model runs were not executed for this change.

1. With candidate x-ranges `[.18,.30]`, `[.455,.525]`, `[.68,.80]`, enter them in a different discovery order. Verify left-to-right rank 3 is the rightmost candidate, and choosing the middle candidate reports a mismatch.
2. Repeat with right-to-left Chinese-character selection; preserve unreadable identity/occlusion as unresolved. Repeatedly enlarge the same crop and verify it does not turn missing identity evidence into a supported lock or raise review confidence.
3. Use ambiguous wording such as “The second first alpaca from left to right”. Keep both readings visible. Verify no lexical default silently forces “first”; a tentative unresolved box remains reviewable.
4. Try to mark a notebook-only or unsupported candidate as locked. Verify `lockDeferred` and the detailed issues, then submit for review. Confirm that no output changes before actual approval, that rejection preserves the pending record, and that upgrading uncertainty needs explicit human acknowledgement.
5. Approve two of three records, restart with the third pending, and directly submit the third using source coordinates. After approval, check `processed=3`, `approvedInJob=3`, `requestedLimitReached=true`, and no “job not started” prompt or extra load.
6. Restore a stale pending entry after its result was already saved. Check that the saved box is unchanged, its approval is counted once, and correction requires reopening the record.
7. Try a view-local subcrop with `viewId + region + coordinateSpace`; check source mapping. Without the coordinate frame, verify that the error offers the three explicit recovery choices.
8. On a copy of an old run only, verify explicit model/auto provenance is rejected without rewriting its progress file. Do not modify real records merely to satisfy validation.

9. On copied outputs, remove/corrupt only the derived JSON/ZIP. Verify status reports the recovery warning without changing files, then `grounding_next_batch` repairs artifacts while approval counts and the progress ledger stay unchanged. Repeat with a completed correction whose final pending-clear entry was lost.
