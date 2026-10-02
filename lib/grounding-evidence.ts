import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { createHash } from "node:crypto";
import { validateGroundingConstraintContract, type GroundingConstraintContract } from "./grounding-constraints";

export type GroundingWorkingState = {
  target?: string;
  facts?: string[];
  hypotheses?: string[];
  openQuestions?: string[];
  ruledOut?: string[];
  contract?: GroundingConstraintContract;
  selection?: {
    status: "locked" | "reconsidering";
    bbox?: [number, number, number, number];
    evidence: string;
  };
};

export const GROUNDING_WORKING_STATE_LIMITS = {
  targetCharacters: 600,
  listItems: 8,
  itemCharacters: 400,
} as const;

export type GroundingEvidenceOptions = {
  active: boolean;
  pinnedViewIds: readonly string[];
  archivedViewIds: readonly string[];
  maxImageBase64Characters?: number;
  requiredViewIds?: readonly string[];
};

// Image ordinal, not content ordinal: text blocks may precede each image.
// A sheet is indivisible; all source-mapped panels belong to the same block.
export type GroundingEvidenceImageBlock = { imageIndex: number; viewIds: string[] };
export const GROUNDING_MAX_PINNED_VIEWS = 8;

/** Validate a concise evidence notebook, not a request for private reasoning. */
export function validateGroundingWorkingState(input: unknown, originalQuery?: string): GroundingWorkingState {
  if (input === null || typeof input !== "object" || Array.isArray(input)) {
    throw new Error("Grounding working state must be an object.");
  }
  const fields = input as Record<string, unknown>;
  const allowed = new Set(["target", "facts", "hypotheses", "openQuestions", "ruledOut", "selection", "contract"]);
  for (const key of Object.keys(fields)) {
    if (!allowed.has(key)) throw new Error(`Unknown grounding working state field: ${key}`);
  }

  const result: GroundingWorkingState = {};
  if (Object.hasOwn(fields, "contract")) {
    result.contract = validateGroundingConstraintContract(fields.contract, originalQuery);
  }
  if (Object.hasOwn(fields, "target")) {
    result.target = validateText(fields.target, "target", GROUNDING_WORKING_STATE_LIMITS.targetCharacters);
  }
  for (const key of ["facts", "hypotheses", "openQuestions", "ruledOut"] as const) {
    if (!Object.hasOwn(fields, key)) continue;
    const value = fields[key];
    if (!Array.isArray(value) || value.length > GROUNDING_WORKING_STATE_LIMITS.listItems) {
      throw new Error(`Grounding working state ${key} must be an array of at most ${GROUNDING_WORKING_STATE_LIMITS.listItems} strings.`);
    }
    result[key] = value.map((item, index) => validateText(
      item,
      `${key}[${index}]`,
      GROUNDING_WORKING_STATE_LIMITS.itemCharacters,
    ));
  }
  if (Object.hasOwn(fields, "selection")) {
    const selection = fields.selection;
    if (selection === null || typeof selection !== "object" || Array.isArray(selection)) {
      throw new Error("Grounding working state selection must be an object.");
    }
    const values = selection as Record<string, unknown>;
    for (const key of Object.keys(values)) {
      if (!["status", "bbox", "evidence"].includes(key)) {
        throw new Error(`Unknown grounding working state selection field: ${key}`);
      }
    }
    if (values.status !== "locked" && values.status !== "selected" && values.status !== "reconsidering") {
      throw new Error("Grounding working state selection status must be locked or reconsidering.");
    }
    const status = values.status === "selected" ? "locked" : values.status;
    const evidence = validateText(values.evidence, "selection.evidence", GROUNDING_WORKING_STATE_LIMITS.itemCharacters);
    if (evidence.length < 8) throw new Error("Grounding working state selection evidence must contain at least 8 characters.");
    let bbox: [number, number, number, number] | undefined;
    if (Object.hasOwn(values, "bbox")) {
      if (!Array.isArray(values.bbox) || values.bbox.length !== 4
        || values.bbox.some((edge) => typeof edge !== "number" || !Number.isFinite(edge))) {
        throw new Error("Grounding working state selection bbox must contain four finite numbers.");
      }
      bbox = [...values.bbox] as [number, number, number, number];
      if (bbox.some((edge) => edge < 0 || edge > 1) || bbox[0] >= bbox[2] || bbox[1] >= bbox[3]) {
        throw new Error("Grounding working state selection bbox must be a valid normalized box.");
      }
    }
    if (status === "locked" && !bbox) {
      throw new Error("Grounding working state locked selection requires bbox.");
    }
    result.selection = { status, ...(bbox ? { bbox } : {}), evidence };
  }
  return result;
}

function validateText(input: unknown, field: string, limit: number): string {
  if (typeof input !== "string" || input.length > limit) {
    throw new Error(`Grounding working state ${field} must be a string of at most ${limit} characters.`);
  }
  return input.trim();
}

function evidenceViewIds(details: unknown): string[] | undefined {
  if (details === null || typeof details !== "object" || Array.isArray(details)) return;
  const ids = (details as Record<string, unknown>).evidenceViewIds;
  if (!Array.isArray(ids) || ids.length === 0 || ids.some((id) => typeof id !== "string" || id.length === 0)) return;
  return ids as string[];
}

/** Resolve exact block ownership. Legacy results conservatively share all IDs. */
function imageViewIds(details: unknown, count: number): (string[] | undefined)[] {
  const fallback = Array.from({ length: count }, () => evidenceViewIds(details));
  if (!details || typeof details !== "object" || Array.isArray(details)) return fallback;
  const blocks = (details as Record<string, unknown>).evidenceImageBlocks;
  if (blocks === undefined) return fallback;
  // Malformed explicit mappings must never authorize dropping unknown pixels.
  const unknown = Array.from({ length: count }, () => undefined);
  if (!Array.isArray(blocks) || blocks.length !== count || count > 64) return unknown;
  const result: (string[] | undefined)[] = [...unknown];
  for (const block of blocks) {
    if (!block || typeof block !== "object" || !Number.isInteger(block.imageIndex)
      || block.imageIndex < 0 || block.imageIndex >= count || result[block.imageIndex]
      || !Array.isArray(block.viewIds) || !block.viewIds.length || block.viewIds.length > 64
      || block.viewIds.some((id: unknown) => typeof id !== "string" || !id)) return unknown;
    result[block.imageIndex] = block.viewIds;
  }
  return result;
}

/**
 * Project a bounded visual working set without modifying transcript or pixels.
 * This is an image-base64 allowance, NOT a serialized provider-body limit: text,
 * schemas, signed thinking and provider envelopes require a downstream guard.
 * Required evidence that cannot fit remains visible with an explicit warning.
 */
export function compactGroundingEvidence(
  messages: AgentMessage[],
  options: GroundingEvidenceOptions,
): AgentMessage[] {
  if (!options.active || (options.archivedViewIds.length === 0 && options.maxImageBase64Characters === undefined)) return messages;
  const archived = new Set(options.archivedViewIds);
  const required = new Set([...options.pinnedViewIds, ...(options.requiredViewIds ?? [])]);
  const budget = options.maxImageBase64Characters;
  const bounded = budget !== undefined && Number.isFinite(budget) && budget >= 0;
  type Entry = { message: number; block: number; ids: string[]; data: string; mimeType: string;
    required: boolean; removed?: "archived" | "budget" | "duplicate"; retained?: Entry };
  const entries: Entry[] = [];
  let total = 0;
  messages.forEach((message, index) => {
    if (!("content" in message) || !Array.isArray(message.content)) return;
    const count = message.content.filter((block) => block.type === "image").length;
    const owned = message.role === "toolResult" && message.toolName.startsWith("grounding_")
      ? imageViewIds(message.details, count) : [];
    let imageIndex = 0;
    message.content.forEach((block, blockIndex) => {
      if (block.type !== "image") return;
      total += block.data.length;
      const ids = owned[imageIndex++];
      if (!ids) return;
      const keep = ids.some((id) => required.has(id));
      const entry: Entry = { message: index, block: blockIndex, ids, data: block.data, mimeType: block.mimeType, required: keep };
      if (!keep && ids.every((id) => archived.has(id))) {
        entry.removed = "archived";
        total -= block.data.length;
      }
      entries.push(entry);
    });
  });
  if (bounded) {
    // Preserve every image in the latest observation, not an arbitrary last panel.
    const latest = entries.findLast((entry) => !entry.removed)?.message;
    for (const entry of entries) if (entry.message === latest && !entry.removed) entry.required = true;
    // Scope exact-byte dedup to this projection. A registry/cache hit alone is
    // never proof that pixels are still available to the model.
    const identical = new Map<string, Entry>();
    for (const entry of [...entries].reverse()) {
      if (entry.removed) continue;
      const signature = createHash("sha256").update(entry.mimeType).update("\0").update(entry.data).digest("hex");
      const retained = identical.get(signature);
      if (retained && retained.data === entry.data && retained.mimeType === entry.mimeType) {
        retained.required ||= entry.required;
        entry.retained = retained;
        entry.removed = "duplicate";
        total -= entry.data.length;
      } else identical.set(signature, entry);
    }
    for (const entry of entries) {
      if (total <= budget!) break;
      if (entry.removed || entry.required) continue;
      entry.removed = "budget";
      total -= entry.data.length;
    }
  }
  const overflow = bounded && total > budget!;
  const warningIndex = overflow ? messages.findLastIndex((message) => message.role === "toolResult" && message.toolName.startsWith("grounding_")) : -1;
  const byMessage = new Map<number, Entry[]>();
  for (const entry of entries) {
    if (!entry.removed) continue;
    const group = byMessage.get(entry.message) ?? [];
    group.push(entry);
    byMessage.set(entry.message, group);
  }
  return messages.map((message, index) => {
    const removed = byMessage.get(index) ?? [];
    if (message.role !== "toolResult" || (!removed.length && index !== warningIndex)) return message;
    const blockIndexes = new Set(removed.map((entry) => entry.block));
    const content = message.content.filter((_block, blockIndex) => !blockIndexes.has(blockIndex));
    const notices = removed.flatMap((entry) => {
      if (entry.removed === "archived") return [];
      if (entry.removed === "duplicate" && entry.retained && !entry.retained.removed) {
        const target = messages[entry.retained.message];
        return [`Identical image pixels for ${entry.ids.join(", ")} remain in tool result ${target.role === "toolResult" ? target.toolCallId : entry.retained.message} (${entry.retained.ids.join(", ")}); reuse those pixels. Source mappings above remain unchanged.`];
      }
      return [`Grounding image omitted by the visual working-set request-size limit: ${entry.ids.join(", ")}. Its pixels are unavailable in this request; saved text is not visual proof. Do not reopen it merely to repeat an already resolved check. Preserve uncertainty if required evidence is missing.`];
    });
    if (index === warningIndex) notices.push(`Grounding evidence capacity warning: retained image base64 uses ${total} characters, above the ${budget} image allowance. Required/current/original evidence and unowned images were not silently discarded or resized. This request may still exceed the provider limit (text, tools and envelopes are additional). Reduce the working set explicitly or use a larger-capacity provider; do not claim lossless budget compliance.`);
    if (notices.length) content.push({ type: "text", text: notices.join("\n") });
    if (!content.length) content.push({ type: "text", text: `Grounding evidence images archived from active context: ${removed.flatMap((entry) => entry.ids).join(", ")}.` });
    return { ...message, content };
  });
}
