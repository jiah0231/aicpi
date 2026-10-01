import type { AgentMessage } from "@earendil-works/pi-agent-core";
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
};

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

/**
 * Build provider context without modifying the transcript. Explicit model
 * archival removes images first. When the remaining grounding images exceed a
 * transport budget, older unpinned view results are omitted until the request
 * fits; the newest such result is retained so the model can inspect the pixels
 * it just requested. A result containing multiple views is handled as one unit,
 * because its image blocks need not correspond one-to-one with its view IDs.
 * Pinned views and unmarked overview/load images always remain available.
 *
 * Keep every message and every non-image block in order. In particular signed
 * thinking blocks, sibling tool calls/results, and user corrections are untouched.
 */
export function compactGroundingEvidence(
  messages: AgentMessage[],
  options: GroundingEvidenceOptions,
): AgentMessage[] {
  if (!options.active || (options.archivedViewIds.length === 0 && options.maxImageBase64Characters === undefined)) return messages;
  const archived = new Set(options.archivedViewIds);
  const pinned = new Set(options.pinnedViewIds);

  const compacted = messages.map((message) => {
    if (message.role !== "toolResult" || !message.toolName.startsWith("grounding_")) return message;
    const ids = evidenceViewIds(message.details);
    if (!ids || ids.some((id) => !archived.has(id) || pinned.has(id))) return message;
    if (!message.content.some((block) => block.type === "image")) return message;

    const content = message.content.filter((block) => block.type !== "image");
    // Even an image-only result must remain a nonempty result for its tool call.
    if (content.length === 0) {
      content.push({ type: "text", text: `Grounding evidence images archived from active context: ${ids.join(", ")}.` });
    }
    return { ...message, content };
  });

  const budget = options.maxImageBase64Characters;
  if (budget === undefined || !Number.isFinite(budget) || budget < 0) return compacted;
  const imageCharacters = (message: AgentMessage) => message.role === "toolResult"
    ? message.content.reduce((sum, block) => sum + (block.type === "image" ? block.data.length : 0), 0)
    : 0;
  let total = compacted.reduce((sum, message) => sum + imageCharacters(message), 0);
  if (total <= budget) return compacted;

  const candidates = compacted.flatMap((message, index) => {
    if (message.role !== "toolResult" || !message.toolName.startsWith("grounding_")) return [];
    const ids = evidenceViewIds(message.details);
    const size = imageCharacters(message);
    if (!ids || size === 0 || ids.some((id) => pinned.has(id))) return [];
    return [{ index, ids, size }];
  });
  // The latest view is the model's current observation. Older views are safe
  // to request again from the registry if the concise evidence state proves
  // insufficient later.
  const removable = candidates.slice(0, -1);
  if (removable.length === 0) return compacted;
  const bounded = [...compacted];
  for (const candidate of removable) {
    if (total <= budget) break;
    const message = bounded[candidate.index];
    if (message.role !== "toolResult") continue;
    const content = message.content.filter((block) => block.type !== "image");
    content.push({ type: "text", text: `Earlier grounding evidence image omitted from active model context to stay within the request-size limit: ${candidate.ids.join(", ")}. Reopen that view if its pixels are still needed.` });
    bounded[candidate.index] = { ...message, content };
    total -= candidate.size;
  }
  return bounded;
}
