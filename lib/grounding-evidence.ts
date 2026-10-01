import type { AgentMessage } from "@earendil-works/pi-agent-core";

export type GroundingWorkingState = {
  target?: string;
  facts?: string[];
  hypotheses?: string[];
  openQuestions?: string[];
  ruledOut?: string[];
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
};

/** Validate a concise evidence notebook, not a request for private reasoning. */
export function validateGroundingWorkingState(input: unknown): GroundingWorkingState {
  if (input === null || typeof input !== "object" || Array.isArray(input)) {
    throw new Error("Grounding working state must be an object.");
  }
  const fields = input as Record<string, unknown>;
  const allowed = new Set(["target", "facts", "hypotheses", "openQuestions", "ruledOut", "selection"]);
  for (const key of Object.keys(fields)) {
    if (!allowed.has(key)) throw new Error(`Unknown grounding working state field: ${key}`);
  }

  const result: GroundingWorkingState = {};
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
 * Build provider context without modifying the transcript. Only explicit model
 * archival removes images; no image count, age, zoom, or recency policy applies.
 * A result containing multiple views is retained until all are archived, because
 * its image blocks need not correspond one-to-one with its view IDs. Pinned views
 * always win. Unmarked overview/load images therefore remain available.
 *
 * Keep every message and every non-image block in order. In particular signed
 * thinking blocks, sibling tool calls/results, and user corrections are untouched.
 */
export function compactGroundingEvidence(
  messages: AgentMessage[],
  options: GroundingEvidenceOptions,
): AgentMessage[] {
  if (!options.active || options.archivedViewIds.length === 0) return messages;
  const archived = new Set(options.archivedViewIds);
  const pinned = new Set(options.pinnedViewIds);

  return messages.map((message) => {
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
}
