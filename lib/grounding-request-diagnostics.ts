/** Explicit, per-user-turn opt-in; this is never restored from session history. */
export const GROUNDING_REQUEST_DIAGNOSTICS_MARKER = "[grounding-benchmark:request-controls]";
export const GROUNDING_REQUEST_DIAGNOSTICS_ENTRY = "grounding:request-controls";

const THINKING_FORMATS = new Set([
  "openai", "openrouter", "deepseek", "together", "baseten", "zai", "qwen",
  "chat-template", "qwen-chat-template", "string-thinking", "ant-ling",
]);
const THINKING_TYPES = new Set(["enabled", "disabled", "adaptive"]);
const REASONING_EFFORTS = new Set(["none", "off", "minimal", "low", "medium", "high", "xhigh", "max"]);

export interface GroundingRequestControls {
  version: 1;
  model: { provider?: string; id?: string; reasoning?: boolean; thinkingFormat?: string };
  controls: {
    thinking?: { type: string } | string;
    reasoning_effort?: string;
    enable_thinking?: boolean;
  };
}

export function hasGroundingRequestDiagnosticsMarker(prompt: string): boolean {
  return prompt.includes(GROUNDING_REQUEST_DIAGNOSTICS_MARKER);
}

/** The diagnostic marker alone must not turn an ordinary chat into grounding. */
export function stripGroundingRequestDiagnosticsMarker(prompt: string): string {
  return prompt.replaceAll(GROUNDING_REQUEST_DIAGNOSTICS_MARKER, "");
}

// Read only named, own data properties. Do not enumerate a request, invoke a
// getter/toJSON, or retain references to model/request objects in the entry.
function ownValue(value: unknown, key: string): unknown {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  try {
    const descriptor = Object.getOwnPropertyDescriptor(value, key);
    return descriptor && "value" in descriptor ? descriptor.value : undefined;
  } catch {
    return undefined;
  }
}

function safeIdentity(value: unknown, allowSlash: boolean): string | undefined {
  if (typeof value !== "string" || value.length > 160) return undefined;
  // Omit URLs, paths, whitespace, query strings and common credential prefixes.
  // Provider/model identities are labels, not arbitrary serialized metadata.
  if (/^(?:sk-|sk_|Bearer|key-|token-)/i.test(value)) return undefined;
  const pattern = allowSlash
    ? /^[A-Za-z0-9][A-Za-z0-9._-]*(?:\/[A-Za-z0-9][A-Za-z0-9._-]*)*$/
    : /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
  return pattern.test(value) ? value : undefined;
}

/**
 * Observe only an allowlist of non-content request controls. Missing/unknown
 * values are omitted, so {} means no recognized control was present. Never
 * serialize the request itself: it contains credentials and conversation data.
 */
export function groundingRequestControls(model: unknown, payload: unknown): GroundingRequestControls {
  const result: GroundingRequestControls = { version: 1, model: {}, controls: {} };
  const provider = safeIdentity(ownValue(model, "provider"), false);
  const id = safeIdentity(ownValue(model, "id"), true);
  const reasoning = ownValue(model, "reasoning");
  const thinkingFormat = ownValue(ownValue(model, "compat"), "thinkingFormat");
  if (provider !== undefined) result.model.provider = provider;
  if (id !== undefined) result.model.id = id;
  if (typeof reasoning === "boolean") result.model.reasoning = reasoning;
  if (typeof thinkingFormat === "string" && THINKING_FORMATS.has(thinkingFormat)) {
    result.model.thinkingFormat = thinkingFormat;
  }

  const thinking = ownValue(payload, "thinking");
  const thinkingType = ownValue(thinking, "type");
  if (typeof thinkingType === "string" && THINKING_TYPES.has(thinkingType)) {
    result.controls.thinking = { type: thinkingType };
  } else if (typeof thinking === "string" && THINKING_TYPES.has(thinking)) {
    result.controls.thinking = thinking;
  }
  const effort = ownValue(payload, "reasoning_effort");
  if (typeof effort === "string" && REASONING_EFFORTS.has(effort)) result.controls.reasoning_effort = effort;
  const enabled = ownValue(payload, "enable_thinking");
  if (typeof enabled === "boolean") result.controls.enable_thinking = enabled;
  return result;
}
