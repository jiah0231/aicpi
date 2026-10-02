import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import type { AgentMessage } from "@earendil-works/pi-agent-core";

export const GROUNDING_GENERATION_CONTROL_ENTRY = "grounding:generation-control";
const MARKER = /\[grounding-generation:([^\]\r\n]*)\]/g;
const LIMIT_FIELDS = ["max_tokens", "max_completion_tokens", "max_output_tokens"] as const;

/** Per-user-turn only. History and dataset/tool content cannot enable a budget. */
export function parseGroundingGenerationControl(prompt: string): {
  prompt: string; maxOutputTokens?: number; error?: string;
} {
  const matches = [...prompt.matchAll(MARKER)];
  const clean = prompt.replace(MARKER, "");
  if (!matches.length) return { prompt: clean };
  const match = matches.length === 1 && /^max-output-tokens=([1-9]\d*)$/.exec(matches[0][1]);
  const limit = match ? Number(match[1]) : NaN;
  if (!Number.isSafeInteger(limit) || limit > 1_000_000_000) {
    return { prompt: clean, error: "Use exactly one [grounding-generation:max-output-tokens=N] marker with a positive integer N." };
  }
  return { prompt: clean, maxOutputTokens: limit };
}

/** Never increase a provider limit or rewrite reasoning settings/model selection. */
export function applyGroundingGenerationLimit(payload: unknown, limit: number):
  | { payload: Record<string, unknown>; fields: string[]; effectiveLimit: number }
  | { error: string } {
  if (!Number.isSafeInteger(limit) || limit <= 0 || limit > 1_000_000_000) return { error: "The output-token limit must be a positive integer." };
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
    return { error: "This provider request does not expose a supported output-token limit." };
  }
  const source = payload as Record<string, unknown>;
  const fields = LIMIT_FIELDS.filter((field) => Object.hasOwn(source, field));
  if (!fields.length || fields.some((field) => !Number.isSafeInteger(source[field]) || (source[field] as number) <= 0)) {
    return { error: "This provider request does not expose a supported positive output-token limit." };
  }
  const effectiveLimit = Math.min(limit, ...fields.map((field) => source[field] as number));
  const thinking = source.thinking;
  const thinkingBudget = thinking && typeof thinking === "object" && !Array.isArray(thinking)
    ? (thinking as Record<string, unknown>).budget_tokens : undefined;
  const budgets = [thinkingBudget, source.thinking_token_budget].filter((budget) => budget !== undefined);
  if (budgets.some((budget) => typeof budget !== "number" || !Number.isSafeInteger(budget) || budget < 0 || budget >= effectiveLimit)) {
    return { error: "The requested output limit conflicts with the provider's explicit thinking budget. Increase or remove the limit; reasoning settings were not changed." };
  }
  return { payload: { ...source, ...Object.fromEntries(fields.map((field) => [field, Math.min(limit, source[field] as number)])) }, fields, effectiveLimit };
}

/**
 * The caller wires these methods into its existing hooks so activation and prompt
 * detection remain owned by grounding safety. No timers or captured async ctx:
 * abort is synchronous and tied to the exact request's run signal.
 */
export function createGroundingGenerationControl(
  pi: Pick<ExtensionAPI, "appendEntry" | "sendMessage">,
  isActive: () => boolean,
  onBlocked: () => void,
) {
  // Observability must not throw out of before_provider_request: the SDK
  // catches hook errors and would otherwise send the original unlimited payload.
  const appendEntry: typeof pi.appendEntry = (...args) => {
    try { pi.appendEntry(...args); } catch { /* Budget enforcement still applies. */ }
  };
  const sendMessage: typeof pi.sendMessage = (...args) => {
    try { pi.sendMessage(...args); } catch { /* Never replace a bounded request with the original payload. */ }
  };
  let control: ReturnType<typeof parseGroundingGenerationControl> = { prompt: "" };
  let requestSignal: AbortSignal | undefined;
  let applied = false;
  let blocked = false;
  let announced = false;
  const reset = () => {
    control = { prompt: "" };
    requestSignal = undefined;
    applied = blocked = announced = false;
  };
  const block = (ctx: ExtensionContext, reason: string) => {
    if (blocked) return;
    blocked = true;
    ctx.abort(); // Must precede notifications: prevents SDK length compaction/retry.
    onBlocked();
    appendEntry(GROUNDING_GENERATION_CONTROL_ENTRY, { version: 1, status: "blocked", maxOutputTokens: control.maxOutputTokens, reason });
    sendMessage({ customType: GROUNDING_GENERATION_CONTROL_ENTRY, display: true,
      content: `Grounding generation blocked: ${reason} No truncated tool call is accepted as a result. The current record remains pending unless it was already saved by a completed tool call. To continue with full reasoning, send a new request without the generation marker, or explicitly increase its limit.` }, { triggerTurn: false });
  };
  return {
    reset,
    beforeCompact(reason: "manual" | "threshold" | "overflow") {
      // SDK 0.87.1 can prepare a next turn after a truncated tool call even
      // with the run signal aborted; compaction has its own independent signal.
      return blocked && reason !== "manual" ? { cancel: true as const } : undefined;
    },
    startTurn(prompt: string) {
      reset();
      control = parseGroundingGenerationControl(prompt);
      return control.prompt;
    },
    beforeRequest(payload: unknown, ctx: ExtensionContext): unknown {
      applied = false;
      requestSignal = undefined;
      if (!isActive() || (!control.error && control.maxOutputTokens === undefined)) return undefined;
      if (ctx.signal?.aborted) return undefined;
      requestSignal = ctx.signal;
      if (control.error) { block(ctx, control.error); return undefined; }
      const result = applyGroundingGenerationLimit(payload, control.maxOutputTokens!);
      if ("error" in result) { block(ctx, result.error); return undefined; }
      applied = true;
      if (!announced) {
        announced = true;
        appendEntry(GROUNDING_GENERATION_CONTROL_ENTRY, { version: 1, status: "requested", maxOutputTokens: control.maxOutputTokens, effectiveLimit: result.effectiveLimit, fields: result.fields });
        sendMessage({ customType: GROUNDING_GENERATION_CONTROL_ENTRY, display: true,
          content: `Grounding output budget requested: ${result.effectiveLimit} tokens per generation. Provider enforcement is not verified; this is not a wall-clock deadline. Model and reasoning settings are unchanged. A length stop blocks this run without automatic retry.` }, { triggerTurn: false });
      }
      return result.payload;
    },
    messageEnd(message: AgentMessage, ctx: ExtensionContext) {
      // Streaming steering/follow-up prompts bypass before_agent_start. Apply
      // their controls only once consumed, never while the old call is in flight.
      if (message.role === "user") {
        reset();
        const prompt = typeof message.content === "string" ? message.content
          : message.content.filter((part) => part.type === "text").map((part) => part.text).join("\n");
        control = parseGroundingGenerationControl(prompt);
        return;
      }
      if (!applied || blocked || !isActive() || message.role !== "assistant" || message.stopReason !== "length") return;
      if (!requestSignal || ctx.signal !== requestSignal || ctx.signal.aborted) return;
      block(ctx, `The provider ended a generation with stopReason=length after an explicit output budget of ${control.maxOutputTokens} tokens was requested.`);
    },
  };
}
