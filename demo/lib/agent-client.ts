// Client-side helper for POST /api/agent/[id].
//
// Every /api/agent/[id] route returns one of:
//   { success: true, data: <result> }
//   { error: string }              (non-2xx)
//
// Transport failures are kept separate from HTTP/API failures. Only commands
// whose repeated delivery is safe are retried; prompt-like commands are never
// retried because a failed fetch cannot tell us whether the server accepted it.

export class AgentCommandError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly code?: string,
    public readonly accepted?: boolean,
  ) {
    super(message);
    this.name = "AgentCommandError";
  }
}

export function isPromptRejectedError(error: unknown): error is AgentCommandError {
  return error instanceof AgentCommandError
    && error.code === "prompt_rejected"
    && error.accepted === false;
}

export function isAgentTransportError(error: unknown): error is AgentCommandError {
  return error instanceof AgentCommandError && error.code === "transport_error";
}

const IDEMPOTENT_COMMAND_TYPES = new Set([
  "abort",
  "abort_bash",
  "get_commands",
  "get_last_assistant_text",
  "get_session_stats",
  "get_state",
  "get_tools",
  "navigate_tree",
  "reload",
  "set_model",
  "set_session_name",
  "set_thinking_level",
]);

// Keep this short: a development-server reload should recover quickly, while
// repeated requests must never turn a persistent outage into a request storm.
const TRANSPORT_RETRY_DELAYS_MS = [150, 500];

type AgentCommandResponse<T> = {
  success?: boolean;
  data?: T;
  error?: string;
  code?: string;
  accepted?: boolean;
};

async function sendOnce<T>(sessionId: string, command: Record<string, unknown>): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api/agent/${encodeURIComponent(sessionId)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(command),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new AgentCommandError(`Agent transport failed: ${message}`, 0, "transport_error");
  }

  let raw = "";
  try {
    raw = await res.text();
  } catch (error) {
    // Fetch can resolve after the headers arrive and still lose the body. A
    // successful status alone does not acknowledge the command. Preserve a
    // known HTTP rejection, but treat an interrupted success body like any
    // other transport failure so only safe commands can be retried.
    if (res.ok) {
      const message = error instanceof Error ? error.message : String(error);
      throw new AgentCommandError(`Agent transport failed: ${message}`, 0, "transport_error");
    }
  }
  let body: AgentCommandResponse<T> = {};
  if (raw) {
    try {
      const parsed: unknown = JSON.parse(raw);
      if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) {
        body = parsed as AgentCommandResponse<T>;
      }
    } catch {
      // Keep the raw response as the diagnostic below.
    }
  }

  const apiError = typeof body.error === "string" ? body.error : undefined;
  if (!res.ok || apiError) {
    const message = apiError ?? (raw.trim() || `HTTP ${res.status}`);
    throw new AgentCommandError(
      message,
      res.status,
      typeof body.code === "string" ? body.code : "http_error",
      typeof body.accepted === "boolean" ? body.accepted : undefined,
    );
  }
  // Proxy HTML, empty/truncated JSON and unrelated JSON are not successful
  // replies. Keep acceptance unknown and do not retry malformed envelopes.
  if (body.success !== true) {
    throw new AgentCommandError(
      "Agent returned an invalid command response; command acceptance is unknown.",
      res.status,
      "invalid_response",
    );
  }
  return body.data as T;
}

export async function sendAgentCommand<T = unknown>(
  sessionId: string,
  command: Record<string, unknown>,
): Promise<T> {
  const retryable = typeof command.type === "string"
    && IDEMPOTENT_COMMAND_TYPES.has(command.type);

  for (let attempt = 0; ; attempt += 1) {
    try {
      return await sendOnce<T>(sessionId, command);
    } catch (error) {
      const delay = retryable && isAgentTransportError(error)
        ? TRANSPORT_RETRY_DELAYS_MS[attempt]
        : undefined;
      if (delay === undefined) throw error;
      await new Promise<void>((resolve) => setTimeout(resolve, delay));
    }
  }
}
