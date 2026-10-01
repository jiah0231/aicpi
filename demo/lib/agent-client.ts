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

  const raw = await res.text().catch(() => "");
  let body: AgentCommandResponse<T> = {};
  if (raw) {
    try {
      body = JSON.parse(raw) as AgentCommandResponse<T>;
    } catch {
      // Keep the raw response as the diagnostic below.
    }
  }

  if (!res.ok || body.error) {
    const message = body.error ?? (raw.trim() || `HTTP ${res.status}`);
    throw new AgentCommandError(
      message,
      res.status,
      body.code ?? "http_error",
      body.accepted,
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
