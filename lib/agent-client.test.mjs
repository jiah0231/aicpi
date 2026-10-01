import assert from "node:assert/strict";
import test from "node:test";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url, {
  interopDefault: true,
  moduleCache: false,
});
const {
  AgentCommandError,
  isAgentTransportError,
  isPromptRejectedError,
  sendAgentCommand,
} = await jiti.import("./agent-client.ts");

function response(body, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("agent command HTTP rejections preserve API diagnostics", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  globalThis.fetch = async () => response({
    error: "Authentication failed",
    code: "prompt_rejected",
    accepted: false,
  }, 500);

  await assert.rejects(
    sendAgentCommand("session-id", { type: "prompt", message: "hello" }),
    (error) => {
      assert.equal(error instanceof AgentCommandError, true);
      assert.equal(error.status, 500);
      assert.equal(error.message, "Authentication failed");
      assert.equal(error.code, "prompt_rejected");
      assert.equal(error.accepted, false);
      assert.equal(isPromptRejectedError(error), true);
      assert.equal(isAgentTransportError(error), false);
      return true;
    },
  );

  globalThis.fetch = async () => response({
    error: "Request Entity Too Large",
    code: "payload_too_large",
  }, 413);
  await assert.rejects(
    sendAgentCommand("session-id", { type: "get_state" }),
    (error) => {
      assert.equal(error instanceof AgentCommandError, true);
      assert.equal(error.status, 413);
      assert.equal(error.message, "Request Entity Too Large");
      assert.equal(error.code, "payload_too_large");
      return true;
    },
  );
});

test("transport failures become structured errors and prompt is never retried", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  let calls = 0;
  const transportError = new TypeError("connection reset");
  globalThis.fetch = async () => {
    calls += 1;
    throw transportError;
  };

  await assert.rejects(
    sendAgentCommand("session-id", { type: "prompt", message: "hello" }),
    (error) => {
      assert.equal(error instanceof AgentCommandError, true);
      assert.equal(error.status, 0);
      assert.equal(error.code, "transport_error");
      assert.match(error.message, /connection reset/);
      assert.equal(isAgentTransportError(error), true);
      assert.equal(isPromptRejectedError(error), false);
      return true;
    },
  );
  assert.equal(calls, 1);
});

test("idempotent commands retry a bounded number of times after transport failure", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    if (calls < 3) throw new TypeError("development server reloading");
    return response({ success: true, data: { running: false } });
  };

  const result = await sendAgentCommand("session-id", { type: "get_state" });
  assert.deepEqual(result, { running: false });
  assert.equal(calls, 3);
});

test("bounded retries still terminate with a structured transport error", async (t) => {
  const originalFetch = globalThis.fetch;
  t.after(() => {
    globalThis.fetch = originalFetch;
  });

  let calls = 0;
  globalThis.fetch = async () => {
    calls += 1;
    throw new TypeError("server offline");
  };

  await assert.rejects(
    sendAgentCommand("session-id", { type: "get_tools" }),
    (error) => isAgentTransportError(error) && error.status === 0,
  );
  assert.equal(calls, 3);
});

test("only an explicit negative prompt acknowledgement is definitive", () => {
  assert.equal(
    isPromptRejectedError(new AgentCommandError("proxy failure", 502)),
    false,
  );
  assert.equal(
    isPromptRejectedError(new AgentCommandError("generic API failure", 500, "internal_error", false)),
    false,
  );
});
