#!/usr/bin/env node
/**
 * Serialize a custom models.json model's off/low requests WITHOUT inference.
 * The SDK payload hook throws before transport; both fetch paths also reject.
 * Only allow-listed thinking controls are printed, never URLs or credentials.
 *
 * node e2e/grounding-thinking-diagnostic.mjs --provider linuxdo --model mimo-v2.5
 */
import { readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";

const HELP = "Usage: node e2e/grounding-thinking-diagnostic.mjs --provider <id> --model <id> [--models-config <path>]";
const INTERCEPT = "GROUNDING_DIAGNOSTIC_CAPTURED_BEFORE_NETWORK";
const CONTROLS = ["reasoning_effort", "thinking", "enable_thinking", "reasoning", "chat_template_kwargs", "chat_template_args", "thinking_token_budget", "reasoning_budget"];
const MODEL_FIELDS = ["id", "name", "api", "baseUrl", "reasoning", "thinkingLevelMap", "input", "inputLimits", "cost", "promptCache", "contextWindow", "maxTokens", "samplingParams", "compat"];
const CONTROL_VALUES = new Set(["off", "none", "minimal", "low", "medium", "high", "xhigh", "max", "enabled", "disabled", "adaptive"]);
const CONTROL_KEYS = new Set(["type", "effort", "enabled", "enable_thinking", "preserve_thinking", "clear_thinking", "budget_tokens", "thinking_budget", "reasoning_budget"]);

function safeControl(value, depth = 0) {
  if (typeof value === "boolean" || (typeof value === "number" && Number.isFinite(value))) return value;
  if (typeof value === "string") return CONTROL_VALUES.has(value) ? value : "REDACTED_NONSTANDARD_VALUE";
  if (value && typeof value === "object" && !Array.isArray(value) && depth < 2) {
    return Object.fromEntries(Object.entries(value).filter(([key]) => CONTROL_KEYS.has(key)).map(([key, entry]) => [key, safeControl(entry, depth + 1)]));
  }
  return "REDACTED_NONSTANDARD_VALUE";
}

function parseArgs(args) {
  const options = {};
  for (let i = 0; i < args.length; i += 1) {
    const arg = args[i];
    if (arg === "--help" || arg === "-h") return { help: true };
    if (!["--provider", "--model", "--models-config"].includes(arg) || !args[i + 1] || args[i + 1].startsWith("--")) {
      throw new Error(HELP);
    }
    options[arg.slice(2)] = args[++i];
  }
  if (!options.provider || !options.model) throw new Error(HELP);
  return options;
}

function pickModelFields(value) {
  return Object.fromEntries(MODEL_FIELDS.filter((key) => Object.hasOwn(value, key)).map((key) => [key, value[key]]));
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    console.log(HELP);
    return;
  }

  // Never resolve models.json credential commands, headers, or actual API keys.
  let source;
  try {
    source = JSON.parse(await readFile(options["models-config"] ?? join(homedir(), ".pi", "agent", "models.json"), "utf8"));
  } catch {
    throw new Error("Could not read a valid models configuration; its contents were not printed.");
  }
  const configuredProvider = source.providers?.[options.provider];
  const configuredModel = configuredProvider?.models?.find((model) => model.id === options.model);
  if (!configuredModel) throw new Error("The requested model is not explicitly defined in this models configuration.");
  const safeModel = pickModelFields(configuredModel);
  const configuredOverride = configuredProvider.modelOverrides?.[options.model];
  const safeConfig = {
    api: configuredProvider.api,
    baseUrl: configuredProvider.baseUrl,
    compat: configuredProvider.compat,
    apiKey: "DIAGNOSTIC_FAKE_KEY_NOT_A_CREDENTIAL",
    models: [safeModel],
    ...(configuredOverride ? { modelOverrides: { [options.model]: pickModelFields(configuredOverride) } } : {}),
  };
  if ((safeModel.api ?? safeConfig.api) !== "openai-completions" || !(safeModel.baseUrl ?? safeConfig.baseUrl)) {
    throw new Error("This diagnostic requires an explicit custom openai-completions model and endpoint.");
  }

  let transportAttempts = 0;
  const denyFetch = async () => {
    transportAttempts += 1;
    throw new Error("GROUNDING_DIAGNOSTIC_NETWORK_FORBIDDEN");
  };
  const originalFetch = globalThis.fetch;
  globalThis.fetch = denyFetch;
  try {
    const { composeModelProvider } = await import("../node_modules/@earendil-works/pi-coding-agent/dist/core/provider-composer.js");
    const { streamSimple } = await import("../node_modules/@earendil-works/pi-ai/dist/api/openai-completions.js");
    // getModels composes metadata only; no auth resolution or provider refresh.
    const provider = composeModelProvider(options.provider, undefined, { getProvider: () => safeConfig });
    const model = provider.getModels().find((entry) => entry.id === options.model);
    if (!model) throw new Error("Model composition did not return the requested model.");
    const results = [];
    for (const level of ["off", "low"]) {
      let controls;
      let intercepted = false;
      const stream = streamSimple(model, {
        messages: [{ role: "user", content: "Serialization-only diagnostic. This message must never be sent.", timestamp: 0 }],
      }, {
        apiKey: "DIAGNOSTIC_FAKE_KEY_NOT_A_CREDENTIAL",
        fetch: denyFetch,
        maxRetries: 0,
        reasoning: level === "off" ? undefined : level,
        onPayload(payload) {
          controls = Object.fromEntries(CONTROLS.map((key) => [key, Object.hasOwn(payload, key) ? safeControl(payload[key]) : "OMITTED"]));
          intercepted = true;
          throw new Error(INTERCEPT);
        },
      });
      const result = await stream.result();
      if (!intercepted || result.stopReason !== "error" || !result.errorMessage?.includes(INTERCEPT) || transportAttempts !== 0) {
        throw new Error("Serialization did not stop at the expected pre-network interception point.");
      }
      results.push({ level, controls, interceptedBeforeNetwork: true });
    }
    console.log(JSON.stringify({
      provider: options.provider,
      model: options.model,
      mode: "serialization-only; no inference",
      requests: results,
      transportAttempts,
    }, null, 2));
  } catch {
    // SDK errors can contain request configuration: deliberately do not echo them.
    throw new Error(`Diagnostic failed safely; transport attempts: ${transportAttempts}. No request or credential details were printed.`);
  } finally {
    globalThis.fetch = originalFetch;
  }
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
