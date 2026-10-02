import { hasJsonContentType, isApiRequestAllowed } from "@/lib/request-security";
import { addGenericLearningSource, listConfirmedGenericSources, mutateLearningRule, readLearningStore } from "@/lib/grounding-learning-store";
import { listConsolidationModels } from "@/lib/grounding-learning-consolidator";
import { readGroundingLearningSettings, updateGroundingLearningSettings } from "@/lib/grounding-learning-settings";
import { getGroundingLearningJobStatus, notifyGroundingLearningSettingsChanged, requestGroundingLearningRetry } from "@/lib/grounding-learning-worker";
import type { LearningRuleAction } from "@/lib/grounding-learning-consolidation-types";

export const dynamic = "force-dynamic";
// Authentication is enforced by the existing /api/:path* proxy, including GET.
const json = (body: unknown, status = 200) => Response.json(body, { status, headers: { "Cache-Control": "no-store" } });
function object(input: unknown, keys: readonly string[]): Record<string, unknown> {
  if (!input || typeof input !== "object" || Array.isArray(input) || Object.keys(input).some((key) => !keys.includes(key))) throw new Error("Invalid learning request fields");
  return input as Record<string, unknown>;
}
function revision(value: unknown): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) throw new Error("expectedRevision is required");
  return value;
}

async function readBoundedBody(req: Request): Promise<string | null> {
  const reader = req.body?.getReader();
  if (!reader) return "";
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > 16000) { await reader.cancel(); return null; }
      chunks.push(value);
    }
  } finally { reader.releaseLock(); }
  const bytes = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  return new TextDecoder().decode(bytes);
}

export async function GET(req: Request) {
  if (!isApiRequestAllowed(req)) return json({ error: "Untrusted API request" }, 403);
  if (new URL(req.url).search) return json({ error: "Query parameters are not supported" }, 400);
  try {
    const [store, sources, settings, job, models] = await Promise.all([readLearningStore(), listConfirmedGenericSources(), readGroundingLearningSettings(), getGroundingLearningJobStatus(), listConsolidationModels()]);
    return json({ store, sources, settings, job, models });
  } catch { return json({ error: "Unable to read generic learning settings" }, 500); }
}

export async function POST(req: Request) {
  if (!isApiRequestAllowed(req)) return json({ error: "Untrusted API request" }, 403);
  if (!hasJsonContentType(req)) return json({ error: "Content-Type must be application/json" }, 415);
  if (new URL(req.url).search) return json({ error: "Query parameters are not supported" }, 400);
  try {
    const raw = await readBoundedBody(req);
    if (raw === null) return json({ error: "Learning request too large" }, 413);
    const body = object(JSON.parse(raw), ["action", "text", "sampleIndependent", "expectedRevision", "settings", "proposalId", "procedure", "ruleId", "enabled", "revision"]);
    if (body.action === "source") {
      object(body, ["action", "text", "sampleIndependent"]);
      if (body.sampleIndependent !== true || typeof body.text !== "string") throw new Error("Confirm the original generic advice is sample-independent");
      await addGenericLearningSource({ text: body.text, sampleIndependent: true });
    } else if (body.action === "retry") {
      object(body, ["action", "expectedRevision"]);
      await requestGroundingLearningRetry(revision(body.expectedRevision));
    } else if (body.action === "settings") {
      object(body, ["action", "settings", "expectedRevision"]);
      const patch = object(body.settings, ["enabled", "provider", "modelId", "maxSources", "maxInputChars", "maxOutputTokens", "timeoutMs", "dailyAttemptLimit"]);
      const current = await readGroundingLearningSettings();
      const next = { ...current, ...patch };
      if (next.enabled || next.provider !== current.provider || next.modelId !== current.modelId) {
        const models = await listConsolidationModels();
        if ((next.enabled || next.provider || next.modelId) && !models.some((model) => model.provider === next.provider && model.id === next.modelId)) throw new Error("Select an existing configured model");
      }
      await updateGroundingLearningSettings(patch, undefined, revision(body.expectedRevision));
      notifyGroundingLearningSettingsChanged();
    } else {
      if (!["activate", "dismiss", "set_enabled", "rollback"].includes(String(body.action))) throw new Error("Unknown learning action");
      const { action, expectedRevision, ...fields } = body;
      await mutateLearningRule({ type: action, ...fields } as LearningRuleAction, revision(expectedRevision));
    }
    return json({ ok: true });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid learning request";
    // Never echo filesystem errors, provider responses, or submitted sample content.
    if (/revision conflict|settings changed/i.test(message)) return json({ error: "Learning revision conflict. Refresh and review again." }, 409);
    return json({ error: "Invalid learning request. Check generic-only content, confirmation, model selection and limits." }, 400);
  }
}
