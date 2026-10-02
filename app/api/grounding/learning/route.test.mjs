import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test, { after } from "node:test";
import { createJiti } from "jiti";
const prior = process.env.PI_CODING_AGENT_DIR;
const priorStore = process.env.PI_WEB_GROUNDING_LEARNING_STORE_PATH;
const priorLessons = process.env.PI_WEB_GROUNDING_LESSONS_PATH;
const directory = await mkdtemp(join(tmpdir(), "generic-learning-route-"));
process.env.PI_CODING_AGENT_DIR = directory;
process.env.PI_WEB_GROUNDING_LEARNING_STORE_PATH = join(directory, "synthetic-store.json");
process.env.PI_WEB_GROUNDING_LESSONS_PATH = join(directory, "synthetic-lessons.json");
const jiti = createJiti(import.meta.url, { alias: { "@": process.cwd() }, moduleCache: false, interopDefault: true });
const { GET, POST } = await jiti.import("./route.ts");
const { updateGroundingLearningSettings, readGroundingLearningSettings } = await jiti.import("../../../../lib/grounding-learning-settings.ts");
after(async () => {
 if (priorStore === undefined) delete process.env.PI_WEB_GROUNDING_LEARNING_STORE_PATH; else process.env.PI_WEB_GROUNDING_LEARNING_STORE_PATH = priorStore;
 if (priorLessons === undefined) delete process.env.PI_WEB_GROUNDING_LESSONS_PATH; else process.env.PI_WEB_GROUNDING_LESSONS_PATH = priorLessons;
 if (prior === undefined) delete process.env.PI_CODING_AGENT_DIR; else process.env.PI_CODING_AGENT_DIR = prior; await rm(directory, { recursive: true, force: true }); });
function request(body, headers = {}, query = "") { return new Request(`http://localhost/api/grounding/learning${query}`, { method: "POST", headers: { Host: "localhost", "Content-Type": "application/json", ...headers }, body: JSON.stringify(body) }); }
const advice = "Compare the complete candidate outline before selecting; verify the same identity across views.";
test("rejects cross-origin, non-JSON, unknown fields and filesystem arguments", async () => {
 assert.equal((await POST(request({}, { Origin: "https://untrusted.example" }))).status, 403);
 assert.equal((await POST(request({}, { "Content-Type": "text/plain" }))).status, 415);
 assert.equal((await POST(request({ action: "source", text: advice, sampleIndependent: true, filePath: "/tmp/other" }))).status, 400);
 assert.equal((await POST(request({ action: "source", text: advice, sampleIndependent: true }, {}, "?filePath=/tmp/other"))).status, 400);
 assert.equal((await GET(new Request("http://localhost/api/grounding/learning?path=/tmp/other", { headers: { Host: "localhost" } }))).status, 400);
});
test("only confirmed generic free-text sources are accepted", async () => {
 assert.equal((await POST(request({ action: "source", text: advice }))).status, 400);
 assert.equal((await POST(request({ action: "source", text: "imagePath: /private/example.png", sampleIndependent: true }))).status, 400);
 assert.equal((await POST(request({ action: "source", text: advice, sampleIndependent: true }))).status, 200);
});
test("mutations require explicit revision and reject stale writes", async () => {
 assert.equal((await POST(request({ action: "set_enabled", ruleId: "synthetic", enabled: true }))).status, 400);
 assert.equal((await POST(request({ action: "set_enabled", ruleId: "synthetic", enabled: true, expectedRevision: 9000 }))).status, 409);
 assert.equal((await POST(request({ action: "settings", expectedRevision: 0, settings: { enabled: false, dailyAttemptLimit: 1 } }))).status, 200);
 assert.equal((await POST(request({ action: "settings", expectedRevision: 0, settings: { enabled: false } }))).status, 409);
 assert.equal((await POST(request({ action: "settings", expectedRevision: 1, settings: { apiKey: "never accepted" } }))).status, 400);
});

test("disabling preserves an unavailable selection without needing registry lookup", async () => {
 const stored = await updateGroundingLearningSettings({ enabled: true, provider: "synthetic-unavailable", modelId: "fake-model" });
 const { revision, ...settings } = stored;
 assert.equal((await POST(request({ action: "settings", expectedRevision: revision, settings: { ...settings, enabled: false } }))).status, 200);
 assert.equal((await readGroundingLearningSettings()).enabled, false);
});
test("oversized streamed request is rejected before parsing or mutation", async () => {
 assert.equal((await POST(request({ action: "source", text: "x".repeat(16001), sampleIndependent: true }))).status, 413);
});

test("manual retry rejects missing revisions and extra control fields", async () => {
 assert.equal((await POST(request({ action: "retry" }))).status, 400);
 assert.equal((await POST(request({ action: "retry", expectedRevision: 0, resetQuota: true }))).status, 400);
});
