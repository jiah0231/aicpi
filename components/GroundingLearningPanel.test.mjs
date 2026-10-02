import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
const source = await readFile(new URL("./GroundingLearningPanel.tsx", import.meta.url), "utf8");
const settings = await readFile(new URL("./SettingsPanel.tsx", import.meta.url), "utf8");
test("generic management is collapsed in existing General settings with no polling or provider credentials", () => {
 assert.match(settings, /<GroundingLearningPanel \/>/);
 assert.match(source, /\[open, setOpen\] = useState\(false\)/);
 assert.match(source, /open && <LearningManager/);
 assert.doesNotMatch(source, /setInterval|setTimeout|apiKey|baseUrl|imagePath|\/api\/sessions/);
 assert.match(source, /data\.models\.map/);
});
test("proposal edits clear confirmation and all version mutations include expectedRevision", () => {
 assert.match(source, /setProcedure\(.*setConfirmed\(false\)/);
 assert.match(source, /disabled=\{!confirmed\}/);
 for (const action of ["activate", "dismiss", "set_enabled", "rollback"]) assert.match(source, new RegExp(`action: "${action}"[^\\n]+expectedRevision`));
 assert.match(source, /response.status === 409/);
 assert.match(source, /key=\{`\$\{proposal.id\}:\$\{data.store.revision\}`\}/);
});
test("all built-in locales contain safety, cost, status and review copy", async () => {
 for (const locale of ["en", "zh-CN", "zh-TW"]) {
  const messages = await readFile(new URL(`../lib/i18n/messages/${locale}.ts`, import.meta.url), "utf8");
  for (const key of ["notice", "confirm", "cost", "calls", "tokens", "disabled", "waiting", "running", "pending", "budget", "backoff", "interrupted", "stale", "activate", "rollback"]) assert.ok(messages.includes(`"groundingLearning.${key}"`), `${locale}: ${key}`);
 }
});

test("retry is explicit, revision guarded, and unavailable while disabled", () => {
 assert.match(source, /data.job.errorCode &&/);
 assert.match(source, /disabled=\{!data.settings.enabled\}/);
 assert.match(source, /action: "retry", expectedRevision: data.settings.revision/);
 assert.match(source, /label\("retryNotice"\)/);
});
