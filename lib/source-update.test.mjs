import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, readFileSync, mkdirSync, rmSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { updateCheckout } from "../bin/web-source-update.mjs";
import { createJiti } from "jiti";
const { readUpdateManager, requestSourceUpdate } = await createJiti(import.meta.url).import("./source-update.ts");
const git = (cwd, args) => execFileSync("git", args, { cwd, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"] }).trim();
function fixture(t) {
  const base = mkdtempSync(path.join(os.tmpdir(), "pi-update-"));
  t.after(() => rmSync(base, { recursive: true, force: true }));
  const upstream = path.join(base, "upstream"), root = path.join(base, "app");
  mkdirSync(upstream); git(upstream, ["init", "-b", "gptdot"]);
  git(upstream, ["config", "user.email", "test@example.com"]); git(upstream, ["config", "user.name", "Test"]);
  writeFileSync(path.join(upstream, "AGENTS.md"), "original\n");
  writeFileSync(path.join(upstream, "app.txt"), "old\n");
  git(upstream, ["add", "."]); git(upstream, ["commit", "-m", "initial"]);
  git(base, ["clone", upstream, root]); git(root, ["remote", "set-url", "origin", "https://github.com/jiah0231/aicpi.git"]);
  const run = args => {
    const copy = [...args];
    if (copy.includes("fetch")) copy[copy.indexOf("origin")] = upstream;
    return git(root, copy);
  };
  const commit = (name, text) => { writeFileSync(path.join(upstream, name), text); git(upstream, ["add", "."]); git(upstream, ["commit", "-m", "next"]); };
  return { root, upstream, run, commit };
}
test("ff-only update preserves unrelated tracked edits and untracked files", t => {
  const f = fixture(t); f.commit("app.txt", "new\n");
  writeFileSync(path.join(f.root, "AGENTS.md"), "local edits\n");
  writeFileSync(path.join(f.root, "untracked.txt"), "keep\n");
  updateCheckout(f.root, f.run);
  assert.equal(readFileSync(path.join(f.root, "app.txt"), "utf8"), "new\n");
  assert.equal(readFileSync(path.join(f.root, "AGENTS.md"), "utf8"), "local edits\n");
  assert.equal(readFileSync(path.join(f.root, "untracked.txt"), "utf8"), "keep\n");
});
test("conflicting local edit fails without losing it or moving HEAD", t => {
  const f = fixture(t); f.commit("app.txt", "upstream\n");
  const before = git(f.root, ["rev-parse", "HEAD"]);
  writeFileSync(path.join(f.root, "app.txt"), "local\n");
  assert.throws(() => updateCheckout(f.root, f.run));
  assert.equal(git(f.root, ["rev-parse", "HEAD"]), before);
  assert.equal(readFileSync(path.join(f.root, "app.txt"), "utf8"), "local\n");
});
test("untracked collision fails without deleting the file", t => {
  const f = fixture(t); f.commit("new.txt", "upstream\n");
  writeFileSync(path.join(f.root, "new.txt"), "local\n");
  assert.throws(() => updateCheckout(f.root, f.run));
  assert.equal(readFileSync(path.join(f.root, "new.txt"), "utf8"), "local\n");
});
test("wrong remote and branch fail before fetching", t => {
  const f = fixture(t);
  git(f.root, ["remote", "set-url", "origin", "https://evil.example/aicpi.git"]);
  assert.throws(() => updateCheckout(f.root, f.run), /requires origin/);
  git(f.root, ["checkout", "-b", "other"]);
  assert.throws(() => updateCheckout(f.root, f.run), /gptdot branch/);
});
test("dependency change refuses merge and restart", t => {
  const f = fixture(t); f.commit("package.json", "{}\n");
  const before = git(f.root, ["rev-parse", "HEAD"]);
  assert.throws(() => updateCheckout(f.root, f.run), /Dependency files/);
  assert.equal(git(f.root, ["rev-parse", "HEAD"]), before);
});
test("manager validates capability, freshness and coalesces requests", async t => {
  const root = mkdtempSync(path.join(os.tmpdir(), "pi-manager-test-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const run = "a".repeat(32); mkdirSync(path.join(root, ".pi-web-run"));
  const state = (capability = "update-v1", timestamp = Date.now()) => writeFileSync(path.join(root, ".pi-web-run/state"), `${run}\nready\n${timestamp}\n1\n30141\ndev\n${capability}\n`);
  state(""); await assert.rejects(readUpdateManager(root, run), /predates/);
  state("update-v1", 0); await assert.rejects(readUpdateManager(root, run), /stale/);
  state(); await assert.rejects(readUpdateManager(root, "b".repeat(32)), /stale/);
  assert.equal((await requestSourceUpdate(root, run)).status, "requested");
  await assert.rejects(requestSourceUpdate(root, run), /already requested/);
});
test("diverged branch refuses update without rewriting local commits", t => {
  const f = fixture(t); f.commit("app.txt", "upstream\n");
  git(f.root, ["config", "user.email", "test@example.com"]); git(f.root, ["config", "user.name", "Test"]);
  writeFileSync(path.join(f.root, "local.txt"), "local\n");
  git(f.root, ["add", "."]); git(f.root, ["commit", "-m", "local"]);
  const before = git(f.root, ["rev-parse", "HEAD"]);
  assert.throws(() => updateCheckout(f.root, f.run));
  assert.equal(git(f.root, ["rev-parse", "HEAD"]), before);
});
test("built-in update asks confirmation and never starts a model session", () => {
  const hook = readFileSync(new URL("../hooks/useAgentSession.ts", import.meta.url), "utf8");
  const block = hook.slice(hook.indexOf('case "update":'), hook.indexOf('case "compact":', hook.indexOf('case "update":')));
  assert.match(block, /window\.confirm\(t\("chat.updateConfirm"\)\)/);
  assert.match(block, /confirmInterruption: true/);
  assert.match(hook, /commandName === "update" \? sessionIdRef\.current :/);
  assert.doesNotMatch(block, /sendAgentCommand|ensureNewSession|executeBash/);
  const host = readFileSync(new URL("../bin/windows-web-host.cs", import.meta.url), "utf8");
  assert.match(host, /if \(ok\) \{ restart = true;/);
  assert.match(host, /updater\.ExitCode == 0/);
  assert.match(host, /ClearUpdateLock\(\); Log\("UPDATE FAILED/);
});
test("ignored untracked collision fails rather than overwriting local content", t => {
  const f = fixture(t);
  writeFileSync(path.join(f.root, ".git/info/exclude"), "new.txt\n");
  writeFileSync(path.join(f.root, "new.txt"), "local ignored\n");
  f.commit("new.txt", "upstream\n");
  const before = git(f.root, ["rev-parse", "HEAD"]);
  assert.throws(() => updateCheckout(f.root, f.run));
  assert.equal(git(f.root, ["rev-parse", "HEAD"]), before);
  assert.equal(readFileSync(path.join(f.root, "new.txt"), "utf8"), "local ignored\n");
});
test("manager consumes only current-generation requests while ready", () => {
  const host = readFileSync(new URL("../bin/windows-web-host.cs", import.meta.url), "utf8");
  const block = host.slice(host.indexOf("private static bool ConsumeUpdate()"), host.indexOf("private static bool Consume(string"));
  assert.match(block, /generation \+ "\.request"/);
  assert.match(block, /String\.Equals\(request, expected, StringComparison\.OrdinalIgnoreCase\) && phase == "ready"/);
  assert.match(block, /File\.Delete\(request\)/);
  assert.doesNotMatch(host, /Consume\("update"\)/);
});
test("status hides a previous manager/generation's result and reports updating", async t => {
  const root = mkdtempSync(path.join(os.tmpdir(), "pi-update-status-"));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const run = "a".repeat(32); mkdirSync(path.join(root, ".pi-web-run"));
  writeFileSync(path.join(root, ".pi-web-run/state"), `${run}\nupdating\n${Date.now()}\n2\n30141\ndev\nupdate-v1\n`);
  writeFileSync(path.join(root, ".pi-web-run/update-result"), `${run}\n1\npulled`);
  assert.equal((await readUpdateManager(root, run)).result, "");
  writeFileSync(path.join(root, ".pi-web-run/update-result"), `${run}\n2\nupdating`);
  assert.equal((await readUpdateManager(root, run)).result, "updating");
  await assert.rejects(requestSourceUpdate(root, run), /READY/);
});
