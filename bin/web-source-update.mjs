// Fixed, manager-owned updater. Never runs model-provided commands or session cwd.
import { execFileSync } from "node:child_process";
import { realpathSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

export function updateCheckout(root, runGit = (args) => execFileSync("git", args, {
  cwd: root, encoding: "utf8", timeout: 120_000, windowsHide: true,
  env: { ...process.env, GIT_TERMINAL_PROMPT: "0" },
}).trim()) {
  const git = (...args) => runGit(["-c", "core.hooksPath=", "-c", "merge.autoStash=false", ...args]);
  if (realpathSync(git("rev-parse", "--show-toplevel")) !== realpathSync(root)) throw new Error("Update requires this application's own Git checkout.");
  if (git("branch", "--show-current") !== "gptdot") throw new Error("Update requires the gptdot branch; no branch was switched.");
  // get-url expands Git's insteadOf rules: validate the effective fetch URL,
  // not just the configured spelling. Allow only equivalent GitHub HTTPS forms;
  // no credentials, ports, query/fragment, or alternate transports.
  if (!/^https:\/\/github\.com\/jiah0231\/aicpi(?:\.git)?$/.test(git("remote", "get-url", "origin"))) throw new Error("Update requires origin https://github.com/jiah0231/aicpi.git (optional .git suffix); no source was changed.");
  if (git("rev-parse", "--abbrev-ref", "@{upstream}") !== "origin/gptdot") throw new Error("Update requires upstream origin/gptdot.");
  // Fetch does not touch working files. A single fixed ref avoids arbitrary refs or commands.
  git("fetch", "--no-tags", "origin", "refs/heads/gptdot");
  const target = git("rev-parse", "FETCH_HEAD");
  if (!/^[a-f0-9]{40,64}$/.test(target)) throw new Error("Invalid fetched commit.");
  git("merge-base", "--is-ancestor", "HEAD", target);
  const dependencies = git("diff", "--name-only", "HEAD", target, "--", "package.json", "package-lock.json");
  if (dependencies) throw new Error("Dependency files changed upstream. Update and install dependencies manually in the external terminal; automatic restart was not requested.");
  // Git protects overlapping tracked edits/untracked paths itself. Disable autostash;
  // never reset, clean, stash, switch branches, install packages, or run a build.
  git("merge", "--ff-only", "--no-edit", "--no-overwrite-ignore", target);
  return target;
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
    console.log(`Updated to ${updateCheckout(root)}. Manager may now restart.`);
  } catch (error) {
    console.error(`Update stopped; no restart: ${error instanceof Error ? error.message : String(error)}`);
    process.exitCode = 1;
  }
}
