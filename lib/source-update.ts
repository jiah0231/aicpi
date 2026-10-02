import { readFile, writeFile, unlink, realpath } from "node:fs/promises";
import path from "node:path";

export async function readUpdateManager(root: string, expectedRun: string) {
  let content: string;
  try { content = await readFile(path.join(root, ".pi-web-run", "state"), "utf8"); }
  catch { throw new Error("No Windows Web manager found. Run start-web.cmd in an external Windows terminal first."); }
  const [runId, phase, heartbeat, generation, port, mode, capability] = content.trim().split(/\r?\n/);
  if (!/^[a-f0-9]{32}$/.test(runId) || runId !== expectedRun || !/^\d+$/.test(generation)
    || !/^\d+$/.test(heartbeat) || !/^\d+$/.test(port)
    || !["starting", "ready", "restarting", "updating", "failed"].includes(phase)
    || Math.abs(Date.now() - Number(heartbeat)) > 20_000) {
    throw new Error("Windows Web manager is unavailable or stale. Check the external launcher; no update was requested.");
  }
  if (capability !== "update-v1") throw new Error("This manager predates /update. Close its external launcher and run start-web.cmd once to enable the command.");
  if (mode !== "dev") throw new Error("/update requires the manager's dev mode. Production builds must be updated manually.");
  let result = "";
  try {
    const [resultRun, resultGeneration, status] = (await readFile(path.join(root, ".pi-web-run", "update-result"), "utf8")).trim().split(/\r?\n/);
    if (resultRun === runId && resultGeneration === generation && ["updating", "pulled", "failed"].includes(status)) result = status;
  } catch { /* no update for this generation yet */ }
  return { runId, phase, generation: Number(generation), result };
}

export async function requestSourceUpdate(root: string, expectedRun: string) {
  const state = await readUpdateManager(root, expectedRun);
  if (state.phase !== "ready") throw new Error("Wait until the external manager reports READY before requesting an update.");
  const directory = path.join(root, ".pi-web-run");
  const lock = path.join(directory, `update-${state.runId}-${state.generation}.lock`);
  try { await writeFile(lock, "requested\n", { flag: "wx", mode: 0o600 }); }
  catch { throw new Error("An update is already requested for this server generation. Check the external launcher and update.log."); }
  try {
    await writeFile(path.join(directory, `update-${state.runId}-${state.generation}.request`), "requested\n", { flag: "wx", mode: 0o600 });
  } catch (error) { await unlink(lock).catch(() => {}); throw error; }
  return { status: "requested", generation: state.generation };
}

export async function managedUpdateRoot() {
  const root = process.env.PI_WEB_MANAGER_ROOT;
  const run = process.env.PI_WEB_MANAGED_RUN;
  if (process.platform !== "win32" || !root || !run || await realpath(root) !== await realpath(process.cwd())) {
    throw new Error("/update is available only in this checkout's Windows Web manager. Start it with start-web.cmd in an external Windows terminal.");
  }
  return { root, run };
}
