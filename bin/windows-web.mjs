#!/usr/bin/env node
// The manager lives in an external terminal. Embedded terminals only submit
// requests: they neither retain PI_WEB_PASSWORD nor survive server shutdown.
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { existsSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const runDir = path.join(root, ".pi-web-run");
const usage = "node bin/windows-web.mjs start [--mode dev|start] [--port 30141]\n" +
  "node bin/windows-web.mjs restart|status|stop";

function readState() {
  let lines;
  try {
    lines = readFileSync(path.join(runDir, "state"), "utf8").trim().split(/\r?\n/);
  } catch {
    throw new Error("No manager found. First run .\\start-web.cmd in an EXTERNAL Windows terminal.");
  }
  const [runId, phase, heartbeat, generation, port, mode] = lines;
  if ((lines.length !== 6 && lines.length !== 7) || !/^[a-f0-9]{32}$/.test(runId) ||
      !["starting", "ready", "restarting", "updating", "failed", "stopped"].includes(phase) ||
      !/^\d+$/.test(heartbeat) || !/^\d+$/.test(generation) || !/^\d+$/.test(port) ||
      !["dev", "start"].includes(mode)) {
    throw new Error("Invalid manager state. Check the external launcher window.");
  }
  if (phase === "stopped" || Math.abs(Date.now() - Number(heartbeat)) > 20_000) {
    throw new Error("Manager is stopped or not responding. Check its external terminal; no process was killed.");
  }
  return { runId, phase, generation, port, mode };
}

function main() {
  if (process.platform !== "win32") throw new Error("This launcher is for native Windows only.");
  const [action, ...options] = process.argv.slice(2);
  if (!["start", "restart", "status", "stop"].includes(action)) throw new Error(usage);
  if (action !== "start" && options.length) throw new Error(usage);

  if (action !== "start") {
    const state = readState();
    if (action === "status") {
      console.log(`${state.phase} | ${state.mode} | 127.0.0.1:${state.port} | generation ${state.generation}`);
      console.log(`Logs: ${runDir}`);
      if (state.phase !== "ready") process.exitCode = 1;
      return;
    }
    // A run-specific filename cannot be replayed against a later manager.
    // Rename makes publication atomic; concurrent requests are coalesced.
    const request = path.join(runDir, `${action}-${state.runId}-${randomUUID()}.request`);
    const temporary = `${request}.tmp`;
    writeFileSync(temporary, "requested\n", { flag: "wx", mode: 0o600 });
    renameSync(temporary, request);
    console.log(`${action} request submitted. The embedded terminal may disconnect.`);
    console.log(action === "restart"
      ? "Wait for READY in the external launcher (or run status), then refresh the web page."
      : "Check the external launcher for STOPPED.");
    return;
  }

  if (process.env.PI_WEB_MANAGED_RUN) {
    throw new Error("Start the manager in an EXTERNAL Windows terminal. Here, use .\\restart.cmd instead.");
  }
  let mode = "dev";
  let port = "30141";
  for (let i = 0; i < options.length; i += 2) {
    if (options[i] === "--mode" && ["dev", "start"].includes(options[i + 1])) mode = options[i + 1];
    else if (options[i] === "--port" && /^\d+$/.test(options[i + 1] ?? "")) port = options[i + 1];
    else throw new Error(usage);
  }
  if (Number(port) < 1 || Number(port) > 65535) throw new Error("Port must be between 1 and 65535.");
  const nextCli = path.join(root, "node_modules", "next", "dist", "bin", "next");
  if (!existsSync(nextCli)) throw new Error("Install this checkout's dependencies first; Next.js was not found.");
  if (mode === "start" && !existsSync(path.join(root, ".next", "BUILD_ID"))) {
    throw new Error("Production build missing. Use dev for source changes; this launcher never builds automatically.");
  }
  const powershell = path.join(process.env.SystemRoot || "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  console.log("Keep this EXTERNAL terminal open. Restart stops this app and all work launched inside it.");
  console.log(`Mode: ${mode}; address: http://127.0.0.1:${Number(port)}; logs: ${runDir}`);
  // Static PowerShell source: paths/arguments are data, never interpolated code.
  // No execution-policy changes, service installation, or elevated privileges.
  const manager = spawn(powershell, ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
    "$ErrorActionPreference = 'Stop'; Add-Type -Path $env:PI_WEB_MANAGER_HELPER; " +
    "[PiWeb.WindowsHost]::Run($env:PI_WEB_MANAGER_ROOT, $env:PI_WEB_MANAGER_NODE, " +
    "$env:PI_WEB_MANAGER_MODE, [int]$env:PI_WEB_MANAGER_PORT)"], {
    cwd: root,
    env: { ...process.env, PI_WEB_MANAGER_HELPER: path.join(root, "bin", "windows-web-host.cs"),
      PI_WEB_MANAGER_ROOT: root, PI_WEB_MANAGER_NODE: process.execPath,
      PI_WEB_MANAGER_MODE: mode, PI_WEB_MANAGER_PORT: String(Number(port)) },
    // This private pipe is a lifetime lease. EOF tells the manager to stop its
    // job even if this Node launcher is killed without running JS handlers.
    stdio: ["pipe", "inherit", "inherit"],
  });
  manager.stdin.on("error", () => {});
  process.once("SIGINT", () => manager.stdin.end());
  process.once("SIGTERM", () => manager.stdin.end());
  manager.once("error", (error) => {
    console.error(`Cannot start Windows manager: ${error.message}`);
    process.exitCode = 1;
  });
  manager.once("exit", (code) => { process.exitCode = code ?? 1; });
}

try { main(); } catch (error) {
  console.error(error instanceof Error ? error.message : String(error));
  process.exitCode = 1;
}
