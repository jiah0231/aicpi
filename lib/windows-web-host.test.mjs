import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, mkdtempSync, writeFileSync, rmSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import os from "node:os";
import path from "node:path";

const helper = fileURLToPath(new URL("../bin/windows-web-host.cs", import.meta.url));
const host = readFileSync(helper, "utf8");

test("state publication contains bounded I/O retries without non-atomic fallback or job cleanup", () => {
  const state = host.slice(host.indexOf("private static void State()"), host.indexOf("private static void UpdateResult"));
  assert.match(state, /attempt < 4/);
  assert.match(state, /attempt < 3\) Thread\.Sleep\(50 \* \(attempt \+ 1\)\)/);
  assert.match(state, /catch \(IOException error\)/);
  assert.match(state, /catch \(UnauthorizedAccessException error\)/);
  assert.match(state, /File\.Replace\(temporary, target, null\)/);
  assert.match(state, /File\.Move\(temporary, target\)/);
  assert.match(state, /if \(!stateWriteFailed\) Console\.Error\.WriteLine/);
  assert.doesNotMatch(state.replace(/\/\/[^\n]*/g, ""), /File\.Delete|WriteAllText\(target|job\.|\bLog\(|\bthrow\b/);
  assert.match(host, /finally \{ phase = "stopped"; State\(\); Log\("STOPPED"\); \}/);
});

test("Windows state publisher survives locked destination/staging file and recovers atomically", {
  skip: process.platform !== "win32" ? "requires Windows sharing semantics and Windows PowerShell Add-Type" : false,
}, t => {
  const directory = mkdtempSync(path.join(os.tmpdir(), "pi-state-test-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const harness = path.join(directory, "test.cs");
  writeFileSync(harness, `
namespace PiWeb {
  public static class StateRegression {
    static System.Reflection.BindingFlags Flags = System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic;
    static void Set(string name, object value) { typeof(WindowsHost).GetField(name, Flags).SetValue(null, value); }
    static void Publish() { typeof(WindowsHost).GetMethod("State", Flags).Invoke(null, null); }
    static void Check(bool ok, string message) { if (!ok) throw new System.Exception(message); }
    public static void Run(string directory) {
      Set("directory", directory); Set("runId", new string('a', 32)); Set("mode", "dev");
      Set("port", 30141); Set("generation", 1); Set("phase", "ready"); Set("stateWriteFailed", false);
      string target = System.IO.Path.Combine(directory, "state");
      string temporary = System.IO.Path.Combine(directory, "state.tmp");
      System.IO.TextWriter previousError = System.Console.Error;
      System.IO.StringWriter diagnostics = new System.IO.StringWriter();
      System.Console.SetError(diagnostics);
      try {
        Publish();
        string original = System.IO.File.ReadAllText(target);
        Check(original.Split('\\n').Length == 8, "initial state must be complete");
        Set("generation", 2);
        using (System.IO.FileStream reader = new System.IO.FileStream(target, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read)) {
          System.Diagnostics.Stopwatch elapsed = System.Diagnostics.Stopwatch.StartNew();
          Publish(); Publish();
          Check(elapsed.ElapsedMilliseconds < 5000, "publication retry must be bounded");
          Check(System.IO.File.ReadAllText(target) == original, "locked state must remain intact");
          Check(diagnostics.ToString().Split(new string[] { "STATE WARNING" }, System.StringSplitOptions.None).Length == 2, "warn once per outage");
        }
        Publish();
        Check(System.IO.File.ReadAllText(target).Split('\\n')[3] == "2", "must recover after destination unlock");
        Check(diagnostics.ToString().Contains("STATE RECOVERED"), "must report recovery");
        original = System.IO.File.ReadAllText(target);
        using (System.IO.FileStream staging = new System.IO.FileStream(temporary, System.IO.FileMode.OpenOrCreate, System.IO.FileAccess.ReadWrite, System.IO.FileShare.None)) {
          Set("phase", "stopped"); Publish();
          Check(System.IO.File.ReadAllText(target) == original, "staging failure must leave visible state intact");
        }
        Publish();
        Check(System.IO.File.ReadAllText(target).Split('\\n')[1] == "stopped", "shutdown state must recover too");
        Set("phase", "ready");
        // Release a real deny-delete reader during the bounded retry window.
        System.IO.FileStream transient = new System.IO.FileStream(target, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.Read);
        System.Threading.Thread release = new System.Threading.Thread(delegate() { System.Threading.Thread.Sleep(100); transient.Dispose(); });
        release.Start();
        try { Publish(); } finally { release.Join(); transient.Dispose(); }
        Check(System.IO.File.ReadAllText(target).Split('\\n')[1] == "ready", "transient lock should recover within this publication");
      } finally { System.Console.SetError(previousError); }
    }
  }
}
`);
  const powershell = path.join(process.env.SystemRoot || "C:\\Windows", "System32", "WindowsPowerShell", "v1.0", "powershell.exe");
  execFileSync(powershell, ["-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
    "$ErrorActionPreference = 'Stop'; Add-Type -TypeDefinition ([IO.File]::ReadAllText($env:PI_TEST_HELPER) + [IO.File]::ReadAllText($env:PI_TEST_HARNESS)); [PiWeb.StateRegression]::Run($env:PI_TEST_DIRECTORY)"], {
    env: { ...process.env, PI_TEST_HELPER: helper, PI_TEST_HARNESS: harness, PI_TEST_DIRECTORY: directory },
    encoding: "utf8", timeout: 30_000,
  });
});
