import { mkdir, readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import lockfile from "proper-lockfile";
import { writePrivateFileAtomicSync } from "./atomic-file";
import { groundingLessonsPath } from "./grounding-learning";
import { readGroundingLearningSettings, withGroundingLearningSettingsLock } from "./grounding-learning-settings";
import { snapshotConsolidationBatch, saveConsolidationProposals, learningStoreOptionsForLegacyPath } from "./grounding-learning-store";
import { consolidateGenericLearning } from "./grounding-learning-consolidator";
import type { ConsolidationCall, LearningStoreOptions } from "./grounding-learning-consolidation-types";

export interface GroundingLearningJobStatus {
  status: "disabled" | "waiting" | "running" | "budget" | "backoff" | "interrupted" | "error";
  attemptsToday: number; dailyAttemptLimit: number;
  lastAttemptAt?: string; lastCompletedAt?: string; nextAttemptAt?: string;
  errorCode?: "request_failed" | "interrupted" | "storage_error";
}
interface JobState {
  version: 1; day: string; attempts: number; processedSourceIds: string[]; attemptedSourceIds?: string[];
  lastAttemptDigest?: string; running?: boolean; lastAttemptAt?: string;
  lastCompletedAt?: string; nextAttemptAt?: string; errorCode?: GroundingLearningJobStatus["errorCode"];
}
interface Runtime { quarantined?: boolean; controller?: AbortController; busy?: boolean; digest?: string; changedAt?: number; }
interface WorkerGlobal { runtimes: Map<string, Runtime>; timer?: ReturnType<typeof setInterval>; }
const root = globalThis as typeof globalThis & { __piGroundingLearningWorker?: WorkerGlobal; __piGroundingLearningChanged?: (path?: string) => void };
function globalWorker(): WorkerGlobal { return root.__piGroundingLearningWorker ??= { runtimes: new Map() }; }
function runtime(path: string): Runtime { const g = globalWorker(); if (!g.runtimes.has(path)) g.runtimes.set(path, {}); return g.runtimes.get(path)!; }
function statePath(path: string): string { return `${path}.consolidation-state.json`; }
export function groundingLearningWorkerStoreOptions(learningPath?: string): LearningStoreOptions {
  return learningStoreOptionsForLegacyPath(learningPath);
}
async function readState(path: string, now: number): Promise<JobState> {
  const day = new Date(now).toISOString().slice(0, 10);
  try {
    const s = JSON.parse(await readFile(statePath(path), "utf8")) as JobState;
    const fields = ["version", "day", "attempts", "processedSourceIds", "attemptedSourceIds", "lastAttemptDigest", "running", "lastAttemptAt", "lastCompletedAt", "nextAttemptAt", "errorCode"];
    const idsValid = (ids: unknown) => Array.isArray(ids) && ids.every(id => typeof id === "string" && /^[a-f0-9]{64}$/.test(id));
    if (!s || typeof s !== "object" || Array.isArray(s) || Object.keys(s).some(key => !fields.includes(key)) ||
      s.version !== 1 || !/^\d{4}-\d{2}-\d{2}$/.test(s.day) || !Number.isSafeInteger(s.attempts) || s.attempts < 0 ||
      !idsValid(s.processedSourceIds) || (s.attemptedSourceIds !== undefined && !idsValid(s.attemptedSourceIds)) ||
      (s.lastAttemptDigest !== undefined && !/^[a-f0-9]{64}$/.test(s.lastAttemptDigest)) ||
      (s.running !== undefined && typeof s.running !== "boolean") ||
      (s.errorCode !== undefined && !["request_failed", "interrupted", "storage_error"].includes(s.errorCode)) ||
      [s.lastAttemptAt, s.lastCompletedAt, s.nextAttemptAt].some(value => value !== undefined &&
        (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/.test(value) || !Number.isFinite(Date.parse(value))))) throw new Error("Invalid state");
    if (!Number.isFinite(Date.parse(s.day)) || new Date(s.day).toISOString().slice(0, 10) !== s.day || s.day > day) throw new Error("Invalid quota day");
    return { ...s, day, attempts: s.day === day ? s.attempts : 0 };
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw new Error("Cannot safely read consolidation quota");
    return { version: 1, day, attempts: 0, processedSourceIds: [] };
  }
}
function writeState(path: string, state: JobState): void { writePrivateFileAtomicSync(statePath(path), JSON.stringify(state)); }
export function notifyGroundingLearningSettingsChanged(learningPath?: string): void {
  runtime(resolve(learningPath ?? groundingLessonsPath())).controller?.abort();
}
root.__piGroundingLearningChanged = notifyGroundingLearningSettingsChanged;
export async function getGroundingLearningJobStatus(learningPath?: string): Promise<GroundingLearningJobStatus> {
  const path = resolve(learningPath ?? groundingLessonsPath());
  const settings = await readGroundingLearningSettings(learningPath);
  const base = { attemptsToday: 0, dailyAttemptLimit: settings.dailyAttemptLimit };
  try {
    const state = await readState(path, Date.now());
    const externallyRunning = state.running && await lockfile.check(statePath(path), { realpath: false, stale: 120000 });
    const status = !settings.enabled ? "disabled" : runtime(path).quarantined ? "interrupted" : (runtime(path).busy || externallyRunning) ? "running" : state.running ? "interrupted" : state.attempts >= settings.dailyAttemptLimit ? "budget" : state.nextAttemptAt && Date.parse(state.nextAttemptAt) > Date.now() ? "backoff" : "waiting";
    return { ...base, status, attemptsToday: state.attempts, lastAttemptAt: state.lastAttemptAt, lastCompletedAt: state.lastCompletedAt, nextAttemptAt: state.nextAttemptAt, errorCode: state.running && !externallyRunning && !runtime(path).busy ? "interrupted" : state.errorCode };
  } catch { return { ...base, status: settings.enabled ? "error" : "disabled", errorCode: "storage_error" }; }
}
/** Server-only entry; tests inject a fake model call and a clock, never samples. */
export async function tickGroundingLearningWorker(options: { learningPath?: string; now?: number; debounceMs?: number; call?: ConsolidationCall } = {}): Promise<void> {
  const path = resolve(options.learningPath ?? groundingLessonsPath());
  const rt = runtime(path);
  const initial = await readGroundingLearningSettings(options.learningPath);
  if (!initial.enabled) { rt.controller?.abort(); return; }
  if (rt.busy) return;
  rt.busy = true;
  const controller = rt.controller = new AbortController();
  let release: (() => Promise<void>) | undefined;
  let providerPending: Promise<unknown> | undefined;
  let providerSettled = true;
  let compromised = false;
  try {
    await mkdir(dirname(path), { recursive: true });
    try { release = await lockfile.lock(statePath(path), { realpath: false, stale: 120000, update: 10000, retries: 0, onCompromised: () => { compromised = true; controller.abort(); } }); }
    catch (error) { if ((error as NodeJS.ErrnoException).code === "ELOCKED") return; throw error; }
    const now = options.now ?? Date.now();
    const settings = await readGroundingLearningSettings(options.learningPath);
    if (!settings.enabled) return;
    const state = await readState(path, now);
    if (compromised || controller.signal.aborted) return;
    if (state.running) { state.running = false; state.errorCode = "interrupted"; state.nextAttemptAt = new Date(now + 60000).toISOString(); writeState(path, state); }
    if (state.attempts >= settings.dailyAttemptLimit || (state.nextAttemptAt && Date.parse(state.nextAttemptAt) > now)) return;
    const storeOptions = groundingLearningWorkerStoreOptions(options.learningPath);
    const batch = await snapshotConsolidationBatch({ ...storeOptions, excludeSourceIds: [...state.processedSourceIds, ...(state.attemptedSourceIds ?? [])], maxItems: settings.maxSources, maxChars: settings.maxInputChars });
    if (!batch.sources.length || batch.inputDigest === state.lastAttemptDigest) return;
    if (rt.digest !== batch.inputDigest) { rt.digest = batch.inputDigest; rt.changedAt = now; }
    if (now - (rt.changedAt ?? now) < (options.debounceMs ?? 60000)) return;
    const reserved = await withGroundingLearningSettingsLock(options.learningPath, async settingsCanCommit => {
      const latest = await readGroundingLearningSettings(options.learningPath);
      if (!settingsCanCommit() || compromised || controller.signal.aborted || !latest.enabled || latest.revision !== settings.revision) return false;
      state.attemptedSourceIds = [...new Set([...(state.attemptedSourceIds ?? []), ...batch.sources.map(source => source.id)])];
      state.attempts++; state.lastAttemptDigest = batch.inputDigest; state.lastAttemptAt = new Date(now).toISOString(); state.running = true; delete state.errorCode;
      writeState(path, state); // Commit the charge reservation BEFORE any network request.
      return true;
    });
    if (!reserved) return;
    const timeout = setTimeout(() => controller.abort(), settings.timeoutMs); timeout.unref();
    // Cross-process settings changes must cancel requests too.
    const poll = setInterval(() => { void readGroundingLearningSettings(options.learningPath).then(s => { if (!s.enabled || s.revision !== settings.revision) controller.abort(); }).catch(() => controller.abort()); }, 1000); poll.unref();
    try {
      const aborted = new Promise<never>((_, reject) => {
        controller.signal.addEventListener("abort", () => reject(new Error("Consolidation cancelled")), { once: true });
      });
      providerSettled = false;
      const response = consolidateGenericLearning(batch, { provider: settings.provider, modelId: settings.modelId, maxOutputTokens: settings.maxOutputTokens, timeoutMs: settings.timeoutMs, signal: controller.signal }, options.call);
      providerPending = response.then(() => { providerSettled = true; }, () => { providerSettled = true; });
      const proposals = await Promise.race([aborted, response]);
      await withGroundingLearningSettingsLock(options.learningPath, async settingsCanCommit => {
        const latest = await readGroundingLearningSettings(options.learningPath);
        if (!settingsCanCommit() || compromised || controller.signal.aborted || !latest.enabled || latest.revision !== settings.revision) throw new Error("Consolidation cancelled");
        await saveConsolidationProposals(batch, proposals, storeOptions, () => !compromised && !controller.signal.aborted && settingsCanCommit());
      });
      state.processedSourceIds = [...new Set([...state.processedSourceIds, ...batch.sources.map(source => source.id)])];
      state.lastCompletedAt = new Date(now).toISOString(); delete state.nextAttemptAt;
    } catch {
      state.errorCode = "request_failed";
      state.nextAttemptAt = new Date(now + Math.min(3600000, 60000 * 2 ** Math.max(0, state.attempts - 1))).toISOString();
    } finally {
      clearTimeout(timeout); clearInterval(poll); rt.controller = undefined;
      state.running = false; if (!compromised) writeState(path, state);
    }
  } finally {
    rt.controller = undefined;
    if (!providerSettled && providerPending && release) {
      // An adapter ignoring cancellation may still be charging. Keep the job
      // lock until it settles; never overlap another paid request with it.
      rt.quarantined = true;
      const retainedRelease = release;
      void providerPending.then(async () => { await retainedRelease(); }).catch(() => {}).finally(() => { rt.busy = false; rt.quarantined = false; });
    } else { rt.busy = false; if (release) await release().catch(() => {}); }
  }
}
/** Explicit user retry only: retains quota, backoff, successes and all safety locks. */
export async function requestGroundingLearningRetry(expectedRevision: number, learningPath?: string): Promise<GroundingLearningJobStatus> {
  const path = resolve(learningPath ?? groundingLessonsPath());
  if (runtime(path).busy) throw new Error("A consolidation request is still running");
  await mkdir(dirname(path), { recursive: true });
  let compromised = false;
  const release = await lockfile.lock(statePath(path), { realpath: false, retries: 0, stale: 120000, onCompromised: () => { compromised = true; } });
  try {
    await withGroundingLearningSettingsLock(learningPath, async settingsCanCommit => {
      const settings = await readGroundingLearningSettings(learningPath);
      if (!settings.enabled || !Number.isSafeInteger(expectedRevision) || expectedRevision !== settings.revision) throw new Error("Consolidation settings changed; refresh and retry");
      const state = await readState(path, Date.now());
      if (state.running) {
        state.running = false; state.errorCode = "interrupted";
        state.nextAttemptAt = new Date(Date.now() + 60000).toISOString();
      }
      state.attemptedSourceIds = [...state.processedSourceIds];
      delete state.lastAttemptDigest;
      if (compromised || !settingsCanCommit()) throw new Error("Consolidation retry lock was interrupted");
      writeState(path, state);
      runtime(path).digest = undefined;
    });
  } finally { await release().catch(() => {}); }
  return getGroundingLearningJobStatus(learningPath);
}
export function startGroundingLearningWorker(): void {
  const g = globalWorker();
  if (g.timer) return;
  const tick = () => { void tickGroundingLearningWorker().catch(() => { /* Fail closed; public status exposes storage failure without details. */ }); };
  g.timer = setInterval(tick, 60000); g.timer.unref(); tick();
}
export function stopGroundingLearningWorker(): void {
  const g = globalWorker(); if (g.timer) clearInterval(g.timer); g.timer = undefined;
  for (const rt of g.runtimes.values()) rt.controller?.abort();
}
