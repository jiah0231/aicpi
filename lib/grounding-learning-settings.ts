import { mkdir, readFile } from "node:fs/promises";
import { dirname } from "node:path";
import lockfile from "proper-lockfile";
import { writePrivateFileAtomicSync } from "./atomic-file";
import { groundingLessonsPath } from "./grounding-learning";

export interface GroundingLearningSettings {
  enabled: boolean; provider: string; modelId: string; revision: number;
  maxSources: number; maxInputChars: number; maxOutputTokens: number;
  timeoutMs: number; dailyAttemptLimit: number;
}
export const DEFAULT_GROUNDING_LEARNING_SETTINGS: GroundingLearningSettings = {
  enabled: false, provider: "", modelId: "", revision: 0, maxSources: 8,
  maxInputChars: 12000, maxOutputTokens: 1600, timeoutMs: 60000, dailyAttemptLimit: 4,
};
export function groundingLearningSettingsPath(learningPath = groundingLessonsPath()): string {
  return `${learningPath}.consolidation-settings.json`;
}
const bounds = { maxSources: [1, 8], maxInputChars: [1000, 12000], maxOutputTokens: [128, 1600], timeoutMs: [1000, 60000], dailyAttemptLimit: [1, 4] } as const;
function validatePatch(value: unknown): Partial<GroundingLearningSettings> {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid consolidation settings");
  const input = value as Record<string, unknown>;
  const result: Record<string, unknown> = {};
  for (const [key, val] of Object.entries(input)) {
    if (key === "enabled") { if (typeof val !== "boolean") throw new Error("Invalid enabled setting"); }
    else if (key === "provider" || key === "modelId") {
      if (typeof val !== "string" || val.length > 200 || /[\r\n\x00-\x1f]/.test(val) || /https?:\/\/|sk-|bearer\s/i.test(val)) throw new Error("Invalid model selection");
    } else if (key in bounds) {
      const [min, max] = bounds[key as keyof typeof bounds];
      if (typeof val !== "number" || !Number.isInteger(val) || val < min || val > max) throw new Error(`Invalid ${key} setting`);
    } else throw new Error("Unknown consolidation setting");
    result[key] = val;
  }
  return result;
}
export async function readGroundingLearningSettings(learningPath?: string): Promise<GroundingLearningSettings> {
  try {
    const stored = JSON.parse(await readFile(groundingLearningSettingsPath(learningPath), "utf8"));
    const { revision, ...values } = stored;
    if (!Number.isSafeInteger(revision) || revision < 0) throw new Error("Invalid revision");
    return { ...DEFAULT_GROUNDING_LEARNING_SETTINGS, ...validatePatch(values), revision };
  } catch { return { ...DEFAULT_GROUNDING_LEARNING_SETTINGS }; }
}
export async function withGroundingLearningSettingsLock<T>(learningPath: string | undefined, operation: (canCommit: () => boolean) => Promise<T>): Promise<T> {
  const path = groundingLearningSettingsPath(learningPath);
  await mkdir(dirname(path), { recursive: true });
  let compromised = false;
  const release = await lockfile.lock(path, { realpath: false, onCompromised: () => { compromised = true; }, retries: { retries: 10, minTimeout: 20, maxTimeout: 100 } });
  try { return await operation(() => !compromised); } finally { await release().catch(() => {}); }
}
export async function updateGroundingLearningSettings(patch: unknown, learningPath?: string, expectedRevision?: number): Promise<GroundingLearningSettings> {
  const validated = validatePatch(patch);
  const result = await withGroundingLearningSettingsLock(learningPath, async canCommit => {
    const current = await readGroundingLearningSettings(learningPath);
    if (expectedRevision !== undefined && expectedRevision !== current.revision) throw new Error("Consolidation settings changed; refresh and retry");
    const next = { ...current, ...validated, revision: current.revision + 1 };
    if (next.enabled && (!next.provider.trim() || !next.modelId.trim())) throw new Error("Select a configured provider and model first");
    if (!canCommit()) throw new Error("Consolidation settings lock was interrupted");
    writePrivateFileAtomicSync(groundingLearningSettingsPath(learningPath), JSON.stringify(next));
    return next;
  });
  // Immediate same-process cancellation; no provider calls are made by this path.
  (globalThis as typeof globalThis & { __piGroundingLearningChanged?: (path?: string) => void }).__piGroundingLearningChanged?.(learningPath);
  return result;
}
