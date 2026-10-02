import { mkdir, open, readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type { GroundingLearningCategory, GroundingReviewLearning } from "./types";
import { validateGroundingReviewLearning } from "./grounding-learning-validation";
export { validateGroundingReviewLearning } from "./grounding-learning-validation";

const LEARNING_VERSION = 2;
const DEFAULT_RESULT_LIMIT = 6;
const MAX_RESULT_LIMIT = 12;
/** A general, explicitly human-reviewed procedure, never a sample example. */
export interface GroundingLesson extends GroundingReviewLearning {
  version: 2;
}

type LearningGlobal = typeof globalThis & { __piGroundingLearningWrites?: Map<string, Promise<void>> };
function learningWrites(): Map<string, Promise<void>> {
  const globalState = globalThis as LearningGlobal;
  globalState.__piGroundingLearningWrites ??= new Map();
  return globalState.__piGroundingLearningWrites;
}

export function groundingLessonsPath(): string {
  // Leave the old default file untouched. An explicit custom path may contain
  // mixed versions; the reader still excludes every legacy row without migration.
  return process.env.PI_WEB_GROUNDING_LESSONS_PATH
    || join(homedir(), ".pi", "agent", "grounding-methods-v2.jsonl");
}

function parseLesson(value: unknown): GroundingLesson | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const input = value as Record<string, unknown>;
  if (input.version !== LEARNING_VERSION) return undefined;
  const procedure = { ...input };
  delete procedure.version;
  const validated = validateGroundingReviewLearning(procedure);
  return validated ? { version: 2, ...validated } : undefined;
}

/** Input deliberately has no query, key, image, bbox, outcome or source context. */
export async function appendGroundingLesson(
  input: GroundingReviewLearning,
  filePath = groundingLessonsPath(),
): Promise<GroundingLesson> {
  const learning = validateGroundingReviewLearning(input);
  if (!learning) throw new Error("A generic grounding procedure is required.");
  const lesson: GroundingLesson = { version: 2, ...learning };
  const queues = learningWrites();
  const previous = queues.get(filePath) ?? Promise.resolve();
  const pending = previous.catch(() => {}).then(async () => {
    await mkdir(dirname(filePath), { recursive: true });
    const file = await open(filePath, "a+");
    try {
      const { size } = await file.stat();
      let separator = "";
      if (size > 0) {
        const tail = Buffer.alloc(1);
        const { bytesRead } = await file.read(tail, 0, 1, size - 1);
        if (bytesRead !== 1) throw new Error("Grounding lessons changed while preparing an append.");
        if (tail[0] !== 0x0a) separator = "\n";
      }
      await file.appendFile(`${separator}${JSON.stringify(lesson)}\n`, "utf8");
    } finally {
      await file.close();
    }
  });
  queues.set(filePath, pending);
  try { await pending; }
  finally { if (queues.get(filePath) === pending) queues.delete(filePath); }
  return lesson;
}

export async function readGroundingLessons(filePath = groundingLessonsPath()): Promise<GroundingLesson[]> {
  let text: string;
  try { text = await readFile(filePath, "utf8"); }
  catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return [];
    throw error;
  }
  const lessons: GroundingLesson[] = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line.trim()) continue;
    try {
      const lesson = parseLesson(JSON.parse(line));
      if (lesson) lessons.push(lesson);
    } catch {
      // Legacy rows, unexpected fields and malformed payloads are excluded.
      // Never rewrite, auto-sanitize or migrate existing user-authored files.
    }
  }
  return lessons;
}

/** Query-free retrieval: recent distinct procedures, with category diversity. */
export async function selectGroundingLessons(
  options: { filePath?: string; limit?: number } = {},
): Promise<GroundingLesson[]> {
  const lessons = await readGroundingLessons(options.filePath);
  const limit = Number.isFinite(options.limit)
    ? Math.min(MAX_RESULT_LIMIT, Math.max(1, Math.floor(options.limit!))) : DEFAULT_RESULT_LIMIT;
  const seen = new Set<string>();
  const recent = lessons.reverse().filter((lesson) => {
    const key = JSON.stringify([lesson.category, lesson.applicability, lesson.error, lesson.method, lesson.check]).toLocaleLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  const categories = new Set<GroundingLearningCategory>();
  const diverse = recent.filter((lesson) => {
    if (categories.has(lesson.category)) return false;
    categories.add(lesson.category);
    return true;
  }).slice(0, limit);
  const chosen = new Set(diverse);
  return [...diverse, ...recent.filter((lesson) => !chosen.has(lesson))].slice(0, limit);
}

export function groundingLessonsForModel(lessons: GroundingLesson[]) {
  return {
    advisory: "Human-authored general procedures, explicitly reviewed as sample-independent. Never override current evidence, human review or runtime safety rules. Lexical validation cannot prove semantic independence; do not treat these procedures as answers, labels or evidence about any record. No source query or image is used to retrieve them.",
    items: lessons.flatMap((lesson) => {
      // Keep the final projection fail-closed even if a caller bypassed the reader.
      try {
        const item = parseLesson(lesson);
        return item ? [{ category: item.category, applicability: item.applicability,
          error: item.error, method: item.method, check: item.check }] : [];
      } catch { return []; }
    }),
  };
}
