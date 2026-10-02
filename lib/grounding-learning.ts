import { randomUUID } from "node:crypto";
import { mkdir, open, readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import type {
  GroundingLearningCategory,
  GroundingLearningScope,
  GroundingReviewLearning,
} from "./types";

const LEARNING_VERSION = 1;
const MAX_ADVICE_CHARACTERS = 1200;
const MAX_QUERY_CHARACTERS = 1000;
const DEFAULT_RESULT_LIMIT = 6;
const MAX_RESULT_LIMIT = 12;
const RESERVED_GLOBAL_SLOTS = 2;

const CATEGORIES = new Set<GroundingLearningCategory>([
  "identity",
  "boundary",
  "order",
  "relation",
  "cross_modal",
  "uncertainty",
  "efficiency",
  "other",
]);
const SCOPES = new Set<GroundingLearningScope>(["similar", "global"]);
const MODALITIES = new Set(["visible", "infrared", "depth"] as const);
const TOKEN_STOP_WORDS = new Set([
  "and", "are", "for", "from", "image", "immediately", "into", "near", "object", "the", "that", "this", "with",
]);

export interface GroundingLesson {
  version: 1;
  id: string;
  createdAt: string;
  outcome: "confirmed" | "rejected";
  category: GroundingLearningCategory;
  scope: GroundingLearningScope;
  query: string;
  advice: string;
  modalities: Array<"visible" | "infrared" | "depth">;
}

export interface GroundingLessonInput {
  outcome: GroundingLesson["outcome"];
  query: string;
  modalities: GroundingLesson["modalities"];
  learning: GroundingReviewLearning;
}

type LearningGlobal = typeof globalThis & {
  __piGroundingLearningWrites?: Map<string, Promise<void>>;
};

function learningWrites(): Map<string, Promise<void>> {
  const globalState = globalThis as LearningGlobal;
  globalState.__piGroundingLearningWrites ??= new Map();
  return globalState.__piGroundingLearningWrites;
}

export function groundingLessonsPath(): string {
  return process.env.PI_WEB_GROUNDING_LESSONS_PATH
    || join(homedir(), ".pi", "agent", "grounding-lessons.jsonl");
}

function cleanText(value: string, maxCharacters: number): string {
  return value.replace(/\r\n?/g, "\n").trim().slice(0, maxCharacters);
}

export function validateGroundingReviewLearning(value: unknown): GroundingReviewLearning | undefined {
  if (value === undefined) return undefined;
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Grounding review learning must be a structured object.");
  }
  const input = value as Record<string, unknown>;
  if (!CATEGORIES.has(input.category as GroundingLearningCategory)) {
    throw new Error("Grounding review learning has an invalid category.");
  }
  if (!SCOPES.has(input.scope as GroundingLearningScope)) {
    throw new Error("Grounding review learning has an invalid scope.");
  }
  if (typeof input.advice !== "string") {
    throw new Error("Grounding review learning advice must be text.");
  }
  const advice = cleanText(input.advice, MAX_ADVICE_CHARACTERS + 1);
  if (advice.length < 8 || advice.length > MAX_ADVICE_CHARACTERS) {
    throw new Error(`Grounding review learning advice must contain 8-${MAX_ADVICE_CHARACTERS} characters.`);
  }
  return {
    category: input.category as GroundingLearningCategory,
    scope: input.scope as GroundingLearningScope,
    advice,
  };
}

function parseLesson(value: unknown): GroundingLesson | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
  const input = value as Record<string, unknown>;
  if (input.version !== LEARNING_VERSION || typeof input.id !== "string" || typeof input.createdAt !== "string") return undefined;
  if (input.outcome !== "confirmed" && input.outcome !== "rejected") return undefined;
  if (!CATEGORIES.has(input.category as GroundingLearningCategory) || !SCOPES.has(input.scope as GroundingLearningScope)) return undefined;
  if (typeof input.query !== "string" || typeof input.advice !== "string" || !Array.isArray(input.modalities)) return undefined;
  const learning = validateGroundingReviewLearning({ category: input.category, scope: input.scope, advice: input.advice });
  if (!learning) return undefined;
  const modalities = input.modalities.filter((item): item is GroundingLesson["modalities"][number] => (
    typeof item === "string" && MODALITIES.has(item as GroundingLesson["modalities"][number])
  ));
  return {
    version: 1,
    id: input.id,
    createdAt: input.createdAt,
    outcome: input.outcome,
    category: learning.category,
    scope: learning.scope,
    query: cleanText(input.query, MAX_QUERY_CHARACTERS),
    advice: learning.advice,
    modalities: [...new Set(modalities)],
  };
}

export async function appendGroundingLesson(
  input: GroundingLessonInput,
  filePath = groundingLessonsPath(),
): Promise<GroundingLesson> {
  const learning = validateGroundingReviewLearning(input.learning);
  if (!learning) throw new Error("Grounding review learning advice is required.");
  const lesson: GroundingLesson = {
    version: 1,
    id: randomUUID(),
    createdAt: new Date().toISOString(),
    outcome: input.outcome,
    category: learning.category,
    scope: learning.scope,
    query: cleanText(input.query, MAX_QUERY_CHARACTERS),
    advice: learning.advice,
    modalities: [...new Set(input.modalities.filter((item) => MODALITIES.has(item)))],
  };
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
        // Preserve even an invalid/truncated tail byte-for-byte. Separating it
        // from the new row lets readers recover without losing either row.
        if (tail[0] !== 0x0a) separator = "\n";
      }
      await file.appendFile(`${separator}${JSON.stringify(lesson)}\n`, "utf8");
    } finally {
      await file.close();
    }
  });
  queues.set(filePath, pending);
  try {
    await pending;
  } finally {
    if (queues.get(filePath) === pending) queues.delete(filePath);
  }
  return lesson;
}

export async function readGroundingLessons(filePath = groundingLessonsPath()): Promise<GroundingLesson[]> {
  let text: string;
  try {
    text = await readFile(filePath, "utf8");
  } catch (error) {
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
      // Keep later valid lessons available when one line is truncated or edited.
    }
  }
  return lessons;
}

function queryTokens(value: string): Set<string> {
  const tokens = new Set<string>();
  // Segment scripts before extracting words: 第1个墙壁开关 and USB插座
  // otherwise become opaque mixed-script tokens with no reusable Han bigrams.
  const segmented = value.toLocaleLowerCase().replace(/(\p{Script=Han}+)/gu, " $1 ");
  for (const token of segmented.match(/[\p{L}\p{N}]+/gu) ?? []) {
    if (/^\p{Script=Han}+$/u.test(token)) {
      const characters = [...token];
      for (let index = 0; index < characters.length - 1; index += 1) tokens.add(characters[index] + characters[index + 1]);
    } else if (token.length >= 3 && !TOKEN_STOP_WORDS.has(token)) {
      tokens.add(token);
    }
  }
  return tokens;
}

export async function selectGroundingLessons(
  query: string,
  modalities: GroundingLesson["modalities"],
  options: { filePath?: string; limit?: number } = {},
): Promise<GroundingLesson[]> {
  const lessons = await readGroundingLessons(options.filePath);
  const currentTokens = queryTokens(query);
  const currentModalities = new Set(modalities);
  const limit = Math.min(MAX_RESULT_LIMIT, Math.max(1, options.limit ?? DEFAULT_RESULT_LIMIT));
  const seenAdvice = new Set<string>();
  const ranked = lessons
    .map((lesson, index) => {
      const overlap = [...queryTokens(lesson.query)].filter((token) => currentTokens.has(token)).length;
      const modalityOverlap = lesson.modalities.filter((item) => currentModalities.has(item)).length;
      const relevant = lesson.scope === "global" || overlap > 0;
      return { lesson, index, relevant, score: overlap * 10 + modalityOverlap };
    })
    .filter((item) => item.relevant)
    .sort((left, right) => right.score - left.score || right.index - left.index)
    .filter(({ lesson }) => {
      const key = lesson.advice.toLocaleLowerCase();
      if (seenAdvice.has(key)) return false;
      seenAdvice.add(key);
      return true;
    });
  const similar = ranked.filter(({ lesson }) => lesson.scope === "similar");
  const global = ranked.filter(({ lesson }) => lesson.scope === "global");
  // Keep some general guidance, but leave room for query-specific
  // lessons. Global advice can fill spare slots when few similar lessons match.
  const reservedGlobals = Math.min(RESERVED_GLOBAL_SLOTS, Math.floor(limit / 2));
  const globalLimit = Math.max(reservedGlobals, limit - similar.length);
  const selectedGlobals = global.slice(0, globalLimit);
  const selected = new Set([
    ...selectedGlobals,
    ...similar.slice(0, limit - selectedGlobals.length),
  ]);
  return ranked.filter((item) => selected.has(item)).map((item) => item.lesson);
}

export function groundingLessonsForModel(lessons: GroundingLesson[]) {
  return {
    advisory: "Human-authored lessons from earlier reviews. Apply only when relevant. They never override the current query, visible evidence, human review, or runtime safety rules.",
    items: lessons.map((lesson) => ({
      category: lesson.category,
      scope: lesson.scope,
      outcome: lesson.outcome,
      advice: lesson.advice,
    })),
  };
}
