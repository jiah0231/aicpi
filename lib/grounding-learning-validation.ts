import type { GroundingLearningCategory, GroundingReviewLearning } from "./types";

const CATEGORIES = new Set<GroundingLearningCategory>([
  "identity", "boundary", "order", "relation", "cross_modal", "uncertainty", "efficiency", "other",
]);
const FIELDS = ["category", "applicability", "error", "method", "check", "sampleIndependent"] as const;
const TEXT_LIMITS = { applicability: [4, 240], error: [4, 400], method: [8, 800], check: [4, 400] } as const;

// Reject obvious sample-bearing payloads, not ordinary discussion of generic
// query interpretation or coordinate measurement. These are lexical checks;
// they cannot prove that paraphrased prose is semantically sample-independent.
const SAMPLE_PAYLOAD = /https?:\/\/|data:|<img\b|\b[a-z]:[\\/]|(?:^|\s)(?:\/|\.\.?[\\/]|~[\\/])\S+|\b\S+\.(?:png|jpe?g|webp|bmp|tiff?|gif|jsonl?|csv|zip)\b|\b(?:originalQuery|query|sampleId|recordId|imagePath|bbox|answer|ground[_ -]?truth)["']?\s*[:=]|(?:原始\s*(?:问题|题目|query)|样本\s*(?:编号|ID)|图片路径|真值|答案)\s*[:：=]|\b(?:novel|record|sample|frame|image)[_-][\w-]*\d[\w-]*\b|\b[\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12}\b|\[\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?(?:\s*,\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?){3}\s*\]|\b(?:x|y|x1|y1|x2|y2)\s*[:=]\s*-?\d|[A-Za-z0-9+/]{120,}={0,2}/i;

function object(input: unknown, allowed: readonly string[]): Record<string, unknown> {
  if (!input || typeof input !== "object" || Array.isArray(input)) {
    throw new Error("Grounding review learning must be a structured generic procedure.");
  }
  const value = input as Record<string, unknown>;
  if (Object.keys(value).some((key) => !allowed.includes(key))) {
    // Never echo a submitted sample field or its value into the model context.
    throw new Error("Grounding learning accepts only generic applicability, error, method and check fields plus category and the sample-independent confirmation. Source/sample fields and legacy advice are not allowed.");
  }
  return value;
}

export function validateGroundingReviewLearning(value: unknown): GroundingReviewLearning | undefined {
  if (value === undefined) return undefined;
  const input = object(value, FIELDS);
  if (!CATEGORIES.has(input.category as GroundingLearningCategory)) throw new Error("Grounding review learning has an invalid category.");
  if (input.sampleIndependent !== true) {
    throw new Error("Confirm this is a general procedure with no original or paraphrased sample query, images, paths, sample IDs, concrete answers, boxes or ground truth.");
  }
  const fields = {} as Pick<GroundingReviewLearning, keyof typeof TEXT_LIMITS>;
  for (const field of Object.keys(TEXT_LIMITS) as Array<keyof typeof TEXT_LIMITS>) {
    const [minimum, maximum] = TEXT_LIMITS[field];
    if (typeof input[field] !== "string") throw new Error(`Grounding learning ${field} must be generic procedural text.`);
    const text = (input[field] as string).replace(/\r\n?/g, "\n").trim();
    if (text.length < minimum || text.length > maximum) throw new Error(`Grounding learning ${field} must contain ${minimum}-${maximum} characters.`);
    if (SAMPLE_PAYLOAD.test(text)) throw new Error(`Grounding learning ${field} contains an apparent sample identifier, file/image payload, answer field or coordinates. Write a general method without sample details; no automatic anonymization is performed.`);
    fields[field] = text;
  }
  return { category: input.category as GroundingLearningCategory, ...fields, sampleIndependent: true };
}


/** Lexical rejection only; human review remains the semantic boundary. */
export function validateGenericLearningText(value: unknown, minimum = 8, maximum = 2400): string {
  if (typeof value !== "string") throw new Error("Generic learning text is required.");
  const text = value.replace(/\r\n?/g, "\n").trim();
  if (text.length < minimum || text.length > maximum || SAMPLE_PAYLOAD.test(text)) {
    throw new Error("Generic learning text is invalid or contains apparent sample data. Remove original or paraphrased sample content; no automatic anonymization is performed.");
  }
  return text;
}
