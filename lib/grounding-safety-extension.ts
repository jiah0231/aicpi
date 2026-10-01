import { execFile } from "node:child_process";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, extname, join, parse, resolve } from "node:path";
import type {
  ExtensionAPI,
  ExtensionContext,
  InlineExtension,
  ToolCallEvent,
  ToolResultEvent,
} from "@earendil-works/pi-coding-agent";
import { resizeImage, withFileMutationQueue } from "@earendil-works/pi-coding-agent";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import JSZip from "jszip";
import sharp from "sharp";
import { Type } from "typebox";
import type { GroundingReviewDetails, GroundingReviewResponse } from "./types";
import { analyzeGroundingColor } from "./grounding-color";
import { buildGroundingComparison, GroundingViewRegistry, type GroundingViewDescriptor } from "./grounding-views";
import { compactGroundingEvidence, validateGroundingWorkingState, type GroundingWorkingState } from "./grounding-evidence";

const EXTENSION_NAME = "pi-web-grounding-safety";
const GROUNDING_SECTION = "grounding_runtime_safety";
const QUERY_FILE = "queries.json";
const GROUNDING_TOOL_NAMES = [
  "grounding_next_batch",
  "grounding_view",
  "grounding_compare",
  "grounding_evidence",
  "grounding_color_region",
  "grounding_status",
  "grounding_reopen_record",
  "grounding_save_result",
  "grounding_save_and_next",
] as const;

// A small image is unreadable to the model: a tiny target inside it, or a
// low-resolution modality frame, cannot be inspected. Small images are enlarged
// with a high-quality kernel up to a bounded factor, crops additionally carry a
// labeled source-normalized grid, and an oversized frame is only bounded by the
// payload limit, never enlarged.
const IMAGE_TARGET_LONG_SIDE = 768;
const IMAGE_MAX_LONG_SIDE = 1600;
const IMAGE_MAX_ZOOM = 12;
// Some OpenAI-compatible gateways reject the HTTP body before the model sees
// it. Keep every model-facing preview below one bounded base64 size even when
// the SDK image resizer is unavailable. Original bytes are still used for
// crops, color analysis and saved source coordinates.
const GROUNDING_PREVIEW_MAX_BASE64_CHARS = 1_250_000;
const GROUNDING_PREVIEW_JPEG_QUALITIES = [82, 70, 58, 46, 38, 30] as const;
const CROP_GRID_STEPS = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2] as const;
const CROP_GRID_MAX_LINES = 8;
const CROP_GRID_COLOR = "#22d3ee";
const CROP_GRID_HALO = "#0b1220";
const CURRENT_OVERLAY_COLOR = "#f59e0b";
const CURRENT_OVERLAY_HALO = "#0b1220";

const PROTECTED_FILE_NAMES = new Set([
  "expected.json",
  "meta.csv",
]);

const PROTECTED_PATH_PARTS = [
  "_annotated",
  "_crops",
  "_crop",
  "_overlay",
  "_overlays",
  "overlays",
];

const PROTECTED_WORDS = /(?:ground[_ -]?truth|annotation|label|expected\.json|meta\.csv|_annotated|_crops?|_overlays?)/i;
// Keep the guard broad enough to catch a dataset-only first message. A user
// will often paste a path and say "process this" rather than repeat the task
// name; waiting for an explicit "grounding" word would leave that first read
// unredacted. These cues are specific to the multimodal query format, so they
// should not affect ordinary file-editing sessions.
const GROUNDING_PROMPT = /(?:grounding|referring expression|bounding box|bbox|图像定位|指代|边界框|框准|热成像|infrared|thermal|queries?\.json|(?:^|[\\/])(?:visible|infrared|depth)(?:[\\/]|$)|(?:visible|infrared|depth).{0,24}(?:image|图像|图片)|(?:image|图像|图片).{0,24}(?:visible|infrared|depth))/i;
const INFRARED_QUERY = /(?:infrared|thermal|heat|hot|temperature|bright (?:thermal|heat|signature)|红外|热成像|热源)/i;
const DEPTH_QUERY = /(?:front|behind|backmost|nearest|farthest|distance|depth|overlap|occlu|前后|最近|最远|距离|深度|遮挡|重叠)/i;

type GroundingSafetyOptions = {
  cwd: string;
  sessionId: string;
  imageArchives?: Partial<Record<GroundingModality, string>>;
};

type SanitizedQueryState = {
  safePath: string;
  allowedPaths: Set<string>;
  safe: Record<string, Record<string, string>>;
  source: Record<string, unknown>;
};

type GroundingProgress = {
  key: string;
  status: "ok" | "low_confidence" | "unresolved";
  confidence: number;
  bbox: [number, number, number, number];
  rawBbox?: [number, number, number, number];
  calibration?: "tiny_target_padding";
  reviewed: true;
  reviewSource?: "human" | "runtime_auto" | "model";
  targetFound: boolean;
  candidateCount: number;
  candidateRank?: number;
  expectedOrdinal?: number;
  reason: string;
};

type GroundingResultInput = {
  queryPath: string;
  outputDir: string;
  key: string;
  bbox: readonly number[];
  status: GroundingProgress["status"];
  confidence: number;
  reason: string;
  coordinateSpace?: string;
  viewId?: string;
};

type GroundingModality = "visible" | "infrared" | "depth";

type LoadedBatchRecord = {
  sourcePath: string;
  outputDirectory: string;
  record: Record<string, string>;
  grantedModalities: Set<GroundingModality>;
  lastCropRegion?: [number, number, number, number];
  currentBbox?: [number, number, number, number];
  settleNudges: number;
  revision?: boolean;
  views: GroundingViewRegistry;
  pinnedViewIds: Set<string>;
  archivedViewIds: Set<string>;
  workingState?: GroundingWorkingState;
};

type PersistedGroundingPending = {
  key: string;
  currentBbox?: [number, number, number, number];
  workingState?: GroundingWorkingState;
  revision?: boolean;
};

type GroundingToolContent =
  | { type: "text"; text: string }
  | { type: "image"; data: string; mimeType: string };

type GroundingImage = {
  content: GroundingToolContent[];
  details: {
    modality: GroundingModality;
    path: string;
    originalWidth?: number;
    originalHeight?: number;
    width?: number;
    height?: number;
    cropNormalized?: [number, number, number, number];
  };
};

type GroundingImageSource = {
  bytes: Buffer;
  label: string;
};

type GroundingImageReader = (
  sourcePath: string,
  record: Record<string, string>,
  modality: GroundingModality,
) => Promise<GroundingImageSource>;

type EncodedGroundingPreview = {
  data: string;
  mimeType: string;
  width: number;
  height: number;
};

async function encodeGroundingPreview(
  bytes: Buffer,
  mimeType: string,
  targetSize?: { width: number; height: number },
): Promise<EncodedGroundingPreview> {
  let workingBytes = bytes;
  let metadata = await sharp(workingBytes, { failOn: "error" }).metadata();
  if (!metadata.width || !metadata.height) throw new Error("Could not determine grounding preview dimensions.");
  if (targetSize && (metadata.width !== targetSize.width || metadata.height !== targetSize.height)) {
    workingBytes = await sharp(workingBytes, { failOn: "error" })
      .resize({ width: targetSize.width, height: targetSize.height, fit: "fill", kernel: "lanczos3" })
      .png()
      .toBuffer();
    metadata = { ...metadata, width: targetSize.width, height: targetSize.height };
  }
  let width = metadata.width;
  let height = metadata.height;
  const originalData = workingBytes.toString("base64");
  if (originalData.length <= GROUNDING_PREVIEW_MAX_BASE64_CHARS) {
    return { data: originalData, mimeType, width, height };
  }

  // Preserve the display grid first and lower JPEG quality. If a deliberately
  // adversarial/high-entropy frame still exceeds the gateway-safe limit, shrink
  // the display and let callers update their coordinate mapping to these exact
  // returned dimensions.
  while (true) {
    let smallest: Buffer | undefined;
    for (const quality of GROUNDING_PREVIEW_JPEG_QUALITIES) {
      const encoded = await sharp(workingBytes, { failOn: "error" })
        .flatten({ background: "#ffffff" })
        .jpeg({ quality, chromaSubsampling: "4:4:4", progressive: true })
        .toBuffer();
      smallest = encoded;
      const data = encoded.toString("base64");
      if (data.length <= GROUNDING_PREVIEW_MAX_BASE64_CHARS) {
        return { data, mimeType: "image/jpeg", width, height };
      }
    }
    const currentLength = smallest!.toString("base64").length;
    const scale = Math.min(0.85, Math.max(0.5, Math.sqrt(GROUNDING_PREVIEW_MAX_BASE64_CHARS / currentLength) * 0.92));
    const nextWidth = Math.max(1, Math.floor(width * scale));
    const nextHeight = Math.max(1, Math.floor(height * scale));
    if (nextWidth === width && nextHeight === height) {
      throw new Error("Could not reduce grounding preview below the transport payload limit.");
    }
    width = nextWidth;
    height = nextHeight;
    workingBytes = await sharp(workingBytes, { failOn: "error" })
      .resize({ width, height, fit: "fill", kernel: "lanczos3" })
      .png()
      .toBuffer();
  }
}

async function encodeGroundingImagePayload(image: GroundingImage): Promise<GroundingImage> {
  const imageBlock = image.content.find((block) => block.type === "image");
  if (!imageBlock || imageBlock.type !== "image") throw new Error("Grounding image payload is missing.");
  const encoded = await encodeGroundingPreview(Buffer.from(imageBlock.data, "base64"), imageBlock.mimeType);
  const originalWidth = image.details.originalWidth ?? encoded.width;
  const originalHeight = image.details.originalHeight ?? encoded.height;
  const content = image.content.map((block) => {
    if (block.type === "image") return { ...block, data: encoded.data, mimeType: encoded.mimeType };
    try {
      const value = JSON.parse(block.text) as Record<string, unknown>;
      value.dimensions = `${originalWidth}x${originalHeight} source pixels; displayed as ${encoded.width}x${encoded.height}`;
      value.displayedSizePixels = [encoded.width, encoded.height];
      value.payloadEncoding = { mimeType: encoded.mimeType, base64Characters: encoded.data.length,
        resizedForTransport: encoded.width !== image.details.width || encoded.height !== image.details.height };
      if (Array.isArray(value.cropSizePixels)) {
        const cropSize = value.cropSizePixels as number[];
        value.magnification = Math.min(encoded.width / cropSize[0], encoded.height / cropSize[1]);
        value.effectiveMagnification = { x: encoded.width / cropSize[0], y: encoded.height / cropSize[1] };
      }
      return { ...block, text: JSON.stringify(value) };
    } catch {
      return block;
    }
  });
  return { ...image, content, details: { ...image.details, width: encoded.width, height: encoded.height } };
}

type GroundingViewDetails = ReturnType<typeof groundingTargetReminder> & {
  key: string;
  action: "view" | "crop";
  currentBbox: [number, number, number, number] | null;
  modality: GroundingModality;
  reason: string;
  image: GroundingImage["details"];
  viewId?: string;
  sourceReuse?: ReturnType<GroundingViewRegistry["sourceReuse"]>;
  evidenceViewIds?: string[];
};

function groundingTargetReminder(loaded: LoadedBatchRecord) {
  return {
    originalQuery: loaded.record.query ?? "",
    taskReminder: "Keep every original query constraint. A new candidate or pixel match does not replace the requested target; discovery order is not spatial order. If a required condition lacks support, keep the result unresolved.",
  };
}

function recordEvidence() {
  return { views: new GroundingViewRegistry(), pinnedViewIds: new Set<string>(), archivedViewIds: new Set<string>() };
}

function registerGroundingView(loaded: LoadedBatchRecord, image: GroundingImage, decorations: GroundingViewDescriptor["decorations"]) {
  const { originalWidth: sourceWidth, originalHeight: sourceHeight, width, height, modality } = image.details;
  if (!sourceWidth || !sourceHeight || !width || !height) throw new Error("The view is missing source/display dimensions.");
  const input = { modality, region: image.details.cropNormalized ?? [0, 0, 1, 1] as [number, number, number, number], sourceWidth, sourceHeight, width, height, decorations };
  const sourceReuse = loaded.views.sourceReuse(input);
  const equivalent = loaded.views.findEquivalent(input);
  const view = loaded.views.register(input);
  // Keep dimensions alongside the image, not just in UI-only tool details.
  const metadata = image.content.find((block) => block.type === "text");
  if (metadata?.type === "text") metadata.text = JSON.stringify({ ...JSON.parse(metadata.text), viewId: view.id,
    coordinateMapping: view,
    ...(sourceReuse ? {
      sourceReuse,
      informationNote: sourceReuse.relation === "exact"
        ? "These exact source pixels were already viewed. A new display scale is not new source detail."
        : `${(sourceReuse.coveredFraction * 100).toFixed(1)}% of this requested source region was already visible in ${sourceReuse.previousViewId}; only ${sourceReuse.newSourcePixels} source pixels are new. A shifted or tighter crop is not independent identity evidence.`,
    } : equivalent ? { repeatedSourceViewId: equivalent.id,
      informationNote: "These source pixels were already viewed. A new display scale is not new source detail; compare identity/context or archive superseded evidence when useful." } : {}) });
  return { ...view, ...(sourceReuse ? { sourceReuse } : {}) };
}

function normalizePath(value: string): string {
  return value.replaceAll("\\", "/").replaceAll(/\/+/g, "/").toLowerCase();
}

function isQueryPath(value: string): boolean {
  return basename(value).toLowerCase() === QUERY_FILE;
}

function isProtectedPath(value: string): boolean {
  const normalized = normalizePath(value);
  const name = basename(normalized);
  return PROTECTED_FILE_NAMES.has(name)
    || PROTECTED_PATH_PARTS.some((part) => normalized.split("/").includes(part))
    || PROTECTED_WORDS.test(normalized);
}

function isSensitiveCommand(value: string): boolean {
  // Query files may be read only through the `read` tool, which redirects them
  // to the sanitized projection. Blocking shell access closes the common
  // Get-Content/type/findstr bypass without preventing normal output writes.
  return PROTECTED_WORDS.test(value)
    || /\b(?:bbox|groundtruth|annotations?|labels?)\b/i.test(value)
    || /(?:^|[\\/\s"'`])queries?\.json(?:$|[\\/\s"'`])/i.test(value);
}

function isFilesystemTraversalCommand(value: string): boolean {
  return /\b(?:get-content|gc|cat|type|more|dir|gci|get-childitem|ls|find|findstr|grep|select-string|where)\b/i.test(value);
}

function isGroundingPrompt(prompt: string): boolean {
  return GROUNDING_PROMPT.test(prompt);
}

function groundingEntryText(entry: unknown): string {
  if (!entry || typeof entry !== "object" || Array.isArray(entry)) return "";
  const message = (entry as { message?: unknown }).message;
  if (!message || typeof message !== "object" || Array.isArray(message)) return "";
  const content = (message as { content?: unknown }).content;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content
    .map((block) => (
      block && typeof block === "object" && typeof (block as { text?: unknown }).text === "string"
        ? (block as { text: string }).text
        : ""
    ))
    .filter(Boolean)
    .join("\n");
}

export function requestedBatchCount(prompt: string): number | undefined {
  const matches = [
    /(?:做|处理|完成|继续|接下来(?:的)?|前)\s*(?:满|下面这组|这组|以下(?:这组)?|下面(?:的)?|这|前)?\s*(\d+)\s*条/i,
    /\b(\d+)\s*(?:records?|queries|items)\b/i,
  ];
  for (const pattern of matches) {
    const match = pattern.exec(prompt);
    if (!match) continue;
    const count = Number(match[1]);
    if (Number.isSafeInteger(count) && count > 0) return count;
  }
  return undefined;
}

function allowedRecord(record: unknown): Record<string, string> | null {
  if (!record || typeof record !== "object" || Array.isArray(record)) return null;
  const input = record as Record<string, unknown>;
  const output: Record<string, string> = {};
  for (const field of ["visible", "infrared", "depth", "query"] as const) {
    if (typeof input[field] === "string") output[field] = input[field];
  }
  return Object.keys(output).length > 0 ? output : null;
}

/**
 * Keep only fields that are needed to make a blind grounding prediction.
 * This is exported so the redaction contract can be tested without starting
 * an AgentSession.
 */
export function sanitizeGroundingQueryJson(value: unknown): Record<string, Record<string, string>> {
  if (!value || typeof value !== "object" || Array.isArray(value)) return {};
  const output: Record<string, Record<string, string>> = {};
  for (const [key, record] of Object.entries(value as Record<string, unknown>)) {
    const safe = allowedRecord(record);
    if (safe) output[key] = safe;
  }
  return output;
}

export function redactGroundingToolText(text: string): string {
  if (/\bbbox\b|expected\.json|ground[_ -]?truth|annotation|meta\.csv|_annotated|_crops?|_overlays?/i.test(text)) {
    return "[grounding safety] Annotation or evaluation content was redacted.";
  }
  return text;
}

/**
 * Keep the transcript auditable on disk while preventing completed records'
 * images and free-form reasoning from being resent on every later provider
 * request. The latest record-loading result is the context boundary because
 * save-and-next attaches the next record's first image in that same result.
 */
export function compactCompletedGroundingContext(
  messages: AgentMessage[],
  hasActiveRecord: boolean,
): AgentMessage[] {
  let boundary = -1;
  let boundaryToolCallId: string | undefined;
  {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (
        message.role === "toolResult"
        && (hasActiveRecord
          ? ["grounding_next_batch", "grounding_save_and_next", "grounding_reopen_record"].includes(message.toolName)
            && message.content.some((block) => block.type === "image")
          : ["grounding_next_batch", "grounding_save_and_next", "grounding_save_result", "grounding_status"].includes(message.toolName))
      ) {
        boundary = index;
        boundaryToolCallId = message.toolCallId;
        break;
      }
    }
  }
  if (boundary < 0) return messages;

  let keepFrom = boundary;
  if (boundaryToolCallId) {
    for (let index = boundary - 1; index >= 0; index -= 1) {
      const message = messages[index];
      if (
        message.role === "assistant"
        && message.content.some((block) => block.type === "toolCall" && block.id === boundaryToolCallId)
      ) {
        keepFrom = index;
        break;
      }
    }
  }

  const compacted: AgentMessage[] = [];
  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];
    if (index < keepFrom) {
      // User/custom messages carry the dataset request and hidden recovery
      // instructions. Old assistant/tool-result pairs are already persisted in
      // the transcript and are irrelevant once their record has been saved.
      if (message.role !== "assistant" && message.role !== "toolResult") compacted.push(message);
      continue;
    }
    if (index === keepFrom && message.role === "assistant" && boundaryToolCallId) {
      compacted.push({
        ...message,
        content: message.content.filter(
          // Keep every call from this assistant entry: their results may be
          // before or after the record-loading result in the same parallel turn.
          (block) => block.type === "toolCall",
        ),
      });
      continue;
    }
    if (!hasActiveRecord && message.role === "toolResult") {
      const content = message.content.filter((block) => block.type !== "image");
      compacted.push({ ...message, content: content.length ? content : [{ type: "text", text: "Previous image omitted. Use grounding_reopen_record to inspect the saved record again." }] });
    } else if (!hasActiveRecord && message.role === "assistant") {
      const content = message.content.filter((block) => block.type !== "thinking");
      if (content.length) compacted.push({ ...message, content });
    } else compacted.push(message);
  }
  return compacted.length === messages.length
    && compacted.every((message, index) => message === messages[index])
    ? messages
    : compacted;
}

function redactToolContent(event: ToolResultEvent): ToolResultEvent["content"] | undefined {
  let changed = false;
  const content = event.content.map((block) => {
    if (block.type !== "text") return block;
    const text = redactGroundingToolText(block.text);
    if (text === block.text) return block;
    changed = true;
    return { ...block, text };
  });
  return changed ? content : undefined;
}

async function makeSanitizedQuery(
  sourcePath: string,
  tempRoot: string,
): Promise<SanitizedQueryState> {
  const parsed = JSON.parse(await readFile(sourcePath, "utf8")) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw new Error("Grounding query JSON must be an object keyed by record id.");
  }
  const source = parsed as Record<string, unknown>;
  const safe = sanitizeGroundingQueryJson(source);
  const allowedPaths = new Set<string>();
  for (const record of Object.values(safe)) {
    for (const field of ["visible", "infrared", "depth"] as const) {
      const imagePath = record[field];
      if (!imagePath) continue;
      // Keep both spellings: the model may repeat the source string exactly,
      // or resolve it relative to the directory containing queries.json.
      allowedPaths.add(normalizePath(imagePath));
      allowedPaths.add(normalizePath(resolve(dirname(sourcePath), imagePath)));
    }
  }
  await mkdir(tempRoot, { recursive: true });
  const safePath = join(tempRoot, "queries-sanitized.json");
  await writeFile(safePath, `${JSON.stringify(safe, null, 2)}\n`, "utf8");
  allowedPaths.add(normalizePath(safePath));
  return { safePath, allowedPaths, safe, source };
}

function resolvedOutputDirectory(cwd: string, sourcePath: string, outputDir: string): string {
  const resolvedOutput = resolve(cwd, outputDir);
  const sourceDirectory = normalizePath(dirname(sourcePath)).replace(/\/$/u, "");
  const normalizedOutput = normalizePath(resolvedOutput).replace(/\/$/u, "");
  if (
    normalizedOutput === sourceDirectory
    || normalizedOutput.startsWith(`${sourceDirectory}/`)
    || isProtectedPath(resolvedOutput)
  ) {
    throw new Error("Grounding output must be outside the source query directory and prior artifact paths.");
  }
  return resolvedOutput;
}

async function readGroundingProgress(path: string): Promise<Map<string, GroundingProgress>> {
  let text: string;
  try {
    text = await readFile(path, "utf8");
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return new Map();
    throw error;
  }

  const progress = new Map<string, GroundingProgress>();
  for (const [index, line] of text.split(/\r?\n/u).entries()) {
    if (!line.trim()) continue;
    let value: unknown;
    try {
      value = JSON.parse(line) as unknown;
    } catch (error) {
      throw new Error(`Malformed grounding progress JSONL at line ${index + 1} in ${path}: ${(error as Error).message}`);
    }
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      throw new Error(`Malformed grounding progress record at line ${index + 1}.`);
    }
    const item = value as Partial<GroundingProgress>;
    if (
      typeof item.key !== "string"
      || !["ok", "low_confidence", "unresolved"].includes(String(item.status))
      || typeof item.confidence !== "number"
      || !Number.isFinite(item.confidence)
      || !Array.isArray(item.bbox)
      || item.bbox.length !== 4
      || item.bbox.some((coordinate) => typeof coordinate !== "number" || !Number.isFinite(coordinate))
      || (item.rawBbox !== undefined && (
        !Array.isArray(item.rawBbox)
        || item.rawBbox.length !== 4
        || item.rawBbox.some((coordinate) => typeof coordinate !== "number" || !Number.isFinite(coordinate))
      ))
      || (item.calibration !== undefined && item.calibration !== "tiny_target_padding")
      || (item.reviewSource !== undefined && item.reviewSource !== "human" && item.reviewSource !== "runtime_auto" && item.reviewSource !== "model")
      || item.reviewed !== true
      || typeof item.targetFound !== "boolean"
      || !Number.isInteger(item.candidateCount)
      || (item.candidateCount as number) < 0
      || (item.candidateRank !== undefined && !Number.isInteger(item.candidateRank))
      || (item.expectedOrdinal !== undefined && !Number.isInteger(item.expectedOrdinal))
      || typeof item.reason !== "string"
    ) {
      throw new Error(
        `Malformed grounding progress record at line ${index + 1} in ${path}. `
        + "Resuming requires the full record written by this runtime "
        + "(key, status, confidence, bbox, reviewed, targetFound, candidateCount, reason); "
        + "a progress.jsonl written by an older build cannot be resumed. Use a new empty outputDir, "
        + "or delete that file first if its saved results are not needed.",
      );
    }
    progress.set(item.key, item as GroundingProgress);
  }
  return progress;
}

async function writeFileAtomic(path: string, content: string, sessionId: string): Promise<void> {
  const temporaryPath = `${path}.${sessionId.replace(/[^a-zA-Z0-9_-]/g, "_")}.tmp`;
  await writeFile(temporaryPath, content, "utf8");
  await rename(temporaryPath, path);
}

async function writeBufferAtomic(path: string, content: Buffer, sessionId: string): Promise<void> {
  const temporaryPath = `${path}.${sessionId.replace(/[^a-zA-Z0-9_-]/g, "_")}.tmp`;
  await writeFile(temporaryPath, content);
  await rename(temporaryPath, path);
}

function validateBoundingBox(value: readonly number[]): [number, number, number, number] {
  if (
    value.length !== 4
    || value.some((coordinate) => !Number.isFinite(coordinate) || coordinate < 0 || coordinate > 1)
    || value[0] >= value[2]
    || value[1] >= value[3]
  ) {
    throw new Error("bbox must contain four finite normalized values with x1 < x2 and y1 < y2.");
  }
  return [value[0], value[1], value[2], value[3]];
}
export function expectedGroundingOrdinal(query: string): number | undefined {
  const chinese = /(?:第)\s*(\d+)/u.exec(query);
  if (chinese) return Number(chinese[1]);
  const numeric = /\b(\d+)(?:st|nd|rd|th)\b/i.exec(query);
  if (numeric) return Number(numeric[1]);
  const words: Record<string, number> = {
    first: 1,
    second: 2,
    third: 3,
    fourth: 4,
    fifth: 5,
    sixth: 6,
    seventh: 7,
    eighth: 8,
    ninth: 9,
    tenth: 10,
  };
  for (const [word, ordinal] of Object.entries(words)) {
    if (new RegExp(`\\b${word}\\b`, "i").test(query)) return ordinal;
  }
  return undefined;
}
type ConfirmedGroundingReview = Extract<GroundingReviewResponse, { action: "confirm" }>;
export function validateGroundingReviewResponse(
  value: unknown,
  details: GroundingReviewDetails,
): GroundingReviewResponse {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Browser grounding review returned no structured response.");
  }
  const input = value as Record<string, unknown>;
  if (input.type !== "grounding_review_response") {
    throw new Error("Browser grounding review returned an unexpected response type.");
  }
  if (input.action === "reject") {
    if (typeof input.reason !== "string" || input.reason.trim().length < 1) {
      throw new Error("A browser review rejection must include a reason.");
    }
    return { type: "grounding_review_response", action: "reject", reason: input.reason.trim() };
  }
  if (input.action !== "confirm") throw new Error("Browser grounding review must confirm or reject the candidate.");
  if (!Array.isArray(input.bbox) || input.bbox.some((coordinate) => typeof coordinate !== "number")) {
    throw new Error("Browser grounding review must return a numeric bbox.");
  }
  const bbox = validateBoundingBox(input.bbox as number[]);
  const status = input.status;
  if (status !== "ok" && status !== "low_confidence" && status !== "unresolved") {
    throw new Error("Browser grounding review returned an invalid status.");
  }
  const confidence = input.confidence;
  if (typeof confidence !== "number" || !Number.isFinite(confidence) || confidence < 0 || confidence > 1) {
    throw new Error("Browser grounding review confidence must be between 0 and 1.");
  }
  const targetFound = input.targetFound;
  if (typeof targetFound !== "boolean") throw new Error("Browser grounding review must state whether the target is present.");
  const candidateCount = input.candidateCount;
  if (typeof candidateCount !== "number" || !Number.isInteger(candidateCount) || candidateCount < 0) {
    throw new Error("Browser grounding review candidate count must be a non-negative integer.");
  }
  const candidateRank = input.candidateRank;
  if (targetFound && candidateCount < 1) throw new Error("A found target requires at least one candidate.");
  if (targetFound && (typeof candidateRank !== "number" || !Number.isInteger(candidateRank) || candidateRank < 1 || candidateRank > candidateCount)) {
    throw new Error("Browser grounding review candidate rank is outside the candidate count.");
  }
  if (!targetFound && status !== "unresolved") throw new Error("A missing target can only be saved as unresolved.");
  if (status === "ok" && confidence < 0.5) throw new Error("An ok grounding result requires confidence of at least 0.5.");
  if (details.expectedOrdinal !== undefined && targetFound && candidateRank !== details.expectedOrdinal) {
    throw new Error(`The query requires candidate ${details.expectedOrdinal}; browser review selected ${String(candidateRank)}.`);
  }
  if (typeof input.reason !== "string" || input.reason.trim().length < 8) {
    throw new Error("Browser grounding review must include an evidence reason of at least 8 characters.");
  }
  const confirmed: ConfirmedGroundingReview = {
    type: "grounding_review_response",
    action: "confirm",
    bbox,
    status,
    confidence,
    targetFound,
    candidateCount,
    ...(targetFound ? { candidateRank: candidateRank as number } : {}),
    reason: input.reason.trim(),
  };
  return confirmed;
}

export function describeGroundingCandidateChange(
  previousBbox: [number, number, number, number],
  nextBbox: [number, number, number, number],
  sourceWidth: number,
  sourceHeight: number,
): NonNullable<GroundingReviewDetails["candidateChange"]> {
  const area = (box: readonly number[]) => (box[2] - box[0]) * (box[3] - box[1]);
  const intersection = Math.max(0, Math.min(previousBbox[2], nextBbox[2]) - Math.max(previousBbox[0], nextBbox[0]))
    * Math.max(0, Math.min(previousBbox[3], nextBbox[3]) - Math.max(previousBbox[1], nextBbox[1]));
  const previousArea = area(previousBbox), nextArea = area(nextBbox);
  const union = previousArea + nextArea - intersection;
  const iou = union > 0 ? intersection / union : 0;
  const centerDeltaPixels: [number, number] = [
    ((nextBbox[0] + nextBbox[2]) - (previousBbox[0] + previousBbox[2])) / 2 * sourceWidth,
    ((nextBbox[1] + nextBbox[3]) - (previousBbox[1] + previousBbox[3])) / 2 * sourceHeight,
  ];
  const edgeDeltaPixels: [number, number, number, number] = [
    (nextBbox[0] - previousBbox[0]) * sourceWidth,
    (nextBbox[1] - previousBbox[1]) * sourceHeight,
    (nextBbox[2] - previousBbox[2]) * sourceWidth,
    (nextBbox[3] - previousBbox[3]) * sourceHeight,
  ];
  const previousMinSide = Math.min(
    (previousBbox[2] - previousBbox[0]) * sourceWidth,
    (previousBbox[3] - previousBbox[1]) * sourceHeight,
  );
  const centerDistance = Math.hypot(...centerDeltaPixels);
  const areaRatio = previousArea > 0 ? nextArea / previousArea : 1;
  const materialChange = iou < 0.5 || centerDistance >= previousMinSide * 0.5 || areaRatio < 0.5 || areaRatio > 2;
  return {
    previousBbox: [...previousBbox],
    iou,
    centerDeltaPixels,
    edgeDeltaPixels,
    areaRatio,
    materialChange,
    note: materialChange
      ? "The candidate moved materially. Re-check target identity, owning object and requested part; cite new visible evidence rather than treating a color match as proof."
      : "The candidate changed only modestly; verify the adjusted edges against the same visible part.",
  };
}

function imageMimeType(path: string): string {
  switch (extname(path).toLowerCase()) {
    case ".png": return "image/png";
    case ".webp": return "image/webp";
    case ".gif": return "image/gif";
    case ".bmp": return "image/bmp";
    case ".jpg":
    case ".jpeg":
    default:
      return "image/jpeg";
  }
}

function recordImagePath(sourcePath: string, record: Record<string, string>, modality: GroundingModality): string {
  const path = record[modality];
  if (!path) {
    throw new Error(
      `The loaded record has no ${modality} image.`
      + " Continue with the views that are available; a missing view is not evidence that the target is absent.",
    );
  }
  return resolve(dirname(sourcePath), path);
}

type ArchiveCommandOptions = {
  encoding: "buffer";
  maxBuffer: number;
  timeout: number;
  killSignal: "SIGKILL";
  windowsHide: true;
};

type ArchiveCommandRunner = (
  command: string,
  args: string[],
  options: ArchiveCommandOptions,
) => Promise<Buffer>;

const runArchiveCommand: ArchiveCommandRunner = (command, args, options) => {
  return new Promise((resolveEntry, rejectEntry) => {
    execFile(
      command,
      args,
      options,
      (error, stdout, stderr) => {
        if (error) {
          rejectEntry(new Error(stderr.toString().trim() || error.message, { cause: error }));
          return;
        }
        resolveEntry(stdout);
      },
    );
  });
};

export async function readArchiveEntry(
  archivePath: string,
  entryPath: string,
  runCommand: ArchiveCommandRunner = runArchiveCommand,
): Promise<Buffer> {
  // Windows/macOS ship a ZIP-capable tar, but GNU tar on Linux does not.
  // Keep extraction streaming: loading a whole dataset ZIP with JSZip can
  // exhaust memory even when the single requested image is small.
  const options: ArchiveCommandOptions = {
    encoding: "buffer", maxBuffer: 64 * 1024 * 1024,
    timeout: 30_000, killSignal: "SIGKILL", windowsHide: true,
  };
  const resolvedArchive = resolve(archivePath);
  const literalEntry = entryPath.replace(/[\\*?[\]]/g, "\\$&");
  const message = (value: unknown) => value instanceof Error ? value.message : String(value);
  let tarError: unknown;
  try {
    let tarEntry = entryPath;
    const selectionOptions: string[] = [];
    if (literalEntry !== entryPath) {
      // BSD tar always treats member names as patterns; GNU tar treats them
      // literally by default. Do not guess when a name contains glob syntax.
      const version = (await runCommand("tar", ["--version"], {
        ...options, maxBuffer: 64 * 1024, timeout: 5_000,
      })).toString();
      if (/bsdtar|libarchive/i.test(version)) tarEntry = literalEntry;
      else if (/GNU tar/i.test(version)) selectionOptions.push("--no-wildcards");
      else throw new Error("Cannot establish literal member matching for this tar implementation");
    }
    return await runCommand("tar", ["-xOf", resolvedArchive, ...selectionOptions, "--", tarEntry], options);
  } catch (error) {
    tarError = error;
  }
  if (extname(resolvedArchive).toLowerCase() !== ".zip") {
    throw new Error(`Could not read ${entryPath} from ${archivePath}: tar: ${message(tarError)}`, { cause: tarError });
  }
  try {
    // GNU-tar systems need unzip for ZIP datasets. Its member arguments are
    // patterns even without a shell; escaping prevents concatenating siblings.
    return await runCommand("unzip", ["-p", resolvedArchive, literalEntry], options);
  } catch (error) {
    throw new Error(
      `Could not read ${entryPath} from ${archivePath}: tar: ${message(tarError)}; unzip: ${message(error)}`,
      { cause: error },
    );
  }
}

/**
 * resizeImage only shrinks an oversized frame, so a low-resolution infrared or
 * depth frame stays unreadable. Enlarge a small frame with the same bounded
 * strategy used for a tiny target crop, and report the factor so the model knows
 * the display is magnified.
 */
async function magnifySmallImage(
  bytes: Buffer,
  knownWidth?: number,
  knownHeight?: number,
): Promise<{
  data: string;
  mimeType: string;
  width: number;
  height: number;
  zoom: number;
  sourceWidth: number;
  sourceHeight: number;
} | null> {
  let sourceWidth = knownWidth;
  let sourceHeight = knownHeight;
  if (!sourceWidth || !sourceHeight) {
    try {
      const metadata = await sharp(bytes, { failOn: "error" }).metadata();
      sourceWidth = metadata.width;
      sourceHeight = metadata.height;
    } catch {
      return null;
    }
  }
  if (!sourceWidth || !sourceHeight) return null;
  const longSide = Math.max(sourceWidth, sourceHeight);
  if (longSide >= IMAGE_TARGET_LONG_SIDE) return null;
  const zoom = Math.min(IMAGE_MAX_ZOOM, Math.max(2, Math.round(IMAGE_TARGET_LONG_SIDE / longSide)));
  const fit = Math.min(1, IMAGE_MAX_LONG_SIDE / (longSide * zoom));
  const width = Math.max(1, Math.round(sourceWidth * zoom * fit));
  const height = Math.max(1, Math.round(sourceHeight * zoom * fit));
  const output = await sharp(bytes, { failOn: "error" })
    .resize({ width, height, kernel: "lanczos3", fit: "fill" })
    .png()
    .toBuffer();
  return {
    data: output.toString("base64"),
    mimeType: "image/png",
    width,
    height,
    zoom: Number((zoom * fit).toFixed(2)),
    sourceWidth,
    sourceHeight,
  };
}

async function loadGroundingImage(
  sourcePath: string,
  record: Record<string, string>,
  modality: GroundingModality,
  readImage: GroundingImageReader,
  currentBbox: readonly number[] | null,
  includeOverlay = true,
): Promise<GroundingImage> {
  const { bytes, label: path } = await readImage(sourcePath, record, modality);
  const metadata = await sharp(bytes).metadata();
  // Crops, color masks and submission coordinates use raw source pixel axes.
  // The SDK auto-orients JPEGs; remove that metadata before it can rotate only
  // the full view and silently make its coordinates disagree with every crop.
  const stripOrientation = metadata.orientation !== undefined && metadata.orientation !== 1;
  const displayBytes = stripOrientation ? await sharp(bytes).png().toBuffer() : bytes;
  const mimeType = stripOrientation ? "image/png" : imageMimeType(record[modality] ?? path);
  const resized = await resizeImage(displayBytes, mimeType);
  const magnified = await magnifySmallImage(
    resized?.data ? Buffer.from(resized.data, "base64") : displayBytes,
    resized?.width,
    resized?.height,
  );
  const normalizedCurrentBbox = currentBbox === null ? null : validateBoundingBox(currentBbox);
  const originalWidth = resized?.originalWidth ?? metadata.width ?? magnified?.sourceWidth;
  const originalHeight = resized?.originalHeight ?? metadata.height ?? magnified?.sourceHeight;
  const width = magnified?.width ?? resized?.width ?? metadata.width;
  const height = magnified?.height ?? resized?.height ?? metadata.height;
  const details = {
    modality,
    path,
    ...(originalWidth !== undefined && originalHeight !== undefined ? { originalWidth, originalHeight } : {}),
    ...(width !== undefined && height !== undefined ? { width, height } : {}),
  };
  const dimensionText =
    originalWidth !== undefined && originalHeight !== undefined && width !== undefined && height !== undefined
      ? `${originalWidth}x${originalHeight} source pixels; displayed as ${width}x${height}`
      : "source dimensions unavailable";
  const image: GroundingImage = {
    content: [
      {
        type: "text",
        text: JSON.stringify({
          image: modality,
          path,
          dimensions: dimensionText,
          ...(magnified
            ? { magnification: magnified.zoom, displayedSizePixels: [magnified.width, magnified.height] }
            : {}),
          currentBbox: normalizedCurrentBbox,
          overlay: !includeOverlay ? "none" : normalizedCurrentBbox ? "current_hypothesis" : "current_hypothesis_none",
        }),
      },
      {
        type: "image",
        data: magnified?.data ?? resized?.data ?? displayBytes.toString("base64"),
        mimeType: magnified?.mimeType ?? resized?.mimeType ?? mimeType,
      },
    ],
    details,
  };
  if (!includeOverlay) return encodeGroundingImagePayload(image);
  return withGroundingOverlay(
    image,
    normalizedCurrentBbox,
    "CURRENT HYPOTHESIS",
    CURRENT_OVERLAY_COLOR,
    CURRENT_OVERLAY_HALO,
  );
}

type CropGridAxis = {
  step: number;
  lines: Array<{ position: number; text: string }>;
};

function cropGridDecimals(step: number): number {
  const text = step.toString();
  const separator = text.indexOf(".");
  return separator < 0 ? 0 : text.length - separator - 1;
}

/**
 * Place grid lines on source-normalized round values so the model can read
 * absolute coordinates off a crop instead of estimating them by eye. The step
 * grows with the crop span to keep the crop readable.
 */
function cropGridAxis(start: number, end: number, displayLength: number): CropGridAxis {
  const span = end - start;
  const step = CROP_GRID_STEPS.find((candidate) => span / candidate <= CROP_GRID_MAX_LINES) ?? 0.5;
  const decimals = cropGridDecimals(step);
  const lines: CropGridAxis["lines"] = [];
  const firstIndex = Math.ceil(start / step - 1e-9);
  const lastIndex = Math.floor(end / step + 1e-9);
  for (let index = firstIndex; index <= lastIndex; index += 1) {
    const normalized = Number((index * step).toFixed(decimals));
    if (normalized <= start + 1e-9 || normalized >= end - 1e-9) continue;
    lines.push({
      position: ((normalized - start) / span) * displayLength,
      text: normalized.toFixed(decimals),
    });
  }
  return { step, lines };
}

function cropGridOverlay(
  vertical: CropGridAxis,
  horizontal: CropGridAxis,
  displayWidth: number,
  displayHeight: number,
): Buffer {
  const longSide = Math.max(displayWidth, displayHeight);
  const stroke = Math.max(1, Math.round(longSide / 900));
  const fontSize = Math.max(11, Math.round(longSide / 42));
  const halo = Math.max(2, Math.round(fontSize / 8));
  const gap = Math.max(stroke + 2, Math.round(fontSize * 0.3));
  // Half-pixel offsets keep a one-pixel rule crisp instead of blurring it.
  const label = (x: number, y: number, text: string) =>
    `<text x="${x.toFixed(1)}" y="${y.toFixed(1)}" font-family="sans-serif" font-size="${fontSize}"`
    + ` fill="${CROP_GRID_COLOR}" stroke="${CROP_GRID_HALO}" stroke-width="${halo}" paint-order="stroke">${text}</text>`;
  const elements: string[] = [];
  for (const line of vertical.lines) {
    const x = Math.round(line.position) + 0.5;
    elements.push(`<line x1="${x}" y1="0" x2="${x}" y2="${displayHeight}" stroke="${CROP_GRID_COLOR}" stroke-width="${stroke}" opacity="0.55"/>`);
    elements.push(label(x + gap, fontSize + gap, line.text));
  }
  for (const line of horizontal.lines) {
    const y = Math.round(line.position) + 0.5;
    elements.push(`<line x1="0" y1="${y}" x2="${displayWidth}" y2="${y}" stroke="${CROP_GRID_COLOR}" stroke-width="${stroke}" opacity="0.55"/>`);
    elements.push(label(gap, Math.min(displayHeight - gap, y + fontSize + gap), line.text));
  }
  return Buffer.from(
    `<svg xmlns="http://www.w3.org/2000/svg" width="${displayWidth}" height="${displayHeight}">${elements.join("")}</svg>`,
  );
}

async function loadGroundingCrop(
  sourcePath: string,
  record: Record<string, string>,
  modality: GroundingModality,
  normalizedRegion: readonly number[],
  currentBbox: readonly number[] | null,
  readImage: GroundingImageReader,
  requestedZoom?: number,
  decorations: "none" | "grid" | "hypothesis" | "all" = "all",
): Promise<GroundingImage> {
  const region = validateBoundingBox(normalizedRegion);
  const normalizedCurrentBbox = currentBbox === null ? null : validateBoundingBox(currentBbox);
  if (requestedZoom !== undefined && (!Number.isFinite(requestedZoom) || requestedZoom < 1)) {
    throw new Error("Crop zoom must be a finite number greater than or equal to 1.");
  }
  const { bytes, label: path } = await readImage(sourcePath, record, modality);
  const metadata = await sharp(bytes, { failOn: "error" }).metadata();
  if (!metadata.width || !metadata.height) throw new Error("Could not determine source image dimensions for the crop.");

  const sourceWidth = metadata.width;
  const sourceHeight = metadata.height;
  const sourceEdge = (value: number, size: number) => {
    const pixel = value * size;
    return Math.abs(pixel - Math.round(pixel)) < 1e-8 ? Math.round(pixel) : pixel;
  };
  const left = Math.min(sourceWidth - 1, Math.max(0, Math.floor(sourceEdge(region[0], sourceWidth))));
  const top = Math.min(sourceHeight - 1, Math.max(0, Math.floor(sourceEdge(region[1], sourceHeight))));
  const right = Math.min(sourceWidth, Math.max(left + 1, Math.ceil(sourceEdge(region[2], sourceWidth))));
  const bottom = Math.min(sourceHeight, Math.max(top + 1, Math.ceil(sourceEdge(region[3], sourceHeight))));
  const cropWidth = right - left;
  const cropHeight = bottom - top;

  // Preserve the whole requested region; bound only the display magnification.
  const automaticZoom = Math.min(IMAGE_MAX_ZOOM, Math.max(2, Math.ceil(IMAGE_TARGET_LONG_SIDE / Math.max(cropWidth, cropHeight))));
  const zoom = requestedZoom ?? automaticZoom;
  const effectiveZoom = Math.min(zoom, IMAGE_MAX_LONG_SIDE / Math.max(cropWidth, cropHeight));
  const zoomAdjusted = effectiveZoom < zoom;
  const displayWidth = Math.max(1, Math.round(cropWidth * effectiveZoom));
  const displayHeight = Math.max(1, Math.round(cropHeight * effectiveZoom));

  // Anchor the grid to the pixel-exact displayed span so every label sits at the
  // source coordinate it names.
  const displayedNormalized: [number, number, number, number] = [
    left / sourceWidth,
    top / sourceHeight,
    right / sourceWidth,
    bottom / sourceHeight,
  ];
  const touchesSourceEdge = left <= 0 || top <= 0 || right >= sourceWidth || bottom >= sourceHeight;
  const displayedSpanX = displayedNormalized[2] - displayedNormalized[0];
  const displayedSpanY = displayedNormalized[3] - displayedNormalized[1];
  const clamp01 = (value: number) => Math.max(0, Math.min(1, value));
  const containsCurrentBbox = normalizedCurrentBbox !== null && normalizedCurrentBbox[0] >= displayedNormalized[0]
    && normalizedCurrentBbox[1] >= displayedNormalized[1]
    && normalizedCurrentBbox[2] <= displayedNormalized[2]
    && normalizedCurrentBbox[3] <= displayedNormalized[3];
  const intersection: [number, number, number, number] | null = normalizedCurrentBbox === null ? null : [
    Math.max(normalizedCurrentBbox[0], displayedNormalized[0]),
    Math.max(normalizedCurrentBbox[1], displayedNormalized[1]),
    Math.min(normalizedCurrentBbox[2], displayedNormalized[2]),
    Math.min(normalizedCurrentBbox[3], displayedNormalized[3]),
  ];
  const intersectsCurrentBbox = intersection !== null && intersection[0] < intersection[2] && intersection[1] < intersection[3];
  const currentBboxInCrop: [number, number, number, number] | null = intersectsCurrentBbox && normalizedCurrentBbox ? [
    clamp01((normalizedCurrentBbox[0] - displayedNormalized[0]) / displayedSpanX),
    clamp01((normalizedCurrentBbox[1] - displayedNormalized[1]) / displayedSpanY),
    clamp01((normalizedCurrentBbox[2] - displayedNormalized[0]) / displayedSpanX),
    clamp01((normalizedCurrentBbox[3] - displayedNormalized[1]) / displayedSpanY),
  ] : null;

  let pipeline = sharp(bytes, { failOn: "error" }).extract({ left, top, width: cropWidth, height: cropHeight });
  if (displayWidth !== cropWidth || displayHeight !== cropHeight) {
    pipeline = pipeline.resize({ width: displayWidth, height: displayHeight, kernel: "lanczos3", fit: "fill" });
  }
  const vertical = cropGridAxis(displayedNormalized[0], displayedNormalized[2], displayWidth);
  const horizontal = cropGridAxis(displayedNormalized[1], displayedNormalized[3], displayHeight);
  const rawCrop = await pipeline.png().toBuffer();
  const showGrid = decorations === "grid" || decorations === "all";
  const showHypothesis = decorations === "hypothesis" || decorations === "all";
  const composed = showGrid ? await sharp(rawCrop)
    .composite([{ input: cropGridOverlay(vertical, horizontal, displayWidth, displayHeight), top: 0, left: 0 }])
    .png().toBuffer() : rawCrop;

  const image: GroundingImage = {
    content: [
      {
        type: "text",
        text: JSON.stringify({
          image: `${modality}_crop`,
          sourcePath: path,
          sourceDimensions: [sourceWidth, sourceHeight],
          cropNormalized: region,
          displayedNormalized,
          cropPixels: [left, top, right, bottom],
          cropSizePixels: [cropWidth, cropHeight],
          displayedSizePixels: [displayWidth, displayHeight],
          gridStep: showGrid ? { x: vertical.step, y: horizontal.step } : null,
          decorations,
          magnification: effectiveZoom,
          effectiveMagnification: { x: displayWidth / cropWidth, y: displayHeight / cropHeight },
          requestedZoom: requestedZoom ?? null,
          zoomAdjusted,
          maxDisplayLongSide: IMAGE_MAX_LONG_SIDE,
          displayNote: zoomAdjusted
            ? `Display magnification was reduced to fit the ${IMAGE_MAX_LONG_SIDE}px limit. The full requested region is preserved; source coordinates are unchanged.`
            : "Requested magnification fits the display limit; source coordinates are unchanged.",
          zoomSource: requestedZoom === undefined ? "runtime_safety_verification" : "model",
          touchesSourceEdge,
          currentBbox: normalizedCurrentBbox,
          currentBboxInCrop,
          containsCurrentBbox,
          intersectsCurrentBbox,
          currentBboxIntersection: intersectsCurrentBbox ? intersection : null,
          overlay: !showHypothesis ? "none" : normalizedCurrentBbox === null ? "current_hypothesis_none" : containsCurrentBbox ? "current_hypothesis" : intersectsCurrentBbox ? "current_hypothesis_intersection" : "current_hypothesis_outside_crop",
          contextNote: containsCurrentBbox ? "The complete current hypothesis is visible." : "This crop shows only the selected local region. Keep the full-image context when judging target identity; the source hypothesis is unchanged.",
          coordinateNote: "Grid labels are full-image normalized coordinates and use coordinateSpace source. For model-requested views, use the returned viewId with view_pixels or view_normalized to map measurements automatically. last_crop is a legacy reference to the most recent crop.",
          informationNote: "Magnification changes display size, not source detail. Repeatedly enlarging the same pixels cannot recover missing texture or establish object identity.",
        }),
      },
      { type: "image", data: composed.toString("base64"), mimeType: "image/png" },
    ],
    details: {
      modality,
      path,
      originalWidth: sourceWidth,
      originalHeight: sourceHeight,
      width: displayWidth,
      height: displayHeight,
      cropNormalized: displayedNormalized,
    },
  };
  if (!showHypothesis || !currentBboxInCrop) return encodeGroundingImagePayload(image);
  return withGroundingOverlay(
    image,
    currentBboxInCrop,
    containsCurrentBbox ? "CURRENT HYPOTHESIS" : "HYPOTHESIS INTERSECTION",
    CURRENT_OVERLAY_COLOR,
    CURRENT_OVERLAY_HALO,
  );
}


function reviewImageFromGroundingImage(image: GroundingImage): GroundingReviewDetails["image"] {
  const imageBlock = image.content.find((block) => block.type === "image");
  if (!imageBlock || imageBlock.type !== "image") throw new Error("Grounding review image payload is missing.");
  const originalWidth = image.details.originalWidth ?? image.details.width ?? 1;
  const originalHeight = image.details.originalHeight ?? image.details.height ?? 1;
  return {
    data: imageBlock.data,
    mimeType: imageBlock.mimeType,
    originalWidth,
    originalHeight,
    ...(image.details.width !== undefined ? { width: image.details.width } : {}),
    ...(image.details.height !== undefined ? { height: image.details.height } : {}),
  };
}
const OVERLAY_COLOR = "#ff2d55";
const OVERLAY_HALO = "#0b1220";

const VERIFY_BOX_AREA_THRESHOLD = 0.03;
const VERIFY_BOX_SIDE_THRESHOLD = 0.06;
const VERIFY_PADDING = 0.6;

/**
 * A box occupying less than 3% of the frame, or thinner than 6% on either side,
 * is too small for a full-frame read to be trusted. Such a box always gets an
 * extra magnified verification crop, so the magnified inspection happens for
 * every small target without human intervention and without model discretion.
 */
function verificationRegionFor(bbox: readonly number[]): [number, number, number, number] | undefined {
  const width = bbox[2] - bbox[0];
  const height = bbox[3] - bbox[1];
  if (width * height >= VERIFY_BOX_AREA_THRESHOLD && Math.min(width, height) >= VERIFY_BOX_SIDE_THRESHOLD) {
    return undefined;
  }
  const padX = width * VERIFY_PADDING;
  const padY = height * VERIFY_PADDING;
  return [
    Math.max(0, bbox[0] - padX),
    Math.max(0, bbox[1] - padY),
    Math.min(1, bbox[2] + padX),
    Math.min(1, bbox[3] + padY),
  ];
}

function escapeXmlText(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll("\"", "&quot;");
}

/**
 * Stamp the agent's current working hypothesis onto every image returned by a
 * grounding tool. `bbox` is normalized to the image payload itself: source
 * resize preserves normalized coordinates, while a crop caller maps its source
 * box into crop-local coordinates before calling this helper.
 */
async function withGroundingOverlay(
  image: GroundingImage,
  bbox: readonly number[] | null,
  label: string,
  color: string,
  haloColor: string,
): Promise<GroundingImage> {
  const imageBlock = image.content.find((block) => block.type === "image");
  if (!imageBlock || imageBlock.type !== "image") throw new Error("Grounding image payload is missing.");
  const bytes = Buffer.from(imageBlock.data, "base64");
  const metadata = await sharp(bytes, { failOn: "error" }).metadata();
  if (!metadata.width || !metadata.height) throw new Error("Could not determine current-hypothesis image dimensions.");
  const width = metadata.width;
  const height = metadata.height;
  const normalized = bbox === null ? null : validateBoundingBox(bbox);
  const longSide = Math.max(width, height);
  const stroke = Math.max(2, Math.round(longSide / 320));
  const fontSize = Math.max(12, Math.round(longSide / 34));
  const textHalo = Math.max(2, Math.round(fontSize / 8));
  const gap = Math.max(stroke + 2, Math.round(fontSize * 0.35));
  const displayLabel = normalized ? label : `${label}: NONE`;
  const textWidth = fontSize * displayLabel.length * 0.6;
  let overlaySvg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">`;
  if (!normalized) {
    const bannerWidth = Math.min(width, Math.max(fontSize * 8, textWidth + gap * 2));
    const bannerHeight = Math.min(height, fontSize + gap * 2);
    const bannerY = 0;
    const textX = Math.max(gap, (width - bannerWidth) / 2 + gap);
    const textY = Math.max(fontSize, bannerY + gap + fontSize);
    overlaySvg += `<rect x="0" y="${bannerY}" width="${bannerWidth}" height="${bannerHeight}" fill="${haloColor}" fill-opacity="0.82"/>`;
    overlaySvg += `<text x="${textX.toFixed(1)}" y="${textY.toFixed(1)}" font-family="sans-serif" font-size="${fontSize}" font-weight="700"`
      + ` fill="${color}" stroke="${haloColor}" stroke-width="${textHalo}" paint-order="stroke">${escapeXmlText(displayLabel)}</text>`;
  } else {
    const x1 = Math.max(0, Math.min(width - 1, Math.round(normalized[0] * width)));
    const y1 = Math.max(0, Math.min(height - 1, Math.round(normalized[1] * height)));
    const x2 = Math.max(x1 + 1, Math.min(width, Math.round(normalized[2] * width)));
    const y2 = Math.max(y1 + 1, Math.min(height, Math.round(normalized[3] * height)));
    const boxWidth = x2 - x1;
    const boxHeight = y2 - y1;
    const textX = Math.max(gap, Math.min(x1, Math.max(gap, width - gap - textWidth)));
    const textY = y1 - gap >= fontSize
      ? y1 - gap
      : Math.max(fontSize, Math.min(height - gap, y2 + fontSize + gap));
    overlaySvg += `<rect x="${x1 + 0.5}" y="${y1 + 0.5}" width="${Math.max(1, boxWidth - 1)}" height="${Math.max(1, boxHeight - 1)}"`
      + ` fill="none" stroke="${color}" stroke-width="${stroke}"/>`;
    overlaySvg += `<text x="${textX.toFixed(1)}" y="${textY.toFixed(1)}" font-family="sans-serif" font-size="${fontSize}" font-weight="700"`
      + ` fill="${color}" stroke="${haloColor}" stroke-width="${textHalo}" paint-order="stroke">${escapeXmlText(displayLabel)}</text>`;
  }
  overlaySvg += `</svg>`;
  const output = await sharp(bytes)
    .composite([{ input: Buffer.from(overlaySvg), top: 0, left: 0 }])
    .png()
    .toBuffer();
  return encodeGroundingImagePayload({
    ...image,
    content: image.content.map((block) => block.type === "image"
      ? { ...block, data: output.toString("base64"), mimeType: "image/png" }
      : block),
  });
}

/**
 * Draw the saved box on the full image so that every result reaches the
 * conversation as an annotated picture, not only as numbers. The surrounding area
 * is dimmed so the framed target stays visually dominant, and the label carries
 * the record key, status, and confidence of the saved box.
 */
async function renderGroundingOverlay(
  image: GroundingImage,
  bbox: readonly number[],
  label: string,
): Promise<{ data: string; mimeType: string; width: number; height: number }> {
  const imageBlock = image.content.find((block) => block.type === "image");
  if (!imageBlock || imageBlock.type !== "image") throw new Error("Grounding overlay source image is missing.");
  const bytes = Buffer.from(imageBlock.data, "base64");
  const metadata = await sharp(bytes, { failOn: "error" }).metadata();
  if (!metadata.width || !metadata.height) throw new Error("Could not determine overlay image dimensions.");
  const width = metadata.width;
  const height = metadata.height;
  const clamped = validateBoundingBox(bbox);
  const x1 = Math.max(0, Math.min(width - 1, Math.round(clamped[0] * width)));
  const y1 = Math.max(0, Math.min(height - 1, Math.round(clamped[1] * height)));
  const x2 = Math.max(x1 + 1, Math.min(width, Math.round(clamped[2] * width)));
  const y2 = Math.max(y1 + 1, Math.min(height, Math.round(clamped[3] * height)));
  const longSide = Math.max(width, height);
  const stroke = Math.max(2, Math.round(longSide / 320));
  const fontSize = Math.max(12, Math.round(longSide / 34));
  const halo = Math.max(2, Math.round(fontSize / 8));
  const gap = Math.max(stroke + 2, Math.round(fontSize * 0.35));
  const boxWidth = x2 - x1;
  const boxHeight = y2 - y1;
  const textWidth = fontSize * label.length * 0.6;
  const textX = Math.max(gap, Math.min(x1, Math.max(gap, width - gap - textWidth)));
  const textY = y1 - gap >= fontSize
    ? y1 - gap
    : Math.max(fontSize, Math.min(height - gap, y2 + fontSize + gap));
  const dimRect = (x: number, y: number, rectWidth: number, rectHeight: number) =>
    `<rect x="${x}" y="${y}" width="${rectWidth}" height="${rectHeight}" fill="#000000" fill-opacity="0.38"/>`;
  const overlaySvg = `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}">`
    + dimRect(0, 0, width, y1)
    + dimRect(0, y2, width, height - y2)
    + dimRect(0, y1, x1, boxHeight)
    + dimRect(x2, y1, width - x2, boxHeight)
    + `<rect x="${x1 + 0.5}" y="${y1 + 0.5}" width="${Math.max(1, boxWidth - 1)}" height="${Math.max(1, boxHeight - 1)}"`
    + ` fill="none" stroke="${OVERLAY_COLOR}" stroke-width="${stroke}"/>`
    + `<text x="${textX.toFixed(1)}" y="${textY.toFixed(1)}" font-family="sans-serif" font-size="${fontSize}"`
    + ` fill="${OVERLAY_COLOR}" stroke="${OVERLAY_HALO}" stroke-width="${halo}" paint-order="stroke">${escapeXmlText(label)}</text>`
    + `</svg>`;
  const output = await sharp(bytes)
    .composite([{ input: Buffer.from(overlaySvg), top: 0, left: 0 }])
    .png()
    .toBuffer();
  const encoded = await encodeGroundingPreview(output, "image/png");
  return { data: encoded.data, mimeType: encoded.mimeType, width: encoded.width, height: encoded.height };
}
function block(reason: string) {
  return { block: true, reason: `[grounding safety] ${reason}` };
}

export function createGroundingSafetyExtension(options: GroundingSafetyOptions): InlineExtension {
  const tempRoot = join(tmpdir(), "pi-grounding", options.sessionId.replace(/[^a-zA-Z0-9_-]/g, "_"));
  const sanitizedQueries = new Map<string, Promise<SanitizedQueryState>>();
  const allowedReadPaths = new Set<string>();
  const loadedBatchRecords = new Map<string, LoadedBatchRecord>();
  // Each result requires the user's decision. Legacy auto/none environment
  // settings must never bypass this gate, and elapsed time is not a decision.

  const imageSourceCache = new Map<string, Promise<GroundingImageSource>>();
  const archivePathFor = (sourcePath: string, modality: GroundingModality): string => {
    const explicit = options.imageArchives?.[modality];
    if (explicit) return explicit;
    const modalityEnv = process.env[`PI_WEB_GROUNDING_${modality.toUpperCase()}_ARCHIVE`];
    if (modalityEnv) return modalityEnv;
    if (modality !== "depth" && process.env.PI_WEB_GROUNDING_VISIBLE_INFRARED_ARCHIVE) {
      return process.env.PI_WEB_GROUNDING_VISIBLE_INFRARED_ARCHIVE;
    }
    const evidenceRoot = process.env.PI_WEB_GROUNDING_EVIDENCE_DIR
      ?? join(parse(sourcePath).root, "evidence");
    return modality === "depth"
      ? join(evidenceRoot, "depth.zip", "depth.zip")
      : join(evidenceRoot, "infrared.zip", "infrared.zip");
  };
  const cachedImageSource = (key: string, loader: () => Promise<GroundingImageSource>) => {
    const existing = imageSourceCache.get(key);
    if (existing) {
      imageSourceCache.delete(key);
      imageSourceCache.set(key, existing);
      return existing;
    }
    const pending = loader().catch((error) => {
      imageSourceCache.delete(key);
      throw error;
    });
    imageSourceCache.set(key, pending);
    while (imageSourceCache.size > 12) {
      const oldest = imageSourceCache.keys().next().value as string | undefined;
      if (!oldest) break;
      imageSourceCache.delete(oldest);
    }
    return pending;
  };
  const readRecordImage: GroundingImageReader = async (sourcePath, record, modality) => {
    const directPath = recordImagePath(sourcePath, record, modality);
    // File identity is case-sensitive on Linux. The safety guard's folded
    // paths must not make A.png reuse a.png's image bytes.
    const directKey = `file:${directPath}`;
    try {
      return await cachedImageSource(directKey, async () => ({
        bytes: await readFile(directPath),
        label: directPath,
      }));
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }

    const archivePath = archivePathFor(sourcePath, modality);
    const entryPath = `${modality}/${basename(record[modality])}`;
    // ZIP member names remain case-sensitive even on Windows.
    const archiveKey = JSON.stringify(["zip", resolve(archivePath), entryPath]);
    return cachedImageSource(archiveKey, async () => {
      try {
        return {
          bytes: await readArchiveEntry(archivePath, entryPath),
          label: `${archivePath}::${entryPath}`,
        };
      } catch (error) {
        throw new Error(
          `The ${modality} image could not be loaded as grounding evidence (${(error as Error).message}).`
          + " Continue with the views that are available; a missing view is not evidence that the target is absent.",
        );
      }
    });
  };
  let active = false;
  let batchMode = false;

  return {
    name: EXTENSION_NAME,
    hidden: true,
    factory: (pi: ExtensionAPI) => {
      let batchToolStarted = false;
      let batchExhausted = false;
      let requestedRecordLimit: number | undefined;
      let sessionSavedCount = 0;
      let startupNudges = 0;
      let continuationNudges = 0;
      let savingRecord = false;
      let reviewInterrupted = false;
      let lastJob: { queryPath: string; outputDir: string } | undefined;
      let jobStartCompleted: number | undefined;
      let requestedCountPending = false;
      let lastApprovedKey: string | undefined;
      let toolsBeforeBatch: string[] | undefined;
      let persistedPending: PersistedGroundingPending | undefined;

      const loadedRecord = (
        sourcePath: string,
        outputDirectory: string,
        record: Record<string, string>,
        pending?: PersistedGroundingPending,
      ): LoadedBatchRecord => {
        const grantedModalities = new Set<GroundingModality>();
        if (record.visible) grantedModalities.add("visible");
        if (record.infrared && INFRARED_QUERY.test(record.query ?? "")) grantedModalities.add("infrared");
        if (record.depth && DEPTH_QUERY.test(record.query ?? "")) grantedModalities.add("depth");
        return {
          sourcePath,
          outputDirectory,
          record,
          grantedModalities,
          settleNudges: 0,
          ...(pending?.currentBbox ? { currentBbox: pending.currentBbox } : {}),
          ...(pending?.workingState ? { workingState: pending.workingState } : {}),
          ...(pending?.revision ? { revision: true } : {}),
          ...recordEvidence(),
        };
      };

      const persistJob = () => {
        const current = loadedBatchRecords.entries().next().value as [string, LoadedBatchRecord] | undefined;
        if (current) {
          persistedPending = {
            key: current[0],
            ...(current[1].currentBbox ? { currentBbox: current[1].currentBbox } : {}),
            ...(current[1].workingState ? { workingState: current[1].workingState } : {}),
            ...(current[1].revision ? { revision: true } : {}),
          };
        }
        if (lastJob) pi.appendEntry?.("grounding:job", { version: 2, ...lastJob, targetCount: requestedRecordLimit,
          startCompleted: jobStartCompleted, lastApprovedKey, pending: persistedPending ?? null });
      };

      const jobSnapshot = () => {
        const current = loadedBatchRecords.entries().next().value as [string, LoadedBatchRecord] | undefined;
        const pendingKey = current?.[0] ?? persistedPending?.key;
        const pendingRevision = current?.[1].revision ?? persistedPending?.revision;
        return { targetCount: requestedRecordLimit ?? null, approvedInJob: sessionSavedCount,
          currentKey: pendingKey ?? null, lastApprovedKey: lastApprovedKey ?? null,
          pendingCorrectionKey: pendingRevision ? pendingKey ?? null : null,
          reviewStatus: savingRecord ? "awaiting_review_or_saving" : pendingKey ? "awaiting_prediction" : "idle",
          requestedLimitReached: requestedRecordLimit !== undefined && sessionSavedCount >= requestedRecordLimit };
      };

      const reportCompletion = (completed: number, total: number, outputDir: string) => {
        pi.sendMessage({
          customType: "grounding-complete",
          content: completed === total
            ? `【标注进度】这组数据已完成 ${completed}/${total} 条，没有下一条。\n\n输出目录：${outputDir}\n\n如需修改已保存的框，请说明记录编号或目标；可以重新打开该条，根据图像证据修正后再次提交人工审核。`
            : `【标注进度】本次请求已完成 ${sessionSavedCount}/${requestedRecordLimit} 条。数据集累计 ${completed}/${total} 条；本次已停止，不会自动处理其余记录。\n\n输出目录：${outputDir}`,
          display: true,
          details: { completed, total, outputDir },
        }, { triggerTurn: false });
      };

      const requestGroundingReview = async (
        ctx: ExtensionContext,
        details: GroundingReviewDetails,
        signal?: AbortSignal,
      ): Promise<GroundingReviewResponse> => {
        signal?.throwIfAborted();
        const response = await ctx.ui.custom<GroundingReviewResponse>(
          (...args: unknown[]) => {
            const done = args[3];
            if (typeof done !== "function") throw new Error("Grounding review UI did not provide a completion callback.");
            const onAbort = () => {
              (done as (value: GroundingReviewResponse) => void)({
                type: "grounding_review_response",
                action: "reject",
                reason: "Grounding review was cancelled; this record remains unsaved.",
              });
            };
            signal?.addEventListener("abort", onAbort, { once: true });
            if (signal?.aborted) onAbort();
            return {
              groundingReview: details,
              render: () => ["Grounding browser review required before saving this record."],
              invalidate: () => {},
              handleInput: (data: string) => {
                try {
                  const parsed = validateGroundingReviewResponse(JSON.parse(data), details);
                  (done as (value: GroundingReviewResponse) => void)(parsed);
                } catch {
                  // Invalid or generic custom-UI input cannot approve a box.
                }
              },
              dispose: () => {
                signal?.removeEventListener("abort", onAbort);
              },
            };
          },
          { overlayOptions: { width: 120 } },
        );
        signal?.throwIfAborted();
        return validateGroundingReviewResponse(response, details);
      };

      const setGroundingToolsActive = (enabled: boolean) => {
        const activeTools = pi.getActiveTools();
        if (enabled && batchMode && !toolsBeforeBatch) {
          toolsBeforeBatch = activeTools.filter((name) => !GROUNDING_TOOL_NAMES.includes(name as typeof GROUNDING_TOOL_NAMES[number]));
        }
        const nextTools = enabled && batchMode ? [] : (toolsBeforeBatch ?? activeTools)
          .filter((name) => !GROUNDING_TOOL_NAMES.includes(name as typeof GROUNDING_TOOL_NAMES[number]));
        if (enabled) nextTools.push(...GROUNDING_TOOL_NAMES);
        if (nextTools.length !== activeTools.length || nextTools.some((name, index) => name !== activeTools[index])) {
          pi.setActiveTools(nextTools);
        }
      };

      pi.on("context", (event) => {
        if (!active || !batchMode) return undefined;
        const current = loadedBatchRecords.values().next().value as LoadedBatchRecord | undefined;
        const completedContext = compactCompletedGroundingContext(event.messages, loadedBatchRecords.size > 0);
        const messages = current ? compactGroundingEvidence(completedContext, { active: true,
          pinnedViewIds: [...current.pinnedViewIds], archivedViewIds: [...current.archivedViewIds] }) : completedContext;
        return messages === event.messages ? undefined : { messages };
      });

      const loadQueryState = async (queryPath: string): Promise<{ sourcePath: string; state: SanitizedQueryState }> => {
        const sourcePath = resolve(options.cwd, queryPath);
        const statePromise = sanitizedQueries.get(sourcePath) ?? makeSanitizedQuery(sourcePath, tempRoot);
        sanitizedQueries.set(sourcePath, statePromise);
        try {
          const state = await statePromise;
          allowedReadPaths.add(normalizePath(state.safePath));
          return { sourcePath, state };
        } catch (error) {
          sanitizedQueries.delete(sourcePath);
          throw error;
        }
      };

      const allowRecordModality = (
        sourcePath: string,
        record: Record<string, string>,
        modality: GroundingModality,
      ) => {
        const imagePath = record[modality];
        if (!imagePath) return;
        allowedReadPaths.add(normalizePath(imagePath));
        allowedReadPaths.add(normalizePath(resolve(dirname(sourcePath), imagePath)));
      };

      const grantRecordModality = (loaded: LoadedBatchRecord, modality: GroundingModality) => {
        if (!loaded.record[modality]) return;
        allowRecordModality(loaded.sourcePath, loaded.record, modality);
        loaded.grantedModalities.add(modality);
      };

      const loadNextRecord = async (queryPath: string, outputDir: string) => {
        // Mark the batch as started before looking for a record. An empty
        // exhausted result is still a completed tool call; treating it as
        // "not started" would let agent_before_settle inject another request.
        batchToolStarted = true;
        if (loadedBatchRecords.size > 0) {
          throw new Error("Cannot load another record until the user approves the current one. Rejected records must be revised and reviewed again.");
        }
        if (!active) throw new Error("Grounding runtime safety is not active for this session.");
        const { sourcePath, state } = await loadQueryState(queryPath);
        const outputDirectory = resolvedOutputDirectory(options.cwd, sourcePath, outputDir);
        const changedJob = Boolean(lastJob
          && (normalizePath(resolve(options.cwd, lastJob.queryPath)) !== normalizePath(sourcePath)
            || normalizePath(lastJob.outputDir) !== normalizePath(outputDirectory)));
        if (changedJob) {
          jobStartCompleted = undefined;
          sessionSavedCount = 0;
          lastApprovedKey = undefined;
          persistedPending = undefined;
          if (!requestedCountPending) requestedRecordLimit = undefined;
        }
        lastJob = { queryPath: sourcePath, outputDir: outputDirectory };
        const progressPath = join(outputDirectory, "progress.jsonl");
        const completed = await readGroundingProgress(progressPath);
        jobStartCompleted ??= completed.size;
        sessionSavedCount = Math.max(0, completed.size - jobStartCompleted);
        requestedCountPending = false;
        let current: [string, LoadedBatchRecord] | undefined;
        const requestedLimitReached = requestedRecordLimit !== undefined && sessionSavedCount >= requestedRecordLimit;
        const pendingRecord = persistedPending ? state.safe[persistedPending.key] : undefined;
        if (persistedPending && (!pendingRecord || completed.has(persistedPending.key) && !persistedPending.revision)) {
          // A process may stop after the result file is committed but before a
          // clearing job entry is appended. The progress file is authoritative
          // for a normal prediction that is already approved.
          persistedPending = undefined;
        }
        const resumedPending = persistedPending && pendingRecord
          ? [persistedPending.key, loadedRecord(sourcePath, outputDirectory, pendingRecord, persistedPending)] as [string, LoadedBatchRecord]
          : undefined;
        if (resumedPending) {
          current = resumedPending;
        } else if (!requestedLimitReached) {
          const next = Object.entries(state.safe).find(([key]) => !completed.has(key));
          if (next) current = [next[0], loadedRecord(sourcePath, outputDirectory, next[1])];
        }

        batchExhausted = !current;
        if (current) {
          continuationNudges = 0;
        }
        const total = Object.keys(state.safe).length;
        const record = current ? { key: current[0], ...current[1].record } : null;
        const payload = {
          records: record ? [record] : [],
          completed: completed.size,
          total,
          remaining: Math.max(0, total - completed.size),
          outputDir: outputDirectory,
          requestedLimitReached,
          targetCount: requestedRecordLimit ?? null,
          approvedInJob: sessionSavedCount,
          includedModalities: current
            ? (["visible", "infrared", "depth"] as GroundingModality[])
                .filter((modality) => current?.[1].grantedModalities.has(modality))
            : [],
        };
        const content: GroundingToolContent[] = [{ type: "text", text: JSON.stringify(payload) }];
        const imageDetails: GroundingImage["details"][] = [];
        if (current) {
          try {
            for (const modality of payload.includedModalities) {
              const image = await loadGroundingImage(sourcePath, current[1].record, modality, readRecordImage,
                current[1].currentBbox ?? null);
              const view = registerGroundingView(current[1], image, "hypothesis");
              current[1].pinnedViewIds.add(view.id);
              content.push(...image.content);
              imageDetails.push(image.details);
            }
          } catch (error) {
            // No query or image was delivered. Leave this key available for a
            // retry, including when a corrupt image is replaced on disk.
            imageSourceCache.clear();
            throw error;
          }
          for (const modality of payload.includedModalities) allowRecordModality(sourcePath, current[1].record, modality);
          loadedBatchRecords.set(current[0], current[1]);
        }
        persistJob();
        return {
          content,
          details: { records: payload.records, completed: completed.size, total, images: imageDetails,
            requestedLimitReached, job: jobSnapshot() },
          payload,
        };
      };

      const saveReviewedGroundingResult = async (
        params: GroundingResultInput,
        ctx: ExtensionContext,
        signal: AbortSignal | undefined,
        continueAfterSave: boolean,
      ) => {
        signal?.throwIfAborted();
        if (!active) throw new Error("Grounding runtime safety is not active for this session.");
        const { sourcePath, state } = await loadQueryState(params.queryPath);
        if (!Object.hasOwn(state.safe, params.key)) throw new Error(`Unknown grounding key: ${params.key}`);
        const outputDirectory = resolvedOutputDirectory(options.cwd, sourcePath, params.outputDir);
        let loaded = loadedBatchRecords.get(params.key);
        const pendingMatchesJob = persistedPending?.key === params.key && lastJob
          && normalizePath(resolve(options.cwd, lastJob.queryPath)) === normalizePath(sourcePath)
          && normalizePath(lastJob.outputDir) === normalizePath(outputDirectory);
        if (!loaded && pendingMatchesJob) {
          loaded = loadedRecord(sourcePath, outputDirectory, state.safe[params.key], persistedPending);
          for (const modality of loaded.grantedModalities) allowRecordModality(sourcePath, loaded.record, modality);
          loadedBatchRecords.set(params.key, loaded);
        }
        if (!loaded || loaded.sourcePath !== sourcePath) {
          throw new Error("Load this record with grounding_next_batch before saving its result.");
        }
        if (normalizePath(outputDirectory) !== normalizePath(loaded.outputDirectory)) {
          throw new Error("Use the same output directory that loaded this record; do not change it during review.");
        }
        const completed = await readGroundingProgress(join(outputDirectory, "progress.jsonl"));
        const coordinateSpace = params.coordinateSpace ?? "source";
        const mappedView = coordinateSpace === "view_pixels" || coordinateSpace === "view_normalized";
        if (mappedView && !params.viewId) throw new Error("viewId is required for view_pixels or view_normalized coordinates.");
        const inputBbox = mappedView
          ? loaded.views.toSource(params.viewId!, params.bbox, coordinateSpace)
          : validateBoundingBox(params.bbox);
        let proposedBbox = inputBbox;
        if (coordinateSpace === "last_crop") {
          const crop = loaded?.lastCropRegion;
          if (!crop) throw new Error("coordinateSpace last_crop requires a successful grounding_view crop for this record.");
          const cropWidth = crop[2] - crop[0];
          const cropHeight = crop[3] - crop[1];
          proposedBbox = validateBoundingBox([
            crop[0] + inputBbox[0] * cropWidth,
            crop[1] + inputBbox[1] * cropHeight,
            crop[0] + inputBbox[2] * cropWidth,
            crop[1] + inputBbox[3] * cropHeight,
          ]);
        } else if (coordinateSpace !== "source" && !mappedView) {
          throw new Error("coordinateSpace must be source, last_crop, view_pixels, or view_normalized.");
        }
        if (typeof params.reason !== "string" || params.reason.trim().length < 8) {
          throw new Error("Provide a short visible-evidence reason of at least 8 characters before requesting review.");
        }
        const rawBbox = proposedBbox;
        // The browser draws the editable box. Baked-in hypothesis pixels would
        // leave a second, stale box visible after a manual edit.
        const visibleImage = await loadGroundingImage(sourcePath, loaded.record, "visible", readRecordImage, null, false);
        const previousBbox = loaded.currentBbox;
        const candidateChange = previousBbox
          && previousBbox.some((edge, index) => Math.abs(edge - rawBbox[index]) > 1e-12)
          ? describeGroundingCandidateChange(previousBbox, rawBbox,
            visibleImage.details.originalWidth!, visibleImage.details.originalHeight!)
          : undefined;
        loaded.currentBbox = rawBbox;
        persistJob();
        const reviewDetails: GroundingReviewDetails = {
          kind: "grounding_review",
          key: params.key,
          query: loaded.record.query ?? "",
          bbox: rawBbox,
          ...(candidateChange ? { previousBbox: candidateChange.previousBbox, candidateChange } : {}),
          status: params.status,
          confidence: params.confidence,
          targetFound: params.status !== "unresolved",
          candidateCount: 1,
          candidateRank: params.status === "unresolved" ? undefined : 1,
          expectedOrdinal: expectedGroundingOrdinal(loaded.record.query ?? ""),
          reason: params.reason.trim(),
          image: reviewImageFromGroundingImage(visibleImage),
          availableModalities: Array.from(loaded.grantedModalities),
          canContinue: !loaded.revision && (continueAfterSave || requestedRecordLimit !== undefined)
            && (requestedRecordLimit === undefined || sessionSavedCount + 1 < requestedRecordLimit)
            && Object.keys(state.safe).some((key) => key !== params.key && !completed.has(key)),
        };
        let review: GroundingReviewResponse;
        try {
          review = await requestGroundingReview(ctx, reviewDetails, signal);
        } catch (error) {
          reviewInterrupted = true;
          throw error;
        }
        if (review.action === "reject") {
          throw new Error(`Browser grounding review rejected candidate: ${review.reason}. Revise this same record and request review again; do not load the next record.`);
        }
        signal?.throwIfAborted();
        const bbox = review.bbox;
        loaded.currentBbox = bbox;
        persistJob();
        const status = review.status;
        const confidence = review.confidence;
        await mkdir(outputDirectory, { recursive: true });
        const progressPath = join(outputDirectory, "progress.jsonl");
        const submissionPath = join(outputDirectory, "queries.json");
        const archivePath = join(outputDirectory, "queries.zip");
        const overlay = await renderGroundingOverlay(
          visibleImage,
          bbox,
          `${params.key} | ${status} ${confidence.toFixed(2)}`,
        );
        // Small targets are the ones a full frame cannot show reliably, so the
        // runtime re-inspects them itself instead of trusting the model to ask for
        // a crop. This is a runtime safety artifact and does not consume or
        // restrict the model's crop requests.
        const verification = verificationRegionFor(bbox);
        const verifyCrop = verification
          ? await loadGroundingCrop(sourcePath, loaded.record, "visible", verification, bbox, readRecordImage)
          : null;
        const verificationContent: GroundingToolContent[] = [];
        const verifyText = verifyCrop?.content.find((block) => block.type === "text");
        if (verifyText && verifyText.type === "text") {
          verificationContent.push({
            type: "text",
            text: JSON.stringify({ automaticMagnifiedVerification: JSON.parse(verifyText.text) }),
          });
        }
        const verifyImage = verifyCrop?.content.find((block) => block.type === "image");
        if (verifyImage && verifyImage.type === "image") verificationContent.push(verifyImage);
        const summary = await withFileMutationQueue(progressPath, async () => {
          signal?.throwIfAborted();
          const progress = await readGroundingProgress(progressPath);
          progress.set(params.key, {
            key: params.key,
            status,
            confidence,
            bbox,
            reviewed: true,
            targetFound: review.targetFound,
            candidateCount: review.candidateCount,
            ...(review.candidateRank !== undefined ? { candidateRank: review.candidateRank } : {}),
            ...(reviewDetails.expectedOrdinal !== undefined ? { expectedOrdinal: reviewDetails.expectedOrdinal } : {}),
            reason: review.reason,
            reviewSource: "human",
          });

          const orderedProgress = Object.keys(state.safe)
            .map((key) => progress.get(key))
            .filter((item): item is GroundingProgress => item !== undefined);
          const progressText = orderedProgress.map((item) => JSON.stringify(item)).join("\n");
          await writeFileAtomic(progressPath, progressText ? `${progressText}\n` : "", options.sessionId);

          const submission: Record<string, unknown> = {};
          for (const key of Object.keys(state.safe)) {
            const result = progress.get(key);
            if (!result) continue;
            const original = state.source[key];
            if (!original || typeof original !== "object" || Array.isArray(original)) continue;
            const originalWithoutBbox = { ...(original as Record<string, unknown>) };
            delete originalWithoutBbox.bbox;
            submission[key] = { ...originalWithoutBbox, bbox: result.bbox };
          }
          const submissionText = `${JSON.stringify(submission, null, 2)}\n`;
          await writeFileAtomic(submissionPath, submissionText, options.sessionId);

          const complete = orderedProgress.length === Object.keys(state.safe).length;
          if (complete) {
            const archive = new JSZip();
            archive.file(QUERY_FILE, submissionText);
            const archiveBuffer = await archive.generateAsync({
              type: "nodebuffer",
              compression: "DEFLATE",
              compressionOptions: { level: 9 },
            });
            await writeBufferAtomic(archivePath, archiveBuffer, options.sessionId);
          }
          return {
            saved: params.key,
            processed: orderedProgress.length,
            total: Object.keys(state.safe).length,
            progressPath,
            submissionPath,
            ...(complete ? { archivePath } : {}),
          };
        });
        const revision = loaded.revision === true;
        if (batchMode && !revision) sessionSavedCount += 1;
        lastApprovedKey = params.key;
        loadedBatchRecords.delete(params.key);
        persistedPending = undefined;
        persistJob();
        return { summary, overlay, verificationContent, revision };
      };

      const saveGroundingResult = async (
        params: GroundingResultInput,
        ctx: ExtensionContext,
        signal?: AbortSignal,
        continueAfterSave = false,
      ) => {
        if (savingRecord) throw new Error("The current record is already awaiting human review or being saved.");
        savingRecord = true;
        try {
          return await saveReviewedGroundingResult(params, ctx, signal, continueAfterSave);
        } finally {
          savingRecord = false;
          if (signal?.aborted) reviewInterrupted = true;
        }
      };

      pi.registerTool({
        name: "grounding_next_batch",
        label: "Grounding next record",
        description: "Load exactly one unfinished sanitized grounding record and attach its needed image modalities, starting with visible. The first image explicitly states when no current hypothesis exists.",
        promptSnippet: "Load the next unfinished grounding record with its visible image and explicit hypothesis state",
        promptGuidelines: [
          "For a query dataset, call grounding_next_batch once to load the next record instead of reading queries.json directly.",
          "This tool loads at most one record: omit limit or set limit: 1; never put the total requested batch count in limit.",
          "Set targetCount once to the total number of records requested for this job. It is separate from limit=1 and persists across prompts/reloads. Stop when requestedLimitReached is true even if the dataset has other unfinished records.",
          "If the result has records: [] and remaining: 0, the dataset is exhausted; stop immediately and do not call this tool again.",
          "The tool already returns the visible image and any modality explicitly required by the query; do not call read for those images.",
          "Every returned image is marked in-image with the current working bbox; the first image says CURRENT HYPOTHESIS: NONE because no bbox has been proposed yet.",
          "Process and save the returned record before requesting another one.",
        ],
        parameters: Type.Object({
          queryPath: Type.String({ description: "Path to the source queries.json" }),
          outputDir: Type.String({ description: "Dedicated output directory outside the source dataset. Use a new or empty directory; an existing progress.jsonl written by another build cannot be resumed." }),
          targetCount: Type.Optional(Type.Integer({ minimum: 1, description: "Total records requested for this job, not the per-call loading limit. Infer only from the user's requested scope." })),
          limit: Type.Optional(Type.Number({
            minimum: 1,
            maximum: 1,
            description: "Compatibility field; serial grounding always loads exactly one record",
          })),
        }),
        executionMode: "sequential",
        async execute(_toolCallId, params) {
          // Tool use is authoritative: users should not need to write a
          // special "batch" keyword merely to unlock record views and the
          // serial save/continue safeguards.
          if (params.targetCount !== undefined) {
            if (!Number.isSafeInteger(params.targetCount) || params.targetCount < 1) throw new Error("targetCount must be a positive integer.");
            if (requestedRecordLimit !== undefined && requestedRecordLimit !== params.targetCount) {
              throw new Error(`targetCount must match the current user-requested count (${requestedRecordLimit}). Start a newly requested job to change it.`);
            }
            requestedRecordLimit = params.targetCount;
            requestedCountPending = true;
          }
          batchMode = true;
          setGroundingToolsActive(true);
          const result = await loadNextRecord(params.queryPath, params.outputDir);
          if (result.payload.records.length === 0) reportCompletion(result.payload.completed, result.payload.total, result.payload.outputDir);
          // An exhausted dataset is a terminal state. Without an explicit
          // termination marker, a model can interpret records: [] as a
          // transient failure and call this tool forever.
          return {
            content: result.content,
            details: result.details,
            terminate: result.payload.records.length === 0,
          };
        },
      });

      const resolveGroundingJob = (params: { queryPath?: string; outputDir?: string }) => {
        const queryPath = params.queryPath ?? lastJob?.queryPath;
        const outputDir = params.outputDir ?? lastJob?.outputDir;
        if (!queryPath || !outputDir) throw new Error("Provide queryPath and outputDir for this dataset once; grounding_status and grounding_reopen_record can then reuse them.");
        return { queryPath, outputDir };
      };

      const resolveLoadedRecord = (params: { key?: string; queryPath?: string }) => {
        if (!active) throw new Error("Grounding runtime safety is not active for this session.");
        const current = params.key ? [params.key, loadedBatchRecords.get(params.key)] as const : loadedBatchRecords.entries().next().value;
        if (!current?.[1]) throw new Error("Load this record with grounding_next_batch or grounding_reopen_record before viewing evidence.");
        if (params.queryPath && normalizePath(resolve(options.cwd, params.queryPath)) !== normalizePath(current[1].sourcePath)) throw new Error("Use the currently loaded record's queryPath.");
        return current as [string, LoadedBatchRecord];
      };

      pi.registerTool({
        name: "grounding_status",
        label: "Grounding progress",
        description: "Read this run's approved predictions and progress without listing directories or reading reference annotations. Reuses the last dataset/output paths. Use before responding to continue or a request to correct a saved box. Returns at most 20 records per page, including keys, query text and this run's approved boxes, never source annotations.",
        parameters: Type.Object({
          queryPath: Type.Optional(Type.String()), outputDir: Type.Optional(Type.String()),
          offset: Type.Optional(Type.Integer({ minimum: 0 })),
          limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 20 })),
        }),
        executionMode: "sequential",
        async execute(_id, params) {
          if (!active) throw new Error("Grounding runtime safety is not active for this session.");
          const job = resolveGroundingJob(params);
          const { sourcePath, state } = await loadQueryState(job.queryPath);
          const outputDir = resolvedOutputDirectory(options.cwd, sourcePath, job.outputDir);
          const progress = await readGroundingProgress(join(outputDir, "progress.jsonl"));
          const entries = Object.entries(state.safe);
          const completed = entries.filter(([key]) => progress.has(key)).length;
          lastJob ??= { queryPath: sourcePath, outputDir };
          const sameJob = normalizePath(lastJob.queryPath) === normalizePath(sourcePath) && normalizePath(lastJob.outputDir) === normalizePath(outputDir);
          if (sameJob && jobStartCompleted !== undefined) sessionSavedCount = Math.max(0, completed - jobStartCompleted);
          const offset = params.offset ?? 0;
          const records = entries.slice(offset, offset + (params.limit ?? 20)).map(([key, record]) => {
            const loaded = loadedBatchRecords.get(key);
            const belongsHere = loaded && normalizePath(loaded.sourcePath) === normalizePath(sourcePath)
              && normalizePath(loaded.outputDirectory) === normalizePath(outputDir);
            return { key, query: record.query,
              state: belongsHere ? "awaiting_prediction_or_review" : progress.has(key) ? "approved" : "unfinished",
              ...(progress.has(key) ? { approvedPrediction: progress.get(key) } : {}) };
          });
          const details = { queryPath: sourcePath, outputDir, completed, total: entries.length,
            remaining: entries.length - completed, records, nextOffset: offset + records.length < entries.length ? offset + records.length : null,
            job: sameJob ? jobSnapshot() : null,
            note: "Approved predictions are this run's own results, not ground truth. Reopen a specific key to revise it; approval is still required." };
          return { content: [{ type: "text" as const, text: JSON.stringify(details) }], details };
        },
      });

      pi.registerTool({
        name: "grounding_reopen_record",
        label: "Grounding revise record",
        description: "Reopen one previously approved key for a user-requested correction. Get its key from grounding_status; no directory listing needed. Original approved output remains unchanged until the revised box passes human review. Revise and finish with grounding_save_result.",
        parameters: Type.Object({
          key: Type.String(), queryPath: Type.Optional(Type.String()), outputDir: Type.Optional(Type.String()),
        }),
        executionMode: "sequential",
        async execute(_id, params) {
          if (!active) throw new Error("Grounding runtime safety is not active for this session.");
          if (loadedBatchRecords.size || savingRecord) throw new Error("Finish reviewing the currently loaded record before reopening another key.");
          const job = resolveGroundingJob(params);
          const { sourcePath, state } = await loadQueryState(job.queryPath);
          const outputDir = resolvedOutputDirectory(options.cwd, sourcePath, job.outputDir);
          const previous = (await readGroundingProgress(join(outputDir, "progress.jsonl"))).get(params.key);
          const record = state.safe[params.key];
          if (!record || !previous) throw new Error("This key has no approved result in the selected run. Use grounding_status to choose an approved key, or grounding_next_batch for unfinished records.");
          const image = await loadGroundingImage(sourcePath, record, "visible", readRecordImage, previous.bbox);
          const loaded: LoadedBatchRecord = { sourcePath, outputDirectory: outputDir, record, grantedModalities: new Set(),
            currentBbox: previous.bbox, settleNudges: 0, revision: true, ...recordEvidence() };
          const view = registerGroundingView(loaded, image, "hypothesis");
          loaded.pinnedViewIds.add(view.id);
          grantRecordModality(loaded, "visible");
          loadedBatchRecords.set(params.key, loaded);
          batchMode = true; batchToolStarted = true; batchExhausted = false;
          requestedRecordLimit = undefined; sessionSavedCount = 0; reviewInterrupted = false;
          requestedCountPending = false;
          jobStartCompleted = undefined;
          lastJob = { queryPath: sourcePath, outputDir };
          persistJob();
          setGroundingToolsActive(true);
          const details = { key: params.key, query: record.query, previousPrediction: previous,
            queryPath: sourcePath, outputDir, revision: true, note: "This is the run's own approved prediction. Apply the user's correction; save only after another human review." };
          return { content: [{ type: "text" as const, text: JSON.stringify(details) }, ...image.content], details };
        },
      });

      pi.registerTool({
        name: "grounding_color_region",
        label: "Grounding local color analysis",
        description: "Optional deterministic pixel measurement for a visually identified target or part whose color reliably contrasts with its surroundings and whose boundary needs measurement. Not a default step for every record: skip it when color is unhelpful or the box is already clear. Analyzes only original visible-image pixels in a chosen local region; returns source-normalized color-component bounds, an optional point sample, clean crop and mask preview. It cannot identify the target, recover hidden boundaries, or save a result.",
        promptGuidelines: [
          "First inspect the image and establish the requested object/part from shape, structure and surrounding context. Use this optional tool only if reliable local color contrast can answer a remaining pixel-boundary question; neither a color word in the query nor a difficult record requires it. If the box is already clear, proceed to review without color analysis.",
          "Skip color analysis when identity is unresolved or color does not distinguish the target. Use existing visual evidence, grounding_view or grounding_compare as needed instead. Shadows, lighting, reflections, similar-colored neighbors, low resolution and occlusion can merge or fragment matches; preserve uncertainty rather than forcing a color-derived box.",
          "Sampling always uses the original visible image, not infrared/depth pixels or their display palettes. A viewId maps coordinates only; it does not change the sampled modality or establish cross-modal alignment. Choose a small ROI around the visually identified part with enough surrounding context to check the boundary.",
          "Use color black/white/gray/red/orange/yellow/green/cyan/blue/purple/pink/brown or #RRGGBB. tolerance (0..1, default .12) broadens matching; #RRGGBB with tolerance 0 matches exact bytes.",
          "selection largest (default) chooses the largest connected matching patch, not the most likely target. Black pixels may be shadows or background, not the requested part. point chooses the patch containing your point, but does not verify its identity. all unions separated letters/parts only when visible evidence shows they belong to the target.",
          "Inspect selectionAssessment, pointSample, clean and mask previews. No match does not prove target absence and largest can be background. A mask filling most of the ROI or reaching several edges calls for checking background/clipping, not increased confidence. clippedRoiEdges lists artificial clipping that a wider ROI may resolve; touchesSourceEdges lists actual image borders with no pixels beyond them. Do not broaden thresholds just to force a match.",
          "When choosing the ROI or point on a returned crop/comparison, pass its viewId with coordinateSpace view_pixels or view_normalized. The runtime maps both region and point to source pixels.",
          "Returned bbox coordinates enclose only the selected matching pixels, not necessarily the full requested target or part. Check them against the clean image; do not treat missing or occluded pixels as recovered. Coordinates are already in full-image source space: pass coordinateSpace source when saving. Submit your chosen/corrected box for human approval; color analysis never auto-saves.",
        ],
        parameters: Type.Object({
          key: Type.Optional(Type.String({ description: "Optional when exactly one record is loaded" })),
          queryPath: Type.Optional(Type.String({ description: "Optional; defaults to the loaded record's dataset" })),
          region: Type.Array(Type.Number(), { minItems: 4, maxItems: 4, description: "Small ROI in coordinateSpace; source-normalized by default, or coordinates on viewId" }),
          color: Type.String({ description: "Named color preset or #RRGGBB" }),
          tolerance: Type.Optional(Type.Number({ minimum: 0, maximum: 1 })),
          minAreaPixels: Type.Optional(Type.Integer({ minimum: 1 })),
          selection: Type.Optional(Type.Union([Type.Literal("largest"), Type.Literal("all"), Type.Literal("point")])),
          point: Type.Optional(Type.Array(Type.Number(), { minItems: 2, maxItems: 2, description: "Required for point selection; uses the same coordinateSpace and viewId as region" })),
          coordinateSpace: Type.Optional(Type.Union([Type.Literal("source"), Type.Literal("view_pixels"), Type.Literal("view_normalized")])),
          viewId: Type.Optional(Type.String({ description: "Required with view_pixels or view_normalized; must belong to the current record" })),
        }),
        executionMode: "sequential",
        async execute(_id, params, signal) {
          signal?.throwIfAborted();
          if (!active) throw new Error("Grounding runtime safety is not active for this session.");
          const current = params.key ? [params.key, loadedBatchRecords.get(params.key)] as const : loadedBatchRecords.entries().next().value;
          if (!current?.[1]) throw new Error("Load a record with grounding_next_batch, or use grounding_status and grounding_reopen_record to revise an approved result.");
          const [key, loaded] = current;
          if (params.queryPath && normalizePath(resolve(options.cwd, params.queryPath)) !== normalizePath(loaded.sourcePath)) throw new Error("Use the currently loaded record's queryPath for color analysis.");
          const coordinateSpace = params.coordinateSpace ?? "source";
          const mappedView = coordinateSpace === "view_pixels" || coordinateSpace === "view_normalized";
          if (mappedView && !params.viewId) throw new Error("viewId is required for view_pixels or view_normalized color coordinates.");
          if (coordinateSpace === "source" && params.viewId) throw new Error("viewId is only used with view_pixels or view_normalized color coordinates.");
          const region = mappedView
            ? loaded.views.toSource(params.viewId!, params.region, coordinateSpace)
            : params.region as [number, number, number, number];
          const point = params.point && mappedView
            ? loaded.views.toSourcePoint(params.viewId!, params.point, coordinateSpace)
            : params.point as [number, number] | undefined;
          const { bytes } = await readRecordImage(loaded.sourcePath, loaded.record, "visible");
          const { rawPreview, maskPreview, ...analysis } = await analyzeGroundingColor(bytes, { ...params, region, point });
          signal?.throwIfAborted();
          const encodedRawPreview = await encodeGroundingPreview(rawPreview, "image/png");
          const encodedMaskPreview = await encodeGroundingPreview(maskPreview, "image/png", {
            width: encodedRawPreview.width,
            height: encodedRawPreview.height,
          });
          const details = { key, ...groundingTargetReminder(loaded), ...analysis,
            previewWidth: encodedRawPreview.width, previewHeight: encodedRawPreview.height,
            coordinateSpace: "source", inputCoordinateSpace: coordinateSpace,
            ...(params.viewId ? { inputViewId: params.viewId } : {}), saved: false,
            note: "Local visible-image color measurement only. selectionAssessment.establishesObjectIdentity is always false. Bounds enclose selected matching pixels, not necessarily the whole target or part; verify identity and boundaries against visible structure, and discard misleading measurements." };
          const view = loaded.views.register({ modality: "visible", region: analysis.region, sourceWidth: analysis.sourceWidth,
            sourceHeight: analysis.sourceHeight, width: encodedRawPreview.width, height: encodedRawPreview.height, decorations: "none", label: "Color ROI (raw and mask share coordinates)" });
          const evidenceDetails = { ...details, viewId: view.id, coordinateMapping: view, evidenceViewIds: [view.id] };
          return { content: [
            { type: "text" as const, text: JSON.stringify(evidenceDetails) },
            { type: "image" as const, data: encodedRawPreview.data, mimeType: encodedRawPreview.mimeType },
            { type: "image" as const, data: encodedMaskPreview.data, mimeType: encodedMaskPreview.mimeType },
          ], details: evidenceDetails };
        },
      });

      pi.registerTool({
        name: "grounding_evidence",
        label: "Grounding working evidence",
        description: "Keep a short evidence state and choose which views remain in model context. Separate direct visible facts from unverified identity/part hypotheses. Pin useful views, archive superseded images, or restore them; full transcript and user corrections are never deleted.",
        promptGuidelines: [
          "facts must contain only direct visible observations. Put interpretations such as 'this dark line is the beak' in hypotheses until structural evidence verifies the object and part. A color match alone never promotes a hypothesis to a fact.",
          "Keep the original query and the user's target requirements intact in state.target; put changing candidate identities in hypotheses, not in place of the requested target. Record hypotheses, open questions and ruled-out candidates concisely, not private deliberation. Before another substantially overlapping view, state what remains unresolved and what visible result would confirm or reject the hypothesis.",
          "Archive only superseded or redundant views after keeping the evidence and counterexamples that matter. Pin overrides archive; the record's original overview stays available. For a comparison image, every panel viewId must be archived before that whole image is omitted.",
        ],
        parameters: Type.Object({
          key: Type.Optional(Type.String()),
          pin: Type.Optional(Type.Array(Type.String(), { maxItems: 100 })),
          archive: Type.Optional(Type.Array(Type.String(), { maxItems: 100 })),
          restore: Type.Optional(Type.Array(Type.String(), { maxItems: 100 })),
          unpin: Type.Optional(Type.Array(Type.String(), { maxItems: 100 })),
          state: Type.Optional(Type.Object({ target: Type.Optional(Type.String({ maxLength: 600 })),
            facts: Type.Optional(Type.Array(Type.String({ maxLength: 400 }), { maxItems: 8 })),
            hypotheses: Type.Optional(Type.Array(Type.String({ maxLength: 400 }), { maxItems: 8 })),
            openQuestions: Type.Optional(Type.Array(Type.String({ maxLength: 400 }), { maxItems: 8 })),
            ruledOut: Type.Optional(Type.Array(Type.String({ maxLength: 400 }), { maxItems: 8 })) })),
          offset: Type.Optional(Type.Integer({ minimum: 0 })), limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 20 })),
        }),
        executionMode: "sequential",
        async execute(_id, params) {
          const [key, loaded] = resolveLoadedRecord(params);
          const state = params.state === undefined ? loaded.workingState : validateGroundingWorkingState(params.state);
          for (const id of [...(params.pin ?? []), ...(params.unpin ?? []), ...(params.archive ?? []), ...(params.restore ?? [])]) {
            if (!loaded.views.get(id)) throw new Error(`Unknown viewId ${id} for this record. List the current record's views first.`);
          }
          for (const id of params.unpin ?? []) loaded.pinnedViewIds.delete(id);
          for (const id of params.archive ?? []) loaded.archivedViewIds.add(id);
          for (const id of params.restore ?? []) loaded.archivedViewIds.delete(id);
          for (const id of params.pin ?? []) loaded.pinnedViewIds.add(id);
          loaded.workingState = state;
          persistJob();
          const all = loaded.views.list();
          const offset = params.offset ?? 0;
          const views = all.slice(offset, offset + (params.limit ?? 20)).map((view) => ({ ...view,
            pinned: loaded.pinnedViewIds.has(view.id), archived: loaded.archivedViewIds.has(view.id) && !loaded.pinnedViewIds.has(view.id) }));
          const details = { key, ...groundingTargetReminder(loaded), state: state ?? {}, views, totalViews: all.length,
            nextOffset: offset + views.length < all.length ? offset + views.length : null,
            note: "Facts are direct observations; hypotheses are unverified object/part interpretations. Color membership never establishes identity. Only explicitly archived image payloads leave subsequent model input; use grounding_view with viewId to inspect a source region again." };
          return { content: [{ type: "text" as const, text: JSON.stringify(details) }], details };
        },
      });

      pi.registerTool({
        name: "grounding_compare",
        label: "Grounding candidate comparison",
        description: "Compare model-chosen candidate regions side by side with a clean full-image overview in one image. Useful for identity, ordering and size ambiguity. No object detection: you choose every ROI. Returns one viewId per panel with exact composite-image coordinates; pass that panel viewId and view_pixels/view_normalized when saving a box measured on the comparison.",
        promptGuidelines: [
          "Use overview for relations between candidates and detail panels for local parts. Every panel label is outside its pixels; different panel scales must not be used to compare real object size. You may call again with more candidates; the 4-panel limit bounds one image only.",
          "For ordinal queries, establish which candidates satisfy the object description, then order them along the requested axis and direction in a common source-image frame. Panel labels and discovery order are not spatial rank. Recompute the order when a candidate is added, removed or reidentified; do not invent a candidate to satisfy the requested number.",
        ],
        parameters: Type.Object({
          key: Type.Optional(Type.String()), queryPath: Type.Optional(Type.String()),
          modality: Type.Optional(Type.Union([Type.Literal("visible"), Type.Literal("infrared"), Type.Literal("depth")])),
          regions: Type.Array(Type.Object({ label: Type.String({ minLength: 1, maxLength: 80 }),
            region: Type.Array(Type.Number(), { minItems: 4, maxItems: 4 }) }), { minItems: 1, maxItems: 4 }),
          reason: Type.String({ minLength: 8 }),
        }),
        executionMode: "sequential",
        async execute(_id, params) {
          const [key, loaded] = resolveLoadedRecord(params);
          if (!params.regions.length || params.regions.length > 4) throw new Error("Choose 1–4 regions per comparison image; more comparisons remain available.");
          const modality = params.modality ?? "visible";
          if (!["visible", "infrared", "depth"].includes(modality)) throw new Error("Unknown modality.");
          if (params.reason.trim().length < 8) throw new Error("Explain which ambiguity this comparison will resolve.");
          const regions = params.regions.map((item) => ({ ...item, region: validateBoundingBox(item.region) }));
          const overview = await loadGroundingImage(loaded.sourcePath, loaded.record, modality, readRecordImage, null, false);
          const crops = await Promise.all(regions.map((item) => loadGroundingCrop(loaded.sourcePath, loaded.record, modality, item.region, null, readRecordImage, 1, "none")));
          const bytes = (image: GroundingImage) => {
            const block = image.content.find((item) => item.type === "image");
            if (!block || block.type !== "image") throw new Error("Missing comparison pixels.");
            return Buffer.from(block.data, "base64");
          };
          const comparison = await buildGroundingComparison(bytes(overview), crops.map((crop, index) => ({ label: regions[index].label, image: bytes(crop) })));
          const encodedComparison = await encodeGroundingPreview(comparison.image, "image/png");
          const scaleX = encodedComparison.width / comparison.width;
          const scaleY = encodedComparison.height / comparison.height;
          const scaleRect = (rect: readonly number[]): [number, number, number, number] => [
            rect[0] * scaleX, rect[1] * scaleY, rect[2] * scaleX, rect[3] * scaleY,
          ];
          const sourceWidth = overview.details.originalWidth!, sourceHeight = overview.details.originalHeight!;
          const registerComparisonView = (input: Omit<GroundingViewDescriptor, "id">) => {
            const sourceReuse = loaded.views.sourceReuse(input);
            const view = loaded.views.register(input);
            return { ...view, ...(sourceReuse ? { sourceReuse } : {}) };
          };
          const overviewView = registerComparisonView({ modality, region: [0, 0, 1, 1], sourceWidth, sourceHeight,
            width: encodedComparison.width, height: encodedComparison.height, displayRect: scaleRect(comparison.overviewRect), decorations: "none", label: "Overview" });
          const panels = comparison.panels.map((panel, index) => registerComparisonView({ modality,
            region: crops[index].details.cropNormalized!, sourceWidth, sourceHeight, width: encodedComparison.width, height: encodedComparison.height,
            displayRect: scaleRect(panel.rect), decorations: "none", label: panel.label }));
          grantRecordModality(loaded, modality);
          const repeatedPanels = panels.filter((panel) => panel.sourceReuse);
          const details = { key, ...groundingTargetReminder(loaded), modality, reason: params.reason, overview: overviewView, panels,
            evidenceViewIds: [overviewView.id, ...panels.map((view) => view.id)],
            ...(repeatedPanels.length ? { decisionCheckpoint: `${repeatedPanels.length} comparison panel(s) substantially reuse prior source pixels. Record what changed before repeating the same identity comparison.` } : {}),
            note: "Use one panel's viewId. view_pixels and view_normalized refer to this WHOLE comparison canvas; the box must lie inside that panel's displayRect. Source coordinates use coordinateSpace source. Panels may have different display scales." };
          return { content: [{ type: "text" as const, text: JSON.stringify(details) },
            { type: "image" as const, data: encodedComparison.data, mimeType: encodedComparison.mimeType }], details };
        },
      });

      pi.registerTool({
        name: "grounding_view",
        label: "Grounding view",
        description: "Inspect a full modality or a model-selected crop, or recall a source region using viewId. bbox is optional while identifying the target. decorations none returns clean pixels; grid/hypothesis/all are optional. Every view gets a stable id and exact coordinate mapping; no manual source conversion is needed.",
        promptSnippet: "View an additional modality or a model-selected zoom crop with the current bbox marked",
        promptGuidelines: [
          "Visible is already attached by grounding_next_batch. Request infrared or depth only when the query or visible ambiguity requires it.",
          "Equal image dimensions do not prove cross-modal alignment. Establish object correspondence and spatial registration before transferring an infrared/depth box to the visible image used for review. A viewId maps display coordinates within its modality, not between sensors. Conflicting positions or structure remain unresolved evidence, not confirmation of identity.",
          "bbox is an optional working hypothesis, not evidence. Omit it while choosing the target; decorations none removes the grid, box and labels without changing pixels or coordinate metadata.",
          "For a crop, choose region and zoom yourself; there is no crop count limit. A region may inspect a small part of the hypothesis. The runtime marks the overlap and reports missing context; compare the full image when selecting among candidates.",
          "The requested region is preserved. If your zoom would exceed the 1600px display limit, only the display magnification is reduced; requestedZoom, magnification, zoomAdjusted and exact source pixel bounds explain the result.",
          "Use one or more focus crops when the target is small, ambiguous, or needs closer inspection. You decide whether to crop, how many crops to request, and the zoom for each crop. The runtime does not force a crop before saving.",
          "For a box read from this display, use its viewId and coordinateSpace view_pixels or view_normalized. Source grid numbers already use source coordinates. last_crop is legacy and refers only to the latest crop.",
          "When repeatedSourceViewId is returned, these source pixels were already examined; explain what new question another rendering answers. Use grounding_compare for identity ambiguity and grounding_evidence to retain facts and archive redundant views, without a fixed crop count.",
          "A crop that reaches the source image border reports touchesSourceEdge, which means no margin exists on that side; do not assume hidden context beyond the reported crop span.",
        ],
        parameters: Type.Object({
          queryPath: Type.Optional(Type.String({ description: "Source queries.json; defaults to the loaded record's dataset" })),
          key: Type.Optional(Type.String({ description: "Record key; defaults to the currently loaded record" })),
          modality: Type.Optional(Type.String({ description: "visible (default), infrared, or depth" })),
          viewId: Type.Optional(Type.String({ description: "Recall an earlier source region from this record; do not also specify region. A fresh viewId will describe the newly rendered display." })),
          decorations: Type.Optional(Type.Union([Type.Literal("none"), Type.Literal("grid"), Type.Literal("hypothesis"), Type.Literal("all")])),
          bbox: Type.Optional(Type.Array(Type.Number(), {
            minItems: 4,
            maxItems: 4,
            description: "Optional source-normalized working hypothesis; omit when target identity is not established",
          })),
          zoom: Type.Optional(Type.Number({ minimum: 1, description: "Model-selected crop magnification; required with region. Oversized display is capped to 1600px while retaining the whole region and reporting the effective magnification." })),
          region: Type.Optional(Type.Array(Type.Number(), {
            minItems: 4,
            maxItems: 4,
            description: "Optional source-normalized crop region [x1,y1,x2,y2]; may inspect only a small part of bbox. The current hypothesis remains in full source coordinates.",
          })),
          reason: Type.String({ description: "Concrete reason this additional view is needed" }),
        }),
        executionMode: "sequential",
        async execute(_toolCallId, params) {
          if (!active) throw new Error("Grounding runtime safety is not active for this session.");
          const [key, loaded] = resolveLoadedRecord(params);
          const targetReminder = groundingTargetReminder(loaded);
          const recalled = params.viewId ? loaded.views.get(params.viewId) : undefined;
          if (params.viewId && !recalled) throw new Error("Unknown or stale viewId. List current record views with grounding_evidence, or request a new region.");
          if (recalled && params.region) throw new Error("Use either viewId or region, not both.");
          if (recalled && params.modality && params.modality !== recalled.modality) throw new Error("The requested modality does not match this viewId.");
          const modality = recalled?.modality ?? params.modality ?? "visible";
          if (!(modality === "visible" || modality === "infrared" || modality === "depth")) {
            throw new Error("modality must be visible, infrared, or depth.");
          }
          const currentBbox = params.bbox ? validateBoundingBox(params.bbox) : loaded.currentBbox ?? null;
          const decorations = params.decorations ?? recalled?.decorations ?? "all";
          if (!["none", "grid", "hypothesis", "all"].includes(decorations)) throw new Error("Unknown view decorations.");
          if (params.reason.trim().length < 8) throw new Error("Give a concrete reason for the additional view.");
          const { sourcePath } = await loadQueryState(params.queryPath ?? loaded.sourcePath);
          if (loaded.sourcePath !== sourcePath) throw new Error("The requested queryPath does not match the loaded record.");

          const requestedRegion = params.region ?? (recalled && !recalled.region.every((value, index) => value === [0, 0, 1, 1][index]) ? recalled.region : undefined);
          if (requestedRegion) {
            if (params.zoom === undefined && !recalled) throw new Error("A crop request must include the model-selected zoom.");
            const zoom = params.zoom ?? Math.max(1, ((recalled!.displayRect?.[2] ?? recalled!.width) - (recalled!.displayRect?.[0] ?? 0)) / (recalled!.sourceWidth * (recalled!.region[2] - recalled!.region[0])));
            const overview = loaded.grantedModalities.has(modality) ? undefined : await loadGroundingImage(sourcePath, loaded.record, modality, readRecordImage, null, false);
            const region = validateBoundingBox(requestedRegion);
            const image = await loadGroundingCrop(sourcePath, loaded.record, modality, region, currentBbox, readRecordImage, zoom, decorations);
            const view = registerGroundingView(loaded, image, decorations);
            if (overview) {
              const overviewView = registerGroundingView(loaded, overview, "none");
              loaded.pinnedViewIds.add(overviewView.id);
            }
            grantRecordModality(loaded, modality);
            loaded.currentBbox = currentBbox ?? undefined;
            loaded.lastCropRegion = image.details.cropNormalized!;
            persistJob();
            const details: GroundingViewDetails = {
              key,
              ...targetReminder,
              action: "crop",
              currentBbox,
              modality,
              reason: params.reason,
              image: image.details,
              viewId: view.id,
              ...(view.sourceReuse ? { sourceReuse: view.sourceReuse } : {}),
              // A newly attached full-modality overview must not be archived
              // together with its first detail crop.
              ...(overview ? {} : { evidenceViewIds: [view.id] }),
            };
            return {
              content: [
                { type: "text" as const, text: JSON.stringify({ key, ...targetReminder, focused: modality, currentBbox, viewId: view.id, requestedZoom: zoom,
                  reason: params.reason, ...(view.sourceReuse ? { sourceReuse: view.sourceReuse,
                    decisionCheckpoint: "This rendering mostly reuses prior source pixels. State the unresolved question and update grounding_evidence before requesting another overlapping view." } : {}) }) },
                ...(overview?.content ?? []),
                ...image.content,
              ],
              details,
            };
          }

          const image = await loadGroundingImage(sourcePath, loaded.record, modality, readRecordImage, currentBbox,
            decorations === "all" || decorations === "hypothesis");
          const view = registerGroundingView(loaded, image, decorations);
          const newModality = !loaded.grantedModalities.has(modality);
          grantRecordModality(loaded, modality);
          if (newModality) loaded.pinnedViewIds.add(view.id);
          loaded.currentBbox = currentBbox ?? undefined;
          persistJob();
          const details: GroundingViewDetails = {
            key,
            ...targetReminder,
            action: "view",
            currentBbox,
            modality,
            reason: params.reason,
            image: image.details,
            viewId: view.id,
            ...(view.sourceReuse ? { sourceReuse: view.sourceReuse } : {}),
            ...(newModality ? {} : { evidenceViewIds: [view.id] }),
          };
          return {
            content: [
              { type: "text" as const, text: JSON.stringify({ key, ...targetReminder, viewed: modality, currentBbox, viewId: view.id, reason: params.reason,
                ...(view.sourceReuse ? { sourceReuse: view.sourceReuse,
                  decisionCheckpoint: "This rendering mostly reuses prior source pixels. State the unresolved question and update grounding_evidence before requesting another overlapping view." } : {}) }) },
              ...image.content,
            ],
            details,
          };
        },
      });

      pi.registerTool({
        name: "grounding_save_result",
        label: "Grounding save result",
        description: "Show the current box for mandatory human review. Wait for approval, then atomically save it and regenerate submission JSON. Rejection keeps the same record for revision.",
        promptSnippet: "Ask the user to review the current box and save only after approval",
        promptGuidelines: [
          "Prefer grounding_save_and_next between requested records; use grounding_save_result for the last record or a revision. If a known job still has requested records, this tool returns nextAction grounding_next_batch instead of ending the batch.",
          "This tool waits for the user's review without a timeout. Never confirm on the user's behalf or treat a rejected candidate as complete.",
          "Provide reason with the direct visible evidence that connects this box to the requested object and part. A color match alone is not sufficient. Material movement from the prior working box is shown to the reviewer.",
          "Before submitting, reconcile the proposed target with the original query and user requirements, including any required identity, attribute, relation or order. Explain the evidence for those requirements; for ordinal targets include the supported count and spatial order. If a required condition or cross-modal correspondence remains unestablished, use status unresolved with low confidence and state what is missing. A measured box is not proof that the request is satisfied.",
          "If bbox coordinates were measured against the last grounding_view crop, set coordinateSpace to last_crop so the runtime maps them back to the full source image.",
          "Do not write grounding progress or submission files with write, edit, bash, or powershell.",
          "The result returns an overlay image of the saved box and, for a small box, an automatic magnified verification crop; confirm the box sits on the target in those images.",
        ],
        parameters: Type.Object({
          queryPath: Type.String({ description: "Path to the source queries.json" }),
          outputDir: Type.String({ description: "Dedicated output directory outside the source dataset. Use the same new or empty directory as grounding_next_batch." }),
          key: Type.String(),
          bbox: Type.Array(Type.Number(), { minItems: 4, maxItems: 4 }),
          status: Type.Union([Type.Literal("ok"), Type.Literal("low_confidence"), Type.Literal("unresolved")]),
          confidence: Type.Number({ minimum: 0, maximum: 1 }),
          reason: Type.String({ minLength: 8, maxLength: 800, description: "Direct visible evidence linking the box to the requested object and part; do not cite color alone" }),
          coordinateSpace: Type.Optional(Type.String({ description: "source (default), last_crop, view_pixels or view_normalized. View coordinates refer to the display canvas described by viewId." })),
          viewId: Type.Optional(Type.String({ description: "Required with view_pixels/view_normalized; stable id returned by the current record's view or comparison panel" })),
        }),
        executionMode: "sequential",
        async execute(_toolCallId, params, signal, _onUpdate, ctx) {
          const { summary, overlay, verificationContent, revision } = await saveGroundingResult(params, ctx, signal);
          const requestedLimitReached = requestedRecordLimit !== undefined && sessionSavedCount >= requestedRecordLimit;
          const shouldContinue = !revision && !requestedLimitReached && requestedRecordLimit !== undefined && summary.processed < summary.total;
          const result = { ...summary, job: jobSnapshot(), nextAction: shouldContinue ? "grounding_next_batch" : "complete" };
          if (!revision && (summary.processed === summary.total || requestedLimitReached)) reportCompletion(summary.processed, summary.total, params.outputDir);
          if (revision) pi.sendMessage({ customType: "grounding-revised", content: `【返修完成】${params.key} 的新框已经人工确认并保存。`, display: true }, { triggerTurn: false });
          return {
            content: [
              { type: "text" as const, text: JSON.stringify(result) },
              { type: "image" as const, data: overlay.data, mimeType: overlay.mimeType },
              ...verificationContent,
            ],
            details: result,
            terminate: !shouldContinue,
          };
        },
      });

      pi.registerTool({
        name: "grounding_save_and_next",
        label: "Grounding save and next",
        description: "Show the current box and wait for the user's approval. Only after approval, save this result and attach one next unfinished record. Rejection keeps the current record unsaved for revision.",
        promptSnippet: "Wait for human approval, save this bbox, then load one next record",
        promptGuidelines: [
          "Use grounding_save_and_next between records. It blocks for manual review; the user must approve before any later record is loaded. Rejection means revise the same record.",
          "Provide reason with direct visible evidence for target identity, owning object and requested part. If the candidate moved materially, explain the new evidence that caused the change.",
          "Before submitting, reconcile the proposed target with the original query and user requirements, including any required identity, attribute, relation or order. Explain the evidence for those requirements; for ordinal targets include the supported count and spatial order. If a required condition or cross-modal correspondence remains unestablished, use status unresolved with low confidence and state what is missing. A measured box is not proof that the request is satisfied.",
          "If bbox coordinates were measured against the last grounding_view crop, set coordinateSpace to last_crop so the runtime maps them back to the full source image.",
          "Use grounding_save_result for the last requested record.",
          "The result returns an overlay image of the saved box and, for a small box, an automatic magnified verification crop; confirm the box sits on the target in those images.",
        ],
        parameters: Type.Object({
          queryPath: Type.String({ description: "Path to the source queries.json" }),
          outputDir: Type.String({ description: "Dedicated output directory outside the source dataset. Use a new or empty directory; an existing progress.jsonl written by another build cannot be resumed." }),
          key: Type.String(),
          bbox: Type.Array(Type.Number(), { minItems: 4, maxItems: 4 }),
          status: Type.Union([Type.Literal("ok"), Type.Literal("low_confidence"), Type.Literal("unresolved")]),
          confidence: Type.Number({ minimum: 0, maximum: 1 }),
          reason: Type.String({ minLength: 8, maxLength: 800 }),
          coordinateSpace: Type.Optional(Type.String({ description: "source (default), last_crop, view_pixels or view_normalized; view coordinates require viewId" })),
          viewId: Type.Optional(Type.String()),
        }),
        executionMode: "sequential",
        async execute(_toolCallId, params, signal, _onUpdate, ctx) {
          const { summary: saved, overlay, verificationContent, revision } = await saveGroundingResult(params, ctx, signal, true);
          signal?.throwIfAborted();
          const overlayBlock = { type: "image" as const, data: overlay.data, mimeType: overlay.mimeType };
          if (revision || (requestedRecordLimit !== undefined && sessionSavedCount >= requestedRecordLimit)) {
            if (!revision) reportCompletion(saved.processed, saved.total, params.outputDir);
            return {
              content: [
                {
                  type: "text" as const,
                  text: JSON.stringify({ saved, next: null, revision, requestedLimitReached: !revision }),
                },
                overlayBlock,
                ...verificationContent,
              ],
              details: { saved, next: null, revision, requestedLimitReached: !revision, job: jobSnapshot() },
              terminate: true,
            };
          }
          const next = await loadNextRecord(params.queryPath, params.outputDir);
          const key = next.payload.records[0]?.key;
          if (!key) reportCompletion(next.payload.completed, next.payload.total, next.payload.outputDir);
          return {
            content: [
              { type: "text" as const, text: JSON.stringify({ saved, next: next.payload }) },
              overlayBlock,
              ...verificationContent,
              ...next.content.slice(1),
            ],
            details: { saved, next: next.details },
            terminate: !key,
          };
        },
      });

      pi.on("agent_before_settle", (event) => {
        if (!active || !batchMode || savingRecord || reviewInterrupted || event.outcome !== "completed" || !event.context.canContinue) return undefined;
        const current = loadedBatchRecords.entries().next().value as [string, LoadedBatchRecord] | undefined;
        if (!current) {
          if (!batchToolStarted) {
            if (startupNudges >= 1) return undefined;
            startupNudges += 1;
            return {
              entries: [{
                type: "custom_message" as const,
                customType: "grounding-required-start",
                display: false,
              content: "The requested dataset run has not started. Do not end the turn. Call grounding_next_batch exactly once now; do not read queries.json or enumerate files.",
              }],
              continue: true,
            };
          }
          if (
            batchExhausted
            || requestedRecordLimit === undefined
            || sessionSavedCount >= requestedRecordLimit
            || continuationNudges >= 1
          ) return undefined;
          continuationNudges += 1;
          return {
            entries: [{
              type: "custom_message" as const,
              customType: "grounding-required-next",
              display: false,
              content: `Only ${sessionSavedCount} of the requested ${requestedRecordLimit} records have been saved in this run. `
                + "Do not end yet. Call grounding_next_batch once to load the next unfinished record.",
            }],
            continue: true,
          };
        }
        const [key, loaded] = current;
        if (loaded.settleNudges >= 1) return undefined;
        loaded.settleNudges += 1;
        return {
          entries: [{
            type: "custom_message" as const,
            customType: "grounding-required-save",
            display: false,
            content: `Grounding record ${key} is still unsaved. Do not end the turn. `
              + `Original query: ${JSON.stringify(loaded.record.query ?? "")}. `
              + "Use the image already in context and reconcile the best-supported single normalized bbox with all original query requirements before calling grounding_save_result or grounding_save_and_next. If a required condition remains unestablished, submit it as unresolved with low confidence and explain what is missing.",
          }],
          continue: true,
        };
      });

      pi.on("session_start", (_event, ctx) => {
        const entries = ctx.sessionManager.getEntries();
        const previousPromptTexts = entries.map(groundingEntryText);
        const hasPersistedJob = entries.some((entry) => entry.type === "custom" && entry.customType === "grounding:job"
          && entry.data && typeof entry.data === "object" && [1, 2].includes((entry.data as Record<string, unknown>).version as number)
          && typeof (entry.data as Record<string, unknown>).queryPath === "string" && typeof (entry.data as Record<string, unknown>).outputDir === "string");
        for (const entry of entries) {
          if (entry.type === "custom" && entry.customType === "grounding:job" && entry.data && typeof entry.data === "object") {
            const data = entry.data as Record<string, unknown>;
            if ((data.version === 1 || data.version === 2) && typeof data.queryPath === "string" && typeof data.outputDir === "string") {
              lastJob = { queryPath: data.queryPath, outputDir: data.outputDir };
              requestedRecordLimit = typeof data.targetCount === "number" && Number.isSafeInteger(data.targetCount) && data.targetCount > 0 ? data.targetCount : undefined;
              jobStartCompleted = typeof data.startCompleted === "number" && Number.isSafeInteger(data.startCompleted) && data.startCompleted >= 0 ? data.startCompleted : undefined;
              lastApprovedKey = typeof data.lastApprovedKey === "string" ? data.lastApprovedKey : undefined;
              persistedPending = undefined;
              if (data.version === 2 && data.pending && typeof data.pending === "object") {
                const pending = data.pending as Record<string, unknown>;
                if (typeof pending.key === "string" && pending.key.length > 0) {
                  let currentBbox: [number, number, number, number] | undefined;
                  let workingState: GroundingWorkingState | undefined;
                  try {
                    if (pending.currentBbox !== undefined) currentBbox = validateBoundingBox(pending.currentBbox as readonly number[]);
                    if (pending.workingState !== undefined) workingState = validateGroundingWorkingState(pending.workingState);
                    persistedPending = {
                      key: pending.key,
                      ...(currentBbox ? { currentBbox } : {}),
                      ...(workingState ? { workingState } : {}),
                      ...(pending.revision === true ? { revision: true } : {}),
                    };
                  } catch {
                    // Ignore malformed optional pending data while retaining
                    // the compatible dataset-level job state.
                  }
                }
              }
            }
          }
          if (hasPersistedJob || entry.type !== "message" || entry.message.role !== "assistant") continue;
          for (const block of entry.message.content) {
            if (block.type !== "toolCall" || !["grounding_next_batch", "grounding_reopen_record", "grounding_save_result", "grounding_save_and_next"].includes(block.name)) continue;
            const input = block.arguments;
            if (typeof input.queryPath === "string" && typeof input.outputDir === "string") {
              lastJob = { queryPath: input.queryPath, outputDir: input.outputDir };
            }
          }
        }
        const persistedGrounding = Boolean(lastJob) || previousPromptTexts.some((text) => isGroundingPrompt(text));
        active = persistedGrounding;
        batchMode = Boolean(lastJob) || persistedGrounding && previousPromptTexts.some((text) => (
          /batch|queries?\.json|visible|infrared|depth|图像|图片|定位/i.test(text)
        ));
        setGroundingToolsActive(active);
      });

      pi.on("before_agent_start", async (event) => {
        reviewInterrupted = false;
        if (isGroundingPrompt(event.prompt)) active = true;
        const requestedCount = requestedBatchCount(event.prompt);
        const currentRecord = loadedBatchRecords.values().next().value as LoadedBatchRecord | undefined;
        const hasPendingRecord = currentRecord !== undefined || persistedPending !== undefined;
        const adjustsRemainingCount = requestedCount !== undefined && hasPendingRecord
          && /(?:只|仅|改为|改成|接下来|剩下|剩余)|\b(?:only|instead|remaining|next)\b/i.test(event.prompt);
        if (
          /批处理/i.test(event.prompt)
          || (/queries?\.json/i.test(event.prompt) && /(?:visible|infrared|depth|处理|定位|全部|所有|全量)/i.test(event.prompt))
        ) batchMode = true;
        if (requestedCount !== undefined && (loadedBatchRecords.size === 0 || adjustsRemainingCount)) {
          batchMode = true;
          requestedRecordLimit = requestedCount;
          requestedCountPending = !hasPendingRecord;
          sessionSavedCount = 0;
          jobStartCompleted = currentRecord
            ? (await readGroundingProgress(join(currentRecord.outputDirectory, "progress.jsonl"))).size
            : persistedPending && lastJob
              ? (await readGroundingProgress(join(lastJob.outputDir, "progress.jsonl"))).size
              : undefined;
          batchToolStarted = hasPendingRecord;
          batchExhausted = false;
          startupNudges = 0;
          continuationNudges = 0;
          if (hasPendingRecord) persistJob();
        }
        if (!active) return;
        setGroundingToolsActive(true);
        event.systemPromptOptions.sections[GROUNDING_SECTION] = [
          "Runtime grounding safety is active.",
          lastJob ? `Current dataset: ${lastJob.queryPath}. Current output directory: ${lastJob.outputDir}.` : "",
          `Job state: ${JSON.stringify(jobSnapshot())}.`,
          "Keep the original query and the user's task requirements as the target throughout observation, candidate changes and review. A newly noticed object, working hypothesis or convenient tool result must not silently replace the requested object or drop a required attribute, relation or order. Before submitting a box, reconcile the proposal with every requirement actually present in the request and explain the supporting evidence. If a requirement is unestablished or contradicted, state what is unresolved and use low confidence with status unresolved instead of reinterpreting the query to fit the candidate. Pixel measurements alone do not establish that the request is satisfied.",
          "For continue/status questions, call grounding_status instead of listing directories. If remaining is zero, explain the dataset is complete; do not claim to begin the first record again. For a user-requested correction, find the key in grounding_status and call grounding_reopen_record, then submit the revised box for human review. Do not treat your run's saved predictions as reference annotations.",
          "grounding_color_region is an optional local pixel-measurement aid, not a required step for every record or every colored target. First establish object/part identity from the image and context. Use it only when reliable local color contrast helps resolve a remaining boundary question; skip it if the box is clear or color is unhelpful. For unresolved identity, inspect existing evidence or use grounding_view/grounding_compare as needed. Uncertainty is preferable to a forced color match. No extra model or shell/file enumeration is needed.",
          "Color analysis samples only the original visible image; infrared/depth palettes and viewId coordinate mapping do not supply color or alignment evidence. Lighting, shadows, reflections, similar colors, low resolution and occlusion can make a mask misleading. Inspect selectionAssessment, pointSample and clean/mask previews when using it. Measured bounds cover matching pixels only, not necessarily the complete target, and cannot recover hidden boundaries.",
          "Human review is mandatory for every record, including unresolved and low-confidence results. Save tools display the current box and wait for the user to approve it. Never approve on the user's behalf. Only an approved record may be saved or followed by another record; rejection means revise that same record and request review again. Waiting for the user is a valid pause, not an error or a reason to retry.",
          "The runtime sanitizes query JSON and blocks reference annotations and prior annotated artifacts.",
          "For query datasets, grounding_next_batch returns one record plus its visible image; do not read the full queries.json or re-read an image already attached by a grounding tool. The first image is explicitly stamped CURRENT HYPOTHESIS: NONE.",
          "grounding_next_batch loads at most one record: omit limit or set limit to 1; pass the user's requested total as targetCount instead. If it returns records: [] with remaining: 0, or requestedLimitReached true, stop immediately and do not call it again.",
          "grounding_view can show a hypothesis or grid, or clean pixels with decorations none. Model-chosen focus regions may inspect a smaller part; overlap metadata reports a clipped or outside hypothesis. Oversized zoom is capped to payload dimensions while preserving the region and reporting actual magnification. sourceReuse and decisionCheckpoint identify exact, contained and near-duplicate source pixels; another scale alone is not new evidence.",
          "When estimating a box inside the last focus crop, pass coordinateSpace last_crop to the save tool instead of manually converting it to full-image coordinates.",
          "Prefer stable viewId plus coordinateSpace view_pixels or view_normalized to manually converting display coordinates. For comparison panels, coordinates refer to the whole composite canvas; the box must be within the chosen panel.",
          "Use grounding_compare to compare candidate identities together with full-image context. Panel magnification does not indicate source object size. grounding_view decorations none returns clean pixels without boxes, grids or labels; bbox is optional during identity selection.",
          "For ordinal queries such as third from the left, establish which candidates match the requested object, then sort their positions along the specified axis and direction in the same source-image frame. Recompute spatial order whenever a candidate is added, removed or reidentified; discovery order and panel labels are not rank. If the candidate count or order is unsupported, keep the target unresolved instead of inventing an object to satisfy the numeral. The review reason should explain the supported count and order.",
          "Across visible, infrared and depth, equal image dimensions do not establish spatial registration or object correspondence. View coordinate mappings do not align sensors. Verify correspondence before transferring a box into the visible review image; conflicting positions or structure are unresolved evidence, not identity confirmation. If identity, requested rank or cross-modal correspondence remains unresolved, use status unresolved with low confidence and explain the missing evidence rather than reporting ok.",
          "Separate target identity, part selection and boundary measurement. Color membership is not object identity: point no_match does not prove absence and largest can be background. Do not invent thermal properties to justify an infrared interpretation.",
          "Keep a short factual state with grounding_evidence (target, facts, hypotheses, openQuestions, ruledOut). Facts are direct visible observations. Put interpretations such as 'the nearby dark line is the beak' in hypotheses until structure verifies both the object and part. Pin important views; archive redundant images when they add no evidence.",
          "Every additional view should resolve a specific remaining question. Repeated zoom of the same source pixels cannot add texture. Preserve unresolved alternatives instead of restarting all guesses. A request to analyze past mistakes does not itself request new annotation.",
          "Use grounding_save_and_next between requested records and grounding_save_result for the final record or revision. Each save requires a reason naming visible structural evidence for the chosen identity, part and boundaries. If the proposed box materially moves from the prior candidate, explain the new visible evidence before submitting it. If save_result returns nextAction grounding_next_batch, continue that same job.",
          "The browser review displays the previous and proposed boxes on a clean full image when the candidate changed. The approved prediction edges are saved exactly; never add padding for a small target. Every saved result is returned as a red annotated overlay image.",
          "A small approved box can return an automatic magnified verification crop. That crop only improves inspection and never changes or expands the saved prediction.",
          "A wrapper restart preserves the pending key, current source-space bbox, evidence state and revision flag. It deliberately invalidates old viewId mappings: call grounding_next_batch to restore the pending record and obtain fresh views, or save directly only with source coordinates.",
          "Resolve relative image paths against the directory containing the query JSON, and keep the original path strings in any submission output.",
          "Process records serially: after locking each box, append its sidecar result before reading any later record; do not prefetch a later record.",
          "The runtime tracks the requested record count. Complete it serially unless the dataset is exhausted, the user is reviewing a box, or the run is cancelled.",
          "Use visible first; read infrared only for thermal/infrared/heat or bright-signature wording, and read depth only for front/back, overlap, distance, ordering, or genuine visible ambiguity.",
          "For a malformed or missing target, make one best-supported candidate box, mark it unresolved with low confidence, and submit it for human review before continuing.",
          "For tiny switches, logos, labels, and brand marks, box the complete physical plate or full visible mark rather than only its center glyphs.",
          "You may request any number of focus crops and choose each zoom. Include all plausible candidates when disambiguating identity, or select a smaller region when measuring a specific part or color. Focus crops carry a labeled source-normalized grid.",
          "Do not install or invoke any additional detector or vision model.",
          "If a tool is blocked, continue using the sanitized query and listed source images.",
        ].join(" ");
      });

      pi.on("tool_call", async (event: ToolCallEvent) => {
        if (!active) return undefined;

        if (event.toolName === "read") {
          const path = String(event.input.path ?? "");
          let effectivePath = path;
          if (isProtectedPath(path)) {
            return block("Reference annotations and prior debug artifacts are unavailable during blind prediction.");
          }
          if (isQueryPath(path)) {
            if (batchMode) {
              return block("Use grounding_next_batch so the complete query file never enters model context.");
            }
            try {
              const { state } = await loadQueryState(path);
              for (const allowedPath of state.allowedPaths) allowedReadPaths.add(allowedPath);
              event.input.path = state.safePath;
              effectivePath = state.safePath;
            } catch {
              return block("The query file could not be sanitized without exposing annotations.");
            }
          }
          const normalizedPath = normalizePath(effectivePath);
          const resolvedPath = normalizePath(resolve(options.cwd, effectivePath));
          if (!allowedReadPaths.has(normalizedPath) && !allowedReadPaths.has(resolvedPath)) {
            return block("Only image paths listed by the sanitized query are readable during grounding.");
          }
          return undefined;
        }

        if (event.toolName === "write" || event.toolName === "edit") {
          const path = String(event.input.path ?? "");
          if (/\b(?:progress|sidecar|submission|queries)\.(?:jsonl?|ndjson)\b/i.test(basename(path))) {
            return block("Use grounding_save_result for progress and official submission files.");
          }
          if (isProtectedPath(path) || isQueryPath(path)) {
            return block("Grounding source and prior annotation paths are read-only during prediction.");
          }
          return undefined;
        }

        if (event.toolName === "bash" || event.toolName === "powershell") {
          const commandInput = event.input as Record<string, unknown>;
          const command = String(commandInput.command ?? commandInput.script ?? "");
          if (isSensitiveCommand(command) || isFilesystemTraversalCommand(command)) {
            return block("Shell access to annotations, bbox fields, and prior debug artifacts is blocked.");
          }
          return undefined;
        }

        if (event.toolName === "grep" || event.toolName === "find") {
          return block("Directory search is disabled during grounding; use the sanitized query and listed image paths.");
        }

        if (event.toolName === "ls") {
          return block("Directory listing is disabled during grounding; use the sanitized query and listed image paths.");
        }

        return undefined;
      });

      pi.on("tool_result", async (event) => {
        if (!active) return undefined;
        if (GROUNDING_TOOL_NAMES.includes(event.toolName as typeof GROUNDING_TOOL_NAMES[number])) return undefined;
        const content = redactToolContent(event);
        return content ? { content } : undefined;
      });
    },
  };
}
