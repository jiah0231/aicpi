import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createJiti } from "jiti";

const { appendGroundingLesson, groundingLessonsForModel, groundingLessonsPath, readGroundingLessons,
  selectGroundingLessons, validateGroundingReviewLearning } = await createJiti(import.meta.url).import("./grounding-learning.ts");
const procedure = (extra = {}) => ({
  category: "boundary", applicability: "Low-contrast or partly occluded boundaries",
  error: "The inner high-contrast region can be mistaken for the whole silhouette.",
  method: "Inspect the complete visible outline before placing the final edges.",
  check: "Check each outer edge and visible protrusions without inventing hidden parts.",
  sampleIndependent: true, ...extra,
});
const stored = (extra = {}) => ({ version: 2, ...procedure(extra) });
const legacy = { version: 1, id: "legacy", createdAt: "2026-01-01", outcome: "confirmed", category: "boundary",
  scope: "global", query: "RAW_SAMPLE_QUERY_SENTINEL", advice: "An old sample-specific answer should not be migrated.", modalities: ["visible"] };
async function fileFixture(t) {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-procedures-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  return join(root, "lessons.jsonl");
}

test("only human-confirmed structured general procedures are accepted", () => {
  assert.deepEqual(validateGroundingReviewLearning(procedure()), procedure());
  assert.equal(validateGroundingReviewLearning(undefined), undefined);
  assert.deepEqual(validateGroundingReviewLearning(procedure({ method: `  ${procedure().method}\r\n` })), procedure());
  for (const input of [null, "text", [], procedure({ sampleIndependent: false }), procedure({ sampleIndependent: undefined }),
    procedure({ category: "unknown" }), procedure({ method: "short" }), procedure({ check: "x".repeat(401) }),
    procedure({ applicability: undefined }), { category: "boundary", scope: "global", advice: "A legacy advice payload." }]) {
    assert.throws(() => validateGroundingReviewLearning(input), /Grounding|Confirm|generic/i);
  }
});

test("sample-bearing fields are rejected rather than stripped into apparent compliance", () => {
  for (const [key, value] of Object.entries({ query: "source wording", originalQuery: "source wording", bbox: [.1, .2, .3, .4],
    image: "base64", imagePath: "visible/frame.png", sampleId: "sample-17", outcome: "confirmed", groundTruth: "answer", modalities: ["visible"] })) {
    assert.throws(() => validateGroundingReviewLearning({ ...procedure(), [key]: value }), /Source\/sample fields/);
  }
});

test("obvious identifiers, image payloads and coordinates are rejected without echoing their text", () => {
  for (const sample of ["https://example.org/secret.png", "visible/frame.png", "D:\\dataset\\frame.png", "/dataset/visible/file",
    "data:image/png;base64,AAAA", "novel_015772_center_alpaca", "01a0fb0d-7977-71c7-9eff-6a1a9dd7c6b2",
    "bbox: [.1,.2,.3,.4]", "[0.1, 0.2, 0.3, 0.4]", "x1=0.553", "originalQuery: a copied sentence", "答案：具体样本答案",
    '"query": "copied source wording"', "answer: the red switch", "[.1, .2, .3, .4]", "[1e-1, 2e-1, 3e-1, 4e-1]",
    "A".repeat(140)]) {
    const method = `Do not retain sample payload ${sample}`;
    assert.throws(() => validateGroundingReviewLearning(procedure({ method })), (error) => {
      assert.match(error.message, /apparent sample/);
      assert.equal(error.message.includes(sample), false);
      return true;
    });
  }
  assert.doesNotThrow(() => validateGroundingReviewLearning(procedure({
    method: "Preserve all query conditions and map display coordinates into the visible source frame.",
  })));
});

test("appends persist only the procedure fields and no automatic sample metadata", async (t) => {
  const filePath = await fileFixture(t);
  const saved = await appendGroundingLesson(procedure(), filePath);
  assert.deepEqual(saved, stored());
  assert.deepEqual(Object.keys(saved).sort(), ["applicability", "category", "check", "error", "method", "sampleIndependent", "version"]);
  assert.equal(await readFile(filePath, "utf8"), `${JSON.stringify(stored())}\n`);
  await assert.rejects(appendGroundingLesson({ ...procedure(), query: "SHOULD_NOT_APPEND" }, filePath), /Source\/sample fields/);
  assert.equal((await readGroundingLessons(filePath)).length, 1);
});

test("legacy and forged v2 rows stay on disk but never enter retrieval or model projection", async (t) => {
  const filePath = await fileFixture(t);
  const text = [legacy, { ...stored(), query: "PARAPHRASED_SAMPLE_SENTINEL" },
    { ...stored(), sampleIndependent: false }, stored(), { ...stored(), method: "Use novel_015772_center_alpaca as the answer." }]
    .map((row) => JSON.stringify(row)).join("\n") + "\n{invalid json\n";
  await writeFile(filePath, text);
  assert.deepEqual(await readGroundingLessons(filePath), [stored()]);
  assert.deepEqual(await selectGroundingLessons({ filePath }), [stored()]);
  assert.equal(await readFile(filePath, "utf8"), text, "readers never migrate or delete old rows");
  assert.deepEqual(groundingLessonsForModel([legacy, { ...stored(), query: "sentinel" }]).items, []);
  const view = groundingLessonsForModel([stored()]);
  assert.deepEqual(Object.keys(view.items[0]).sort(), ["applicability", "category", "check", "error", "method"]);
  assert.match(view.advisory, /Lexical validation cannot prove semantic independence/);
  assert.doesNotMatch(JSON.stringify(view), /SAMPLE.*SENTINEL|legacy|sampleIndependent/);
});

test("retrieval is query-free, bounded, deduplicated and category-diverse", async (t) => {
  const filePath = await fileFixture(t);
  const rows = [stored({ category: "identity" }), stored({ category: "order" }),
    ...Array.from({ length: 8 }, (_, index) => stored({ method: `Inspect each outer edge using general procedural variation ${index}.` }))];
  await writeFile(filePath, [...rows, rows.at(-1)].map((row) => JSON.stringify(row)).join("\n"));
  const selected = await selectGroundingLessons({ filePath });
  assert.equal(selected.length, 6);
  assert.deepEqual(selected.slice(0, 3).map((item) => item.category), ["boundary", "order", "identity"]);
  assert.equal(new Set(selected.map((item) => JSON.stringify(item))).size, selected.length);
  assert.equal((await selectGroundingLessons({ filePath, limit: 1 })).length, 1);
  assert.ok((await selectGroundingLessons({ filePath, limit: 100 })).length <= 12);
  assert.equal((await selectGroundingLessons({ filePath, limit: NaN })).length, 6);
});

for (const tail of ["valid", "malformed", "partial_utf8", "legacy"]) {
  test(`queued appends preserve a ${tail} unterminated tail and keep new procedures readable`, async (t) => {
    const filePath = await fileFixture(t);
    const existing = tail === "legacy" ? legacy : stored();
    const original = Buffer.concat([Buffer.from(JSON.stringify(existing)),
      ...(tail === "malformed" || tail === "partial_utf8" ? [Buffer.from('\n{"method":"unfinished')] : []),
      ...(tail === "partial_utf8" ? [Buffer.from([0xe5, 0xa2])] : [])]);
    await writeFile(filePath, original);
    const added = await Promise.all(Array.from({ length: 4 }, (_, index) => appendGroundingLesson(
      procedure({ method: `Inspect complete boundaries using general review approach ${index}.` }), filePath)));
    const saved = await readFile(filePath);
    assert.deepEqual(saved.subarray(0, original.length), original);
    const rows = saved.subarray(original.length).toString("utf8").split("\n");
    assert.equal(rows[0], "");
    assert.equal(rows.at(-1), "");
    assert.deepEqual(rows.slice(1, -1).map((row) => JSON.parse(row)), added);
    assert.deepEqual(await readGroundingLessons(filePath), [...(tail === "legacy" ? [] : [existing]), ...added]);
  });
}

test("newlines are not duplicated and old default store is not selected", async (t) => {
  const filePath = await fileFixture(t);
  const first = await appendGroundingLesson(procedure(), filePath);
  const second = await appendGroundingLesson(procedure(), filePath);
  assert.equal(await readFile(filePath, "utf8"), `${JSON.stringify(first)}\n${JSON.stringify(second)}\n`);
  if (!process.env.PI_WEB_GROUNDING_LESSONS_PATH) assert.match(groundingLessonsPath(), /grounding-methods-v2\.jsonl$/);
});

test("integration does not send a record query, image metadata or result outcome into the lesson API", async () => {
  const source = await readFile(new URL("./grounding-safety-extension.ts", import.meta.url), "utf8");
  assert.match(source, /selectGroundingLessons\(\{ filePath: options\.learningPath \}\)/);
  assert.match(source, /appendGroundingLesson\(review\.learning, options\.learningPath\)/);
  assert.doesNotMatch(source, /review\.learning\.advice/);
});
