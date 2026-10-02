import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { createJiti } from "jiti";

const {
  appendGroundingLesson,
  groundingLessonsForModel,
  readGroundingLessons,
  selectGroundingLessons,
  validateGroundingReviewLearning,
} = await createJiti(import.meta.url).import("./grounding-learning.ts");

function storedLesson(id, query, scope = "similar", modalities = ["visible"]) {
  return {
    version: 1,
    id,
    createdAt: "2026-10-02T00:00:00.000Z",
    outcome: "confirmed",
    category: "boundary",
    scope,
    query,
    advice: `Keep the complete visible boundary for lesson ${id}.`,
    modalities,
  };
}

test("review learning accepts bounded structured advice and rejects malformed input", () => {
  assert.deepEqual(validateGroundingReviewLearning({
    category: "boundary",
    scope: "global",
    advice: "  Include the complete low-contrast outer plate, not only its bright center.  ",
  }), {
    category: "boundary",
    scope: "global",
    advice: "Include the complete low-contrast outer plate, not only its bright center.",
  });
  assert.equal(validateGroundingReviewLearning(undefined), undefined);
  assert.throws(() => validateGroundingReviewLearning({ category: "code", scope: "global", advice: "Change the runtime." }), /invalid category/);
  assert.throws(() => validateGroundingReviewLearning({ category: "boundary", scope: "forever", advice: "Use complete visible edges." }), /invalid scope/);
  assert.throws(() => validateGroundingReviewLearning({ category: "boundary", scope: "similar", advice: "short" }), /8-1200/);
});

test("lessons persist across readers, ignore malformed rows, and select only relevant similar advice", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-learning-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const filePath = join(root, "lessons.jsonl");
  await appendGroundingLesson({
    outcome: "rejected",
    query: "The small square wall switch beside the door",
    modalities: ["visible"],
    learning: { category: "boundary", scope: "similar", advice: "Include the complete switch plate rather than only the inner rocker." },
  }, filePath);
  await appendGroundingLesson({
    outcome: "confirmed",
    query: "The thermal hand holding a banana",
    modalities: ["visible", "infrared"],
    learning: { category: "cross_modal", scope: "similar", advice: "Match the hand across visible and thermal frames before measuring its box." },
  }, filePath);
  await appendGroundingLesson({
    outcome: "confirmed",
    query: "Any grounding task",
    modalities: ["visible"],
    learning: { category: "uncertainty", scope: "global", advice: "Keep unsupported identity claims unresolved instead of inventing certainty." },
  }, filePath);
  await writeFile(filePath, `${await readFile(filePath, "utf8")}{broken json\n`, "utf8");

  assert.equal((await readGroundingLessons(filePath)).length, 3);
  const selected = await selectGroundingLessons("Locate the wall switch near the door", ["visible"], { filePath });
  assert.deepEqual(selected.map((lesson) => lesson.category), ["boundary", "uncertainty"]);
  assert.equal(selected.some((lesson) => lesson.category === "cross_modal"), false);
  const modelView = groundingLessonsForModel(selected);
  assert.match(modelView.advisory, /never override/i);
  assert.deepEqual(Object.keys(modelView.items[0]).sort(), ["advice", "category", "outcome", "scope"]);
});

test("similar lessons match Han text beside changing ordinals and Latin device names", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-learning-tokens-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const filePath = join(root, "lessons.jsonl");
  for (const [previousQuery, currentQuery] of [
    ["第1个墙壁开关", "第2个墙壁开关"],
    ["左侧USB插座", "usb connector"],
    ["USB connector", "右边USB插座"],
    ["第1个USB3端口", "USB3 connector"],
  ]) {
    const expected = storedLesson("matching", previousQuery);
    const unrelated = storedLesson("unrelated", "第1个香蕉");
    await writeFile(filePath, [expected, unrelated].map((lesson) => JSON.stringify(lesson)).join("\n"), "utf8");
    const selected = await selectGroundingLessons(currentQuery, ["visible"], { filePath });
    assert.deepEqual(selected.map((lesson) => lesson.id), [expected.id], `${previousQuery} -> ${currentQuery}`);
  }
  await writeFile(filePath, JSON.stringify(storedLesson("usb3", "第1个USB3端口")), "utf8");
  assert.deepEqual(await selectGroundingLessons("USB2", ["visible"], { filePath }), []);
});

for (const tail of ["valid", "malformed", "partial_utf8"]) {
  test(`queued appends preserve a ${tail} unterminated tail and keep new rows readable`, async (t) => {
    const root = await mkdtemp(join(tmpdir(), "pi-grounding-learning-tail-"));
    t.after(() => rm(root, { recursive: true, force: true }));
    const filePath = join(root, "lessons.jsonl");
    const existing = storedLesson("existing", "墙壁开关");
    const original = Buffer.concat([
      Buffer.from(JSON.stringify(existing)),
      ...(tail === "valid" ? [] : [Buffer.from('\n{"advice":"unfinished')]),
      ...(tail === "partial_utf8" ? [Buffer.from([0xe5, 0xa2])] : []),
    ]);
    await writeFile(filePath, original);

    const added = await Promise.all(Array.from({ length: 4 }, (_, index) => appendGroundingLesson({
      outcome: "confirmed",
      query: "墙壁开关",
      modalities: ["visible"],
      learning: { category: "boundary", scope: "similar", advice: `Preserve the full switch plate for appended lesson ${index}.` },
    }, filePath)));

    const saved = await readFile(filePath);
    assert.deepEqual(saved.subarray(0, original.length), original);
    const appendedRows = saved.subarray(original.length).toString("utf8").split("\n");
    assert.equal(appendedRows[0], "", "the old tail must be separated from the first new row");
    assert.equal(appendedRows.at(-1), "", "new rows must remain newline-terminated");
    assert.deepEqual(appendedRows.slice(1, -1).map((line) => JSON.parse(line).id), added.map((lesson) => lesson.id));
    assert.deepEqual((await readGroundingLessons(filePath)).map((lesson) => lesson.id), [existing.id, ...added.map((lesson) => lesson.id)]);
  });
}

test("appends do not insert blank rows after a terminated tail or into an empty file", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-learning-newline-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const filePath = join(root, "lessons.jsonl");
  const input = {
    outcome: "confirmed",
    query: "Wall switch",
    modalities: ["visible"],
    learning: { category: "boundary", scope: "similar", advice: "Preserve the complete outer wall switch plate." },
  };
  const first = await appendGroundingLesson(input, filePath);
  const second = await appendGroundingLesson(input, filePath);
  assert.equal(await readFile(filePath, "utf8"), `${JSON.stringify(first)}\n${JSON.stringify(second)}\n`);
});

test("global lessons retain bounded slots without crowding out specific matches", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-learning-ranking-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const filePath = join(root, "lessons.jsonl");
  const specific = Array.from({ length: 5 }, (_, index) => storedLesson(`specific-${index}`, "wall switch plate"));
  const usefulGlobal = storedLesson("useful-global", "wall switch plate", "global");
  const broadGlobal = storedLesson("visible-global", "Any grounding task", "global");
  const otherGlobals = Array.from({ length: 8 }, (_, index) => storedLesson(`infrared-global-${index}`, "Any grounding task", "global", ["infrared"]));
  const rows = [...specific, usefulGlobal, broadGlobal, ...otherGlobals, storedLesson("irrelevant", "thermal banana")];
  await writeFile(filePath, rows.map((lesson) => JSON.stringify(lesson)).join("\n"), "utf8");

  const selected = await selectGroundingLessons("wall switch plate", ["visible"], { filePath });
  assert.equal(selected.length, 6);
  assert.deepEqual(selected.filter((lesson) => lesson.scope === "similar").map((lesson) => lesson.id), ["specific-4", "specific-3", "specific-2", "specific-1"]);
  assert.deepEqual(selected.filter((lesson) => lesson.scope === "global").map((lesson) => lesson.id), [usefulGlobal.id, broadGlobal.id]);
  assert.deepEqual((await selectGroundingLessons("wall switch plate", ["visible"], { filePath, limit: 1 })).map((lesson) => lesson.id), ["specific-4"]);
  assert.deepEqual((await selectGroundingLessons("wall switch plate", ["visible"], { filePath, limit: 2 })).map((lesson) => lesson.id), [usefulGlobal.id, "specific-4"]);
  assert.deepEqual((await selectGroundingLessons("wall switch plate", ["visible"], { filePath, limit: 3 })).map((lesson) => lesson.id), [usefulGlobal.id, "specific-4", "specific-3"]);
});

test("global lessons fill spare slots and duplicate advice consumes only one slot", async (t) => {
  const root = await mkdtemp(join(tmpdir(), "pi-grounding-learning-dedupe-"));
  t.after(() => rm(root, { recursive: true, force: true }));
  const filePath = join(root, "lessons.jsonl");
  const specific = storedLesson("specific", "wall switch plate");
  const duplicate = { ...storedLesson("duplicate", "Any grounding task", "global"), advice: specific.advice.toUpperCase() };
  const globals = Array.from({ length: 7 }, (_, index) => storedLesson(`global-${index}`, "Any grounding task", "global"));
  await writeFile(filePath, [specific, duplicate, ...globals].map((lesson) => JSON.stringify(lesson)).join("\n"), "utf8");

  const selected = await selectGroundingLessons("wall switch plate", ["visible"], { filePath });
  assert.equal(selected.length, 6);
  assert.equal(selected[0].id, specific.id);
  assert.equal(selected.some((lesson) => lesson.id === duplicate.id), false);
  assert.equal(new Set(selected.map((lesson) => lesson.advice.toLowerCase())).size, 6);
  const globalOnly = await selectGroundingLessons("unrelated zebra", ["visible"], { filePath, limit: 3 });
  assert.deepEqual(globalOnly.map((lesson) => lesson.id), ["global-6", "global-5", "global-4"]);
});
