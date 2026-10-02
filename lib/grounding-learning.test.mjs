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
  assert.deepEqual(selected.map((lesson) => lesson.category), ["uncertainty", "boundary"]);
  assert.equal(selected.some((lesson) => lesson.category === "cross_modal"), false);
  const modelView = groundingLessonsForModel(selected);
  assert.match(modelView.advisory, /never override/i);
  assert.deepEqual(Object.keys(modelView.items[0]).sort(), ["advice", "category", "outcome", "scope"]);
});
