import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const source = await readFile(new URL("./GroundingReviewPanel.tsx", import.meta.url), "utf8");

test("grounding review exposes explicit cross-session learning controls", () => {
  assert.match(source, /aria-label="Long-term grounding improvement advice"/);
  assert.match(source, /aria-label="Learning category"/);
  assert.match(source, /aria-label="Learning scope"/);
  assert.match(source, /rememberLearning && learningAdvice\.trim\(\)\.length > 0/);
  assert.match(source, /\.\.\.learningPayload\(\)/);
  assert.match(source, /跨会话保存/);
});
