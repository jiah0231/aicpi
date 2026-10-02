import assert from "node:assert/strict";
import test from "node:test";
import { getToolRoundTripSeconds } from "./tool-timing.ts";

test("measures the complete model-to-result interval instead of claiming tool execution time", () => {
  const generationStart = 1_000;
  const generationEnd = 31_000;
  const toolResult = 33_000;

  assert.equal(getToolRoundTripSeconds(generationStart, toolResult), 32);
  assert.notEqual(getToolRoundTripSeconds(generationStart, toolResult), (toolResult - generationEnd) / 1_000);
});

test("later tool results share the generation start, including any scheduling and confirmation wait", () => {
  assert.equal(getToolRoundTripSeconds(1_000, 11_000), 10);
  assert.equal(getToolRoundTripSeconds(1_000, 81_000), 80);
});

test("keeps whole-second rounding and accepts a valid zero timestamp", () => {
  assert.equal(getToolRoundTripSeconds(0, 1_499), 1);
  assert.equal(getToolRoundTripSeconds(0, 1_500), 2);
  assert.equal(getToolRoundTripSeconds(1_000, 1_499), undefined);
  assert.equal(getToolRoundTripSeconds(1_000, 1_000), undefined);
});

test("omits missing, invalid, and reversed timestamps instead of displaying a misleading duration", () => {
  for (const invalid of [undefined, null, NaN, Infinity, -Infinity, -1, "1000"]) {
    assert.equal(getToolRoundTripSeconds(invalid, 5_000), undefined);
    assert.equal(getToolRoundTripSeconds(1_000, invalid), undefined);
  }
  assert.equal(getToolRoundTripSeconds(5_000, 1_000), undefined);
});
