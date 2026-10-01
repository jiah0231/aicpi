import assert from "node:assert/strict";
import test from "node:test";
import sharp from "sharp";
import { analyzeGroundingColor } from "./grounding-color.ts";

test("normalized pixel edges and points round-trip without leaking a neighboring column", async () => {
  const bytes = await sharp({ create: { width: 1920, height: 1080, channels: 3, background: "black" } }).png().toBuffer();
  const result = await analyzeGroundingColor(bytes, {
    region: [123 / 1920, 39 / 1080, 125 / 1920, 79 / 1080], color: "black",
    selection: "point", point: [123 / 1920, 39 / 1080],
  });
  assert.deepEqual(result.regionPixels, [123, 39, 125, 79]);
  assert.equal(result.selectedAreaPixels, 80);
  assert.deepEqual(result.bbox, result.region);
});

async function fixture(width, height, rectangles = [], background = [255, 255, 255, 255]) {
  const pixels = Buffer.alloc(width * height * 4);
  for (let index = 0; index < width * height; index += 1) pixels.set(background, index * 4);
  for (const [left, top, right, bottom, color] of rectangles) {
    for (let y = top; y < bottom; y += 1) {
      for (let x = left; x < right; x += 1) pixels.set(color, (y * width + x) * 4);
    }
  }
  return sharp(pixels, { raw: { width, height, channels: 4 } }).png().toBuffer();
}

const black = [0, 0, 0, 255];
const region = [0, 0, 1, 1];

test("black component is measured at exact original pixel edges, separate from gray", async () => {
  const input = await fixture(20, 10, [[3, 2, 7, 6, black], [10, 1, 18, 9, [120, 120, 120, 255]]]);
  const result = await analyzeGroundingColor(input, { region, color: "black" });
  assert.equal(result.status, "matched");
  assert.deepEqual(result.bbox, [3 / 20, 2 / 10, 7 / 20, 6 / 10]);
  assert.equal(result.selectedAreaPixels, 16);
  assert.equal(result.candidateCount, 1);
  assert.deepEqual(result.candidates[0].pixelBbox, [3, 2, 7, 6]);
  assert.equal(result.candidates[0].matchingFraction, 1);
  assert.equal(result.matchingFraction, 16 / 200);
  const gray = await analyzeGroundingColor(input, { region, color: "gray" });
  assert.deepEqual(gray.bbox, [.5, .1, .9, .9]);
});

test("red hue wraps across zero and low-saturation pixels are not chromatic", async () => {
  const input = await fixture(12, 8, [
    [1, 1, 3, 3, [255, 0, 30, 255]],
    [7, 4, 9, 6, [255, 30, 0, 255]],
    [4, 1, 6, 3, [200, 190, 190, 255]],
  ]);
  const result = await analyzeGroundingColor(input, { region, color: "red", selection: "all" });
  assert.equal(result.candidateCount, 2);
  assert.equal(result.selectedAreaPixels, 8);
  assert.deepEqual(result.bbox, [1 / 12, 1 / 8, 9 / 12, 6 / 8]);
});

test("all named chromatic presets recognize their representative colors", async () => {
  const examples = {
    red: [255, 0, 0, 255], orange: [255, 128, 0, 255], yellow: [255, 255, 0, 255],
    green: [0, 220, 0, 255], cyan: [0, 220, 220, 255], blue: [0, 0, 255, 255],
    purple: [128, 0, 220, 255], pink: [255, 0, 128, 255], brown: [120, 60, 20, 255],
  };
  for (const [color, rgb] of Object.entries(examples)) {
    const result = await analyzeGroundingColor(await fixture(4, 4, [[1, 1, 3, 3, rgb]]), { region, color });
    assert.equal(result.selectedAreaPixels, 4, color);
  }
});

test("white requires brightness and low saturation", async () => {
  const input = await fixture(10, 10, [
    [1, 1, 3, 3, [245, 245, 245, 255]],
    [5, 5, 9, 9, [255, 255, 0, 255]],
  ], black);
  const result = await analyzeGroundingColor(input, { region, color: "white" });
  assert.equal(result.selectedAreaPixels, 4);
  assert.deepEqual(result.bbox, [.1, .1, .3, .3]);
});

test("hex tolerance zero matches only the exact RGB bytes", async () => {
  const input = await fixture(12, 8, [
    [1, 1, 3, 3, [32, 64, 128, 255]],
    [6, 3, 10, 6, [33, 64, 128, 255]],
  ]);
  const exact = await analyzeGroundingColor(input, { region, color: "#204080", tolerance: 0 });
  assert.equal(exact.selectedAreaPixels, 4);
  const broader = await analyzeGroundingColor(input, { region, color: "#204080", tolerance: .01, selection: "all" });
  assert.equal(broader.selectedAreaPixels, 16);
});

test("ROI excludes unrelated larger matching objects outside the chosen region", async () => {
  const input = await fixture(20, 20, [[1, 1, 8, 9, black], [13, 12, 16, 16, black]]);
  const result = await analyzeGroundingColor(input, { region: [.5, .5, .9, .9], color: "black" });
  assert.equal(result.candidateCount, 1);
  assert.equal(result.selectedAreaPixels, 12);
  assert.deepEqual(result.bbox, [.65, .6, .8, .8]);
  assert.equal(result.touchesRegionEdge, false);
});

test("fractional ROI coordinates report actual intersecting pixel edges and clipping", async () => {
  const input = await fixture(10, 10, [[1, 2, 6, 7, black]]);
  const result = await analyzeGroundingColor(input, { region: [.21, .21, .59, .59], color: "black" });
  assert.deepEqual(result.regionPixels, [2, 2, 6, 6]);
  assert.deepEqual(result.region, [.2, .2, .6, .6]);
  assert.deepEqual(result.bbox, [.2, .2, .6, .6]);
  assert.equal(result.selectedAreaPixels, 16);
  assert.equal(result.touchesRegionEdge, true);
  assert.match(result.warnings.join(" "), /may clip/);
});

test("color edges distinguish artificial bottom clipping from true top/right source boundaries", async () => {
  const input = await fixture(10, 10, [[6, 0, 10, 8, black]]);
  const result = await analyzeGroundingColor(input, { region: [0, 0, 1, .6], color: "black" });
  assert.equal(result.touchesRegionEdge, true);
  assert.deepEqual(result.touchesRoiEdges, ["top", "right", "bottom"]);
  assert.deepEqual(result.touchesSourceEdges, ["top", "right"]);
  assert.deepEqual(result.clippedRoiEdges, ["bottom"]);
  assert.deepEqual(result.candidates[0].touchesRoiEdges, result.touchesRoiEdges);
  assert.deepEqual(result.candidates[0].touchesSourceEdges, result.touchesSourceEdges);
  assert.deepEqual(result.candidates[0].clippedRoiEdges, result.clippedRoiEdges);
  const expandWarning = result.warnings.find((warning) => /wider region/.test(warning));
  assert.match(expandWarning, /bottom/);
  assert.doesNotMatch(expandWarning, /top|right/);
  assert.match(result.warnings.join(" "), /source image boundary at top, right/);
});

test("rounded ROI at source edges never asks to expand beyond the source", async () => {
  const input = await fixture(10, 10, [], black);
  const result = await analyzeGroundingColor(input, { region: [.01, .01, .99, .99], color: "black" });
  assert.deepEqual(result.regionPixels, [0, 0, 10, 10]);
  assert.deepEqual(result.touchesRoiEdges, ["left", "top", "right", "bottom"]);
  assert.deepEqual(result.touchesSourceEdges, result.touchesRoiEdges);
  assert.deepEqual(result.clippedRoiEdges, []);
  assert.doesNotMatch(result.warnings.join(" "), /wider region|may clip/);
});

test("edge summaries describe selected components and exclude unselected or filtered edge noise", async () => {
  const input = await fixture(20, 10, [[1, 1, 5, 5, black], [10, 3, 12, 5, black], [19, 9, 20, 10, black]]);
  const options = { region: [.05, .1, 1, 1], color: "black" };
  const point = await analyzeGroundingColor(input, { ...options, selection: "point", point: [.525, .35] });
  assert.deepEqual(point.touchesRoiEdges, []);
  assert.deepEqual(point.touchesSourceEdges, []);
  assert.deepEqual(point.clippedRoiEdges, []);
  assert.deepEqual(point.candidates[0].clippedRoiEdges, ["left", "top"]);
  const all = await analyzeGroundingColor(input, { ...options, selection: "all" });
  assert.deepEqual(all.touchesRoiEdges, ["left", "top"]);
  assert.deepEqual(all.touchesSourceEdges, []);
  assert.deepEqual(all.clippedRoiEdges, ["left", "top"]);
});

test("minimum area filters noise without expanding or joining selected components", async () => {
  const input = await fixture(20, 10, [[1, 1, 2, 2, black], [3, 4, 6, 8, black], [11, 3, 13, 6, black]]);
  const largest = await analyzeGroundingColor(input, { region, color: "black" });
  assert.equal(largest.matchingPixels, 19);
  assert.equal(largest.candidateCount, 2);
  assert.equal(largest.selectedAreaPixels, 12);
  assert.deepEqual(largest.bbox, [.15, .4, .3, .8]);
  const all = await analyzeGroundingColor(input, { region, color: "black", selection: "all" });
  assert.equal(all.selectedAreaPixels, 18);
  assert.equal(all.selectedComponentCount, 2);
  assert.deepEqual(all.bbox, [.15, .3, .65, .8]);
  assert.match(all.warnings.join(" "), /disconnected/);
});

test("diagonally adjoining pixels use 8-neighbor connectivity", async () => {
  const input = await fixture(6, 6, [[1, 1, 2, 2, black], [2, 2, 3, 3, black], [3, 3, 4, 4, black]]);
  const result = await analyzeGroundingColor(input, { region, color: "black" });
  assert.equal(result.candidateCount, 1);
  assert.equal(result.selectedAreaPixels, 3);
  assert.equal(result.candidates[0].matchingFraction, 1 / 3);
});

test("point chooses its smaller component without substituting a nearby match", async () => {
  const input = await fixture(20, 10, [[1, 1, 7, 7, black], [12, 3, 14, 5, black]]);
  const options = { region, color: "black", selection: "point" };
  const result = await analyzeGroundingColor(input, { ...options, point: [.625, .35] });
  assert.equal(result.selectedAreaPixels, 4);
  assert.deepEqual(result.bbox, [.6, .3, .7, .5]);
  for (const point of [[.75, .4], [1, 1], [.95, .95]]) {
    const empty = await analyzeGroundingColor(input, { ...options, point });
    assert.equal(empty.status, "no_match");
    assert.equal(empty.bbox, null);
    assert.match(empty.warnings.join(" "), /No nearest component/);
  }
});

test("point diagnostics report the sampled color and do not turn a miss into identity evidence", async () => {
  const input = await fixture(10, 10, [[6, 6, 9, 9, black]], [80, 100, 120, 255]);
  const miss = await analyzeGroundingColor(input, {
    region, color: "black", selection: "point", point: [.2, .3],
  });
  assert.equal(miss.status, "no_match");
  assert.deepEqual(miss.pointSample.sourcePixel, [2, 3]);
  assert.deepEqual(miss.pointSample.rgba, [80, 100, 120, 255]);
  assert.equal(miss.pointSample.hex, "#506478");
  assert.equal(miss.pointSample.matchesRequestedColor, false);
  assert.equal(miss.pointSample.retainedComponentId, null);
  assert.equal(miss.selectionAssessment.boundaryStatus, "no_selection");
  assert.equal(miss.selectionAssessment.establishesObjectIdentity, false);
  assert.match(miss.selectionAssessment.recommendation, /Do not infer absence/);

  const tiny = await analyzeGroundingColor(input, {
    region, color: "black", selection: "point", point: [.65, .65], minAreaPixels: 1,
  });
  assert.equal(tiny.pointSample.matchesRequestedColor, true);
  assert.ok(tiny.pointSample.retainedComponentId > 0);
  assert.equal(tiny.selectionAssessment.role, "pixel_measurement_only");
});

test("empty, transparent and below-threshold components return no_match, not an invented box", async () => {
  for (const input of [
    await fixture(4, 4),
    await fixture(4, 4, [], [0, 0, 0, 0]),
    await fixture(4, 4, [], [0, 0, 0, 127]),
    await fixture(4, 4, [[1, 1, 2, 2, black]]),
  ]) {
    const result = await analyzeGroundingColor(input, { region, color: "black" });
    assert.equal(result.status, "no_match");
    assert.equal(result.bbox, null);
    assert.equal(result.selectedAreaPixels, 0);
    assert.equal(result.touchesRegionEdge, false);
    assert.deepEqual(result.touchesRoiEdges, []);
    assert.deepEqual(result.touchesSourceEdges, []);
    assert.deepEqual(result.clippedRoiEdges, []);
  }
});

test("bounded candidate output still measures all components and includes a selected point", async () => {
  const rectangles = [];
  for (let y = 0; y < 10; y += 1) {
    for (let x = 0; x < 10; x += 1) rectangles.push([x * 3, y * 3, x * 3 + 1, y * 3 + 1, black]);
  }
  const input = await fixture(30, 30, rectangles);
  const options = { region, color: "black", minAreaPixels: 1 };
  const result = await analyzeGroundingColor(input, { ...options, selection: "all" });
  assert.equal(result.candidateCount, 100);
  assert.equal(result.candidates.length, 32);
  assert.equal(result.candidatesTruncated, true);
  assert.equal(result.selectedComponentCount, 100);
  assert.equal(result.selectedAreaPixels, 100);
  const point = await analyzeGroundingColor(input, { ...options, selection: "point", point: [27.5 / 30, 27.5 / 30] });
  assert.equal(point.selectedAreaPixels, 1);
  assert.ok(point.candidates.some((item) => item.id === point.selectedIds[0]));
});

test("raw and marked previews have the same cropped geometry and preserve source measurements", async () => {
  const input = await fixture(2200, 200, [[900, 50, 1200, 100, black]]);
  const result = await analyzeGroundingColor(input, { region, color: "black" });
  assert.deepEqual(result.bbox, [900 / 2200, .25, 1200 / 2200, .5]);
  assert.equal(result.previewWidth, 1600);
  assert.ok(result.previewHeight <= 1600);
  const raw = await sharp(result.rawPreview).metadata();
  const marked = await sharp(result.maskPreview).metadata();
  assert.equal(raw.format, "png");
  assert.equal(marked.format, "png");
  assert.equal(raw.width, result.previewWidth);
  assert.equal(raw.height, result.previewHeight);
  assert.equal(marked.width, raw.width);
  assert.equal(marked.height, raw.height);
  assert.notDeepEqual(result.rawPreview, result.maskPreview);
});

test("raw preview remains clean and mask tints only selected pixels", async () => {
  const input = await fixture(10, 10, [[2, 2, 8, 8, black]]);
  const result = await analyzeGroundingColor(input, { region, color: "black" });
  const raw = await sharp(result.rawPreview).raw().toBuffer({ resolveWithObject: true });
  const marked = await sharp(result.maskPreview).raw().toBuffer({ resolveWithObject: true });
  const pixel = (image, x, y) => [...image.data.subarray((y * image.info.width + x) * 4, (y * image.info.width + x) * 4 + 4)];
  assert.deepEqual(pixel(raw, 40, 40), black);
  assert.deepEqual(pixel(marked, 40, 40), [0, 153, 153, 255]);
  assert.deepEqual(pixel(raw, 4, 4), [255, 255, 255, 255]);
  assert.deepEqual(pixel(marked, 4, 4), [255, 255, 255, 255]);
});

test("invalid inputs and oversized ROIs are rejected before full region allocation", async () => {
  const input = await fixture(5, 5);
  for (const options of [
    { region: [0, 0, 0, 1], color: "black" },
    { region: [-.1, 0, 1, 1], color: "black" },
    { region, color: "unknown" },
    { region, color: "black", tolerance: Number.NaN },
    { region, color: "black", minAreaPixels: 0 },
    { region, color: "black", selection: "point" },
    { region, color: "black", selection: "point", point: [0, 2] },
  ]) await assert.rejects(analyzeGroundingColor(input, options));
  const large = await sharp({ create: { width: 3000, height: 3000, channels: 3, background: "white" } }).png().toBuffer();
  await assert.rejects(analyzeGroundingColor(large, { region, color: "black" }), /choose a smaller region/);
});
