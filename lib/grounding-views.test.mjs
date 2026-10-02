import assert from "node:assert/strict";
import test from "node:test";
import sharp from "sharp";
import { GroundingViewRegistry, buildGroundingComparison, groundingCandidateGeometry, groundingBoundaryRegions } from "./grounding-views.ts";

const descriptor = {
  modality: "visible",
  region: [17 / 1920, 23 / 1080, 418 / 1920, 622 / 1080],
  sourceWidth: 1920,
  sourceHeight: 1080,
  width: 803,
  height: 1199,
};

function closeBox(actual, expected) {
  actual.forEach((edge, index) => assert.ok(Math.abs(edge - expected[index]) < 1e-12, `${actual} != ${expected}`));
}

test("non-square pixel and normalized views map to exact source edges without coordinate rounding", () => {
  const registry = new GroundingViewRegistry();
  const view = registry.register(descriptor);
  assert.deepEqual(registry.toSource(view.id, [0, 0, 803, 1199], "view_pixels"), descriptor.region);
  assert.deepEqual(registry.toSource(view.id, [0, 0, 1, 1], "view_normalized"), descriptor.region);
  const fractional = [123.25, 79.125, 751.5, 1004.625];
  const pixels = registry.toSource(view.id, fractional, "view_pixels");
  const normalized = registry.toSource(view.id, fractional.map((edge, index) => edge / (index % 2 ? 1199 : 803)), "view_normalized");
  closeBox(pixels, normalized);
  closeBox(pixels, [
    (17 + 123.25 / 803 * 401) / 1920,
    (23 + 79.125 / 1199 * 599) / 1080,
    (17 + 751.5 / 803 * 401) / 1920,
    (23 + 1004.625 / 1199 * 599) / 1080,
  ]);
});

test("padded montage coordinates use the whole canvas and reject its labels and padding", () => {
  const registry = new GroundingViewRegistry();
  const displayRect = [851, 419, 1517, 988];
  const view = registry.register({ ...descriptor, width: 1600, height: 1500, displayRect });
  const normalizedRect = displayRect.map((edge, index) => edge / (index % 2 ? 1500 : 1600));
  assert.deepEqual(registry.toSource(view.id, displayRect, "view_pixels"), descriptor.region);
  assert.deepEqual(registry.toSource(view.id, normalizedRect, "view_normalized"), descriptor.region);
  closeBox(registry.toSource(view.id, [1017.5, 561.25, 1350.5, 845.75], "view_pixels"), [
    descriptor.region[0] + .25 * (descriptor.region[2] - descriptor.region[0]),
    descriptor.region[1] + .25 * (descriptor.region[3] - descriptor.region[1]),
    descriptor.region[0] + .75 * (descriptor.region[2] - descriptor.region[0]),
    descriptor.region[1] + .75 * (descriptor.region[3] - descriptor.region[1]),
  ]);
  for (const bad of [[850, 419, 1517, 988], [851, 418, 1517, 988], [851, 419, 1518, 988], [851, 419, 1517, 989]]) {
    assert.throws(() => registry.toSource(view.id, bad, "view_pixels"), /boundary/);
  }
  assert.throws(() => registry.toSource(view.id, [0, 0, 1, 1], "view_normalized"), /boundary/);
  assert.throws(() => registry.toSource(view.id, [normalizedRect[0] - 1e-10, ...normalizedRect.slice(1)], "view_normalized"), /boundary/);
});

test("view IDs cannot accidentally identify views in another record registry", () => {
  const first = new GroundingViewRegistry();
  const second = new GroundingViewRegistry();
  const a = first.register(descriptor);
  const b = second.register(descriptor);
  assert.notEqual(a.id, b.id);
  assert.equal(second.get(a.id), undefined);
  assert.throws(() => second.toSource(a.id, [0, 0, 1, 1], "view_normalized"), /stale viewId/);
});

test("unknown or stale view errors supply bounded actual IDs without changing the coordinate reference", () => {
  const registry = new GroundingViewRegistry();
  const oldRegistry = new GroundingViewRegistry();
  const stale = oldRegistry.register(descriptor);
  const views = Array.from({ length: 8 }, (_, index) => registry.register({
    ...descriptor, modality: index % 2 ? "infrared" : "visible",
  }));
  for (const [method, coordinates] of [
    ["toSource", [0, 0, 1, 1]], ["toSourcePoint", [.5, .5]],
    ["toVisibleSource", [0, 0, 1, 1]], ["toVisibleSourcePoint", [.5, .5]],
  ]) {
    assert.throws(() => registry[method](stale.id, coordinates, "view_normalized"), (error) => {
      assert.match(error.message, /Unknown or stale viewId/);
      assert.match(error.message, /view that supplied the measured coordinates/);
      assert.match(error.message, /most recent 6 of 8/);
      for (const view of views.slice(-6)) assert.ok(error.message.includes(`${view.id} (${view.modality})`));
      for (const view of views.slice(0, 2)) assert.ok(!error.message.includes(view.id));
      assert.ok(!error.message.includes(stale.id), "must not advertise IDs from a previous record");
      return true;
    });
  }
  const empty = new GroundingViewRegistry();
  assert.throws(() => empty.toVisibleSource("missing", [0, 0, 1, 1], "view_normalized"), /No viewIds are registered for the current record/);
});

test("visible-source wrappers preserve exact crop and montage mappings and boundary validation", () => {
  const registry = new GroundingViewRegistry();
  for (const input of [descriptor, { ...descriptor, width: 1600, height: 1500, displayRect: [851, 419, 1517, 988] }]) {
    const view = registry.register(input);
    const rect = view.displayRect ?? [0, 0, view.width, view.height];
    const point = [(rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2];
    for (const space of ["view_pixels", "view_normalized"]) {
      const convert = (values) => space === "view_pixels" ? values : values.map((value, index) => value / (index % 2 ? view.height : view.width));
      assert.deepEqual(registry.toVisibleSource(view.id, convert(rect), space), descriptor.region);
      assert.deepEqual(registry.toVisibleSourcePoint(view.id, convert(point), space), registry.toSourcePoint(view.id, convert(point), space));
      assert.throws(() => registry.toVisibleSource(view.id, convert([rect[0] - 1, ...rect.slice(1)]), space), /boundary/);
      assert.throws(() => registry.toVisibleSourcePoint(view.id, convert([rect[0] - 1, point[1]]), space), /boundary/);
    }
    assert.throws(() => registry.toVisibleSource(view.id, [0, 0, NaN, 1], "view_normalized"), /finite edges/);
    assert.throws(() => registry.toVisibleSourcePoint(view.id, [NaN, .5], "view_normalized"), /finite coordinates/);
    assert.throws(() => registry.toVisibleSource(view.id, rect, "source"), /coordinate space/);
    assert.throws(() => registry.toVisibleSourcePoint(view.id, point, "source"), /coordinate space/);
  }
});

test("visible-source mapping rejects infrared and depth even when their dimensions match visible", () => {
  const registry = new GroundingViewRegistry();
  const visible = registry.register(descriptor);
  for (const modality of ["infrared", "depth"]) {
    const view = registry.register({ ...descriptor, modality });
    for (const space of ["view_pixels", "view_normalized"]) {
      const box = space === "view_pixels" ? [0, 0, view.width, view.height] : [0, 0, 1, 1];
      const point = space === "view_pixels" ? [view.width / 2, view.height / 2] : [.5, .5];
      // Evidence and further crops can still be mapped within their own modality.
      assert.deepEqual(registry.toSource(view.id, box, space), descriptor.region);
      closeBox(registry.toSourcePoint(view.id, point, space), [
        (descriptor.region[0] + descriptor.region[2]) / 2,
        (descriptor.region[1] + descriptor.region[3]) / 2,
      ]);
      for (const map of [() => registry.toVisibleSource(view.id, box, space), () => registry.toVisibleSourcePoint(view.id, point, space)]) {
        assert.throws(map, (error) => {
          assert.ok(error.message.includes(`${view.id} is ${modality}, not visible`));
          assert.match(error.message, /registration is unsupported/);
          assert.match(error.message, /equal image dimensions do not establish alignment/);
          assert.match(error.message, /Remeasure the coordinates on a visible view/);
          assert.ok(error.message.includes(`Available visible viewIds: ${visible.id} (visible)`));
          return true;
        });
      }
    }
  }
});

test("callers cannot mutate registered coordinate metadata through input or returned descriptors", () => {
  const registry = new GroundingViewRegistry();
  const input = { ...descriptor, region: [...descriptor.region], displayRect: [0, 0, 803, 1199] };
  const first = registry.register(input);
  input.region[0] = .5;
  first.region[0] = .7;
  first.displayRect[0] = 80;
  registry.get(first.id).region[0] = .9;
  registry.list()[0].displayRect[0] = 44;
  assert.deepEqual(registry.get(first.id).region, descriptor.region);
  assert.deepEqual(registry.get(first.id).displayRect, [0, 0, 803, 1199]);
});

test("equivalent views identify the same source pixels despite changed display size or decorations", () => {
  const registry = new GroundingViewRegistry();
  const first = registry.register({ ...descriptor, decorations: "all" });
  assert.equal(registry.findEquivalent({ ...descriptor, width: 10, decorations: "none" }).id, first.id);
  assert.equal(registry.findEquivalent({ ...descriptor, modality: "infrared" }), undefined);
  assert.equal(registry.findEquivalent({ ...descriptor, region: [0, ...descriptor.region.slice(1)] }), undefined);
  assert.equal(registry.findEquivalent({ ...descriptor, sourceWidth: 3840 }), undefined);
  assert.notEqual(registry.register({ ...descriptor, decorations: "none" }).id, first.id);
});

test("source reuse catches contained and shifted local rerenders without treating an overview as redundant", () => {
  const registry = new GroundingViewRegistry();
  registry.register({ ...descriptor, region: [0, 0, 1, 1], width: 1600, height: 900, decorations: "none" });
  const firstInput = {
    ...descriptor,
    region: [1121 / 1920, 509 / 1080, 1183 / 1920, 571 / 1080],
    width: 1364,
    height: 1364,
  };
  assert.equal(registry.sourceReuse(firstInput), undefined, "the initial overview must not suppress a useful local crop");
  const first = registry.register(firstInput);
  const contained = registry.sourceReuse({
    ...descriptor,
    region: [1136 / 1920, 520 / 1080, 1179 / 1920, 566 / 1080],
    width: 1495,
    height: 1600,
  });
  assert.equal(contained.relation, "contained_rerender");
  assert.equal(contained.previousViewId, first.id);
  assert.equal(contained.coveredFraction, 1);
  assert.equal(contained.newSourcePixels, 0);
  assert.ok(contained.displayScaleRatio > 1);

  const shiftedRegistry = new GroundingViewRegistry();
  const original = shiftedRegistry.register({ ...descriptor, region: [100 / 1920, 100 / 1080, 200 / 1920, 200 / 1080] });
  const shifted = shiftedRegistry.sourceReuse({ ...descriptor, region: [101 / 1920, 100 / 1080, 201 / 1920, 200 / 1080] });
  assert.equal(shifted.relation, "near_duplicate");
  assert.equal(shifted.previousViewId, original.id);
  assert.equal(shifted.newSourcePixels, 100);
  assert.equal(shifted.coveredFraction, .99);
  assert.ok(shifted.iou > .98);
  assert.equal(shiftedRegistry.sourceReuse({ ...descriptor, modality: "infrared", region: shiftedRegistry.get(original.id).region }), undefined);
});

test("view points map through content rectangles and reject montage labels", () => {
  const registry = new GroundingViewRegistry();
  const view = registry.register({ ...descriptor, width: 1600, height: 1500, displayRect: [800, 400, 1400, 1300] });
  closeBox(registry.toSourcePoint(view.id, [1100, 850], "view_pixels"), [
    (descriptor.region[0] + descriptor.region[2]) / 2,
    (descriptor.region[1] + descriptor.region[3]) / 2,
  ]);
  closeBox(registry.toSourcePoint(view.id, [1100 / 1600, 850 / 1500], "view_normalized"), [
    (descriptor.region[0] + descriptor.region[2]) / 2,
    (descriptor.region[1] + descriptor.region[3]) / 2,
  ]);
  assert.throws(() => registry.toSourcePoint(view.id, [100, 100], "view_pixels"), /boundary/);
});

test("invalid views and boxes fail instead of silently clamping or accepting NaN", () => {
  const registry = new GroundingViewRegistry();
  for (const change of [
    { sourceWidth: 0 }, { sourceHeight: Infinity }, { width: 2.5 }, { height: -1 },
    { region: [-.01, 0, 1, 1] }, { region: [0, 0, 1.01, 1] }, { region: [0, 0, 0, 1] },
    { displayRect: [0, 0, 804, 1199] }, { displayRect: [0, 0, NaN, 1199] },
  ]) assert.throws(() => registry.register({ ...descriptor, ...change }));
  const view = registry.register(descriptor);
  for (const box of [[0, 0, NaN, 1], [0, 0, Infinity, 1], [1, 0, 0, 1], [0, 0, 1], [0, 0, 0, 1], [-.001, 0, 1, 1], [0, 0, 1.00001, 1]]) {
    assert.throws(() => registry.toSource(view.id, box, "view_normalized"));
  }
  assert.throws(() => registry.toSource(view.id, [0, 0, 1, 1], "last_crop"), /coordinate space/);
});

async function solidImage(width, height, color) {
  return sharp({ create: { width, height, channels: 3, background: color } }).png().toBuffer();
}

test("comparison keeps image rectangles separate from labels and preserves non-square aspect ratios", async () => {
  const overview = await solidImage(1920, 1080, "#ff0000");
  const candidates = await Promise.all([
    [1200, 300, "#00ff00"], [240, 960, "#0000ff"], [20, 10, "#ffff00"], [801, 599, "#00ffff"],
  ].map(async ([width, height, color], index) => ({ label: `target <${index}> & test`, image: await solidImage(width, height, color) })));
  const result = await buildGroundingComparison(overview, candidates);
  assert.ok(Math.max(result.width, result.height) <= 1600);
  assert.equal(result.panels.length, 4);
  assert.deepEqual(result.panels.map((panel) => panel.label.slice(0, 2)), ["A:", "B:", "C:", "D:"]);
  const raw = await sharp(result.image).removeAlpha().raw().toBuffer({ resolveWithObject: true });
  assert.equal(raw.info.width, result.width);
  assert.equal(raw.info.height, result.height);
  const expected = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255]];
  const sizes = [[1920, 1080], [1200, 300], [240, 960], [20, 10], [801, 599]];
  const rects = [result.overviewRect, ...result.panels.map((panel) => panel.rect)];
  const registry = new GroundingViewRegistry();
  for (const [index, rect] of rects.entries()) {
    assert.ok(rect.every(Number.isInteger));
    assert.ok(rect[0] >= 0 && rect[1] >= 32 && rect[2] <= result.width && rect[3] <= result.height);
    const [sourceWidth, sourceHeight] = sizes[index];
    const contentWidth = rect[2] - rect[0];
    const contentHeight = rect[3] - rect[1];
    if (index === 0) assert.ok(contentWidth <= sourceWidth && contentHeight <= sourceHeight, "overview should not enlarge");
    else assert.ok(contentWidth === 776 || contentHeight === 500, "candidate should fill its panel in at least one dimension");
    assert.ok(Math.abs(contentHeight - contentWidth / sourceWidth * sourceHeight) <= 1.1, "aspect ratio preserved to pixel rounding");
    // Every content pixel remains the original solid color: no title/grid was painted over it.
    for (let y = rect[1]; y < rect[3]; y += 1) {
      for (let x = rect[0]; x < rect[2]; x += 1) {
        const offset = (y * result.width + x) * 3;
        for (let channel = 0; channel < 3; channel += 1) {
          assert.equal(raw.data[offset + channel], expected[index][channel]);
        }
      }
    }
    const view = registry.register({ modality: "visible", region: [0, 0, 1, 1], sourceWidth, sourceHeight, width: result.width, height: result.height, displayRect: rect });
    assert.deepEqual(registry.toSource(view.id, rect, "view_pixels"), [0, 0, 1, 1]);
    assert.deepEqual(registry.toSource(view.id, rect.map((edge, axis) => edge / (axis % 2 ? result.height : result.width)), "view_normalized"), [0, 0, 1, 1]);
    assert.throws(() => registry.toSource(view.id, [rect[0], rect[1] - 1, rect[2], rect[3]], "view_pixels"), /boundary/);
  }
});

test("single-candidate comparison remains bounded; panel-count errors allow later calls", async () => {
  const image = await solidImage(90, 70, "red");
  const result = await buildGroundingComparison(image, [{ label: "one", image }]);
  assert.ok(Math.max(result.width, result.height) <= 1600);
  assert.equal(result.panels.length, 1);
  assert.equal(result.overviewRect[2] - result.overviewRect[0], 90, "small overview remains native size");
  assert.equal(result.panels[0].rect[3] - result.panels[0].rect[1], 540, "small candidate is magnified to its panel");
  await assert.rejects(buildGroundingComparison(image, []), /1 to 4/);
  await assert.rejects(buildGroundingComparison(image, Array(5).fill({ label: "extra", image })), /another comparison/);
  const second = await buildGroundingComparison(image, [{ label: "retry", image }]);
  assert.equal(second.panels.length, 1);
});

test("comparison preserves source pixel axes even when JPEG has EXIF rotation metadata", async () => {
  const pixels = Buffer.alloc(120 * 60 * 3);
  for (let y = 0; y < 60; y += 1) for (let x = 0; x < 120; x += 1) {
    const offset = (y * 120 + x) * 3;
    pixels[offset + (x < 60 ? 0 : 2)] = 255;
  }
  const jpeg = await sharp(pixels, { raw: { width: 120, height: 60, channels: 3 } }).withMetadata({ orientation: 6 }).jpeg({ quality: 100 }).toBuffer();
  const result = await buildGroundingComparison(jpeg, [{ label: "horizontal target", image: jpeg }]);
  const rect = result.panels[0].rect;
  assert.equal(rect[2] - rect[0], 2 * (rect[3] - rect[1]));
  const raw = await sharp(result.image).removeAlpha().raw().toBuffer();
  const y = Math.floor((rect[1] + rect[3]) / 2);
  const left = (y * result.width + Math.floor(rect[0] + (rect[2] - rect[0]) * .25)) * 3;
  const right = (y * result.width + Math.floor(rect[0] + (rect[2] - rect[0]) * .75)) * 3;
  assert.ok(raw[left] > 240 && raw[left + 2] < 10, "source left remains red");
  assert.ok(raw[right] < 10 && raw[right + 2] > 240, "source right remains blue");
});

test("candidate geometry sorts object boxes, never ROI centers or discovery order, and reports ties", () => {
  const result = groundingCandidateGeometry([
    { region: [0, 0, .2, .2], bbox: [.7, .2, .9, .4] },
    { region: [.8, .8, 1, 1], bbox: [.1, .2, .3, .4] },
    { region: [.4, .4, .6, .6] },
  ], 100, 200);
  assert.deepEqual(result.leftToRight, ["B", "A"]);
  assert.deepEqual(result.missingObjectBoxes, ["C"]);
  assert.deepEqual(result.tiedY, [["A", "B"]]);
  assert.equal(result.candidates[2].objectCenter, undefined);
  closeBox([...result.candidates[0].objectCenter, ...result.candidates[0].objectSizePixels], [.8, .3, 20, 40]);
});

test("boundary strips straddle proposal edges and disclose unavailable source context", () => {
  const box = [0, .2, .6, 1];
  const strips = groundingBoundaryRegions(box, 100, 100);
  assert.deepEqual(box, [0, .2, .6, 1], "never alter proposal edges");
  assert.deepEqual(strips.map((strip) => strip.edge), ["top", "bottom", "left", "right"]);
  for (const strip of strips) {
    const axis = strip.axis === "x" ? 0 : 1;
    assert.ok(strip.region[axis] <= strip.position && strip.position <= strip.region[axis + 2]);
    assert.ok(strip.region.every((value) => value >= 0 && value <= 1));
  }
  assert.equal(strips[1].outsideSourcePixels, 0);
  assert.equal(strips[2].outsideSourcePixels, 0);
  assert.equal(strips[1].contextClipped, true);
  assert.equal(strips[2].contextClipped, true);
});

test("overview annotations and external edge markers preserve panel pixel content and mapping", async () => {
  const image = await solidImage(100, 100, "#ff0000");
  const plain = await buildGroundingComparison(image, [{ label: "top", image }]);
  const decorated = await buildGroundingComparison(image, [{ label: "top", image, edgeMarker: { axis: "y", fraction: .5 } }], [
    { label: "A ROI", kind: "roi", region: [.1, .1, .9, .9] },
    { label: "A object", kind: "object", region: [.2, .2, .8, .8] },
  ]);
  assert.deepEqual(decorated.panels, plain.panels);
  assert.deepEqual(decorated.overviewRect, plain.overviewRect);
  const [left, top, right, bottom] = decorated.panels[0].rect;
  const extract = { left, top, width: right - left, height: bottom - top };
  assert.deepEqual(await sharp(decorated.image).extract(extract).raw().toBuffer(), await sharp(plain.image).extract(extract).raw().toBuffer());
});

test("tangential context clipping does not imply zero pixels beyond the inspected edge", () => {
  const top = groundingBoundaryRegions([0, .4, .6, .8], 100, 100)[0];
  assert.equal(top.edge, "top");
  assert.equal(top.contextClipped, true, "left side of the top strip is clipped");
  assert.equal(top.outsideSourcePixels, 40, "forty source pixels still exist above the proposal");
});
