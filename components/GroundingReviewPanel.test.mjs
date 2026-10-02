import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url, {
  jsx: { runtime: "automatic" },
  tsconfigPaths: true,
});
const { mapBboxToBoundaryPreview } = await jiti.import("./GroundingReviewPanel.tsx");

const source = await readFile(new URL("./GroundingReviewPanel.tsx", import.meta.url), "utf8");
const typesSource = await readFile(new URL("../lib/types.ts", import.meta.url), "utf8");

test("grounding review exposes explicit cross-session learning controls", () => {
  assert.match(source, /aria-label="Long-term grounding improvement advice"/);
  assert.match(source, /aria-label="Learning category"/);
  assert.match(source, /aria-label="Learning scope"/);
  assert.match(source, /rememberLearning && learningAdvice\.trim\(\)\.length > 0/);
  assert.match(source, /\.\.\.learningPayload\(\)/);
  assert.match(source, /跨会话保存/);
});

test("boundary preview metadata is optional and carries clean pixels plus actual source crop edges", () => {
  assert.match(typesSource, /interface GroundingReviewBoundaryPreview \{[\s\S]*?image: GroundingReviewImage;[\s\S]*?region: \[number, number, number, number\];/);
  assert.match(typesSource, /boundaryPreview\?: GroundingReviewBoundaryPreview;/);
  assert.match(source, /data:\$\{boundaryPreview\.image\.mimeType\};base64,\$\{boundaryPreview\.image\.data\}/);
  assert.match(source, /src=\{boundaryImageUrl\}/);
  assert.match(source, /aspectRatio: `\$\{boundaryWidth\} \/ \$\{boundaryHeight\}`/);
  assert.match(source, /boundaryPreview\?\.image\.width \|\| boundaryPreview\?\.image\.originalWidth/);
  assert.match(source, /boundaryPreview\?\.image\.height \|\| boundaryPreview\?\.image\.originalHeight/);
  assert.match(source, /alt=\{`Full image for \$\{details\.key\}`\}/);
});

test("preview overlay maps the edited box into a non-square source region", () => {
  const region = [0.125, 0.25, 0.875, 0.5];
  assert.deepEqual(mapBboxToBoundaryPreview([0.3125, 0.3125, 0.6875, 0.4375], region), {
    left: 0.25, top: 0.25, width: 0.5, height: 0.5, extendsBeyondPreview: false,
  });
  assert.deepEqual(mapBboxToBoundaryPreview([0.125, 0.25, 0.5, 0.5], region), {
    left: 0, top: 0, width: 0.5, height: 1, extendsBeyondPreview: false,
  });
  assert.match(source, /mapBboxToBoundaryPreview\(bbox, boundaryPreview\.region\)/);
});

test("preview clipping never changes the source box or invents a crop-edge boundary", () => {
  const bbox = Object.freeze([0.125, 0.125, 0.875, 0.875]);
  const region = Object.freeze([0.25, 0.25, 0.75, 0.75]);
  assert.deepEqual(mapBboxToBoundaryPreview(bbox, region), {
    left: -0.25, top: -0.25, width: 1.5, height: 1.5, extendsBeyondPreview: true,
  });
  assert.deepEqual(bbox, [0.125, 0.125, 0.875, 0.875]);
  assert.deepEqual(region, [0.25, 0.25, 0.75, 0.75]);
  assert.match(source, /aspectRatio: `\$\{boundaryWidth\} \/ \$\{boundaryHeight\}`, overflow: "hidden"/);
  assert.match(source, /left: `\$\{boundaryBox\.left \* 100\}%`/);
  assert.match(source, /width: `\$\{boundaryBox\.width \* 100\}%`/);
  assert.match(source, /boundaryBox\?\.extendsBeyondPreview/);
  assert.match(source, /请在上方全图核对超出部分；提交坐标保持不变/);
  assert.match(source, /void send\("confirm", \{\s*bbox,/);
});

test("each edge outside the fixed preview raises the informational warning", () => {
  const region = [0.25, 0.25, 0.75, 0.75];
  for (const bbox of [
    [0.125, 0.375, 0.625, 0.625],
    [0.375, 0.125, 0.625, 0.625],
    [0.375, 0.375, 0.875, 0.625],
    [0.375, 0.375, 0.625, 0.875],
  ]) {
    assert.equal(mapBboxToBoundaryPreview(bbox, region).extendsBeyondPreview, true);
  }
  assert.equal(mapBboxToBoundaryPreview(region, region).extendsBeyondPreview, false);
  assert.deepEqual(mapBboxToBoundaryPreview([0, 0, 1, 1], [0, 0, 1, 1]), {
    left: 0, top: 0, width: 1, height: 1, extendsBeyondPreview: false,
  });
});

test("invalid boxes and preview regions do not produce misleading overlays", () => {
  const valid = [0.25, 0.25, 0.75, 0.75];
  for (const invalid of [
    [0.5, 0.25, 0.5, 0.75],
    [0.25, 0.75, 0.75, 0.25],
    [-0.1, 0.25, 0.75, 0.75],
    [0.25, 0.25, 1.1, 0.75],
    [NaN, 0.25, 0.75, 0.75],
    [0.25, 0.25, 0.75, Infinity],
  ]) {
    assert.equal(mapBboxToBoundaryPreview(invalid, valid), null);
    assert.equal(mapBboxToBoundaryPreview(valid, invalid), null);
  }
});

test("clean-pixel mode hides a transparent outline and respects submission busy state", () => {
  assert.match(source, /aria-label="Show boundary preview outline"\s+aria-pressed=\{showBoundaryOutline\}\s+disabled=\{busy\}/);
  assert.match(source, /onClick=\{\(\) => setShowBoundaryOutline\(\(current\) => !current\)\}/);
  assert.match(source, /boundaryImageReady && showBoundaryOutline && boundaryBox/);
  assert.match(source, /boxSizing: "border-box", border: "1px solid #f97316", background: "transparent"/);
  assert.match(source, /隐藏边框，查看干净像素/);
});

test("preview issues and boundary reminder do not add an approval or learning gate", () => {
  assert.match(source, /const approvalDisabled = busy \|\| !imageReady \|\| validationMessages\.length > 0;/);
  assert.match(source, /if \(!imageReady \|\| validationMessages\.length > 0\) return;/);
  const validationSource = source.slice(source.indexOf("const validationMessages ="), source.indexOf("const learningPayload ="));
  assert.doesNotMatch(validationSource, /boundaryPreview|boundaryImage|boundaryBox|showBoundaryOutline/);
  assert.match(source, /局部预览加载失败，请使用上方全图审核/);
  assert.match(source, /左、上、右、下四条边/);
  assert.match(source, /完整的可见外轮廓，包括尾部、果皮、圆顶和低对比度边缘/);
  assert.match(source, /除非题目明确只要求某个部分/);
  assert.equal(source.match(/type="checkbox"/g)?.length, 3);
  assert.ok(source.indexOf('aria-label="Boundary detail preview"') < source.indexOf("onClick={confirm}"));
});
