import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url, {
  jsx: { runtime: "automatic" },
  tsconfigPaths: true,
});
const { mapBboxToBoundaryPreview, prepareGroundingReviewLearning } = await jiti.import("./GroundingReviewPanel.tsx");

const source = await readFile(new URL("./GroundingReviewPanel.tsx", import.meta.url), "utf8");
const typesSource = await readFile(new URL("../lib/types.ts", import.meta.url), "utf8");

const authoredLesson = {
  category: "uncertainty",
  applicability: "When required evidence is unavailable",
  error: "Treating a plausible guess as established evidence",
  method: "List the missing evidence and retain an unresolved result until it can be checked.",
  check: "Check that every claimed conclusion has observable support.",
  sampleIndependent: true,
};

test("grounding review exposes a collapsed opt-in form for newly authored general procedures", () => {
  assert.match(source, /<details[^>]*>\s*<summary[^>]*>通用经验（可选，新写的方法）/);
  assert.match(source, /\[rememberLearning, setRememberLearning\] = useState\(false\)/);
  assert.match(source, /\[learningSampleIndependent, setLearningSampleIndependent\] = useState\(false\)/);
  for (const field of ["Applicability", "Error", "Method", "Check"]) {
    assert.match(source, new RegExp(`\\[learning${field}, setLearning${field}\\] = useState\\(""\\)`));
    assert.match(source, new RegExp(`aria-label="Learning ${field.toLowerCase()}"`));
  }
  assert.match(source, /aria-label="Learning category"/);
  assert.match(source, /aria-label="Save general grounding lesson"/);
  assert.match(source, /aria-label="Confirm sample-independent lesson"/);
  assert.match(source, /<fieldset disabled=\{!rememberLearning\}/);
  assert.doesNotMatch(source, /learningAdvice|learningScope|GroundingLearningScope|Learning scope/);
  assert.match(source, /\.\.\.learningPayload\(\)/);
  assert.match(source, /不会自动填入样本内容/);
  assert.match(source, /不含原始问题或其改写、图像内容或路径、样本 ID、具体答案、坐标框或标准答案/);
  assert.match(source, /长度与词面校验不能证明语义上与样本无关/);
});

test("authored lesson fields and category edits require a fresh human acknowledgment", () => {
  for (const field of ["Applicability", "Error", "Method", "Check", "Category"]) {
    assert.match(source, new RegExp(`setLearning${field}\\(event\\.target\\.value(?: as GroundingLearningCategory)?\\); setLearningSampleIndependent\\(false\\);`));
  }
  assert.match(source, /setRememberLearning\(event\.target\.checked\); setLearningSampleIndependent\(false\);/);
  const preparationSource = source.slice(source.indexOf("export function prepareGroundingReviewLearning"), source.indexOf("export function GroundingReviewPanel"));
  assert.doesNotMatch(preparationSource, /details|query|rejectionReason|image|bbox|groundTruth/);
});

test("disabled and wholly empty lessons never block or add to annotation submissions", () => {
  assert.deepEqual(prepareGroundingReviewLearning(false, authoredLesson), {});
  assert.deepEqual(prepareGroundingReviewLearning(false, { ...authoredLesson, method: "x", sampleIndependent: false }), {});
  for (const sampleIndependent of [false, true]) {
    assert.deepEqual(prepareGroundingReviewLearning(true, {
      category: "boundary", applicability: "  ", error: "\n", method: "", check: "\t", sampleIndependent,
    }), {});
  }
});

test("a valid authored lesson submits only the general fields after explicit acknowledgment", () => {
  assert.deepEqual(prepareGroundingReviewLearning(true, authoredLesson), { learning: authoredLesson });
  assert.deepEqual(prepareGroundingReviewLearning(true, {
    ...authoredLesson, applicability: ` ${authoredLesson.applicability} `,
    query: "Not a source for the lesson", reason: "Not copied", bbox: [0, 0, 1, 1],
  }), { learning: authoredLesson });
  for (const sampleIndependent of [false, undefined, "true"]) {
    const unacknowledged = prepareGroundingReviewLearning(true, { ...authoredLesson, sampleIndependent });
    assert.equal(unacknowledged.learning, undefined);
    assert.match(unacknowledged.error, /人工确认/);
    assert.match(unacknowledged.error, /关闭跨会话保存后继续审核/);
  }
});

test("partially authored and out-of-range lessons ask the author to finish or disable saving", () => {
  for (const [field, min, max] of [
    ["applicability", 4, 240], ["error", 4, 400], ["method", 8, 800], ["check", 4, 400],
  ]) {
    for (const length of [0, min - 1, max + 1]) {
      const prepared = prepareGroundingReviewLearning(true, { ...authoredLesson, [field]: "x".repeat(length) });
      assert.equal(prepared.learning, undefined);
      assert.match(prepared.error, /补全通用经验/);
      assert.match(prepared.error, /关闭跨会话保存后继续审核/);
    }
    for (const length of [min, max]) {
      const draft = { ...authoredLesson, [field]: "界".repeat(length) };
      assert.deepEqual(prepareGroundingReviewLearning(true, draft), { learning: draft });
    }
  }
  assert.match(source, /if \(learningValidationError\) \{\s*messages\.push\(learningValidationError\);/);
  assert.match(source, /const reject = \(\) => \{\s*if \(learningInvalid\) return;/);
  assert.match(source, /disabled=\{busy \|\| learningInvalid\} onClick=\{reject\}/);
  assert.match(source, /const learningPayload = \(\) => preparedLearning\.learning\s*\? \{ learning: preparedLearning\.learning \}/);
});

test("sample-bearing text is rejected visibly before submitting review and can be disabled", () => {
  for (const method of ['Copied "query": "the source sentence"', "An explicit answer: the red switch", "Use bbox: [0.1,0.2,0.3,0.4] here."]) {
    const draft = { ...authoredLesson, method };
    const prepared = prepareGroundingReviewLearning(true, draft);
    assert.equal(prepared.learning, undefined);
    assert.match(prepared.error, /通用经验未通过校验/);
    assert.match(prepared.error, /关闭跨会话保存后继续审核/);
    assert.deepEqual(prepareGroundingReviewLearning(false, draft), {});
  }
  assert.match(source, /validateGroundingReviewLearning.*grounding-learning-validation/);
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
  assert.equal(source.match(/type="checkbox"/g)?.length, 4);
  assert.ok(source.indexOf('aria-label="Boundary detail preview"') < source.indexOf("onClick={confirm}"));
});

test("proposal geometry follows editable coordinates and remains explicitly advisory", () => {
  assert.match(source, /const proposalGeometry = isValidBbox\(bbox\) && details\.modelContract/);
  assert.match(source, /assessGroundingProposalGeometry\(details\.modelContract\.candidates, details\.modelContract\.selectedCandidateId, bbox\)/);
  assert.match(source, /proposalGeometry\.selectedBoxCoverage/);
  assert.match(source, /proposalGeometry\.proposalInsideSelectedBox/);
  assert.match(source, /proposalGeometry\.otherCandidateOverlaps/);
  assert.match(source, /遮挡可正常重叠/);
  assert.match(source, /声明框本身错误也可能显示 100%/);
});

test("model reasons are labelled as proposals and unresolved previews never claim verification", async () => {
  assert.match(source, /t\("groundingReview\.modelReason"\)/);
  assert.match(source, /unresolvedChecks && <div>\{t\("groundingReview\.unresolvedPreviewNote"\)\}/);
  assert.match(source, /value=\{reason\} onChange=\{\(event\) => setReason\(event\.target\.value\)\}/, "preserve editable model text rather than censoring claims");
  for (const locale of ["en", "zh-CN", "zh-TW"]) {
    const { [locale === "en" ? "enLocale" : locale === "zh-CN" ? "zhCNLocale" : "zhTWLocale"]: plugin } = await jiti.import(`../lib/i18n/messages/${locale}.ts`);
    for (const key of ["groundingReview.modelReason", "groundingReview.unresolvedPreviewNote"]) assert.ok(plugin.messages[key]?.length > 8, `${locale}: ${key}`);
    if (locale === "en") assert.match(plugin.messages["groundingReview.unresolvedPreviewNote"], /does not verify missing identity, rank, or pose evidence/);
  }
});
