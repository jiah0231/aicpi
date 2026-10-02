import assert from "node:assert/strict";
import test from "node:test";
import { createJiti } from "jiti";

const jiti = createJiti(import.meta.url);
const { applyGroundingPromptOptions, compactGroundingContract, GROUNDING_PROMPT_SECTION,
  GROUNDING_RETAINED_GUIDELINES_SECTION, GROUNDING_PROMPT_TOOL_NAMES } = await jiti.import("./grounding-prompt.ts");
// Exercise the installed SDK renderer, not a locally invented approximation.
const { normalizeBuildSystemPromptOptions, buildSystemPromptSections, buildSystemPromptState, buildSystemPrompt } =
  await import(new URL("./core/system-prompt.js", import.meta.resolve("@earendil-works/pi-coding-agent")));

function options(extra = {}) {
  return normalizeBuildSystemPromptOptions({ cwd: "D:\\grounding-workspace", selectedTools: [...GROUNDING_PROMPT_TOOL_NAMES], ...extra });
}

test("compact grounding contract covers the domain safeguards within a bounded instruction budget", () => {
  assert.ok(compactGroundingContract.length >= 3000, compactGroundingContract.length);
  assert.ok(compactGroundingContract.length <= 5200, compactGroundingContract.length);
  for (const text of ["original query", "owning object", "requested part", "axis/direction", "unresolved",
    "discovery order", "sensor alignment", "registration", "viewId", "view_pixels/view_normalized", "whole canvas",
    "partial update", "omitted fields/selection persist", "([] clears)", "complete status and evidence",
    "state.contract", "originalQuery/queryCoverage", "plausible interpretation", "selectedCandidateId",
    "spatialOrder", "candidateSet", "visual_structure", "constraintAssessment",
    "status locked", "reconsidering", "new visible counterevidence", "No notebook update is required",
    "If more observations are needed", "Waiting for approval is a valid pause",
    "optional boundary measurement", "no obligatory crop", "no_match", "sourceReuse/decisionCheckpoint",
    "limit=1", "targetCount=user-requested total", "without prefetch", "requestedLimitReached",
    "grounding_save_and_next", "grounding_save_result", "human approval", "Never approve on the user's behalf",
    "rejection keeps the same record", "revisions never auto-advance", "No ground truth is available",
    "independent visual inspection", "extra detectors or vision models", "Never pad small boxes"]) {
    assert.ok(compactGroundingContract.includes(text), `missing contract: ${text}`);
  }
});

test("SDK keeps the specialized prompt structured, without an exact prompt override", () => {
  const input = options();
  assert.equal(applyGroundingPromptOptions(input, "Job state: {\"targetCount\":3}"), true);
  assert.equal(input.forceSystemPrompt, undefined);
  const state = buildSystemPromptState(input);
  assert.equal(state.content, "");
  assert.match(state.sections.preamble, /visual grounding specialist/);
  assert.equal(state.sections.tools, undefined);
  assert.equal(state.sections.rules, undefined);
  assert.equal(state.sections.docs, undefined);
  assert.ok(state.sections[GROUNDING_PROMPT_SECTION].includes(compactGroundingContract));
  assert.match(state.sections[GROUNDING_PROMPT_SECTION], /Job state: \{"targetCount":3\}/);
  assert.equal(state.sections.cwd, "<cwd>\nD:/grounding-workspace\n</cwd>");
});

test("sparse options apply the same contract without inventing collection defaults", () => {
  for (const input of [{ sections: {} }, {}, { selectedTools: ["third_party"] },
    { sections: { user_rules: "USER\r\n " }, promptGuidelines: ["KEEP USER RULE"] },
    { selectedTools: ["third_party"], toolGuidelines: { third_party: ["KEEP TOOL RULE"] } }]) {
    const before = structuredClone(input);
    assert.equal(applyGroundingPromptOptions(input, "SPARSE RUNTIME"), true);
    assert.match(input.customPrompt, /visual grounding specialist/);
    assert.ok(input.sections[GROUNDING_PROMPT_SECTION].includes(compactGroundingContract));
    assert.ok(input.sections[GROUNDING_PROMPT_SECTION].endsWith("SPARSE RUNTIME"));
    for (const field of ["selectedTools", "toolGuidelines", "promptGuidelines"]) {
      assert.deepEqual(input[field], before[field]);
      assert.equal(Object.hasOwn(input, field), Object.hasOwn(before, field));
    }
    if (before.sections?.user_rules) assert.equal(input.sections.user_rules, before.sections.user_rules);
    const rendered = buildSystemPrompt({ ...input, cwd: "D:/grounding-workspace" });
    assert.match(rendered, /visual grounding specialist/);
    for (const rule of [...(before.promptGuidelines ?? []), ...(before.toolGuidelines?.third_party ?? [])]) {
      assert.ok(rendered.includes(rule));
    }
  }
});

test("SDK preserves user addendum, project files, skills and all custom sections byte-for-byte", () => {
  const input = options({
    selectedTools: ["read", ...GROUNDING_PROMPT_TOOL_NAMES],
    appendSystemPrompt: "  USER ADDENDUM\r\n第二行\t ",
    contextFiles: [{ path: "C:/project/AGENTS.md", content: "  USER PROJECT\r\nKeep spacing.\t\n" },
      { path: "C:/project/extra.md", content: "ANOTHER FILE\n" }],
    sections: { user_section: "  USER SECTION\r\n ", rules: "USER RULES OVERRIDE\n", docs: "USER DOCS\t", tools: "USER TOOL NOTES " },
    skills: [{ name: "domain-user-skill", description: "USER SKILL", filePath: "C:/skills/domain/SKILL.md",
      baseDir: "C:/skills/domain", sourceInfo: { source: "user" }, disableModelInvocation: false }],
  });
  const before = structuredClone(input);
  const beforeSections = buildSystemPromptSections(input);
  const runtime = "  Current dataset: C:/data/queries.json\r\nJob state: {\"currentKey\":\"k2\"}\t ";
  assert.equal(applyGroundingPromptOptions(input, runtime), true);
  for (const field of ["appendSystemPrompt", "contextFiles", "skills", "selectedTools", "toolSnippets", "toolGuidelines", "promptGuidelines"]) {
    assert.deepEqual(input[field], before[field], `${field} changed`);
  }
  for (const [name, value] of Object.entries(before.sections)) assert.equal(input.sections[name], value);
  const afterSections = buildSystemPromptSections(input);
  for (const name of ["addendum", "project_context", "skills", "user_section", "rules", "docs", "tools", "cwd"]) {
    assert.equal(afterSections[name], beforeSections[name], `SDK-rendered ${name} changed`);
  }
  assert.ok(input.sections[GROUNDING_PROMPT_SECTION].endsWith(runtime));
});

test("selected third-party guidelines and prompt rules survive suppression of SDK rules unchanged", () => {
  const thirdParty = "  THIRD PARTY\r\nKeep all spaces.\t ";
  const prefixLookalike = "THIRD PARTY grounding_* TOOL RULE";
  const userRule = "  USER RULE mentions grounding_view but is not app-owned\r\n ";
  const input = options({ selectedTools: ["read", "grounding_view", "grounding_third_party"],
    toolGuidelines: { read: [thirdParty], grounding_view: ["APP DUPLICATE GUIDELINE"],
      grounding_third_party: [prefixLookalike], inactive_tool: ["INACTIVE RULE"] },
    promptGuidelines: [userRule, userRule, ""],
  });
  const before = structuredClone(input);
  assert.equal(applyGroundingPromptOptions(input, ""), true);
  const preserved = input.sections[GROUNDING_RETAINED_GUIDELINES_SECTION];
  assert.equal(preserved, `Tool read:\n${thirdParty}\n\nTool grounding_third_party:\n${prefixLookalike}\n\nAdditional instructions:\n${[userRule, userRule, ""].join("\n")}`);
  const rendered = buildSystemPrompt(input);
  for (const rule of [thirdParty, prefixLookalike, userRule]) assert.ok(rendered.includes(rule));
  assert.doesNotMatch(rendered, /APP DUPLICATE GUIDELINE|INACTIVE RULE/);
  assert.deepEqual(input.toolGuidelines, before.toolGuidelines);
  assert.deepEqual(input.promptGuidelines, before.promptGuidelines);
});

test("explicit custom or forced prompts are never changed, including empty strings", () => {
  for (const choice of [{ customPrompt: "USER PREFIX\n" }, { customPrompt: "" },
    { forceSystemPrompt: "USER EXACT PROMPT\r\n" }, { forceSystemPrompt: "" }]) {
    const input = options({ ...choice, sections: { user_rules: "KEEP" }, promptGuidelines: ["KEEP TOO"] });
    const before = structuredClone(input);
    const rendered = buildSystemPrompt(input);
    assert.equal(applyGroundingPromptOptions(input, "new runtime"), false);
    assert.deepEqual(input, before);
    assert.equal(buildSystemPrompt(input), rendered);
  }
});

test("only the app-owned safety section is replaced; an unrelated section collision leaves options intact", () => {
  const collision = options({ sections: { [GROUNDING_RETAINED_GUIDELINES_SECTION]: "USER OWNED\r\n" }, promptGuidelines: ["preserve me"] });
  const before = structuredClone(collision);
  assert.equal(applyGroundingPromptOptions(collision, "runtime"), false);
  assert.deepEqual(collision, before);
  const legacy = options({ sections: { [GROUNDING_PROMPT_SECTION]: "OLD APP CONTRACT", user_rules: "USER OWNED\r\n" } });
  assert.equal(applyGroundingPromptOptions(legacy, "runtime"), true);
  assert.doesNotMatch(legacy.sections[GROUNDING_PROMPT_SECTION], /OLD APP CONTRACT/);
  assert.equal(legacy.sections.user_rules, "USER OWNED\r\n");
  const input = options({ sections: { [GROUNDING_RETAINED_GUIDELINES_SECTION]: "already here" } });
  assert.equal(applyGroundingPromptOptions(input, ""), true);
  assert.equal(input.sections[GROUNDING_RETAINED_GUIDELINES_SECTION], "already here");
  const after = structuredClone(input);
  assert.equal(applyGroundingPromptOptions(input, "different runtime"), false);
  assert.deepEqual(input, after);
});

test("real grounding tool guidelines compact substantially under the installed SDK renderer", async () => {
  const { createGroundingSafetyExtension } = await jiti.import("./grounding-safety-extension.ts");
  const registered = new Map();
  createGroundingSafetyExtension({ cwd: process.cwd(), sessionId: "prompt-unit-test" }).factory({
    on() {}, registerTool(tool) { registered.set(tool.name, tool); }, getActiveTools() { return []; }, setActiveTools() {},
  });
  assert.deepEqual([...registered.keys()].sort(), [...GROUNDING_PROMPT_TOOL_NAMES].sort(), "explicit app ownership must match registered tools");
  const input = options({
    toolSnippets: Object.fromEntries([...registered].map(([name, tool]) => [name, tool.promptSnippet ?? ""])),
    toolGuidelines: Object.fromEntries([...registered].map(([name, tool]) => [name, tool.promptGuidelines ?? []])),
    promptGuidelines: ["THIRD_PARTY_RULE_SURVIVES"],
  });
  const baselineLength = buildSystemPrompt(input).length;
  assert.equal(applyGroundingPromptOptions(input, "Job state: {\"targetCount\":1}"), true);
  const rendered = buildSystemPrompt(input);
  assert.ok(rendered.length < baselineLength * .5, `${rendered.length} compact vs ${baselineLength} original chars`);
  assert.match(rendered, /THIRD_PARTY_RULE_SURVIVES/);
  assert.equal(input.selectedTools.length, 9, "prompt specialization must not alter tool availability");
  assert.equal(input.forceSystemPrompt, undefined);
});
