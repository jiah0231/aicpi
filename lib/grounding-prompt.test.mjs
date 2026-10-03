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
  for (const text of ["grounding_refine_box", "bounds filter only", "rankScore is geometry, not confidence", "original query", "owning object", "requested part", "axis/direction", "unresolved",
    "discovery order", "sensor alignment", "registration", "viewId", "view_pixels/view_normalized", "whole canvas",
    "partial update", "omitted fields/selection persist", "([] clears)", "complete status and evidence",
    "state.contract", "originalQuery", "queryCoverage", "plausible interpretation", "selectedCandidateId",
    "spatialOrder", "candidateSet", "visual_structure", "constraintAssessment",
    "status locked", "reconsidering", "new visible counterevidence", "no notebook update is required",
    "With identity and rank/relations established", "regions[i].modality", "viewId+region needs explicit coordinateSpace", "contract.interpretations[i].spatialOrder", "Waiting for approval is a valid pause", "grounding_evidence clarification", "without requiring a bbox",
    "boundary measurement after identity", "no obligatory crop/color/edge processing", "no_match", "sourceReuse/decisionCheckpoint",
    "limit=1", "targetCount=user-requested total", "without prefetch", "requestedLimitReached",
    "grounding_save_and_next", "grounding_save_result", "human approval", "contract inline", "requirements[].queryText", "all four edges", "Never approve on the user's behalf",
    "rejection keeps the same record", "revisions never auto-advance", "No ground truth is available",
    "independent visual inspection", "extra detectors or vision models", "Never pad small boxes"]) {
    assert.ok(compactGroundingContract.includes(text), `missing contract: ${text}`);
  }
});

test("optional trials follow actual uncertainty and observed outcomes, without a tool quota or forced confidence", () => {
  const decision = compactGroundingContract.split("\n\n").find((section) => section.startsWith("Decide:"));
  for (const rule of ["Decide directly from adequate pixels", "optional tools for uncertain conditions",
    "observable distinction", "no obligatory crop/color/edge processing, fixed tool cap or forced guess",
    "Trials may fail or be inconclusive", "useful/inconclusive/contradictory/failed", "update candidate support and uncertainty",
    "Tool calls/new images are not proof", "Change hypothesis/method across tools when justified",
    "no useful next observation", "best-supported proposal as unresolved", "submit for human review", "without requiring a bbox"])
    assert.ok(decision.includes(rule), rule);
  const feedback = compactGroundingContract.split("\n\n").find((section) => section.startsWith("Feedback:"));
  for (const rule of ["Optional state.lastTrial", "your assessment, not verification", "Piggyback on the next tool",
    "no extra call or long narration"]) assert.ok(feedback.includes(rule), rule);
  assert.ok(compactGroundingContract.includes("Establish counting-set membership and order before selecting"));
  assert.ok(compactGroundingContract.includes("axis/direction/ordinal"));
  assert.ok(compactGroundingContract.includes("Never mix unregistered modality coordinates/ranks"));
});

test("compact instructions expose inspection and candidate provenance fields from the registered schemas", async () => {
  const { createGroundingSafetyExtension } = await jiti.import("./grounding-safety-extension.ts");
  const registered = new Map(), handlers = new Map();
  createGroundingSafetyExtension({ cwd: process.cwd(), sessionId: "prompt-schema-test" }).factory({
    on(name, handler) { handlers.set(name, handler); }, registerTool(tool) { registered.set(tool.name, tool); },
    getActiveTools() { return []; }, setActiveTools() {},
  });
  const input = options();
  await handlers.get("before_agent_start")({ prompt: "批处理图像定位", systemPromptOptions: input });
  const rendered = buildSystemPrompt(input);
  assert.ok(rendered.includes(compactGroundingContract), "active extension must inject these decision rules");
  for (const name of ["grounding_view", "grounding_compare"]) {
    const schema = registered.get(name).parameters.properties;
    assert.deepEqual(Object.keys(schema.inspectionTest.properties), ["condition", "observable"]);
    assert.equal(schema.inspectionTest.properties.observable.minLength, 8);
    for (const intent of schema.inspectionIntent.anyOf.map((item) => item.const)) {
      assert.ok(compactGroundingContract.includes(intent), `${name}: missing ${intent}`);
    }
  }
  // Actual registration must allow direct inspection and optional previous-trial
  // feedback. Requiring a state/trial would turn the advisory into a new ritual.
  const evidenceState = registered.get("grounding_evidence").parameters.properties.state;
  const trial = evidenceState.properties.lastTrial;
  assert.ok(trial, "the advertised feedback field must exist in the live schema");
  assert.ok(!(evidenceState.required ?? []).includes("lastTrial"));
  assert.deepEqual(Object.keys(trial.properties), ["question", "outcome", "observation", "remainingUnknown", "nextObservation"]);
  assert.deepEqual(trial.properties.outcome.anyOf.map((item) => item.const), ["useful", "inconclusive", "contradictory", "failed"]);
  assert.ok(trial.properties.nextObservation.anyOf.some((item) => item.type === "null"));
  for (const name of ["grounding_view", "grounding_compare", "grounding_color_region", "grounding_process_image", "grounding_refine_box"]) {
    const schema = registered.get(name).parameters;
    assert.ok(!(schema.required ?? []).includes("state"), `${name}: feedback must be optional`);
    assert.deepEqual(schema.properties.state.properties.lastTrial, trial, `${name}: advertised piggyback path must use the same schema`);
    assert.ok(!(schema.required ?? []).includes("inspectionTest"), `${name}: a trial must not need a predeclared revisit test`);
  }
  const view = registered.get("grounding_view").parameters.properties;
  const compare = registered.get("grounding_compare").parameters.properties;
  assert.ok(view.region && view.zoom && view.reason);
  assert.ok(compare.regions && compare.reason);
  assert.equal(compare.region, undefined);
  assert.equal(compare.zoom, undefined);
  assert.ok(rendered.includes("grounding_view: region+reason, optional zoom; grounding_compare: regions[]+reason"));
  const candidate = registered.get("grounding_evidence").parameters.properties.state.properties.contract.properties.candidates.items;
  assert.equal(candidate.properties.measurementViewId.type, "string");
  assert.ok(rendered.includes("measurementViewId from an existing current-record visible view covering their full bbox"));
  assert.ok(rendered.includes("no image just to fill a field"));
  assert.ok(rendered.includes("inspectionTest {condition, observable}"));
  assert.ok(rendered.includes("condition = unresolved requirement ID or issue code (query before a contract, boundary when supported)"));
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
  const input = options({ selectedTools: ["read", "grounding_view", "grounding_refine_box", "grounding_third_party"],
    toolGuidelines: { read: [thirdParty], grounding_view: ["APP DUPLICATE GUIDELINE"], grounding_refine_box: ["APP DUPLICATE GUIDELINE"],
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
  const originalSelectedTools = [...input.selectedTools];
  const baselineLength = buildSystemPrompt(input).length;
  assert.equal(applyGroundingPromptOptions(input, "Job state: {\"targetCount\":1}"), true);
  const rendered = buildSystemPrompt(input);
  assert.ok(rendered.length < baselineLength * .5, `${rendered.length} compact vs ${baselineLength} original chars`);
  assert.match(rendered, /THIRD_PARTY_RULE_SURVIVES/);
  assert.deepEqual(input.selectedTools, originalSelectedTools, "prompt specialization must not alter tool availability");
  assert.equal(input.forceSystemPrompt, undefined);
});
