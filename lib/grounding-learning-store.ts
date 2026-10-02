import { createHash } from "node:crypto";
import { mkdir, readFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import lockfile from "proper-lockfile";
import { writePrivateFileAtomicSync } from "./atomic-file";
import { validateGenericLearningText, validateGroundingReviewLearning } from "./grounding-learning-validation";
import type { ConsolidationBatch, ConsolidationProposalInput, GenericLearningSource, GenericLearningStore, LearningRuleAction, LearningStoreOptions } from "./grounding-learning-consolidation-types";

const digest = (value: unknown) => createHash("sha256").update(JSON.stringify(value)).digest("hex");
const idValid = (value: unknown): value is string => typeof value === "string" && /^[a-f0-9]{64}$/.test(value);
export function groundingLearningStorePath(): string {
  return process.env.PI_WEB_GROUNDING_LEARNING_STORE_PATH || join(homedir(), ".pi", "agent", "grounding-learning-v3.json");
}
const empty = (): GenericLearningStore => ({ version: 3, revision: 0, sources: [], proposals: [], rules: [] });
function strictObject(value: unknown, fields: string[]): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value) || Object.keys(value).some(key => !fields.includes(key))) throw new Error("Invalid generic learning schema.");
  return value as Record<string, unknown>;
}
function source(value: unknown): GenericLearningSource {
  const row = strictObject(value, ["id", "text", "sampleIndependent", "origin"]);
  const normalized = validateGenericLearningText(row.text);
  if (typeof row.text !== "string" || row.text.length > 2400) throw new Error("Generic source exceeds its budget.");
  const text = row.text;
  if (row.sampleIndependent !== true || !["human", "v2"].includes(String(row.origin)) || row.id !== digest(normalized)) throw new Error("Invalid generic learning source.");
  return { id: row.id as string, text, sampleIndependent: true, origin: row.origin as "human" | "v2" };
}
export function validateConsolidationProposal(value: unknown, sourceIds: readonly string[]): ConsolidationProposalInput {
  const row = strictObject(value, ["sourceIds", "operation", "procedure", "note"]);
  if (!Array.isArray(row.sourceIds) || row.sourceIds.length < 1 || row.sourceIds.length > 8 || row.sourceIds.some(id => !idValid(id) || !sourceIds.includes(id)) || new Set(row.sourceIds).size !== row.sourceIds.length) throw new Error("Invalid generic proposal source references.");
  if (!["normalize", "possible_duplicate", "possible_conflict"].includes(String(row.operation))) throw new Error("Invalid generic proposal operation.");
  const raw = strictObject(row.procedure, ["category", "applicability", "error", "method", "check"]);
  // Use the same lexical checks, but do not attach a human attestation to a proposal.
  const checked = validateGroundingReviewLearning({ ...raw, sampleIndependent: true })!;
  const { sampleIndependent: _confirmation, ...procedure } = checked;
  void _confirmation;
  return { sourceIds: [...row.sourceIds].sort(), operation: row.operation as ConsolidationProposalInput["operation"], procedure, note: validateGenericLearningText(row.note, 4, 600) };
}
function parseStore(value: unknown): GenericLearningStore {
  const row = strictObject(value, ["version", "revision", "sources", "proposals", "rules"]);
  if (row.version !== 3 || !Number.isSafeInteger(row.revision) || Number(row.revision) < 0 || !Array.isArray(row.sources) || !Array.isArray(row.proposals) || !Array.isArray(row.rules)) throw new Error("Invalid generic learning store.");
  const sources = row.sources.map(source);
  const proposals = row.proposals.map(value => {
    const p = strictObject(value, ["id", "sourceIds", "operation", "procedure", "note", "status"]);
    if (!idValid(p.id) || !["pending", "activated", "dismissed"].includes(String(p.status))) throw new Error("Invalid generic proposal.");
    const { id, status, ...input } = p;
    const checked = validateConsolidationProposal(input, sources.map(source => source.id));
    if (id !== digest(checked)) throw new Error("Invalid generic proposal digest.");
    return { id: id as string, ...checked, status: status as "pending" | "activated" | "dismissed" };
  });
  const rules = row.rules.map(value => {
    const rule = strictObject(value, ["id", "revisions", "currentRevision", "enabled"]);
    if (!idValid(rule.id) || !Array.isArray(rule.revisions) || rule.revisions.length < 1 || typeof rule.enabled !== "boolean" || !Number.isSafeInteger(rule.currentRevision) || Number(rule.currentRevision) < 1 || Number(rule.currentRevision) > rule.revisions.length) throw new Error("Invalid generic rule.");
    return { id: rule.id, revisions: rule.revisions.map(value => {
      const result = validateGroundingReviewLearning(value);
      if (!result) throw new Error("Invalid generic rule revision.");
      return result;
    }), currentRevision: Number(rule.currentRevision), enabled: rule.enabled };
  });
  return { version: 3, revision: Number(row.revision), sources, proposals, rules };
}
export async function readLearningStore(options: LearningStoreOptions = {}): Promise<GenericLearningStore> {
  try { return parseStore(JSON.parse(await readFile(options.filePath ?? groundingLearningStorePath(), "utf8"))); }
  catch (error) { if ((error as NodeJS.ErrnoException).code === "ENOENT") return empty(); throw new Error("Generic learning store is unreadable or invalid."); }
}
async function mutate(options: LearningStoreOptions, fn: (store: GenericLearningStore) => void, canCommit?: () => boolean): Promise<GenericLearningStore> {
  const path = options.filePath ?? groundingLearningStorePath();
  await mkdir(dirname(path), { recursive: true });
  let compromised = false;
  const release = await lockfile.lock(path, { realpath: false, retries: { retries: 20, minTimeout: 10, maxTimeout: 100 }, stale: 30_000, onCompromised: () => { compromised = true; } });
  const checkCommit = () => {
    if (compromised || canCommit?.() === false) throw new Error("Generic learning commit was cancelled.");
  };
  try {
    const store = await readLearningStore(options);
    const before = JSON.stringify(store);
    checkCommit();
    fn(store);
    checkCommit();
    if (before === JSON.stringify(store)) return store;
    store.revision++;
    const validated = parseStore(store);
    checkCommit();
    writePrivateFileAtomicSync(path, JSON.stringify(validated, null, 2));
    return validated;
  } finally { await release(); }
}
export async function addGenericLearningSource(input: { text: string; sampleIndependent: true }, options: LearningStoreOptions = {}): Promise<GenericLearningStore> {
  const row = strictObject(input, ["text", "sampleIndependent"]);
  if (row.sampleIndependent !== true) throw new Error("Confirm the advice is sample-independent.");
  const normalized = validateGenericLearningText(row.text);
  if (typeof row.text !== "string" || row.text.length > 2400) throw new Error("Generic source exceeds its budget.");
  const text = row.text;
  const item: GenericLearningSource = { id: digest(normalized), text, sampleIndependent: true, origin: "human" };
  return mutate(options, store => { if (!store.sources.some(s => s.id === item.id)) store.sources.push(item); });
}
export async function listConfirmedGenericSources(options: LearningStoreOptions = {}): Promise<GenericLearningSource[]> {
  const store = await readLearningStore(options);
  // Dynamic import avoids a module initialization cycle with active retrieval.
  const { readGroundingLessons } = await import("./grounding-learning");
  const legacy = await readGroundingLessons(options.legacyPath);
  const result = [...store.sources];
  for (const item of legacy) {
    const text = [item.applicability, item.error, item.method, item.check].join("\n");
    if (text.length > 2400) continue;
    const projected: GenericLearningSource = { id: digest(text), text: validateGenericLearningText(text), sampleIndependent: true, origin: "v2" };
    if (!result.some(s => s.id === projected.id)) result.push(projected);
  }
  return result;
}
export async function snapshotConsolidationBatch(options: LearningStoreOptions & { excludeSourceIds?: string[]; maxItems?: number; maxChars?: number } = {}): Promise<ConsolidationBatch> {
  const all = await listConfirmedGenericSources(options);
  if (options.maxItems !== undefined && !Number.isSafeInteger(options.maxItems) || options.maxChars !== undefined && !Number.isSafeInteger(options.maxChars)) throw new Error("Invalid consolidation batch limits.");
  const limit = Math.min(8, Math.max(1, Math.floor(options.maxItems ?? 8)));
  const maxChars = Math.min(16000, Math.max(1, Math.floor(options.maxChars ?? 12000)));
  const sources: GenericLearningSource[] = [];
  let chars = 0;
  for (const item of all) {
    if (options.excludeSourceIds?.includes(item.id)) continue;
    if (sources.length >= limit) break;
    if (chars + item.text.length > maxChars) continue;
    sources.push(item); chars += item.text.length;
  }
  return { inputDigest: digest(sources), sources };
}
export async function saveConsolidationProposals(batch: ConsolidationBatch, inputs: ConsolidationProposalInput[], options: LearningStoreOptions = {}, canCommit?: () => boolean): Promise<GenericLearningStore> {
  const sources = batch.sources.map(source);
  if (batch.inputDigest !== digest(sources) || !Array.isArray(inputs) || inputs.length > 4) throw new Error("Invalid consolidation batch.");
  const current = await listConfirmedGenericSources(options);
  if (sources.some(s => !current.some(c => c.id === s.id))) throw new Error("Consolidation sources changed.");
  const proposals = inputs.map(p => validateConsolidationProposal(p, sources.map(s => s.id)));
  return mutate(options, store => {
    // Preserve the generic source snapshot behind each proposal, including v2
    // projections, without ever rewriting the original legacy file.
    for (const item of sources) {
      if (proposals.some(p => p.sourceIds.includes(item.id)) && !store.sources.some(s => s.id === item.id)) store.sources.push(item);
    }
    for (const input of proposals) {
      const id = digest(input);
      if (!store.proposals.some(p => p.id === id)) store.proposals.push({ id, ...input, status: "pending" });
    }
  }, canCommit);
}
export async function mutateLearningRule(action: LearningRuleAction, expectedRevision: number, options: LearningStoreOptions = {}): Promise<GenericLearningStore> {
  return mutate(options, store => {
    if (store.revision !== expectedRevision) throw new Error("Learning store revision conflict.");
    if (action.type === "activate" || action.type === "dismiss") {
      strictObject(action, action.type === "activate" ? ["type", "proposalId", "sampleIndependent", "procedure", "ruleId"] : ["type", "proposalId"]);
      const proposal = store.proposals.find(p => p.id === action.proposalId);
      if (!proposal || proposal.status !== "pending") throw new Error("Pending proposal not found.");
      if (action.type === "dismiss") { proposal.status = "dismissed"; return; }
      if (action.sampleIndependent !== true) throw new Error("Confirm the revised procedure is sample-independent.");
      const procedure = validateGroundingReviewLearning(action.procedure ?? { ...proposal.procedure, sampleIndependent: true })!;
      if (!procedure) throw new Error("Invalid generic procedure.");
      if (action.ruleId) {
        const rule = store.rules.find(r => r.id === action.ruleId);
        if (!rule) throw new Error("Generic rule not found.");
        rule.revisions.push(procedure); rule.currentRevision = rule.revisions.length;
      } else {
        const id = digest(procedure);
        const existing = store.rules.find(r => r.id === id);
        if (!existing) store.rules.push({ id, revisions: [procedure], currentRevision: 1, enabled: true });
      }
      proposal.status = "activated";
    } else {
      strictObject(action, action.type === "set_enabled" ? ["type", "ruleId", "enabled"] : ["type", "ruleId", "revision"]);
      const rule = store.rules.find(r => r.id === action.ruleId);
      if (!rule) throw new Error("Generic rule not found.");
      if (action.type === "set_enabled" && typeof action.enabled === "boolean") rule.enabled = action.enabled;
      else if (action.type === "rollback" && Number.isSafeInteger(action.revision) && action.revision >= 1 && action.revision <= rule.revisions.length) rule.currentRevision = action.revision;
      else throw new Error("Invalid generic rule action.");
    }
  });
}
export async function readActiveGenericLessons(options: LearningStoreOptions = {}) {
  const store = await readLearningStore(options);
  return store.rules.filter(r => r.enabled).map(r => ({ version: 2 as const, ...r.revisions[r.currentRevision - 1] }));
}
export function learningStoreOptionsForLegacyPath(learningPath?: string): LearningStoreOptions {
  return learningPath ? { legacyPath: learningPath, filePath: `${learningPath}.consolidation-v3.json` } : {};
}
