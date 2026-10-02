import type { GroundingReviewLearning } from "./types";

export type GenericProcedure = Omit<GroundingReviewLearning, "sampleIndependent">;
export interface GenericLearningSource {
  id: string;
  text: string;
  sampleIndependent: true;
  origin: "human" | "v2";
}
export interface GenericLearningProposal {
  id: string;
  sourceIds: string[];
  operation: "normalize" | "possible_duplicate" | "possible_conflict";
  procedure: GenericProcedure;
  note: string;
  status: "pending" | "activated" | "dismissed";
}
export interface GenericLearningRule {
  id: string;
  revisions: GroundingReviewLearning[];
  currentRevision: number;
  enabled: boolean;
}
export interface GenericLearningStore {
  version: 3;
  revision: number;
  sources: GenericLearningSource[];
  proposals: GenericLearningProposal[];
  rules: GenericLearningRule[];
}
export interface LearningStoreOptions { filePath?: string; legacyPath?: string }
export interface ConsolidationBatch {
  inputDigest: string;
  sources: GenericLearningSource[];
}
export interface ConsolidationProposalInput {
  sourceIds: string[];
  operation: GenericLearningProposal["operation"];
  procedure: GenericProcedure;
  note: string;
}
export interface ConsolidationModelOptions {
  provider: string;
  modelId: string;
  maxOutputTokens: number;
  timeoutMs: number;
  signal?: AbortSignal;
}
export type ConsolidationCall = (request: { system: string; text: string }, options: ConsolidationModelOptions) => Promise<string>;
export type LearningRuleAction =
  | { type: "activate"; proposalId: string; sampleIndependent: true; procedure?: GroundingReviewLearning; ruleId?: string }
  | { type: "dismiss"; proposalId: string }
  | { type: "set_enabled"; ruleId: string; enabled: boolean }
  | { type: "rollback"; ruleId: string; revision: number };
