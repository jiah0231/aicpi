import { dirname, resolve } from "node:path";

const MODALITIES = ["visible", "infrared", "depth"];
const TOKEN_FIELDS = ["input", "output", "cacheRead", "cacheWrite", "totalTokens"];
const COST_FIELDS = ["input", "output", "cacheRead", "cacheWrite", "total"];

function nonemptyString(value) {
  return typeof value === "string" && value.trim().length > 0 && !value.includes("\0");
}

/** Build the model-visible record, never copying annotations or arbitrary fields. */
export function sanitizeBenchmarkRecord(record, queryPath) {
  if (!record || typeof record !== "object" || Array.isArray(record)) {
    throw new TypeError("A benchmark record must be an object.");
  }
  if (!nonemptyString(queryPath)) throw new TypeError("queryPath must be a nonempty file path.");
  if (!nonemptyString(record.query)) throw new TypeError("A benchmark record requires a nonempty query.");
  if (!nonemptyString(record.visible)) throw new TypeError("A benchmark record requires a visible image path.");
  const safe = { query: record.query };
  const sourceDirectory = dirname(resolve(queryPath));
  for (const modality of MODALITIES) {
    const value = record[modality];
    if (value == null || (typeof value === "string" && value.trim() === "")) continue;
    if (!nonemptyString(value)) throw new TypeError(`${modality} must be an image path string.`);
    safe[modality] = resolve(sourceDirectory, value);
  }
  return safe;
}

function validBbox(value) {
  return Array.isArray(value) && value.length === 4
    // Array#every skips holes, so explicitly access each coordinate.
    && [0, 1, 2, 3].every((index) => typeof value[index] === "number"
      && Number.isFinite(value[index]) && value[index] >= 0 && value[index] <= 1)
    && value[0] < value[2] && value[1] < value[3];
}

/** Normalized source xyxy only: never clamp, reorder, or infer coordinate units. */
export function normalizedBboxIou(prediction, reference) {
  if (!validBbox(prediction) || !validBbox(reference)) return null;
  const pw = prediction[2] - prediction[0];
  const ph = prediction[3] - prediction[1];
  const rw = reference[2] - reference[0];
  const rh = reference[3] - reference[1];
  const iw = Math.max(0, Math.min(prediction[2], reference[2]) - Math.max(prediction[0], reference[0]));
  const ih = Math.max(0, Math.min(prediction[3], reference[3]) - Math.max(prediction[1], reference[1]));
  const directIntersection = iw * ih;
  const directUnion = pw * ph + rw * rh - directIntersection;
  if (directUnion > 0) return Math.min(1, Math.max(0, directIntersection / directUnion));
  // Axis scaling avoids 0/0 when two legitimate extremely small boxes underflow.
  const sx = Math.max(pw, rw);
  const sy = Math.max(ph, rh);
  const intersection = (iw / sx) * (ih / sy);
  const union = (pw / sx) * (ph / sy) + (rw / sx) * (rh / sy) - intersection;
  return Math.min(1, Math.max(0, intersection / union));
}

function nonnegativeFinite(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function average(values) {
  if (values.length === 0) return null;
  // Divide before adding to avoid overflowing when the finite values are large.
  return values.reduce((sum, value) => sum + value / values.length, 0);
}

function latencySummary(values) {
  const sorted = values.filter(nonnegativeFinite).sort((a, b) => a - b);
  const count = sorted.length;
  if (!count) return { count: 0, mean: null, median: null, p95: null };
  const midpoint = Math.floor(count / 2);
  return {
    count,
    mean: average(sorted),
    median: count % 2 ? sorted[midpoint] : sorted[midpoint - 1] / 2 + sorted[midpoint] / 2,
    // Nearest-rank percentile: no interpolation that hides the slowest small-run case.
    p95: sorted[Math.ceil(count * .95) - 1],
  };
}

/**
 * Results: {status, prediction, reference, predictionStatus?, elapsedMs?,
 * toolElapsedMs?, providerElapsedMs?, usage?}.
 *
 * Main metrics include EVERY valid reference, including timeout/error/no-box cases
 * as IoU 0. An unresolved proposal is still scored geometrically and also counted
 * separately; confidence is not ground truth. Cases without a valid reference
 * never enter an accuracy denominator. Metrics are fractions, not percentages.
 */
export function summarizeBenchmarkResults(results) {
  if (!Array.isArray(results)) throw new TypeError("Benchmark results must be an array.");
  const summary = {
    total: results.length, proposals: 0, missingPredictions: 0, invalidPredictions: 0,
    failures: 0, timeouts: 0, settledWithoutProposal: 0, unresolved: 0, lowConfidence: 0,
    noGroundTruth: 0, invalidReferences: 0, validReferences: 0,
    scoredProposals: 0, scoredMisses: 0,
    meanIoU: null, accuracyAt50: null, accuracyAt75: null, proposalOnlyMeanIoU: null,
    statusCounts: {},
    latency: {},
    usage: Object.fromEntries(TOKEN_FIELDS.map((field) => [field, 0])),
  };
  summary.usage.cost = Object.fromEntries(COST_FIELDS.map((field) => [field, 0]));
  const ious = [];
  const proposalIous = [];
  const timings = { elapsedMs: [], toolElapsedMs: [], providerElapsedMs: [] };
  const statusCounts = new Map();
  for (const item of results) {
    const result = item && typeof item === "object" && !Array.isArray(item) ? item : {};
    const status = typeof result.status === "string" && result.status ? result.status : "unknown";
    statusCounts.set(status, (statusCounts.get(status) ?? 0) + 1);
    if (status === "error") summary.failures++;
    if (status === "timeout") summary.timeouts++;
    if (status === "settled_without_proposal") summary.settledWithoutProposal++;
    if (result.predictionStatus === "unresolved") summary.unresolved++;
    if (result.predictionStatus === "low_confidence") summary.lowConfidence++;

    const hasPrediction = validBbox(result.prediction);
    if (hasPrediction) summary.proposals++;
    else if (result.prediction == null) summary.missingPredictions++;
    else summary.invalidPredictions++;

    if (validBbox(result.reference)) {
      summary.validReferences++;
      const iou = hasPrediction ? normalizedBboxIou(result.prediction, result.reference) : 0;
      ious.push(iou);
      if (hasPrediction) {
        summary.scoredProposals++;
        proposalIous.push(iou);
      } else summary.scoredMisses++;
    } else if (result.reference == null) summary.noGroundTruth++;
    else summary.invalidReferences++;

    for (const field of Object.keys(timings)) timings[field].push(result[field]);
    for (const field of TOKEN_FIELDS) {
      if (nonnegativeFinite(result.usage?.[field])) summary.usage[field] += result.usage[field];
    }
    if (nonnegativeFinite(result.usage?.cost)) summary.usage.cost.total += result.usage.cost;
    else for (const field of COST_FIELDS) {
      if (nonnegativeFinite(result.usage?.cost?.[field])) summary.usage.cost[field] += result.usage.cost[field];
    }
  }
  summary.statusCounts = Object.fromEntries(statusCounts);
  summary.meanIoU = average(ious);
  summary.proposalOnlyMeanIoU = average(proposalIous);
  if (ious.length) {
    summary.accuracyAt50 = ious.filter((iou) => iou >= .5).length / ious.length;
    summary.accuracyAt75 = ious.filter((iou) => iou >= .75).length / ious.length;
  }
  summary.latency = Object.fromEntries(Object.entries(timings).map(([name, values]) => [name, latencySummary(values)]));
  return summary;
}
