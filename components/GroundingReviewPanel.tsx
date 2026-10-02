"use client";
/* eslint-disable @next/next/no-img-element -- review images are data URLs produced by the grounding runtime. */

import { useMemo, useRef, useState } from "react";
import { validateGroundingReviewLearning } from "@/lib/grounding-learning-validation";
import type {
  ExtensionUiRequest,
  GroundingLearningCategory,
  GroundingReviewDetails,
  GroundingReviewLearning,
} from "@/lib/types";

export type GroundingReviewRequest = Extract<ExtensionUiRequest, { method: "custom" }> & {
  details: GroundingReviewDetails;
};

type Props = {
  request: GroundingReviewRequest;
  onInput: (request: GroundingReviewRequest, data: string) => void | Promise<void>;
};

type Bbox = [number, number, number, number];

function asBbox(value: readonly number[]): Bbox {
  return [value[0] ?? 0, value[1] ?? 0, value[2] ?? 1, value[3] ?? 1];
}

function formatCoordinate(value: number): string {
  return Number.isFinite(value) ? value.toFixed(4) : "—";
}

function isValidBbox(bbox: Bbox): boolean {
  return bbox.every((value) => Number.isFinite(value) && value >= 0 && value <= 1)
    && bbox[0] < bbox[2]
    && bbox[1] < bbox[3];
}

export function mapBboxToBoundaryPreview(bbox: Bbox, region: Bbox) {
  if (!isValidBbox(region) || !isValidBbox(bbox)) return null;
  const regionWidth = region[2] - region[0];
  const regionHeight = region[3] - region[1];
  // Preserve the source box, including edges outside this fixed crop. The
  // preview container clips the drawing only, never the submitted coordinates.
  return {
    left: (bbox[0] - region[0]) / regionWidth,
    top: (bbox[1] - region[1]) / regionHeight,
    width: (bbox[2] - bbox[0]) / regionWidth,
    height: (bbox[3] - bbox[1]) / regionHeight,
    extendsBeyondPreview: bbox[0] < region[0] || bbox[1] < region[1]
      || bbox[2] > region[2] || bbox[3] > region[3],
  };
}

type GroundingLearningDraft = Omit<GroundingReviewLearning, "sampleIndependent"> & {
  sampleIndependent: boolean;
};

export function prepareGroundingReviewLearning(
  remember: boolean,
  draft: GroundingLearningDraft,
): { learning?: GroundingReviewLearning; error?: string } {
  if (!remember) return {};
  const applicability = draft.applicability.trim();
  const error = draft.error.trim();
  const method = draft.method.trim();
  const check = draft.check.trim();
  if (![applicability, error, method, check].some(Boolean)) return {};

  if (applicability.length < 4 || applicability.length > 240
    || error.length < 4 || error.length > 400
    || method.length < 8 || method.length > 800
    || check.length < 4 || check.length > 400) {
    return { error: "请补全通用经验：适用条件 4–240 字符、常见错误 4–400 字符、改进方法 8–800 字符、复查步骤 4–400 字符；或关闭跨会话保存后继续审核。" };
  }
  if (draft.sampleIndependent !== true) {
    return { error: "请人工确认新写的经验与具体样本无关；或关闭跨会话保存后继续审核。" };
  }
  // Length checks and lexical validation cannot establish semantic independence.
  // The author must review the newly written procedure and explicitly attest it.
  try {
    return { learning: validateGroundingReviewLearning({ category: draft.category, applicability, error, method, check, sampleIndependent: true }) };
  } catch (validationError) {
    return { error: `通用经验未通过校验：${validationError instanceof Error ? validationError.message : "请检查内容"} 可修改经验，或关闭跨会话保存后继续审核。` };
  }
}

export function GroundingReviewPanel({ request, onInput }: Props) {
  const details = request.details;
  const initialBbox = asBbox(details.bbox);
  const [bbox, setBbox] = useState<Bbox>(initialBbox);
  const [status, setStatus] = useState(details.status);
  const [targetFound, setTargetFound] = useState(details.targetFound);
  const [candidateCount, setCandidateCount] = useState(details.candidateCount);
  const [candidateRank, setCandidateRank] = useState<number | undefined>(details.candidateRank);
  const [confidence, setConfidence] = useState(details.confidence);
  const [reason, setReason] = useState(details.reason);
  const [constraintsResolved, setConstraintsResolved] = useState(false);
  const unresolvedChecks = details.constraintAssessment !== undefined && !details.constraintAssessment.canLock;
  const [rejectionReason, setRejectionReason] = useState("");
  const [learningCategory, setLearningCategory] = useState<GroundingLearningCategory>("other");
  const [learningApplicability, setLearningApplicability] = useState("");
  const [learningError, setLearningError] = useState("");
  const [learningMethod, setLearningMethod] = useState("");
  const [learningCheck, setLearningCheck] = useState("");
  const [learningSampleIndependent, setLearningSampleIndependent] = useState(false);
  const [rememberLearning, setRememberLearning] = useState(false);
  const [submission, setSubmission] = useState<"confirm" | "reject" | "submitted" | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  // State alone does not guard two clicks delivered before React re-renders.
  const submissionLocked = useRef(false);
  const [loadedImage, setLoadedImage] = useState<string | null>(null);
  const [failedImage, setFailedImage] = useState<string | null>(null);
  const [loadedBoundaryImage, setLoadedBoundaryImage] = useState<string | null>(null);
  const [failedBoundaryImage, setFailedBoundaryImage] = useState<string | null>(null);
  const [showBoundaryOutline, setShowBoundaryOutline] = useState(true);
  const imageUrl = `data:${details.image.mimeType};base64,${details.image.data}`;
  const imageReady = loadedImage === imageUrl && failedImage !== imageUrl;
  const busy = submission !== null;
  const preparedLearning = prepareGroundingReviewLearning(rememberLearning, {
    category: learningCategory,
    applicability: learningApplicability,
    error: learningError,
    method: learningMethod,
    check: learningCheck,
    sampleIndependent: learningSampleIndependent,
  });
  const learningValidationError = preparedLearning.error;
  const learningInvalid = learningValidationError !== undefined;

  const width = details.image.originalWidth || details.image.width || 1;
  const height = details.image.originalHeight || details.image.height || 1;
  const boundaryPreview = details.boundaryPreview;
  const boundaryWidth = boundaryPreview?.image.width || boundaryPreview?.image.originalWidth || 0;
  const boundaryHeight = boundaryPreview?.image.height || boundaryPreview?.image.originalHeight || 0;
  const boundaryPreviewValid = boundaryPreview !== undefined && isValidBbox(boundaryPreview.region)
    && Number.isFinite(boundaryWidth) && boundaryWidth > 0
    && Number.isFinite(boundaryHeight) && boundaryHeight > 0;
  const boundaryImageUrl = boundaryPreviewValid
    ? `data:${boundaryPreview.image.mimeType};base64,${boundaryPreview.image.data}`
    : null;
  const boundaryImageReady = boundaryImageUrl !== null
    && loadedBoundaryImage === boundaryImageUrl && failedBoundaryImage !== boundaryImageUrl;
  const boundaryBox = boundaryPreviewValid ? mapBboxToBoundaryPreview(bbox, boundaryPreview.region) : null;
  const pixelBbox = bbox.map((value, index) => Math.round(value * (index % 2 === 0 ? width : height)));
  const validationMessages = useMemo(() => {
    const messages: string[] = [];
    if (!isValidBbox(bbox)) messages.push("The bbox must stay inside the image with x1 < x2 and y1 < y2.");
    if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) messages.push("Confidence must be between 0 and 1.");
    if (!Number.isInteger(candidateCount) || candidateCount < 0) messages.push("Candidate count must be a non-negative integer.");
    if (targetFound && candidateCount < 1) messages.push("A found target needs at least one candidate.");
    if (!targetFound && status !== "unresolved") messages.push("A missing target must be saved as unresolved.");
    if (targetFound && (candidateRank !== undefined || details.expectedOrdinal !== undefined)
      && (!Number.isInteger(candidateRank) || (candidateRank ?? 0) < 1 || (candidateRank ?? 0) > candidateCount)) {
      messages.push("Candidate rank must be within the candidate count.");
    }
    if (details.expectedOrdinal !== undefined && targetFound && candidateRank !== details.expectedOrdinal) {
      messages.push(`The query asks for candidate ${details.expectedOrdinal}; review the rank before confirming.`);
    }
    if (reason.trim().length < 8) messages.push("Add a short evidence reason (at least 8 characters).");
    if (status === "ok" && confidence < 0.5) messages.push("An ok result needs confidence of at least 0.5; use low_confidence instead.");
    if (unresolvedChecks && (status !== "unresolved" || confidence > 0.49) && !constraintsResolved) {
      messages.push("Keep unresolved checks at status unresolved and confidence ≤ 0.49, or explicitly confirm you resolved them below.");
    }
    if (learningValidationError) {
      messages.push(learningValidationError);
    }
    return messages;
  }, [bbox, candidateCount, candidateRank, confidence, details.expectedOrdinal, reason, status, targetFound, unresolvedChecks,
    constraintsResolved, learningValidationError]);

  const learningPayload = () => preparedLearning.learning
    ? { learning: preparedLearning.learning }
    : {};

  const send = async (action: "confirm" | "reject", payload: Record<string, unknown>) => {
    if (submissionLocked.current) return;
    submissionLocked.current = true;
    setSubmission(action);
    setSubmitError(null);
    try {
      await onInput(request, JSON.stringify({ type: "grounding_review_response", action, ...payload }));
      // Keep this request locked until the server closes it or sends a new one.
      setSubmission("submitted");
    } catch (error) {
      submissionLocked.current = false;
      setSubmission(null);
      setSubmitError(error instanceof Error ? error.message : "提交失败，请重试。");
    }
  };

  const confirm = () => {
    if (!imageReady || validationMessages.length > 0) return;
    void send("confirm", {
      bbox,
      status,
      confidence,
      targetFound,
      candidateCount,
      ...(candidateRank === undefined ? {} : { candidateRank }),
      ...(constraintsResolved ? { constraintsResolved: true } : {}),
      reason: reason.trim(),
      ...learningPayload(),
    });
  };

  const reject = () => {
    if (learningInvalid) return;
    void send("reject", {
      reason: rejectionReason.trim() || "The browser review rejected this candidate.",
      ...learningPayload(),
    });
  };

  const approvalDisabled = busy || !imageReady || validationMessages.length > 0;
  const boxStyle = {
    position: "absolute" as const,
    left: `${bbox[0] * 100}%`,
    top: `${bbox[1] * 100}%`,
    width: `${Math.max(0, (bbox[2] - bbox[0]) * 100)}%`,
    height: `${Math.max(0, (bbox[3] - bbox[1]) * 100)}%`,
    border: "2px solid #f97316",
    background: "rgba(249,115,22,0.16)",
    boxShadow: "0 0 0 1px rgba(255,255,255,0.75), 0 0 18px rgba(249,115,22,0.35)",
    pointerEvents: "none" as const,
  };
  const previousBoxStyle = details.previousBbox ? {
    position: "absolute" as const,
    left: `${details.previousBbox[0] * 100}%`,
    top: `${details.previousBbox[1] * 100}%`,
    width: `${Math.max(0, (details.previousBbox[2] - details.previousBbox[0]) * 100)}%`,
    height: `${Math.max(0, (details.previousBbox[3] - details.previousBbox[1]) * 100)}%`,
    border: "2px solid #22d3ee",
    background: "rgba(34,211,238,0.08)",
    pointerEvents: "none" as const,
  } : undefined;

  return (
    <div
      role="dialog"
      aria-label="Grounding review"
      aria-busy={busy}
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 120,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: "clamp(8px, 2vw, 16px)",
        background: "rgba(0,0,0,0.34)",
        pointerEvents: "auto",
      }}
    >
      <div
        style={{
          width: "min(1120px, 100%)",
          maxHeight: "100%",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          border: "1px solid var(--border)",
          borderRadius: 10,
          background: "var(--bg)",
          color: "var(--text)",
          boxShadow: "0 22px 72px rgba(0,0,0,0.34)",
        }}
      >
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-start", gap: 12, padding: "12px 16px", borderBottom: "1px solid var(--border)" }}>
          <div style={{ minWidth: 0, flex: 1 }}>
            <div style={{ fontSize: 15, fontWeight: 700 }}>Grounding review · {details.key}</div>
            <div style={{ marginTop: 4, color: "var(--text-muted)", fontSize: 13, lineHeight: 1.45, overflowWrap: "anywhere" }}>{details.query || "No query text was supplied."}</div>
            <div style={{ marginTop: 7, color: "var(--text-muted)", fontSize: 12, lineHeight: 1.5 }}>
              {details.canContinue === false
                ? "等待你审核当前图片和标注框。通过后保存并完成；退回后修改当前结果。"
                : "等待你审核当前图片和标注框。通过后才保存并继续下一张；退回后修改当前结果。"}
            </div>
          </div>
          <div style={{ flexShrink: 0, color: "var(--text-dim)", fontSize: 11, fontFamily: "var(--font-mono)" }}>
            {width} × {height}px
          </div>
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(min(100%, 300px), 1fr))", gap: 16, minHeight: 0, overflow: "auto", padding: 16 }}>
          <div style={{ minWidth: 0 }}>
            <div style={{ marginBottom: 7, fontSize: 12, fontWeight: 650 }}>全图概览</div>
            <div style={{ position: "relative", width: "100%", aspectRatio: `${width} / ${height}`, overflow: "hidden", border: "1px solid var(--border)", borderRadius: 7, background: "#111" }}>
              <img src={imageUrl} alt={`Full image for ${details.key}`} draggable={false} onLoad={() => { setLoadedImage(imageUrl); setFailedImage(null); }} onError={() => setFailedImage(imageUrl)} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain", userSelect: "none" }} />
              {imageReady && previousBoxStyle && <div aria-label="Previous candidate bounding box" style={previousBoxStyle} />}
              {imageReady && <div aria-label="Candidate bounding box" style={boxStyle} />}
              {imageReady && details.previousBbox && (
                <div style={{ position: "absolute", left: `${details.previousBbox[0] * 100}%`, top: `${details.previousBbox[1] * 100}%`, transform: "translateY(-100%)", padding: "2px 5px", borderRadius: 3, background: "#22d3ee", color: "#082f49", fontSize: 11, fontWeight: 700, fontFamily: "var(--font-mono)", pointerEvents: "none" }}>
                  previous
                </div>
              )}
              <div style={{ position: "absolute", left: `${bbox[0] * 100}%`, top: `${bbox[1] * 100}%`, transform: "translateY(-100%)", padding: "2px 5px", borderRadius: 3, background: "#f97316", color: "#111", fontSize: 11, fontWeight: 700, fontFamily: "var(--font-mono)", pointerEvents: "none" }}>
                proposed
              </div>
            </div>
            {!imageReady && (
              <div role={failedImage === imageUrl ? "alert" : "status"} style={{ marginTop: 8, color: "var(--text-muted)", fontSize: 12, lineHeight: 1.5 }}>
                {failedImage === imageUrl ? "图片加载失败，无法通过审核。可退回要求重新提供图片。" : "图片加载中，显示后才能通过审核。"}
              </div>
            )}
            <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 8, color: "var(--text-muted)", fontSize: 11, fontFamily: "var(--font-mono)" }}>
              <span>normalized [{bbox.map(formatCoordinate).join(", ")}]</span>
              <span>pixels [{pixelBbox.join(", ")}]</span>
            </div>
            {details.rawBbox && (
              <div style={{ marginTop: 4, color: "var(--text-dim)", fontSize: 11, fontFamily: "var(--font-mono)" }}>
                model proposal [{details.rawBbox.map(formatCoordinate).join(", ")}]
              </div>
            )}
            {details.candidateChange && (
              <div role={details.candidateChange.materialChange ? "alert" : "status"} style={{ marginTop: 8, padding: "7px 9px", borderRadius: 5, background: details.candidateChange.materialChange ? "rgba(245,158,11,0.12)" : "rgba(34,211,238,0.08)", color: "var(--text-muted)", fontSize: 11, lineHeight: 1.45 }}>
                Candidate change: IoU {details.candidateChange.iou.toFixed(3)} · center Δ [{details.candidateChange.centerDeltaPixels.map((value) => value.toFixed(1)).join(", ")}] px · area ×{details.candidateChange.areaRatio.toFixed(2)}. {details.candidateChange.note}
              </div>
            )}
            <div style={{ marginTop: 12, padding: "9px 10px", border: "1px solid var(--border)", borderRadius: 6, background: "var(--bg-panel)", color: "var(--text-muted)", fontSize: 12, lineHeight: 1.5 }}>
              通过前请逐一检查左、上、右、下四条边，覆盖目标完整的可见外轮廓，包括尾部、果皮、圆顶和低对比度边缘；除非题目明确只要求某个部分。请结合全图确认目标，局部放大供你人工复核边界。
            </div>
            {boundaryImageUrl !== null ? (
              <section aria-label="Boundary detail preview" style={{ marginTop: 12, minWidth: 0 }}>
                <div style={{ display: "flex", flexWrap: "wrap", justifyContent: "space-between", alignItems: "center", gap: 8, marginBottom: 7 }}>
                  <div style={{ fontSize: 12, fontWeight: 650 }}>边界放大 · 可见光原图裁剪</div>
                  <button
                    type="button"
                    aria-label="Show boundary preview outline"
                    aria-pressed={showBoundaryOutline}
                    disabled={busy}
                    onClick={() => setShowBoundaryOutline((current) => !current)}
                    style={{ minHeight: 36, padding: "5px 8px", border: "1px solid var(--border)", borderRadius: 5, background: "var(--bg-panel)", color: "var(--text-muted)", cursor: busy ? "not-allowed" : "pointer", fontSize: 11 }}
                  >
                    {showBoundaryOutline ? "隐藏边框，查看干净像素" : "显示当前边框"}
                  </button>
                </div>
                <div style={{ position: "relative", width: "100%", aspectRatio: `${boundaryWidth} / ${boundaryHeight}`, overflow: "hidden", border: "1px solid var(--border)", borderRadius: 7, background: "#111" }}>
                  <img
                    src={boundaryImageUrl}
                    alt={`Clean visible-image boundary crop for ${details.key}`}
                    draggable={false}
                    onLoad={() => { setLoadedBoundaryImage(boundaryImageUrl); setFailedBoundaryImage(null); }}
                    onError={() => setFailedBoundaryImage(boundaryImageUrl)}
                    style={{ position: "absolute", inset: 0, width: "100%", height: "100%", objectFit: "contain", userSelect: "none" }}
                  />
                  {boundaryImageReady && showBoundaryOutline && boundaryBox && (
                    <div
                      aria-label="Boundary preview current bounding box"
                      style={{ position: "absolute", left: `${boundaryBox.left * 100}%`, top: `${boundaryBox.top * 100}%`, width: `${boundaryBox.width * 100}%`, height: `${boundaryBox.height * 100}%`, boxSizing: "border-box", border: "1px solid #f97316", background: "transparent", pointerEvents: "none" }}
                    />
                  )}
                </div>
                {!boundaryImageReady && (
                  <div role={failedBoundaryImage === boundaryImageUrl ? "alert" : "status"} style={{ marginTop: 7, color: "var(--text-muted)", fontSize: 11, lineHeight: 1.5 }}>
                    {failedBoundaryImage === boundaryImageUrl ? "局部预览加载失败，请使用上方全图审核。" : "局部预览加载中…"}
                  </div>
                )}
                <div style={{ marginTop: 7, color: "var(--text-dim)", fontSize: 11, lineHeight: 1.5 }}>
                  裁剪范围固定，边框随坐标编辑更新；隐藏边框可检查被线条遮住的边缘。
                </div>
                {boundaryBox?.extendsBeyondPreview && (
                  <div role="status" style={{ marginTop: 7, padding: "7px 9px", borderRadius: 5, background: "rgba(245,158,11,0.12)", color: "var(--text-muted)", fontSize: 11, lineHeight: 1.5 }}>
                    当前标注框超出局部预览范围，只显示落在裁剪内的边框。请在上方全图核对超出部分；提交坐标保持不变。
                  </div>
                )}
              </section>
            ) : boundaryPreview !== undefined && (
              <div role="status" style={{ marginTop: 8, color: "var(--text-muted)", fontSize: 11, lineHeight: 1.5 }}>
                局部预览信息无效，请使用上方全图审核。
              </div>
            )}
          </div>

          <fieldset disabled={busy} style={{ minWidth: 0, display: "flex", flexDirection: "column", gap: 10, margin: 0, padding: 0, border: 0 }}>
            {details.constraintAssessment && (
              <div role={unresolvedChecks ? "alert" : "status"} style={{ display: "grid", gap: 6, padding: "9px 10px", border: "1px solid var(--border)", borderRadius: 6, background: unresolvedChecks ? "rgba(245,158,11,0.10)" : "var(--bg-panel)", fontSize: 12, lineHeight: 1.5, overflowWrap: "anywhere" }}>
                <strong>{unresolvedChecks ? "原始问题 / 目标证据仍有未解决项" : "模型声明的约束已通过一致性检查，仍需人工审核"}</strong>
                {details.modelProposal && <div>模型原提议：{details.modelProposal.status} / {details.modelProposal.confidence.toFixed(2)}。审核默认值已按未解决证据降低。</div>}
                {details.constraintAssessment.issues.map((issue, index) => <div key={`${issue.code}-${index}`}>• {issue.message}</div>)}
                {details.constraintAssessment.orders.map((order) => (
                  <div key={order.interpretationId}>
                    {order.interpretationId}: {order.axis} / {order.direction}，请求第 {order.ordinal} 个；坐标排序 {order.orderedCandidateIds.join(" → ") || "无"}。
                    支持 / 可能候选 {order.supportedCount} / {order.possibleCount}；所选 {order.selectedCandidateId} 排名 {order.selectedRank ?? "未确定"}
                  </div>
                ))}
                {details.modelContract && <details>
                  <summary>模型声明的候选、原文约束与不同解读</summary>
                  {details.modelContract.candidates.map((candidate) => <div key={candidate.id}>
                    {candidate.id}: {candidate.identity.label} · {candidate.identity.status} · {candidate.identity.basis} · [{candidate.bbox.map(formatCoordinate).join(", ")}] · {candidate.identity.evidence}
                  </div>)}
                  {details.modelContract.interpretations.map((reading) => <div key={reading.id} style={{ marginTop: 6 }}>
                    <strong>{reading.reading} · {reading.status}</strong><div>{reading.evidence}</div>
                    {reading.requirements.map((requirement) => <div key={requirement.id}>“{requirement.queryText}” · {requirement.status}: {requirement.evidence}</div>)}
                  </div>)}
                </details>}
                <div style={{ color: "var(--text-muted)" }}>这里只检查模型声明与坐标的一致性，不能识别物体或证明候选完整。放大、重复裁剪和颜色测量不会补充缺失的身份依据。可以保留 unresolved 后人工确认保存。</div>
                {unresolvedChecks && <label style={{ display: "flex", gap: 7, alignItems: "flex-start" }}>
                  <input type="checkbox" checked={constraintsResolved} onChange={(event) => setConstraintsResolved(event.target.checked)} />
                  我已对照原始问题和图像，亲自解决上述未确定项；如提高状态或置信度，请在理由中说明
                </label>}
              </div>
            )}
            <fieldset style={{ display: "grid", gridTemplateColumns: "repeat(4, minmax(0, 1fr))", gap: 7, margin: 0, padding: 0, border: 0 }}>
              {(["x1", "y1", "x2", "y2"] as const).map((label, index) => (
                <label key={label} style={{ display: "grid", gap: 4, color: "var(--text-muted)", fontSize: 11 }}>
                  {label}
                  <input
                    aria-label={`${label} normalized`}
                    type="number"
                    min={0}
                    max={1}
                    step={0.001}
                    value={Number.isFinite(bbox[index]) ? bbox[index] : ""}
                    onChange={(event) => {
                      setConstraintsResolved(false);
                      const next = Number(event.target.value);
                      setBbox((current) => current.map((value, currentIndex) => currentIndex === index ? next : value) as Bbox);
                    }}
                    style={{ width: "100%", minWidth: 0, padding: "6px 5px", border: "1px solid var(--border)", borderRadius: 5, background: "var(--bg-panel)", color: "var(--text)", fontFamily: "var(--font-mono)", fontSize: 12 }}
                  />
                </label>
              ))}
            </fieldset>

            <label style={{ display: "flex", alignItems: "center", gap: 8, color: "var(--text)", fontSize: 13 }}>
              <input type="checkbox" checked={targetFound} onChange={(event) => { setTargetFound(event.target.checked); if (!event.target.checked) setStatus("unresolved"); }} />
              Target is visibly present
            </label>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <label style={{ display: "grid", gap: 4, color: "var(--text-muted)", fontSize: 11 }}>
                Candidate count
                <input aria-label="Candidate count" type="number" min={0} step={1} value={candidateCount} onChange={(event) => setCandidateCount(Number(event.target.value))} style={{ padding: "6px 7px", border: "1px solid var(--border)", borderRadius: 5, background: "var(--bg-panel)", color: "var(--text)", fontSize: 12 }} />
              </label>
              <label style={{ display: "grid", gap: 4, color: "var(--text-muted)", fontSize: 11 }}>
                Selected rank
                <input aria-label="Candidate rank" type="number" min={1} step={1} disabled={!targetFound} value={candidateRank ?? ""} onChange={(event) => setCandidateRank(event.target.value === "" ? undefined : Number(event.target.value))} style={{ padding: "6px 7px", border: "1px solid var(--border)", borderRadius: 5, background: targetFound ? "var(--bg-panel)" : "var(--bg-hover)", color: "var(--text)", fontSize: 12 }} />
              </label>
            </div>

            {details.expectedOrdinal !== undefined && (
              <div style={{ padding: "7px 9px", borderRadius: 5, background: "rgba(59,130,246,0.10)", color: "var(--text-muted)", fontSize: 12 }}>
                Query ordinal: <strong>{details.expectedOrdinal}</strong>. Rank the complete candidate list using the query&apos;s stated ordering.
              </div>
            )}

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <label style={{ display: "grid", gap: 4, color: "var(--text-muted)", fontSize: 11 }}>
                Status
                <select aria-label="Review status" value={status} onChange={(event) => setStatus(event.target.value as typeof status)} style={{ padding: "6px 7px", border: "1px solid var(--border)", borderRadius: 5, background: "var(--bg-panel)", color: "var(--text)", fontSize: 12 }}>
                  <option value="ok">ok</option>
                  <option value="low_confidence">low_confidence</option>
                  <option value="unresolved">unresolved</option>
                </select>
              </label>
              <label style={{ display: "grid", gap: 4, color: "var(--text-muted)", fontSize: 11 }}>
                Confidence
                <input aria-label="Review confidence" type="number" min={0} max={1} step={0.01} value={confidence} onChange={(event) => setConfidence(Number(event.target.value))} style={{ padding: "6px 7px", border: "1px solid var(--border)", borderRadius: 5, background: "var(--bg-panel)", color: "var(--text)", fontSize: 12 }} />
              </label>
            </div>

            <label style={{ display: "grid", gap: 4, color: "var(--text-muted)", fontSize: 11 }}>
              Evidence / reason
              <textarea aria-label="Evidence reason" value={reason} onChange={(event) => setReason(event.target.value)} rows={3} style={{ resize: "vertical", padding: "7px 8px", border: "1px solid var(--border)", borderRadius: 5, background: "var(--bg-panel)", color: "var(--text)", fontSize: 12, lineHeight: 1.4 }} />
            </label>

            <details style={{ padding: "9px 10px", border: "1px solid var(--border)", borderRadius: 6, background: "var(--bg-panel)" }}>
              <summary style={{ fontSize: 12, fontWeight: 650, cursor: "pointer" }}>通用经验（可选，新写的方法）</summary>
              <div style={{ marginTop: 8, color: "var(--text-muted)", fontSize: 11, lineHeight: 1.5 }}>
                请独立撰写可用于其他任务的流程，不复制或改写当前题目、审核理由或答案。此处不会自动填入样本内容；留空或关闭保存不影响当前标注。
              </div>
              <label style={{ display: "flex", gap: 7, alignItems: "flex-start", marginTop: 8, color: "var(--text-muted)", fontSize: 11, lineHeight: 1.4 }}>
                <input aria-label="Save general grounding lesson" type="checkbox" checked={rememberLearning} onChange={(event) => { setRememberLearning(event.target.checked); setLearningSampleIndependent(false); }} />
                跨会话保存这条通用经验
              </label>
              <fieldset disabled={!rememberLearning} style={{ display: "grid", gap: 8, margin: "8px 0 0", padding: 0, border: 0 }}>
                <label style={{ display: "grid", gap: 4, color: "var(--text-muted)", fontSize: 11 }}>
                  问题类别
                  <select aria-label="Learning category" value={learningCategory} onChange={(event) => { setLearningCategory(event.target.value as GroundingLearningCategory); setLearningSampleIndependent(false); }} style={{ padding: "6px 7px", border: "1px solid var(--border)", borderRadius: 5, background: "var(--bg)", color: "var(--text)", fontSize: 12 }}>
                    <option value="identity">目标身份</option>
                    <option value="boundary">边界完整性</option>
                    <option value="order">顺序 / 排名</option>
                    <option value="relation">所属 / 空间关系</option>
                    <option value="cross_modal">跨模态对应</option>
                    <option value="uncertainty">不确定性处理</option>
                    <option value="efficiency">效率 / 工具使用</option>
                    <option value="other">其他</option>
                  </select>
                </label>
                <label style={{ display: "grid", gap: 4, color: "var(--text-muted)", fontSize: 11 }}>
                  适用条件（4–240 字符）
                  <textarea aria-label="Learning applicability" value={learningApplicability} onChange={(event) => { setLearningApplicability(event.target.value); setLearningSampleIndependent(false); }} rows={2} minLength={4} maxLength={240} placeholder="这种方法适用于什么通用情形？" style={{ resize: "vertical", padding: "7px 8px", border: "1px solid var(--border)", borderRadius: 5, background: "var(--bg)", color: "var(--text)", fontSize: 12, lineHeight: 1.4 }} />
                </label>
                <label style={{ display: "grid", gap: 4, color: "var(--text-muted)", fontSize: 11 }}>
                  常见错误（4–400 字符）
                  <textarea aria-label="Learning error" value={learningError} onChange={(event) => { setLearningError(event.target.value); setLearningSampleIndependent(false); }} rows={2} minLength={4} maxLength={400} placeholder="要避免哪类推理或操作错误？" style={{ resize: "vertical", padding: "7px 8px", border: "1px solid var(--border)", borderRadius: 5, background: "var(--bg)", color: "var(--text)", fontSize: 12, lineHeight: 1.4 }} />
                </label>
                <label style={{ display: "grid", gap: 4, color: "var(--text-muted)", fontSize: 11 }}>
                  改进方法（8–800 字符）
                  <textarea aria-label="Learning method" value={learningMethod} onChange={(event) => { setLearningMethod(event.target.value); setLearningSampleIndependent(false); }} rows={3} minLength={8} maxLength={800} placeholder="下次应按什么步骤处理？" style={{ resize: "vertical", padding: "7px 8px", border: "1px solid var(--border)", borderRadius: 5, background: "var(--bg)", color: "var(--text)", fontSize: 12, lineHeight: 1.4 }} />
                </label>
                <label style={{ display: "grid", gap: 4, color: "var(--text-muted)", fontSize: 11 }}>
                  复查步骤（4–400 字符）
                  <textarea aria-label="Learning check" value={learningCheck} onChange={(event) => { setLearningCheck(event.target.value); setLearningSampleIndependent(false); }} rows={2} minLength={4} maxLength={400} placeholder="如何检查方法是否落实，并保留未解决的不确定性？" style={{ resize: "vertical", padding: "7px 8px", border: "1px solid var(--border)", borderRadius: 5, background: "var(--bg)", color: "var(--text)", fontSize: 12, lineHeight: 1.4 }} />
                </label>
                <label style={{ display: "flex", gap: 7, alignItems: "flex-start", color: "var(--text-muted)", fontSize: 11, lineHeight: 1.5 }}>
                  <input aria-label="Confirm sample-independent lesson" type="checkbox" checked={learningSampleIndependent} onChange={(event) => setLearningSampleIndependent(event.target.checked)} />
                  我已人工检查：这是一条新写的通用经验，不含原始问题或其改写、图像内容或路径、样本 ID、具体答案、坐标框或标准答案（ground truth）。
                </label>
              </fieldset>
              <div style={{ marginTop: 8, color: "var(--text-dim)", fontSize: 11, lineHeight: 1.5 }}>
                长度与词面校验不能证明语义上与样本无关，仍需人工判断。保存的经验只作参考，不覆盖当前题目、图像证据或安全规则。
              </div>
            </details>

            {validationMessages.length > 0 && (
              <div role="alert" style={{ display: "grid", gap: 4, padding: "8px 9px", border: "1px solid rgba(245,158,11,0.45)", borderRadius: 5, background: "rgba(245,158,11,0.10)", color: "var(--text-muted)", fontSize: 11, lineHeight: 1.4 }}>
                {validationMessages.map((message) => <div key={message}>• {message}</div>)}
              </div>
            )}

            <label style={{ display: "grid", gap: 4, color: "var(--text-muted)", fontSize: 11 }}>
              Rejection note (optional)
              <input aria-label="Rejection note" value={rejectionReason} onChange={(event) => setRejectionReason(event.target.value)} placeholder="Why should the agent revise this candidate?" style={{ padding: "7px 8px", border: "1px solid var(--border)", borderRadius: 5, background: "var(--bg-panel)", color: "var(--text)", fontSize: 12 }} />
            </label>
          </fieldset>
        </div>

        <div style={{ padding: "10px 16px", borderTop: "1px solid var(--border)", background: "var(--bg-panel)" }}>
          {submitError && <div role="alert" style={{ marginBottom: 8, color: "var(--text)", fontSize: 12, lineHeight: 1.5, overflowWrap: "anywhere" }}>提交失败：{submitError}</div>}
          <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", justifyContent: "flex-end", gap: 8 }}>
            {busy && <span role="status" style={{ marginRight: "auto", color: "var(--text-muted)", fontSize: 12 }}>{submission === "submitted" ? "已提交，等待处理…" : "正在提交…"}</span>}
            <button type="button" disabled={busy || learningInvalid} onClick={reject} style={{ minHeight: 40, padding: "7px 11px", border: "1px solid var(--border)", borderRadius: 6, background: "var(--bg)", color: "var(--text-muted)", cursor: busy || learningInvalid ? "not-allowed" : "pointer", fontSize: 12 }}>退回修改</button>
            <button type="button" disabled={approvalDisabled} onClick={confirm} style={{ minHeight: 40, padding: "7px 12px", border: "1px solid var(--accent)", borderRadius: 6, background: approvalDisabled ? "var(--bg-hover)" : "var(--accent)", color: approvalDisabled ? "var(--text-dim)" : "var(--accent-contrast)", cursor: approvalDisabled ? "not-allowed" : "pointer", fontSize: 12, fontWeight: 650 }}>{details.canContinue === false ? "通过并完成" : "通过 / 下一张"}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
