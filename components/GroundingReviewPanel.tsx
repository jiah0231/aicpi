"use client";
/* eslint-disable @next/next/no-img-element -- review images are data URLs produced by the grounding runtime. */

import { useMemo, useRef, useState } from "react";
import type {
  ExtensionUiRequest,
  GroundingLearningCategory,
  GroundingLearningScope,
  GroundingReviewDetails,
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
  const [learningAdvice, setLearningAdvice] = useState("");
  const [learningCategory, setLearningCategory] = useState<GroundingLearningCategory>("other");
  const [learningScope, setLearningScope] = useState<GroundingLearningScope>("similar");
  const [rememberLearning, setRememberLearning] = useState(true);
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
  const learningInvalid = rememberLearning && learningAdvice.trim().length > 0 && learningAdvice.trim().length < 8;

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
    if (learningInvalid) {
      messages.push("长期改进建议至少需要 8 个字符，或留空不保存。");
    }
    return messages;
  }, [bbox, candidateCount, candidateRank, confidence, details.expectedOrdinal, reason, status, targetFound, unresolvedChecks,
    constraintsResolved, learningInvalid]);

  const learningPayload = () => rememberLearning && learningAdvice.trim().length > 0
    ? { learning: { category: learningCategory, scope: learningScope, advice: learningAdvice.trim() } }
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

            <div style={{ display: "grid", gap: 8, padding: "9px 10px", border: "1px solid var(--border)", borderRadius: 6, background: "var(--bg-panel)" }}>
              <div style={{ fontSize: 12, fontWeight: 650 }}>长期改进建议</div>
              <textarea
                aria-label="Long-term grounding improvement advice"
                value={learningAdvice}
                onChange={(event) => setLearningAdvice(event.target.value)}
                rows={3}
                maxLength={1200}
                placeholder="例如：锁定小目标后仍要检查完整外轮廓，不能只框内部高对比区域。"
                style={{ resize: "vertical", padding: "7px 8px", border: "1px solid var(--border)", borderRadius: 5, background: "var(--bg)", color: "var(--text)", fontSize: 12, lineHeight: 1.4 }}
              />
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <label style={{ display: "grid", gap: 4, color: "var(--text-muted)", fontSize: 11 }}>
                  问题类别
                  <select aria-label="Learning category" value={learningCategory} onChange={(event) => setLearningCategory(event.target.value as GroundingLearningCategory)} style={{ padding: "6px 7px", border: "1px solid var(--border)", borderRadius: 5, background: "var(--bg)", color: "var(--text)", fontSize: 12 }}>
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
                  适用范围
                  <select aria-label="Learning scope" value={learningScope} onChange={(event) => setLearningScope(event.target.value as GroundingLearningScope)} style={{ padding: "6px 7px", border: "1px solid var(--border)", borderRadius: 5, background: "var(--bg)", color: "var(--text)", fontSize: 12 }}>
                    <option value="similar">仅相似题目</option>
                    <option value="global">所有定位任务</option>
                  </select>
                </label>
              </div>
              <label style={{ display: "flex", gap: 7, alignItems: "flex-start", color: "var(--text-muted)", fontSize: 11, lineHeight: 1.4 }}>
                <input type="checkbox" checked={rememberLearning} onChange={(event) => setRememberLearning(event.target.checked)} />
                跨会话保存。后续任务只把它作为人工经验，不覆盖当前题目、图像证据或安全规则。
              </label>
            </div>

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
