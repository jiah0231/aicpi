"use client";
/* eslint-disable @next/next/no-img-element -- review images are data URLs produced by the grounding runtime. */

import { useMemo, useRef, useState } from "react";
import type { ExtensionUiRequest, GroundingReviewDetails } from "@/lib/types";

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
  const [rejectionReason, setRejectionReason] = useState("");
  const [submission, setSubmission] = useState<"confirm" | "reject" | "submitted" | null>(null);
  const [submitError, setSubmitError] = useState<string | null>(null);
  // State alone does not guard two clicks delivered before React re-renders.
  const submissionLocked = useRef(false);
  const [loadedImage, setLoadedImage] = useState<string | null>(null);
  const [failedImage, setFailedImage] = useState<string | null>(null);
  const imageUrl = `data:${details.image.mimeType};base64,${details.image.data}`;
  const imageReady = loadedImage === imageUrl && failedImage !== imageUrl;
  const busy = submission !== null;

  const width = details.image.originalWidth || details.image.width || 1;
  const height = details.image.originalHeight || details.image.height || 1;
  const pixelBbox = bbox.map((value, index) => Math.round(value * (index % 2 === 0 ? width : height)));
  const validationMessages = useMemo(() => {
    const messages: string[] = [];
    if (!isValidBbox(bbox)) messages.push("The bbox must stay inside the image with x1 < x2 and y1 < y2.");
    if (!Number.isFinite(confidence) || confidence < 0 || confidence > 1) messages.push("Confidence must be between 0 and 1.");
    if (!Number.isInteger(candidateCount) || candidateCount < 0) messages.push("Candidate count must be a non-negative integer.");
    if (targetFound && candidateCount < 1) messages.push("A found target needs at least one candidate.");
    if (!targetFound && status !== "unresolved") messages.push("A missing target must be saved as unresolved.");
    if (targetFound && (!Number.isInteger(candidateRank) || (candidateRank ?? 0) < 1 || (candidateRank ?? 0) > candidateCount)) {
      messages.push("Candidate rank must be within the candidate count.");
    }
    if (details.expectedOrdinal !== undefined && targetFound && candidateRank !== details.expectedOrdinal) {
      messages.push(`The query asks for candidate ${details.expectedOrdinal}; review the rank before confirming.`);
    }
    if (reason.trim().length < 8) messages.push("Add a short evidence reason (at least 8 characters).");
    if (status === "ok" && confidence < 0.5) messages.push("An ok result needs confidence of at least 0.5; use low_confidence instead.");
    return messages;
  }, [bbox, candidateCount, candidateRank, confidence, details.expectedOrdinal, reason, status, targetFound]);

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
      reason: reason.trim(),
    });
  };

  const reject = () => {
    void send("reject", {
      reason: rejectionReason.trim() || "The browser review rejected this candidate.",
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
          </div>

          <fieldset disabled={busy} style={{ minWidth: 0, display: "flex", flexDirection: "column", gap: 10, margin: 0, padding: 0, border: 0 }}>
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
            <button type="button" disabled={busy} onClick={reject} style={{ minHeight: 40, padding: "7px 11px", border: "1px solid var(--border)", borderRadius: 6, background: "var(--bg)", color: "var(--text-muted)", cursor: busy ? "not-allowed" : "pointer", fontSize: 12 }}>退回修改</button>
            <button type="button" disabled={approvalDisabled} onClick={confirm} style={{ minHeight: 40, padding: "7px 12px", border: "1px solid var(--accent)", borderRadius: 6, background: approvalDisabled ? "var(--bg-hover)" : "var(--accent)", color: approvalDisabled ? "var(--text-dim)" : "var(--accent-contrast)", cursor: approvalDisabled ? "not-allowed" : "pointer", fontSize: 12, fontWeight: 650 }}>{details.canContinue === false ? "通过并完成" : "通过 / 下一张"}</button>
          </div>
        </div>
      </div>
    </div>
  );
}
