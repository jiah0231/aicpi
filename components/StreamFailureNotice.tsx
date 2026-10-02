"use client";

import { useI18n } from "@/hooks/useI18n";
import { getStreamFailureDiagnostic } from "@/lib/stream-failure";
import type { AssistantMessage } from "@/lib/types";

/** Explain a displayed failure without presenting it as the current run state. */
export function StreamFailureNotice({ message }: { message: AssistantMessage }) {
  const { t } = useI18n();
  const diagnostic = getStreamFailureDiagnostic(message);
  if (!diagnostic) return null;
  return (
    <div style={{ marginTop: 6, fontSize: 12, color: "var(--text-dim)", lineHeight: 1.5 }}>
      <div>{t("chat.streamFailureHistory")}</div>
      <details>
        <summary style={{ cursor: "pointer" }}>{t("chat.streamFailureDiagnostic")}</summary>
        <div>{t("chat.streamFailureCauseUnknown")}</div>
        <div>{t("chat.streamFailureCounts", {
          blocks: diagnostic.receivedBlocks,
          tools: diagnostic.pendingToolCalls,
        })}</div>
        <div>{t("chat.streamFailureSafety")}</div>
        <code>{diagnostic.errorClass} / {diagnostic.terminationKind}</code>
      </details>
    </div>
  );
}
