"use client";

import { useEffect, useState } from "react";
import { useI18n } from "@/hooks/useI18n";
import type { GenericLearningProposal, GenericLearningSource, GenericLearningStore, GenericProcedure } from "@/lib/grounding-learning-consolidation-types";

type Settings = { enabled: boolean; provider: string; modelId: string; maxSources: number; maxInputChars: number; maxOutputTokens: number; timeoutMs: number; dailyAttemptLimit: number; revision: number };
type Snapshot = { store: GenericLearningStore; sources: GenericLearningSource[]; settings: Settings; job: { status: string; attemptsToday: number; errorCode?: string }; models: { provider: string; id: string; name: string }[] };
type Mutate = (body: Record<string, unknown>) => Promise<boolean>;
const categories = ["identity", "boundary", "order", "relation", "cross_modal", "uncertainty", "efficiency", "other"] as const;

function ProposalEditor({ proposal, store, busy, mutate }: { proposal: GenericLearningProposal; store: GenericLearningStore; busy: boolean; mutate: Mutate }) {
  const { t } = useI18n();
  const label = (key: string) => t(`groundingLearning.${key}`);
  const [procedure, setProcedure] = useState<GenericProcedure>(proposal.procedure);
  const [confirmed, setConfirmed] = useState(false);
  const [ruleId, setRuleId] = useState("");
  const edit = (patch: Partial<GenericProcedure>) => { setProcedure((current) => ({ ...current, ...patch })); setConfirmed(false); };
  return <fieldset disabled={busy} className="settings-general-section" style={{ minWidth: 0 }}>
    <legend>{label(proposal.operation)} · {proposal.id}</legend>
    <p>{proposal.note}</p>
    <p>{label("sources")}: {proposal.sourceIds.join(", ")}</p>
    <label>{label("category")} <select value={procedure.category} onChange={(event) => edit({ category: event.target.value as GenericProcedure["category"] })}>{categories.map((category) => <option key={category} value={category}>{label(`category.${category}`)}</option>)}</select></label>
    {(["applicability", "error", "method", "check"] as const).map((field) => <label key={field} style={{ display: "block" }}>{label(field)}<textarea style={{ width: "100%" }} value={procedure[field]} maxLength={field === "method" ? 800 : field === "applicability" ? 240 : 400} onChange={(event) => edit({ [field]: event.target.value })} /></label>)}
    <label>{label("target")} <select value={ruleId} onChange={(event) => { setRuleId(event.target.value); setConfirmed(false); }}><option value="">{label("newrule")}</option>{store.rules.map((rule) => <option key={rule.id} value={rule.id}>{rule.id} · {label("version")} {rule.currentRevision} · {label(rule.enabled ? "enabled" : "disabled")}</option>)}</select></label>
    {store.rules.some((rule) => rule.id === ruleId && !rule.enabled) && <p>{label("disabledTarget")}</p>}
    <label style={{ display: "block" }}><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /> {label("confirm")}</label>
    <button type="button" disabled={!confirmed} onClick={() => void mutate({ action: "activate", proposalId: proposal.id, sampleIndependent: true, procedure: { ...procedure, sampleIndependent: true }, ...(ruleId ? { ruleId } : {}), expectedRevision: store.revision })}>{label(store.rules.some((rule) => rule.id === ruleId && !rule.enabled) ? "saveReviewed" : "activate")}</button>{" "}
    <button type="button" onClick={() => void mutate({ action: "dismiss", proposalId: proposal.id, expectedRevision: store.revision })}>{label("dismiss")}</button>
  </fieldset>;
}

function LearningManager() {
  const { t } = useI18n();
  const label = (key: string) => t(`groundingLearning.${key}`);
  const [data, setData] = useState<Snapshot | null>(null);
  const [settings, setSettings] = useState<Settings | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [text, setText] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const [rollback, setRollback] = useState<Record<string, number>>({});
  async function refresh() {
    setBusy(true); setError("");
    try {
      const response = await fetch("/api/grounding/learning", { cache: "no-store" });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
      setData(result); setSettings(result.settings);
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); }
    finally { setBusy(false); }
  }
  // Opening the panel performs one read. No polling or model calls from UI.
  useEffect(() => { void refresh(); }, []);
  const mutate: Mutate = async (body) => {
    setBusy(true); setError("");
    try {
      const response = await fetch("/api/grounding/learning", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      const result = await response.json();
      if (!response.ok) throw new Error(response.status === 409 ? label("stale") : result.error || `HTTP ${response.status}`);
      await refresh(); return true;
    } catch (cause) { setError(cause instanceof Error ? cause.message : String(cause)); return false; }
    finally { setBusy(false); }
  };
  const pending = data?.store.proposals.filter((proposal) => proposal.status === "pending") ?? [];
  return <div style={{ minWidth: 0, overflowWrap: "anywhere" }}>
    <p>{label("notice")}</p>
    <button type="button" disabled={busy} onClick={() => void refresh()}>{label("refresh")}</button>
    {error && <p role="alert">{error}</p>}
    {!data || !settings ? <p>{label("loading")}</p> : <>
      <fieldset disabled={busy} className="settings-general-section"><legend>{label("background")}</legend>
        <p>{label("status")}: {label(data.job.status === "error" ? "errorStatus" : data.job.status)} · {data.job.attemptsToday}/{data.settings.dailyAttemptLimit} · {label("pending")}: {pending.length}</p>
        <p>{label("cost")}</p>
        {data.job.errorCode && <div><p>{label("retryNotice")}</p><button type="button" disabled={!data.settings.enabled} onClick={() => void mutate({ action: "retry", expectedRevision: data.settings.revision })}>{label("retry")}</button></div>}
        <label style={{ display: "block" }}>{label("model")} <select value={JSON.stringify([settings.provider, settings.modelId])} onChange={(event) => { const [provider, modelId] = JSON.parse(event.target.value); setSettings({ ...settings, provider, modelId }); }}><option value={JSON.stringify(["", ""])}>{label("choose")}</option>{data.models.map((model) => <option key={`${model.provider}:${model.id}`} value={JSON.stringify([model.provider, model.id])}>{model.provider} / {model.name || model.id}</option>)}</select></label>
        {([{ key: "dailyAttemptLimit", label: "calls", min: 1, max: 4 }, { key: "maxOutputTokens", label: "tokens", min: 128, max: 1600 }, { key: "maxInputChars", label: "input", min: 1000, max: 12000 }] as const).map((field) => <label key={field.key} style={{ display: "block" }}>{label(field.label)} <input type="number" min={field.min} max={field.max} value={settings[field.key]} onChange={(event) => setSettings({ ...settings, [field.key]: Number(event.target.value) })} /></label>)}
        <label style={{ display: "block" }}><input type="checkbox" checked={settings.enabled} disabled={!settings.provider || !settings.modelId} onChange={(event) => setSettings({ ...settings, enabled: event.target.checked })} />{label("enabled")}</label>
        <button type="button" onClick={() => { const { revision, ...patch } = settings; void mutate({ action: "settings", expectedRevision: revision, settings: patch }); }}>{label("save")}</button>
      </fieldset>
      <fieldset disabled={busy} className="settings-general-section"><legend>{label("source")}</legend>
        <textarea aria-label={label("source")} style={{ width: "100%" }} value={text} maxLength={2400} onChange={(event) => { setText(event.target.value); setConfirmed(false); }} />
        <label style={{ display: "block" }}><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /> {label("confirm")}</label>
        <button type="button" disabled={!confirmed || text.trim().length < 8} onClick={async () => { if (await mutate({ action: "source", text, sampleIndependent: true })) { setText(""); setConfirmed(false); } }}>{label("save")}</button>
      </fieldset>
      <h4>{label("sources")}</h4>{data.sources.length ? data.sources.map((source) => <details key={source.id}><summary>{source.id}</summary><p style={{ whiteSpace: "pre-wrap", overflowWrap: "anywhere" }}>{source.text}</p></details>) : <p>{label("empty")}</p>}
      <h4>{label("proposals")}</h4>{pending.length ? pending.map((proposal) => <ProposalEditor key={`${proposal.id}:${data.store.revision}`} proposal={proposal} store={data.store} busy={busy} mutate={mutate} />) : <p>{label("empty")}</p>}
      <h4>{label("rules")}</h4>{data.store.rules.length ? data.store.rules.map((rule) => <fieldset disabled={busy} key={rule.id}><legend>{rule.id} · {label("version")} {rule.currentRevision} · {label(rule.enabled ? "enabled" : "disabled")}</legend>
        <button type="button" onClick={() => void mutate({ action: "set_enabled", ruleId: rule.id, enabled: !rule.enabled, expectedRevision: data.store.revision })}>{label(rule.enabled ? "disable" : "restore")}</button>
        {rule.revisions.map((revision, index) => <details key={index} open={index + 1 === rule.currentRevision}><summary>{label("version")} {index + 1}</summary>{(["applicability", "error", "method", "check"] as const).map((field) => <p key={field}><strong>{label(field)}: </strong>{revision[field]}</p>)}</details>)}
        <select aria-label={label("version")} value={rollback[rule.id] ?? rule.currentRevision} onChange={(event) => setRollback({ ...rollback, [rule.id]: Number(event.target.value) })}>{rule.revisions.map((_, index) => <option key={index} value={index + 1}>{index + 1}</option>)}</select>{" "}<button type="button" disabled={!rollback[rule.id] || rollback[rule.id] === rule.currentRevision} onClick={() => void mutate({ action: "rollback", ruleId: rule.id, revision: rollback[rule.id], expectedRevision: data.store.revision })}>{label("rollback")}</button>
      </fieldset>) : <p>{label("empty")}</p>}
    </>}
  </div>;
}

export function GroundingLearningPanel() {
  const { t } = useI18n();
  const [open, setOpen] = useState(false);
  return <details className="settings-general-section" onToggle={(event) => setOpen(event.currentTarget.open)}><summary>{t("groundingLearning.title")}</summary>{open && <LearningManager />}</details>;
}
