import { useEffect, useMemo, useState } from "react"
import { Link, useSearchParams } from "react-router-dom"
import { BACKEND_URL } from "../constants"
import SkillPanel from "./SkillPanel"
import TrendChart from "./TrendChart"
import { fmtChange, fmtDate, fmtValue } from "./format"
import { SECTION_TITLES, SECTIONS } from "./types"
import type { Comparison, RunSummary, Status } from "./types"
import "./Bench.css"

const STATUS: Record<Status, { icon: string, text: string }> = {
    improved: { icon: "▲", text: "improved" },
    regressed: { icon: "▼", text: "regressed" },
    unchanged: { icon: "●", text: "unchanged" },
    new: { icon: "+", text: "new" },
    removed: { icon: "−", text: "removed" },
    info: { icon: "i", text: "info" },
}

function StatusBadge({ status }: { status: Status }) {
    const s = STATUS[status]
    return <span className={`status status-${status}`}><span className="status-icon" aria-hidden>{s.icon}</span>{s.text}</span>
}

function runName(run: RunSummary) {
    const commit = `${run.git.short ?? "?"}${run.git.dirty ? "*" : ""}`
    return `${fmtDate(run.started_at)} · ${commit}${run.label ? ` · ${run.label}` : ""} · ${run.suite}`
}

/** Latest earlier run of the same suite that measured every section the head did (as the CLI does). */
function defaultBase(runs: RunSummary[], head: RunSummary | undefined) {
    if (!head) return undefined
    const earlier = runs.filter(r => r.started_at < head.started_at)
    const covers = (r: RunSummary) => head.sections.every(sec => r.sections.includes(sec))
    return [...earlier].reverse().find(r => r.suite === head.suite && covers(r)) ?? earlier[earlier.length - 1]
}

/** First win rate, else the first agent's decision loss, else the first metric: a useful chart to open with. */
function defaultMetric(run: RunSummary | undefined) {
    if (!run) return undefined
    const keys = Object.keys(run.metrics)
    return keys.find(k => run.metrics[k].kind === "rate")
        ?? keys.find(k => /^skill\[.+\]\.loss$/.test(k))
        ?? keys[0]
}

export default function BenchDashboard() {
    const [runs, setRuns] = useState<RunSummary[] | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [headId, setHeadId] = useState<string>()
    const [baseId, setBaseId] = useState<string>()
    const [comparison, setComparison] = useState<Comparison | null>(null)
    const [selected, setSelected] = useState<string>()
    const [changesOnly, setChangesOnly] = useState(false)
    // ?base=&head=&metric= make a view shareable and survive reloads.
    const [params, setParams] = useSearchParams()

    useEffect(() => {
        fetch(`${BACKEND_URL}/api/bench/runs`)
            .then(r => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
            .then((data: RunSummary[]) => {
                setRuns(data)
                const find = (id: string | null) => data.find(r => r.id === id)
                const head = find(params.get("head")) ?? data[data.length - 1]
                setHeadId(head?.id)
                setBaseId((find(params.get("base")) ?? defaultBase(data, head))?.id)
                const metric = params.get("metric")
                setSelected(metric && head?.metrics[metric] ? metric : defaultMetric(head))
            })
            .catch(e => setError(`Could not load benchmark runs from the backend (${e.message}).`))
        // Read the URL once on load; afterwards state drives the URL.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [])

    useEffect(() => {
        if (!runs) return
        const next: Record<string, string> = {}
        if (baseId) next.base = baseId
        if (headId) next.head = headId
        if (selected) next.metric = selected
        setParams(next, { replace: true })
    }, [runs, baseId, headId, selected, setParams])

    useEffect(() => {
        if (!headId || !baseId) { setComparison(null); return }
        const params = new URLSearchParams({ base: baseId, head: headId })
        let stale = false
        fetch(`${BACKEND_URL}/api/bench/compare?${params}`)
            .then(r => r.json())
            .then(data => {
                if (stale) return
                if (data.error) { setError(data.error); return }
                setError(null)
                setComparison(data)
            })
        return () => { stale = true }
    }, [baseId, headId])

    const head = runs?.find(r => r.id === headId)
    const rowsBySection = useMemo(() => {
        const rows = comparison?.rows ?? []
        return SECTIONS.map(section => ({
            section,
            rows: rows.filter(r => r.section === section && (!changesOnly || !["unchanged", "info"].includes(r.status))),
        })).filter(g => g.rows.length)
    }, [comparison, changesOnly])

    if (error && !runs) return <div className="bench-page"><p className="bench-error">{error}</p></div>
    if (!runs) return <div className="bench-page"><p className="muted">Loading…</p></div>

    const selectedMeta = selected && head?.metrics[selected]

    return (
        <div className="bench-page viz-root">
            <header className="bench-header">
                <div>
                    <Link to="/" className="back">← Home</Link>
                    <h1>Benchmarks</h1>
                </div>
                <p className="muted">Record a run with <code>python -m bench run</code> in <code>backend/</code>, then refresh.</p>
            </header>

            {runs.length === 0 && (
                <section className="panel">
                    <p>No runs recorded yet. In <code>backend/</code>:</p>
                    <pre>python -m bench run --label baseline</pre>
                </section>
            )}

            {runs.length > 0 && (
                <section className="panel controls">
                    <label>
                        <span>Base</span>
                        <select value={baseId ?? ""} onChange={e => setBaseId(e.target.value || undefined)}>
                            <option value="">—</option>
                            {[...runs].reverse().map(r => <option key={r.id} value={r.id}>{runName(r)}</option>)}
                        </select>
                    </label>
                    <span className="arrow" aria-hidden>→</span>
                    <label>
                        <span>Head</span>
                        <select value={headId ?? ""} onChange={e => setHeadId(e.target.value)}>
                            {[...runs].reverse().map(r => <option key={r.id} value={r.id}>{runName(r)}</option>)}
                        </select>
                    </label>
                    <label className="toggle">
                        <input type="checkbox" checked={changesOnly} onChange={e => setChangesOnly(e.target.checked)} />
                        Only changes
                    </label>
                </section>
            )}

            {error && runs.length > 0 && <p className="bench-error">{error}</p>}
            {runs.length === 1 && <p className="muted">Only one run recorded: record another to compare.</p>}

            {comparison && (
                <>
                    <section className="summary">
                        {(["regressed", "improved", "unchanged"] as Status[]).map(s => (
                            <div key={s} className="stat">
                                <div className="stat-value">{comparison.counts[s] ?? 0}</div>
                                <StatusBadge status={s} />
                            </div>
                        ))}
                        <div className="stat stat-commits">
                            <div><span className="muted">base</span> {comparison.base.git.short} {comparison.base.git.subject}</div>
                            <div><span className="muted">head</span> {comparison.head.git.short}{comparison.head.git.dirty ? " + local changes" : ""} {comparison.head.git.subject}</div>
                        </div>
                    </section>
                    {comparison.warnings.length > 0 && (
                        <ul className="warnings">
                            {comparison.warnings.map(w => <li key={w}><span aria-hidden>⚠</span> {w}</li>)}
                        </ul>
                    )}
                </>
            )}

            {head && Object.keys(head.metrics).some(k => k.startsWith("skill[")) && (
                <section className="panel">
                    <SkillPanel run={head} selected={selected} onSelect={setSelected} />
                </section>
            )}

            {selected && selectedMeta && (
                <section className="panel">
                    <TrendChart runs={runs} metricKey={selected} baseId={baseId} headId={headId} />
                </section>
            )}

            {rowsBySection.map(({ section, rows }) => (
                <section key={section} className="panel">
                    <h2>{SECTION_TITLES[section] ?? section}</h2>
                    <table className="metrics">
                        <thead>
                            <tr><th>Metric</th><th className="num">Base</th><th className="num">Head</th><th className="num">Change</th><th>Status</th></tr>
                        </thead>
                        <tbody>
                            {rows.map(row => (
                                <tr key={row.key}
                                    className={row.key === selected ? "selected" : ""}
                                    onClick={() => setSelected(row.key)}
                                    title={[row.test, row.note, row.threshold_pct ? `changes under ${(row.threshold_pct * 100).toFixed(0)}% treated as noise` : ""].filter(Boolean).join(" · ")}>
                                    <td>
                                        <button className="metric-link" onClick={() => setSelected(row.key)}>{row.label}</button>
                                        {row.note && <div className="row-note">{row.note}</div>}
                                    </td>
                                    <td className="num">{fmtValue(row.base, row.unit)}</td>
                                    <td className="num">{fmtValue(row.head, row.unit)}</td>
                                    <td className="num">{fmtChange(row)}</td>
                                    <td><StatusBadge status={row.status} /></td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </section>
            ))}

            {runs.length > 0 && (
                <section className="panel">
                    <h2>All runs</h2>
                    <table className="metrics runs">
                        <thead>
                            <tr>
                                <th>Run</th><th>Commit</th><th>Suite</th>
                                {selectedMeta && <th className="num">{selectedMeta.label}</th>}
                                <th className="num">Time</th><th></th>
                            </tr>
                        </thead>
                        <tbody>
                            {[...runs].reverse().map(r => (
                                <tr key={r.id} className={r.id === headId ? "selected" : ""}>
                                    <td>{fmtDate(r.started_at)}{r.label && <span className="label">{r.label}</span>}</td>
                                    <td title={r.git.sha ?? ""}>
                                        <code>{r.git.short}{r.git.dirty ? "*" : ""}</code> {r.git.subject}
                                        {r.warnings.length > 0 && <span className="row-note" title={r.warnings.join("\n")}> ⚠ {r.warnings.length}</span>}
                                    </td>
                                    <td>{r.suite}</td>
                                    {selectedMeta && <td className="num">{fmtValue(r.metrics[selected!]?.value, selectedMeta.unit)}</td>}
                                    <td className="num">{Math.round(r.duration_s)}s</td>
                                    <td className="actions">
                                        <button onClick={() => setBaseId(r.id)} disabled={r.id === baseId}>base</button>
                                        <button onClick={() => setHeadId(r.id)} disabled={r.id === headId}>head</button>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </section>
            )}
        </div>
    )
}
