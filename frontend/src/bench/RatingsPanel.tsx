import { useEffect, useState } from "react"
import { BACKEND_URL } from "../constants"
import { RAMP, niceTicks, rampInk, shortName } from "./format"
import type { Matchup, RunRecord } from "./types"

interface Rating {
    spec: string
    elo: number
    ci: [number, number] | null
    games: number
    score: number | null
    bound: "lower" | "upper" | null
    anchor: boolean
}

const BAR_W = 220
const BAR_PAD = 8

function ratingsOf(run: RunRecord): { ratings: Rating[], matchups: Matchup[], games: number } | null {
    const ratings: Rating[] = []
    for (const m of run.metrics) {
        const k = m.key.match(/^ratings\[(.+)\]\.elo$/)
        if (!k || m.value === null) continue
        const d = (m.detail ?? {}) as { games?: number, score?: number | null, bound?: "lower" | "upper" | null, anchor?: boolean }
        ratings.push({ spec: k[1], elo: m.value, ci: m.ci ?? null, games: d.games ?? 0, score: d.score ?? null,
            bound: d.bound ?? null, anchor: !!d.anchor })
    }
    if (!ratings.length) return null
    const g = run.metrics.find(m => m.key === "ratings.games")
    const matchups = ((g?.detail as { matchups?: Matchup[] } | undefined)?.matchups) ?? []
    return { ratings: ratings.sort((a, b) => b.elo - a.elo), matchups, games: g?.value ?? 0 }
}

/** Elo standings with 95% intervals, and every matchup's score: the results of a tournament or a ratings section. */
export default function RatingsPanel({ runId }: { runId: string }) {
    const [run, setRun] = useState<RunRecord | null>(null)
    const [hover, setHover] = useState<string | null>(null)

    useEffect(() => {
        let stale = false
        setRun(null)
        fetch(`${BACKEND_URL}/api/bench/runs/${encodeURIComponent(runId)}`)
            .then(r => r.ok ? r.json() : null)
            .then(data => { if (!stale) setRun(data) })
            .catch(() => {})
        return () => { stale = true }
    }, [runId])

    const data = run && ratingsOf(run)
    if (!data) return null
    const { ratings, matchups, games } = data

    // Shared Elo axis for the interval marks.
    const lo = Math.min(...ratings.map(r => r.ci?.[0] ?? r.elo))
    const hi = Math.max(...ratings.map(r => r.ci?.[1] ?? r.elo))
    const { ticks } = niceTicks(lo, hi, 3)
    const dLo = Math.min(lo, ticks[0] ?? lo), dHi = Math.max(hi, ticks[ticks.length - 1] ?? hi)
    const x = (v: number) => BAR_PAD + (v - dLo) / (dHi - dLo || 1) * (BAR_W - 2 * BAR_PAD)

    const score: Record<string, { s: number | null, n: number }> = {}
    for (const m of matchups) {
        score[`${m.a}|${m.b}`] = { s: m.score, n: m.games }
        score[`${m.b}|${m.a}`] = { s: m.score === null ? null : 1 - m.score, n: m.games }
    }
    const anchor = ratings.find(r => r.anchor)

    return (
        <section className="panel ratings-panel">
            <h2>{run.suite === "tournament" ? "Tournament results" : "Elo ratings"}</h2>
            <p className="muted small">
                {games.toLocaleString()} games · {anchor ? `${shortName(anchor.spec)} fixed at ${anchor.elo.toFixed(0)} Elo` : ""} ·
                bars are 95% bootstrap intervals
            </p>

            <table className="metrics standings">
                <thead>
                    <tr>
                        <th className="num">#</th><th>Agent</th><th className="num">Elo</th>
                        <th className="interval-head">
                            <svg width={BAR_W} height={16} aria-hidden>
                                {ticks.map(t => <text key={t} className="tick" x={x(t)} y={12} textAnchor="middle">{t}</text>)}
                            </svg>
                        </th>
                        <th className="num">95% interval</th><th className="num">Games</th><th className="num">Score</th>
                    </tr>
                </thead>
                <tbody>
                    {ratings.map((r, i) => (
                        <tr key={r.spec} title={r.spec}>
                            <td className="num">{i + 1}</td>
                            <td>
                                {shortName(r.spec)}
                                {r.anchor && <span className="label">anchor</span>}
                                {r.bound && <div className="row-note">{r.bound === "lower" ? "won every game: a lower bound" : "lost every game: an upper bound"}</div>}
                            </td>
                            <td className="num">{r.bound === "lower" ? "≥ " : r.bound === "upper" ? "≤ " : ""}{r.elo.toFixed(0)}</td>
                            <td className="interval">
                                <svg width={BAR_W} height={18} role="img" aria-label={`${r.elo.toFixed(0)} Elo${r.ci ? `, 95% interval ${r.ci[0].toFixed(0)} to ${r.ci[1].toFixed(0)}` : ""}`}>
                                    {ticks.map(t => <line key={t} className="grid" x1={x(t)} x2={x(t)} y1={0} y2={18} />)}
                                    {r.ci && <line className="whisker" x1={x(r.ci[0])} x2={x(r.ci[1])} y1={9} y2={9} />}
                                    <circle className="pt" cx={x(r.elo)} cy={9} r={4.5} />
                                </svg>
                            </td>
                            <td className="num">{r.anchor ? "fixed" : r.ci ? `${r.ci[0].toFixed(0)} – ${r.ci[1].toFixed(0)}` : "—"}</td>
                            <td className="num">{r.games.toLocaleString()}</td>
                            <td className="num">{r.score === null ? "—" : `${(r.score * 100).toFixed(1)}%`}</td>
                        </tr>
                    ))}
                </tbody>
            </table>

            {matchups.length > 0 && (
                <>
                    <h3>Head to head</h3>
                    <p className="muted small">Row agent's score against the column agent. Hover a cell for the games played.</p>
                    <div className="matrix-wrap">
                        <table className="matrix" onMouseLeave={() => setHover(null)}>
                            <thead>
                                <tr>
                                    <th></th>
                                    {ratings.map((r, j) => <th key={r.spec} scope="col" title={r.spec}>{j + 1}</th>)}
                                </tr>
                            </thead>
                            <tbody>
                                {ratings.map((row, i) => (
                                    <tr key={row.spec}>
                                        <th scope="row" title={row.spec}>{i + 1}. {shortName(row.spec)}</th>
                                        {ratings.map(col => {
                                            if (row.spec === col.spec) return <td key={col.spec} className="diag" aria-hidden />
                                            const cell = score[`${row.spec}|${col.spec}`]
                                            if (!cell || cell.s === null) return <td key={col.spec} className="empty">—</td>
                                            const step = Math.min(RAMP.length - 1, Math.floor(cell.s * RAMP.length))
                                            const id = `${row.spec}|${col.spec}`
                                            return (
                                                <td key={col.spec} tabIndex={0} onMouseEnter={() => setHover(id)} onFocus={() => setHover(id)}
                                                    onBlur={() => setHover(null)}>
                                                    <span className="cell" style={{ background: RAMP[step], color: rampInk(step) }}>
                                                        {(cell.s * 100).toFixed(0)}%
                                                    </span>
                                                    {hover === id && (
                                                        <div className="trend-tip cell-tip" role="tooltip">
                                                            <div className="tip-value">{(cell.s * 100).toFixed(1)}%</div>
                                                            <div>{shortName(row.spec)} against {shortName(col.spec)}</div>
                                                            <div className="tip-row muted">{cell.n.toLocaleString()} games ({cell.n / 2} colour-swapped pairs)</div>
                                                        </div>
                                                    )}
                                                </td>
                                            )
                                        })}
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                    <div className="scale-legend" aria-hidden>
                        <span>0%</span>
                        {RAMP.map(c => <span key={c} className="swatch" style={{ background: c }} />)}
                        <span>100%</span>
                        <span className="muted">row's score</span>
                    </div>
                </>
            )}
        </section>
    )
}
