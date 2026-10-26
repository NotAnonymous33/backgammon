import { useLayoutEffect, useMemo, useRef, useState } from "react"
import type { MetricSummary, RunSummary } from "./types"
import { fmtTick, niceTicks } from "./format"

const PHASES = ["opening", "contact", "bar", "race", "exact"] as const
// Sequential blue ramp for loss cells; ink flips to white from the fourth step.
const RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
const RAMP_MAX = 0.09

type Family = "nn" | "other" | "mcts" | "emm"
const FAMILIES: Record<Family, { label: string, color: string }> = {
    nn: { label: "Networks", color: "var(--series-1)" },
    other: { label: "Other agents", color: "var(--series-2)" },
    mcts: { label: "MCTS", color: "var(--series-3)" },
    emm: { label: "Expectiminimax", color: "var(--series-4)" },
}

interface Agent {
    spec: string
    name: string
    family: Family
    loss: MetricSummary
    phases: (number | null)[]
    best?: number
    blunders?: number
    timeMs?: number
    iterations?: number
    /** Search settings other than the budget (MCTS iterations, expectiminimax ply); points sharing one form a ladder. */
    config?: string
    budget?: number
    pointLabel: string
}

const HEIGHT = 300
const PAD = { top: 20, right: 24, bottom: 40, left: 52 }

function value(run: RunSummary, key: string): number | undefined {
    const v = run.metrics[key]?.value
    return v === null || v === undefined ? undefined : v
}

function shortName(spec: string) {
    if (!spec.startsWith("nn:")) return spec
    const file = spec.slice(3).split("/").pop() ?? spec
    return `nn:${file.replace(/(_checkpoint)?(_latest)?\.pt$/, "")}`
}

/** Every agent the run's skill section scored. */
function skillAgents(run: RunSummary): Agent[] {
    const agents: Agent[] = []
    for (const key of Object.keys(run.metrics)) {
        const m = key.match(/^skill\[(.+)\]\.loss$/)
        if (!m || run.metrics[key].value === null) continue
        const spec = m[1]
        const base = `skill[${spec}]`
        const family: Family = spec.startsWith("mcts") ? "mcts" : spec.startsWith("emm") ? "emm"
            : spec.startsWith("nn") ? "nn" : "other"
        const agent: Agent = {
            spec, family, name: shortName(spec), loss: run.metrics[key],
            phases: PHASES.map(p => value(run, `${base}.loss.${p}`) ?? null),
            best: value(run, `${base}.best_rate`),
            blunders: value(run, `${base}.blunder_rate`),
            timeMs: value(run, `${base}.decision_ms_mean`) ?? value(run, `${base}.decision_ms`),
            iterations: value(run, `${base}.iterations`),
            pointLabel: shortName(spec),
        }
        const settings = spec.split(":")[1]?.split(",").filter(Boolean) ?? []
        const get = (k: string) => settings.find(s => s.startsWith(`${k}=`))?.split("=")[1]
        if (family === "mcts") {
            const iters = get("iters")
            const time = get("time")
            agent.budget = iters ? Number(iters) : undefined
            agent.config = settings.filter(s => !/^(iters|time)=/.test(s)).sort().join(",")
            const budget = iters ? iters : time ? `${time} s` : ""
            // Variants of the default search are named after their mode (or settings), e.g. "heuristic 400".
            const variant = get("mode") ?? agent.config
            agent.pointLabel = agent.config ? `${variant} ${budget}`.trim() : budget || "mcts"
        } else if (family === "emm") {
            const ply = Number(get("ply") ?? 2)
            agent.budget = ply
            agent.config = settings.filter(s => !s.startsWith("ply=")).sort().join(",")
            agent.pointLabel = agent.config ? `${agent.config} ${ply}-ply` : `${ply}-ply`
        }
        agents.push(agent)
    }
    const order: Record<Family, number> = { nn: 0, other: 1, emm: 2, mcts: 3 }
    return agents.sort((a, b) =>
        order[a.family] - order[b.family]
        || (a.config ?? "").localeCompare(b.config ?? "")
        || (a.budget ?? Infinity) - (b.budget ?? Infinity)
        || a.loss.value! - b.loss.value!)
}

const pct = (v: number | null | undefined, digits = 1) =>
    v === null || v === undefined ? "—" : `${(v * 100).toFixed(digits)}%`

function fmtMs(ms: number) {
    if (ms >= 1000) return `${(ms / 1000).toFixed(ms >= 10000 ? 0 : 1)} s`
    return ms >= 10 ? `${ms.toFixed(0)} ms` : `${ms.toFixed(1)} ms`
}

interface Props {
    run: RunSummary
    selected?: string
    onSelect: (metricKey: string) => void
}

/** Decision quality of every agent in one run: skill against cost, and where each loses. */
export default function SkillPanel({ run, selected, onSelect }: Props) {
    const agents = useMemo(() => skillAgents(run), [run])
    const wrapRef = useRef<HTMLDivElement>(null)
    const [width, setWidth] = useState(720)
    const [hover, setHover] = useState<string | null>(null)

    useLayoutEffect(() => {
        const el = wrapRef.current
        if (!el) return
        setWidth(Math.max(320, el.clientWidth))
        const ro = new ResizeObserver(entries => setWidth(Math.max(320, entries[0].contentRect.width)))
        ro.observe(el)
        return () => ro.disconnect()
    }, [agents.length])

    if (agents.length === 0) return null

    // An agent whose decisions all failed has no loss to plot; with none plotted the chart is left out.
    const plotted = agents.filter(a => a.timeMs !== undefined && a.timeMs > 0 && Number.isFinite(a.loss.value))
    const times = plotted.length > 0 ? plotted.map(a => a.timeMs!) : [1]
    // Extend to whole decades so both ends carry a tick.
    const tLo = Math.min(Math.log10(Math.min(...times) / 1.6), Math.floor(Math.log10(Math.min(...times))))
    const tHi = Math.max(Math.log10(Math.max(...times) * 1.6), Math.ceil(Math.log10(Math.max(...times))))
    const yHi = Math.max(0, ...plotted.map(a => a.loss.ci?.[1] ?? a.loss.value!)) * 1.1 || 0.01
    const innerW = width - PAD.left - PAD.right
    const innerH = HEIGHT - PAD.top - PAD.bottom
    const x = (ms: number) => PAD.left + (Math.log10(ms) - tLo) / (tHi - tLo || 1) * innerW
    const y = (v: number) => PAD.top + innerH - v / yHi * innerH
    const { ticks, step } = niceTicks(0, yHi)
    const xTicks: number[] = []
    for (let p = Math.ceil(tLo); p <= Math.floor(tHi); p++) xTicks.push(10 ** p)

    const ladders = new Map<string, Agent[]>()
    for (const a of plotted) {
        if (a.budget !== undefined) {
            const key = `${a.family}:${a.config}`
            ladders.set(key, [...(ladders.get(key) ?? []), a])
        }
    }
    const families = (Object.keys(FAMILIES) as Family[]).filter(f => agents.some(a => a.family === f))
    const hp = plotted.find(a => a.spec === hover)
    const tipLeft = hp ? Math.min(Math.max(x(hp.timeMs!) + 14, 0), width - 230) : 0

    return (
        <>
            <div className="skill-head">
                <div>
                    <h2>Decision quality by agent</h2>
                    <p className="skill-sub">
                        Winning chance lost per decision on the labelled reference positions (lower is better), against
                        mean decision time. Whiskers are 95% intervals; MCTS iteration budgets and expectiminimax depths are joined into ladders.
                    </p>
                </div>
                <div className="skill-legend">
                    {families.map(f => (
                        <span key={f}><span className="key" style={{ background: FAMILIES[f].color }} />{FAMILIES[f].label}</span>
                    ))}
                </div>
            </div>

            <div className="skill-chart" ref={wrapRef}>
                {plotted.length > 0 && <svg width={width} height={HEIGHT} role="img"
                    aria-label="Winning chance lost per decision against mean decision time, one point per agent">
                    {ticks.map(t => (
                        <g key={t}>
                            <line className="grid" x1={PAD.left} x2={width - PAD.right} y1={y(t)} y2={y(t)} />
                            <text className="tick" x={PAD.left - 8} y={y(t)} dy="0.32em" textAnchor="end">{fmtTick(t, "fraction", step)}</text>
                        </g>
                    ))}
                    {xTicks.map(t => (
                        <g key={t}>
                            <line className="grid" x1={x(t)} x2={x(t)} y1={PAD.top} y2={PAD.top + innerH} />
                            <text className="tick" x={x(t)} y={PAD.top + innerH + 16} textAnchor="middle">{t >= 1000 ? `${t / 1000} s` : `${t} ms`}</text>
                        </g>
                    ))}
                    <line className="axis" x1={PAD.left} x2={width - PAD.right} y1={PAD.top + innerH} y2={PAD.top + innerH} />
                    <text className="axis-title" x={PAD.left + innerW} y={HEIGHT - 4} textAnchor="end">mean time per decision (log scale)</text>

                    {[...ladders.values()].filter(l => l.length > 1).map(l => (
                        <polyline key={`${l[0].family}:${l[0].config}`} className="ladder" stroke={FAMILIES[l[0].family].color}
                            points={[...l].sort((a, b) => a.timeMs! - b.timeMs!).map(a => `${x(a.timeMs!)},${y(a.loss.value!)}`).join(" ")} />
                    ))}

                    {plotted.map(a => {
                        const cx = x(a.timeMs!)
                        const cy = y(a.loss.value!)
                        const color = FAMILIES[a.family].color
                        // Default search budgets are labelled above; variants get a hollow marker and a label below.
                        const searched = a.budget !== undefined || a.family === "mcts"
                        const variant = searched && !!a.config
                        const above = searched && !variant
                        return (
                            <g key={a.spec} className={`target${hover === a.spec ? " active" : ""}`} tabIndex={0}
                                aria-label={`${a.spec}: ${pct(a.loss.value, 2)} lost per decision, ${fmtMs(a.timeMs!)} per decision`}
                                onMouseEnter={() => setHover(a.spec)} onMouseLeave={() => setHover(null)}
                                onFocus={() => setHover(a.spec)} onBlur={() => setHover(null)}
                                onClick={() => onSelect(`skill[${a.spec}].loss`)}
                                onKeyDown={e => { if (e.key === "Enter") onSelect(`skill[${a.spec}].loss`) }}>
                                {a.loss.ci && <line className="whisker" stroke={color} x1={cx} x2={cx} y1={y(a.loss.ci[0])} y2={y(a.loss.ci[1])} />}
                                {variant
                                    ? <circle className="pt-hollow" cx={cx} cy={cy} r={4.5} stroke={color} />
                                    : <circle className="pt" cx={cx} cy={cy} r={5} fill={color} />}
                                <circle className="hit" cx={cx} cy={cy} r={13} />
                                <text className="pt-label" x={above || variant ? cx : cx + 9} y={above ? cy - 10 : variant ? cy + 18 : cy}
                                    dy={above || variant ? 0 : "0.32em"} textAnchor={above || variant ? "middle" : "start"}>{a.pointLabel}</text>
                            </g>
                        )
                    })}
                </svg>}
                {hp && (
                    <div className="trend-tip" style={{ left: tipLeft, top: Math.max(0, y(hp.loss.value!) - 20) }}>
                        <div className="tip-value">{pct(hp.loss.value, 2)} lost</div>
                        {hp.loss.ci && <div className="tip-row">95% CI {pct(hp.loss.ci[0], 2)} – {pct(hp.loss.ci[1], 2)}</div>}
                        <div className="tip-row"><b>{hp.spec}</b></div>
                        <div className="tip-row muted">
                            {fmtMs(hp.timeMs!)} per decision{hp.iterations ? ` · ${Math.round(hp.iterations)} iterations when searching` : ""}
                        </div>
                    </div>
                )}
            </div>

            <table className="metrics skill">
                <thead>
                    <tr>
                        <th>Agent</th>
                        <th className="num">Iterations</th>
                        <th className="num">Mean time</th>
                        <th className="num">Lost per decision</th>
                        {PHASES.map(p => <th key={p} className="phase">{p}</th>)}
                        <th className="num">Best play</th>
                        <th className="num">Blunders</th>
                    </tr>
                </thead>
                <tbody>
                    {agents.map(a => {
                        const key = `skill[${a.spec}].loss`
                        return (
                            <tr key={a.spec} className={selected === key ? "selected" : ""} onClick={() => onSelect(key)} title={a.spec}>
                                <td><span className="key" style={{ background: FAMILIES[a.family].color }} />{a.name}</td>
                                <td className={`num${a.iterations ? "" : " dim"}`}>{a.iterations ? Math.round(a.iterations).toLocaleString() : "—"}</td>
                                <td className="num">{a.timeMs !== undefined ? fmtMs(a.timeMs) : "—"}</td>
                                <td className="num">
                                    {pct(a.loss.value, 2)}
                                    {a.loss.ci && <div className="row-note">{pct(a.loss.ci[0], 2)} – {pct(a.loss.ci[1], 2)}</div>}
                                </td>
                                {a.phases.map((v, i) => {
                                    if (v === null) return <td key={PHASES[i]} className="phase dim">—</td>
                                    const step = Math.min(RAMP.length - 1, Math.max(0, Math.floor(v / RAMP_MAX * RAMP.length)))
                                    return (
                                        <td key={PHASES[i]} className="phase">
                                            <span className="cell" style={{ background: RAMP[step], color: step >= 3 ? "#ffffff" : "#0b0b0b" }}>
                                                {(v * 100).toFixed(1)}
                                            </span>
                                        </td>
                                    )
                                })}
                                <td className="num">{pct(a.best)}</td>
                                <td className="num">{pct(a.blunders)}</td>
                            </tr>
                        )
                    })}
                </tbody>
            </table>
            <p className="row-note">
                Phase cells are percentage points of winning chance lost. <i>exact</i> is scored against a solved bearoff
                database; the other phases against rollouts with the main network. Iterations count only decisions where
                MCTS searched: it skips search for forced moves and races. Click a row to chart it across runs.
            </p>
        </>
    )
}
