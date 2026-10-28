import { useEffect, useState } from "react"
import { BACKEND_URL } from "../constants"
import { AgentList, MatchList } from "./AgentBuilder"
import { fmtDuration } from "./format"
import { SECTION_TITLES } from "./types"
import type { BenchConfig, BenchOptions, Estimate, JobRequest } from "./types"

const PRESET_TEXT: Record<string, string> = {
    quick: "A fast check while iterating.",
    standard: "Before merging: more games, the learning check and more agents.",
    full: "For milestones: slow MCTS budgets, more networks and longer training.",
}

const SECTION_TEXT: Record<string, string> = {
    rules: "Move generation on the fixed position suite; any change is real.",
    speed: "Engine throughput and per-decision time of each agent.",
    search: "MCTS iterations per second for each configuration.",
    training: "Training throughput, and optionally networks trained from scratch.",
    strength: "Head-to-head matches on seeded, colour-swapped dice.",
    skill: "Winning chance each agent gives up per decision on labelled positions.",
    ratings: "Elo ratings for a pool of agents from adaptively scheduled games.",
}

export function EstimateText({ estimate, sections }: { estimate: Estimate | null, sections?: string[] }) {
    if (!estimate) return <span className="muted">…</span>
    const all = sections ?? Object.keys(estimate.sections)
    const unknown = estimate.unknown.filter(n => all.includes(n))
    // A total covering only a sliver of the work would mislead more than it helps.
    if (estimate.seconds === null || unknown.length * 2 > all.length) {
        return <span className="muted">No earlier run with these settings; estimated once it starts</span>
    }
    return (
        <span>
            ≈ {fmtDuration(estimate.seconds)}
            {unknown.length > 0 && <span className="muted">, plus {unknown.map(n => SECTION_TITLES[n] ?? n).join(", ")} (not estimated)</span>}
        </span>
    )
}

function NumberField({ label, value, onChange, min = 1, step = 1, help }: {
    label: string, value: number, onChange: (v: number) => void, min?: number, step?: number, help?: string
}) {
    return (
        <label title={help}>
            <span>{label}</span>
            <input type="number" min={min} step={step} value={value}
                onChange={e => { const v = Number(e.target.value); if (Number.isFinite(v)) onChange(v) }} />
        </label>
    )
}

interface Props {
    options: BenchOptions
    busy: boolean
    onStart: (request: JobRequest) => Promise<void>
}

/** Start a preset benchmark, or a custom one with sections and their settings chosen here. */
export default function RunPanel({ options, busy, onStart }: Props) {
    const presets = options.preset_order
    const [mode, setMode] = useState<string>(presets[0])
    const [label, setLabel] = useState("")
    const [base, setBase] = useState(presets[0])
    const [cfg, setCfg] = useState<BenchConfig>(() => structuredClone(options.presets[presets[0]].config))
    const [estimate, setEstimate] = useState<Estimate | null>(null)
    const [error, setError] = useState<string | null>(null)
    const [starting, setStarting] = useState(false)
    const custom = mode === "custom"

    // Ask the backend what a custom configuration should take, from earlier runs.
    useEffect(() => {
        if (!custom) return
        setEstimate(null)
        const timer = setTimeout(() => {
            fetch(`${BACKEND_URL}/api/bench/estimate`, {
                method: "POST", headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ kind: "custom", suite: base, config: cfg }),
            }).then(r => r.json()).then(data => {
                if (data.error) { setError(data.error); return }
                setError(null)
                setEstimate(data)
            }).catch(() => {})
        }, 400)
        return () => clearTimeout(timer)
    }, [custom, base, cfg])

    const set = (patch: Partial<BenchConfig>) => setCfg(prev => ({ ...prev, ...patch }))
    const setRatings = (patch: Partial<BenchConfig["ratings"]>) => setCfg(prev => ({ ...prev, ratings: { ...prev.ratings, ...patch } }))
    const setTraining = (patch: Partial<BenchConfig["training"]>) => setCfg(prev => ({ ...prev, training: { ...prev.training, ...patch } }))
    const toggle = (section: string) => set({
        sections: options.sections.filter(s => s === section ? !cfg.sections.includes(s) : cfg.sections.includes(s)),
    })
    const on = (section: string) => cfg.sections.includes(section)

    const start = async () => {
        setStarting(true)
        setError(null)
        try {
            const l = label.trim() || undefined
            await onStart(custom ? { kind: "custom", suite: base, config: cfg, label: l } : { kind: "suite", suite: mode, label: l })
        } catch (e) {
            setError((e as Error).message)
        } finally {
            setStarting(false)
        }
    }

    const preset = !custom ? options.presets[mode] : null
    return (
        <section className="panel run-panel">
            <div className="seg-control" role="tablist" aria-label="Benchmark">
                {[...presets, "custom"].map(p => (
                    <button key={p} type="button" role="tab" aria-selected={mode === p} className={mode === p ? "active" : ""}
                        onClick={() => { setMode(p); setError(null) }}>
                        {p[0].toUpperCase() + p.slice(1)}
                    </button>
                ))}
            </div>

            {preset && (
                <div className="preset">
                    <p>{PRESET_TEXT[mode]}</p>
                    <dl className="facts">
                        <div><dt>Expected time</dt><dd><EstimateText estimate={preset.estimate} /></dd></div>
                        <div><dt>Matches</dt><dd>{preset.config.matches.length} ({preset.config.matches[0]?.pairs} pairs each)</dd></div>
                        <div><dt>Skill agents</dt><dd>{preset.config.skill_agents.length}</dd></div>
                        <div><dt>Rating pool</dt><dd>{preset.config.ratings.pool.length} agents, up to {preset.config.ratings.max_games.toLocaleString()} games</dd></div>
                        <div><dt>Learning check</dt><dd>{preset.config.training.learn_games ? `${preset.config.training.learn_seeds} runs × ${preset.config.training.learn_games.toLocaleString()} games` : "off"}</dd></div>
                    </dl>
                    <p className="muted small">
                        To change agents or turn sections off, choose <button type="button" className="link" onClick={() => {
                            setBase(mode); setCfg(structuredClone(options.presets[mode].config)); setMode("custom")
                        }}>Custom, starting from {mode}</button>.
                    </p>
                </div>
            )}

            {custom && (
                <div className="custom">
                    <div className="row">
                        <label>
                            <span>Start from</span>
                            <select value={base} onChange={e => { setBase(e.target.value); setCfg(structuredClone(options.presets[e.target.value].config)) }}>
                                {presets.map(p => <option key={p} value={p}>{p}</option>)}
                            </select>
                        </label>
                        <button type="button" className="ghost" onClick={() => setCfg(structuredClone(options.presets[base].config))}>Reset to {base}</button>
                    </div>

                    <fieldset className="sections">
                        <legend>Sections</legend>
                        {options.sections.map(s => (
                            <label key={s} className="section-toggle" title={SECTION_TEXT[s]}>
                                <input type="checkbox" checked={on(s)} onChange={() => toggle(s)}
                                    disabled={s === "skill" && !options.has_reference} />
                                <span>{SECTION_TITLES[s] ?? s}</span>
                            </label>
                        ))}
                    </fieldset>
                    {!options.has_reference && <p className="muted small">Decision quality needs <code>python -m bench reference --build</code> first.</p>}

                    {on("speed") && (
                        <div className="section-form">
                            <h3>{SECTION_TITLES.speed}</h3>
                            <p className="muted small">{SECTION_TEXT.speed}</p>
                            <div className="field-label">Agents timed per decision</div>
                            <AgentList options={options} value={cfg.latency_agents} onChange={v => set({ latency_agents: v })} />
                            <div className="row">
                                <NumberField label="Repeats" value={cfg.repeats} onChange={v => set({ repeats: v })} />
                                <NumberField label="Positions" value={cfg.latency_positions} onChange={v => set({ latency_positions: v })} />
                            </div>
                        </div>
                    )}

                    {on("search") && (
                        <div className="section-form">
                            <h3>{SECTION_TITLES.search}</h3>
                            <p className="muted small">{SECTION_TEXT.search}</p>
                            <div className="field-label">MCTS configurations</div>
                            <AgentList options={options} kinds={["mcts"]} value={cfg.mcts_agents} onChange={v => set({ mcts_agents: v })} />
                            <div className="row">
                                <NumberField label="Positions" value={cfg.search_positions} onChange={v => set({ search_positions: v })} />
                                <NumberField label="Repeats" value={cfg.search_repeats} onChange={v => set({ search_repeats: v })} />
                            </div>
                        </div>
                    )}

                    {on("training") && (
                        <div className="section-form">
                            <h3>{SECTION_TITLES.training}</h3>
                            <p className="muted small">{SECTION_TEXT.training} Learning shows only after a few thousand games; 0 skips it.</p>
                            <div className="row">
                                <NumberField label="Learning games per run" min={0} step={1000} value={cfg.training.learn_games} onChange={v => setTraining({ learn_games: v })} />
                                <NumberField label="Learning runs" value={cfg.training.learn_seeds} onChange={v => setTraining({ learn_seeds: v })} />
                                <NumberField label="Evaluation pairs" value={cfg.training.learn_pairs} onChange={v => setTraining({ learn_pairs: v })} />
                                <NumberField label="Epoch games" value={cfg.training.epoch_games} onChange={v => setTraining({ epoch_games: v })} />
                            </div>
                        </div>
                    )}

                    {on("strength") && (
                        <div className="section-form">
                            <h3>{SECTION_TITLES.strength}</h3>
                            <p className="muted small">{SECTION_TEXT.strength} Each pair is two games.</p>
                            <MatchList options={options} value={cfg.matches} onChange={v => set({ matches: v })} />
                        </div>
                    )}

                    {on("skill") && (
                        <div className="section-form">
                            <h3>{SECTION_TITLES.skill}</h3>
                            <p className="muted small">{SECTION_TEXT.skill}</p>
                            <AgentList options={options} value={cfg.skill_agents} onChange={v => set({ skill_agents: v })} />
                            <div className="row">
                                <NumberField label="Repeats (random and MCTS agents)" value={cfg.skill_repeats} onChange={v => set({ skill_repeats: v })} />
                            </div>
                        </div>
                    )}

                    {on("ratings") && (
                        <div className="section-form">
                            <h3>{SECTION_TITLES.ratings}</h3>
                            <p className="muted small">{SECTION_TEXT.ratings}</p>
                            <AgentList options={options} value={cfg.ratings.pool} onChange={v => setRatings({
                                pool: v, anchor: v.includes(cfg.ratings.anchor) ? cfg.ratings.anchor : (v.includes("heuristic") ? "heuristic" : v[0] ?? ""),
                            })} />
                            <div className="row">
                                <label>
                                    <span>Fixed at {cfg.ratings.anchor_elo} Elo</span>
                                    <select value={cfg.ratings.anchor} onChange={e => setRatings({ anchor: e.target.value })}>
                                        {cfg.ratings.pool.map(s => <option key={s} value={s}>{s}</option>)}
                                    </select>
                                </label>
                                <NumberField label="Game budget" step={100} value={cfg.ratings.max_games} onChange={v => setRatings({ max_games: v })} />
                                <NumberField label="Target ± Elo (95%)" step={5} value={Math.round(cfg.ratings.target_se * 1.96)}
                                    onChange={v => setRatings({ target_se: Math.max(1, v / 1.96) })} />
                                <NumberField label="Max pairs per matchup" value={cfg.ratings.max_pairs} onChange={v => setRatings({ max_pairs: v })} />
                            </div>
                        </div>
                    )}

                    <div className="row">
                        <NumberField label="Worker processes" value={cfg.workers} onChange={v => set({ workers: v })}
                            help={`This machine has ${options.cpu_count} logical CPUs`} />
                        <NumberField label="Seed" min={0} value={cfg.seed} onChange={v => set({ seed: v })}
                            help="Keep the default so results pair with earlier runs game by game" />
                    </div>
                    <p className="muted small">
                        Expected time: <EstimateText estimate={estimate} sections={cfg.sections} />
                        {estimate?.suite === "custom" && " · recorded as a custom run, compared with other custom runs"}
                    </p>
                </div>
            )}

            <div className="run-foot">
                <label>
                    <span>Label (optional)</span>
                    <input value={label} onChange={e => setLabel(e.target.value)} placeholder="e.g. movegen-fix" />
                </label>
                <button type="button" className="primary" disabled={busy || starting} onClick={start}
                    title={busy ? "Another benchmark is running" : undefined}>
                    {starting ? "Starting…" : `Start ${custom ? "custom" : mode} benchmark`}
                </button>
                {busy && <span className="muted small">One benchmark runs at a time.</span>}
            </div>
            {error && <p className="bench-error">{error}</p>}
        </section>
    )
}
