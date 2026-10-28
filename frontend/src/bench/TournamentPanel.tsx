import { useEffect, useState } from "react"
import { BACKEND_URL } from "../constants"
import { AgentList } from "./AgentBuilder"
import { EstimateText } from "./RunPanel"
import type { BenchOptions, Estimate, JobRequest } from "./types"

interface Props {
    options: BenchOptions
    busy: boolean
    onStart: (request: JobRequest) => Promise<void>
}

/** Pick agents (with any MCTS settings), play every pair of them, and rate them on one Elo scale. */
export default function TournamentPanel({ options, busy, onStart }: Props) {
    const [pool, setPool] = useState<string[]>(["heuristic", "nn", "mcts:iters=200"])
    const [anchor, setAnchor] = useState("heuristic")
    const [adaptive, setAdaptive] = useState(false)
    const [pairs, setPairs] = useState(50)
    const [maxPairs, setMaxPairs] = useState(200)
    const [target, setTarget] = useState(40)  // ± Elo at 95%
    const [budget, setBudget] = useState(4000)
    const [workers, setWorkers] = useState(options.cpu_count)
    const [label, setLabel] = useState("")
    const [error, setError] = useState<string | null>(null)
    const [starting, setStarting] = useState(false)
    const [estimate, setEstimate] = useState<Estimate | null>(null)

    const matchups = pool.length * (pool.length - 1) / 2
    const games = adaptive ? null : 2 * pairs * matchups
    const request: JobRequest = {
        kind: "tournament", pool, anchor: pool.includes(anchor) ? anchor : pool[0], pairs, adaptive,
        max_pairs: maxPairs, target_se: target / 1.96, max_games: budget, workers, label: label.trim() || undefined,
    }
    const key = JSON.stringify({ ...request, label: undefined })

    useEffect(() => {
        if (pool.length < 2) { setEstimate(null); return }
        const timer = setTimeout(() => {
            fetch(`${BACKEND_URL}/api/bench/estimate`, { method: "POST", headers: { "Content-Type": "application/json" }, body: key })
                .then(r => r.json()).then(data => { if (!data.error) setEstimate(data) }).catch(() => {})
        }, 400)
        return () => clearTimeout(timer)
    }, [key, pool.length])

    const changePool = (next: string[]) => {
        setPool(next)
        if (!next.includes(anchor)) setAnchor(next.includes("heuristic") ? "heuristic" : next[0] ?? "")
    }

    const start = async () => {
        setStarting(true)
        setError(null)
        try {
            await onStart(request)
        } catch (e) {
            setError((e as Error).message)
        } finally {
            setStarting(false)
        }
    }

    return (
        <section className="panel run-panel">
            <h2>Tournament</h2>
            <p className="muted small">
                Every pair of agents plays seeded, colour-swapped game pairs; ratings are fitted on one Elo scale with the
                anchor fixed at 1500. Click an agent to change its settings, including MCTS budget, leaf evaluation, rollout
                depth and exploration.
            </p>

            <div className="field-label">Agents</div>
            <AgentList options={options} value={pool} onChange={changePool} empty="Add at least two agents" />

            <div className="row">
                <label>
                    <span>Fixed at 1500 Elo</span>
                    <select value={request.anchor} onChange={e => setAnchor(e.target.value)} disabled={pool.length === 0}>
                        {pool.map(s => <option key={s} value={s}>{s}</option>)}
                    </select>
                </label>
            </div>

            <fieldset className="format">
                <legend>Format</legend>
                <label className="inline">
                    <input type="radio" checked={!adaptive} onChange={() => setAdaptive(false)} />
                    Round robin: every matchup plays the same number of pairs
                </label>
                <label className="inline">
                    <input type="radio" checked={adaptive} onChange={() => setAdaptive(true)} />
                    Adaptive: add pairs to close matchups until every rating is precise enough
                </label>
            </fieldset>

            <div className="row">
                <label>
                    <span>{adaptive ? "Starting pairs per matchup" : "Pairs per matchup"}</span>
                    <input type="number" min={1} value={pairs} onChange={e => setPairs(Math.max(1, Number(e.target.value) || 1))} />
                </label>
                {adaptive && (
                    <>
                        <label>
                            <span>Target ± Elo (95%)</span>
                            <input type="number" min={2} step={5} value={target} onChange={e => setTarget(Math.max(2, Number(e.target.value) || 2))} />
                        </label>
                        <label>
                            <span>Max pairs per matchup</span>
                            <input type="number" min={pairs} value={maxPairs} onChange={e => setMaxPairs(Math.max(pairs, Number(e.target.value) || pairs))} />
                        </label>
                        <label>
                            <span>Game budget</span>
                            <input type="number" min={2} step={500} value={budget} onChange={e => setBudget(Math.max(2, Number(e.target.value) || 2))} />
                        </label>
                    </>
                )}
                <label title={`This machine has ${options.cpu_count} logical CPUs`}>
                    <span>Worker processes</span>
                    <input type="number" min={1} value={workers} onChange={e => setWorkers(Math.max(1, Number(e.target.value) || 1))} />
                </label>
            </div>

            <p className="muted small">
                {pool.length} agents, {matchups} matchups
                {games !== null ? `, ${games.toLocaleString()} games` : `, at least ${(2 * pairs * matchups).toLocaleString()} and at most ${budget.toLocaleString()} games`}
                {pool.length >= 2 && <> · expected time: <EstimateText estimate={estimate} /></>}
            </p>

            <div className="run-foot">
                <label>
                    <span>Label (optional)</span>
                    <input value={label} onChange={e => setLabel(e.target.value)} placeholder="e.g. mcts-exploration" />
                </label>
                <button type="button" className="primary" disabled={busy || starting || pool.length < 2} onClick={start}
                    title={busy ? "Another benchmark is running" : undefined}>
                    {starting ? "Starting…" : "Start tournament"}
                </button>
                {busy && <span className="muted small">One benchmark runs at a time.</span>}
            </div>
            {error && <p className="bench-error">{error}</p>}
        </section>
    )
}
