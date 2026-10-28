import { useState } from "react"
import { shortName } from "./format"
import type { BenchOptions, Match } from "./types"

// Agent specs (backend/bench/agents.py): `kind[:target][,key=value...]`.

type Kind = "random" | "first" | "heuristic" | "nn" | "mcts" | "emm"
const KINDS: { kind: Kind, label: string }[] = [
    { kind: "nn", label: "Neural network" },
    { kind: "mcts", label: "MCTS" },
    { kind: "emm", label: "Expectiminimax" },
    { kind: "heuristic", label: "Heuristic" },
    { kind: "first", label: "First legal move" },
    { kind: "random", label: "Random" },
]

const MODE_HELP: Record<string, string> = {
    neural: "the network scores each leaf",
    rollout: "random play from each leaf to the end of the game",
    heuristic: "random plies from each leaf, then the heuristic scores it",
}

interface Draft {
    kind: Kind
    nn: string
    budget: "iters" | "time"
    iters: string
    time: string
    mode: string
    depth: string
    c: string
    ply: string
    /** Settings the form has no field for (e.g. sim, width), kept so editing does not drop them. */
    extra: string[]
}

function parseSpec(spec: string, defaultNn: string): Draft {
    const draft: Draft = { kind: "mcts", nn: defaultNn, budget: "iters", iters: "400", time: "1.0", mode: "neural",
        depth: "", c: "", ply: "2", extra: [] }
    const [head, ...rest] = spec.trim().split(":")
    const parts = rest.join(":").split(",").filter(Boolean)
    draft.kind = (KINDS.some(k => k.kind === head) ? head : "mcts") as Kind
    for (const part of parts) {
        const [key, value] = part.includes("=") ? part.split("=", 2) : ["path", part]
        if (draft.kind === "nn" && key === "path") draft.nn = value
        else if (key === "nn") draft.nn = value
        else if (draft.kind === "mcts" && key === "iters") { draft.budget = "iters"; draft.iters = value }
        else if (draft.kind === "mcts" && key === "time" && !parts.some(p => p.startsWith("iters="))) { draft.budget = "time"; draft.time = value }
        else if (draft.kind === "mcts" && key === "mode") draft.mode = value === "default" ? "rollout" : value
        else if (draft.kind === "mcts" && key === "depth") draft.depth = value
        else if (draft.kind === "mcts" && key === "c") draft.c = value
        else if (draft.kind === "emm" && key === "ply") draft.ply = value
        else draft.extra.push(part)
    }
    return draft
}

function buildSpec(d: Draft, defaultNn: string): string {
    const nn = d.nn && d.nn !== defaultNn ? d.nn : null
    if (d.kind === "nn") return nn ? `nn:${nn}` : "nn"
    if (d.kind === "mcts") {
        const parts = [d.budget === "iters" ? `iters=${d.iters}` : `time=${d.time}`]
        if (d.mode !== "neural") parts.push(`mode=${d.mode}`)
        if (d.depth !== "") parts.push(`depth=${d.depth}`)
        if (d.c !== "") parts.push(`c=${d.c}`)
        if (d.mode === "neural" && nn) parts.push(`nn=${nn}`)
        return `mcts:${[...parts, ...d.extra].join(",")}`
    }
    if (d.kind === "emm") return `emm:${[`ply=${d.ply}`, ...(nn ? [`nn=${nn}`] : []), ...d.extra].join(",")}`
    return d.kind
}

function problem(d: Draft): string | null {
    const whole = (v: string) => /^\d+$/.test(v)
    const positive = (v: string) => Number(v) > 0 && Number.isFinite(Number(v))
    if (d.kind === "mcts") {
        if (d.budget === "iters" && !(whole(d.iters) && Number(d.iters) > 0)) return "Iterations must be a whole number above 0."
        if (d.budget === "time" && !positive(d.time)) return "Time must be above 0."
        if (d.depth !== "" && !whole(d.depth)) return "Depth must be a whole number."
        if (d.c !== "" && !(Number(d.c) >= 0)) return "Exploration must be 0 or more."
    }
    if (d.kind === "emm" && !["1", "2", "3"].includes(d.ply)) return "Ply must be 1, 2 or 3."
    return null
}

/** Checkpoints grouped by folder, default first. */
function CheckpointSelect({ options, value, onChange }: { options: BenchOptions, value: string, onChange: (v: string) => void }) {
    const groups: Record<string, string[]> = {}
    for (const path of options.checkpoints) {
        const folder = path.split("/").slice(0, -1).join("/")
        ;(groups[folder] ??= []).push(path)
    }
    return (
        <select value={value} onChange={e => onChange(e.target.value)}>
            <option value={options.default_nn}>default ({options.default_nn.split("/").pop()})</option>
            {Object.entries(groups).map(([folder, paths]) => (
                <optgroup key={folder} label={folder}>
                    {paths.filter(p => p !== options.default_nn).map(p => <option key={p} value={p}>{p.split("/").pop()}</option>)}
                </optgroup>
            ))}
        </select>
    )
}

interface BuilderProps {
    options: BenchOptions
    initial?: string
    /** Restrict the kinds offered, e.g. MCTS only for search configurations. */
    kinds?: Kind[]
    submitLabel?: string
    onSubmit: (spec: string) => void
    onCancel?: () => void
}

/** Form for one agent spec, with every MCTS setting. */
export function AgentBuilder({ options, initial, kinds, submitLabel = "Add", onSubmit, onCancel }: BuilderProps) {
    const [d, setD] = useState<Draft>(() => {
        const draft = parseSpec(initial ?? "mcts:iters=400", options.default_nn)
        if (kinds && !kinds.includes(draft.kind)) draft.kind = kinds[0]
        return draft
    })
    const [raw, setRaw] = useState<string | null>(null)
    const set = (patch: Partial<Draft>) => setD(prev => ({ ...prev, ...patch }))
    const spec = raw ?? buildSpec(d, options.default_nn)
    const error = raw === null ? problem(d) : (raw.trim() ? null : "Type a spec.")
    const offered = KINDS.filter(k => !kinds || kinds.includes(k.kind))
    const defaultC = options.mcts.exploration[d.mode]
    const defaultDepth = d.mode === "rollout" ? `${options.mcts.sim} (to game end)` : "0"

    return (
        <div className="builder">
            <div className="builder-fields">
                <label>
                    <span>Agent</span>
                    <select value={d.kind} onChange={e => { setRaw(null); set({ kind: e.target.value as Kind, extra: [] }) }}>
                        {offered.map(k => <option key={k.kind} value={k.kind}>{k.label}</option>)}
                    </select>
                </label>

                {d.kind === "nn" && (
                    <label className="wide">
                        <span>Checkpoint</span>
                        <CheckpointSelect options={options} value={d.nn} onChange={nn => set({ nn })} />
                    </label>
                )}

                {d.kind === "mcts" && (
                    <>
                        <fieldset className="budget">
                            <legend>Budget per move</legend>
                            <label className="inline">
                                <input type="radio" checked={d.budget === "iters"} onChange={() => set({ budget: "iters" })} />
                                <input type="number" min={1} step={100} value={d.iters} aria-label="Iterations"
                                    onFocus={() => set({ budget: "iters" })} onChange={e => set({ iters: e.target.value })} />
                                iterations
                            </label>
                            <label className="inline">
                                <input type="radio" checked={d.budget === "time"} onChange={() => set({ budget: "time" })} />
                                <input type="number" min={0.01} step={0.1} value={d.time} aria-label="Seconds"
                                    onFocus={() => set({ budget: "time" })} onChange={e => set({ time: e.target.value })} />
                                seconds
                            </label>
                        </fieldset>
                        <label>
                            <span>Leaf evaluation</span>
                            <select value={d.mode} onChange={e => set({ mode: e.target.value })}>
                                {options.mcts.modes.map(m => <option key={m} value={m}>{m}</option>)}
                            </select>
                        </label>
                        {d.mode === "neural" && (
                            <label className="wide">
                                <span>Network</span>
                                <CheckpointSelect options={options} value={d.nn} onChange={nn => set({ nn })} />
                            </label>
                        )}
                        <label>
                            <span>Rollout depth (plies)</span>
                            <input type="number" min={0} value={d.depth} placeholder={`default ${defaultDepth}`}
                                onChange={e => set({ depth: e.target.value })} />
                        </label>
                        <label>
                            <span>Exploration c</span>
                            <input type="number" min={0} step={0.01} value={d.c} placeholder={`default ${defaultC}`}
                                onChange={e => set({ c: e.target.value })} />
                        </label>
                        <p className="builder-help">
                            Leaf evaluation {d.mode}: {MODE_HELP[d.mode]}. Rollout depth is the random plies played from a leaf before it is scored.
                            Iteration budgets are reproducible; time budgets depend on the machine.
                        </p>
                    </>
                )}

                {d.kind === "emm" && (
                    <>
                        <label>
                            <span>Ply</span>
                            <select value={d.ply} onChange={e => set({ ply: e.target.value })}>
                                {["1", "2", "3"].map(p => <option key={p} value={p}>{p}</option>)}
                            </select>
                        </label>
                        <label className="wide">
                            <span>Network</span>
                            <CheckpointSelect options={options} value={d.nn} onChange={nn => set({ nn })} />
                        </label>
                    </>
                )}
            </div>

            <div className="builder-foot">
                <label className="spec">
                    <span>Spec</span>
                    <input value={spec} onChange={e => setRaw(e.target.value)} spellCheck={false}
                        title="Edit directly for settings the form does not cover" />
                </label>
                {raw !== null && <button type="button" className="ghost" onClick={() => setRaw(null)}>Use form</button>}
                {onCancel && <button type="button" className="ghost" onClick={onCancel}>Cancel</button>}
                <button type="button" className="primary" disabled={!!error} onClick={() => onSubmit(spec.trim())}>{submitLabel}</button>
            </div>
            {error && <p className="field-error">{error}</p>}
        </div>
    )
}

interface ListProps {
    options: BenchOptions
    value: string[]
    onChange: (specs: string[]) => void
    kinds?: Kind[]
    empty?: string
}

/** Removable agent chips; click one to change its settings. */
export function AgentList({ options, value, onChange, kinds, empty = "No agents" }: ListProps) {
    const [editing, setEditing] = useState<number | "new" | null>(null)
    const submit = (spec: string) => {
        if (editing === "new") {
            if (!value.includes(spec)) onChange([...value, spec])
        } else if (editing !== null) {
            onChange(value.map((s, i) => i === editing ? spec : s).filter((s, i, all) => all.indexOf(s) === i))
        }
        setEditing(null)
    }
    return (
        <div className="agent-list">
            <div className="chips">
                {value.length === 0 && <span className="muted">{empty}</span>}
                {value.map((spec, i) => (
                    <span key={spec} className={`chip${editing === i ? " editing" : ""}`}>
                        <button type="button" className="chip-name" title={`${spec}\nClick to change its settings`}
                            onClick={() => setEditing(editing === i ? null : i)}>{shortName(spec)}</button>
                        <button type="button" className="chip-remove" aria-label={`Remove ${spec}`}
                            onClick={() => { onChange(value.filter(s => s !== spec)); setEditing(null) }}>×</button>
                    </span>
                ))}
                {editing !== "new" && <button type="button" className="chip-add" onClick={() => setEditing("new")}>+ Add agent</button>}
            </div>
            {editing !== null && (
                <AgentBuilder key={editing} options={options} kinds={kinds}
                    initial={editing === "new" ? undefined : value[editing]}
                    submitLabel={editing === "new" ? "Add" : "Update"} onSubmit={submit} onCancel={() => setEditing(null)} />
            )}
        </div>
    )
}

/** Head-to-head matches for the strength section. */
export function MatchList({ options, value, onChange }: { options: BenchOptions, value: Match[], onChange: (m: Match[]) => void }) {
    const [adding, setAdding] = useState(false)
    const [a, setA] = useState<string | null>(null)
    const [pairs, setPairs] = useState(100)
    const update = (i: number, patch: Partial<Match>) => onChange(value.map((m, j) => j === i ? { ...m, ...patch } : m))
    return (
        <div className="match-list">
            {value.length > 0 && (
                <table className="metrics compact">
                    <thead><tr><th>Agent A</th><th></th><th>Agent B</th><th className="num">Game pairs</th><th></th></tr></thead>
                    <tbody>
                        {value.map((m, i) => (
                            <tr key={`${m.a}|${m.b}|${i}`}>
                                <td title={m.a}>{shortName(m.a)}</td>
                                <td className="muted">vs</td>
                                <td title={m.b}>{shortName(m.b)}</td>
                                <td className="num">
                                    <input type="number" min={1} value={m.pairs} aria-label={`Game pairs for ${m.a} vs ${m.b}`}
                                        onChange={e => update(i, { pairs: Math.max(1, Number(e.target.value) || 1) })} />
                                </td>
                                <td className="actions">
                                    <button type="button" aria-label={`Remove ${m.a} vs ${m.b}`} onClick={() => onChange(value.filter((_, j) => j !== i))}>remove</button>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            )}
            {!adding && <button type="button" className="chip-add" onClick={() => { setAdding(true); setA(null) }}>+ Add match</button>}
            {adding && (
                <div className="match-builder">
                    <p className="builder-step">
                        {a === null ? "Agent A" : <>Agent A: <code>{a}</code>. Now agent B</>}
                    </p>
                    <AgentBuilder key={a ?? "a"} options={options} initial={a === null ? "nn" : "heuristic"}
                        submitLabel={a === null ? "Next" : "Add match"}
                        onCancel={() => setAdding(false)}
                        onSubmit={spec => {
                            if (a === null) { setA(spec); return }
                            onChange([...value, { a, b: spec, pairs }])
                            setAdding(false)
                        }} />
                    {a !== null && (
                        <label className="inline">
                            Game pairs <input type="number" min={1} value={pairs} onChange={e => setPairs(Math.max(1, Number(e.target.value) || 1))} />
                        </label>
                    )}
                </div>
            )}
        </div>
    )
}
