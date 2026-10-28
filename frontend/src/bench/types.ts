// Shapes returned by the backend's /api/bench endpoints (backend/bench/api.py).

export type Status = "improved" | "regressed" | "unchanged" | "new" | "removed" | "info"
export type Better = "higher" | "lower" | "none"
export type Kind = "timing" | "count" | "rate" | "mean" | "rating" | "value"

export interface MetricSummary {
    label: string
    value: number | null
    unit: string
    better: Better
    kind: Kind
    section: string
    ci?: [number, number] | null
}

export interface RunSummary {
    id: string
    label: string | null
    started_at: string
    duration_s: number
    suite: string
    sections: string[]
    git: { sha: string | null, short: string | null, branch: string | null, dirty: boolean, subject: string | null }
    host: string
    warnings: string[]
    metrics: Record<string, MetricSummary>
}

export interface CompareRow {
    key: string
    section: string
    label: string
    unit: string
    kind: Kind
    better: Better
    base: number | null
    head: number | null
    delta?: number | null
    delta_pct?: number | null
    diff?: number
    diff_ci?: [number, number]
    threshold_pct?: number
    status: Status
    note?: string
    test?: string
}

export interface Comparison {
    base: RunSummary
    head: RunSummary
    warnings: string[]
    counts: Partial<Record<Status, number>>
    rows: CompareRow[]
}

export const SECTIONS = ["rules", "speed", "search", "training", "strength", "skill", "ratings"] as const

export const SECTION_TITLES: Record<string, string> = {
    rules: "Rules correctness",
    speed: "Speed",
    search: "MCTS search",
    training: "Neural-network training",
    strength: "Playing strength",
    skill: "Decision quality",
    ratings: "Elo ratings",
}

// Starting runs from the dashboard (backend/bench/jobs.py).

export interface RatingsConfig {
    pool: string[]
    anchor: string
    anchor_elo: number
    initial_pairs: number
    batch_pairs: number
    max_pairs: number
    target_se: number
    max_games: number
}

export interface TrainingConfig {
    learn_games: number
    learn_seeds: number
    learn_pairs: number
    epoch_games: number
    epoch_repeats: number
    speed_repeats: number
    [key: string]: unknown
}

export interface Match { a: string, b: string, pairs: number }

export interface BenchConfig {
    sections: string[]
    matches: Match[]
    latency_agents: string[]
    mcts_agents: string[]
    skill_agents: string[]
    skill_repeats: number
    repeats: number
    latency_positions: number
    search_positions: number
    search_repeats: number
    workers: number
    seed: number
    training: TrainingConfig
    ratings: RatingsConfig
    [key: string]: unknown
}

export interface SectionEstimate { seconds: number | null, source: string | null }

export interface Estimate {
    sections: Record<string, SectionEstimate>
    seconds: number | null
    unknown: string[]
    suite?: string
}

export interface BenchOptions {
    presets: Record<string, { config: BenchConfig, estimate: Estimate }>
    preset_order: string[]
    sections: string[]
    checkpoints: string[]
    default_nn: string
    cpu_count: number
    has_reference: boolean
    mcts: { modes: string[], exploration: Record<string, number>, sim: number, time: number }
}

export type JobState = "starting" | "running" | "finished" | "failed" | "cancelled"

export interface JobSection {
    name: string
    state: "done" | "running" | "pending" | "skipped"
    estimate: number | null
    elapsed: number | null
    source: string | null
    done: number | null
    total: number | null
    unit: string | null
}

export interface Job {
    id: string
    kind: "suite" | "custom" | "tournament"
    suite: string
    label: string | null
    created: string
    sections: string[]
    state: JobState
    run_id: string | null
    error: string | null
    estimate: Estimate
    pool?: string[]
    sections_detail?: JobSection[]
    elapsed?: number
    remaining?: number
    unknown?: string[]
    log?: string[]
}

export type JobRequest =
    | { kind: "suite", suite: string, label?: string }
    | { kind: "custom", suite: string, label?: string, config: Partial<BenchConfig> }
    | { kind: "tournament", label?: string, pool: string[], anchor: string, pairs: number, adaptive: boolean,
        max_pairs: number, target_se: number, max_games: number, workers: number }

/** Full run record (GET /api/bench/runs/<id>); only the parts the dashboard reads. */
export interface RunRecord {
    id: string
    suite: string
    config: { ratings?: RatingsConfig, sections: string[] }
    metrics: {
        key: string, label: string, value: number | null, unit: string, ci?: [number, number] | null,
        detail?: Record<string, unknown> | null
    }[]
}

export interface Matchup { a: string, b: string, pairs: number, games: number, score: number | null }
