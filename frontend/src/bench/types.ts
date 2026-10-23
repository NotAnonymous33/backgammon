// Shapes returned by the backend's /api/bench endpoints (backend/bench/api.py).

export type Status = "improved" | "regressed" | "unchanged" | "new" | "removed" | "info"
export type Better = "higher" | "lower" | "none"
export type Kind = "timing" | "count" | "rate" | "value"

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

export const SECTIONS = ["rules", "speed", "search", "training", "strength"] as const

export const SECTION_TITLES: Record<string, string> = {
    rules: "Rules correctness",
    speed: "Speed",
    search: "MCTS search",
    training: "Neural-network training",
    strength: "Playing strength",
}
