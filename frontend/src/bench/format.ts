import type { CompareRow } from "./types"

/** Sequential blue ramp (steps 100–700) for magnitude cells, the same in both modes. */
export const RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

/** Ink for text inside a ramp cell: white from the fourth step. */
export const rampInk = (step: number) => step >= 3 ? "#ffffff" : "#0b0b0b"

/** `nn:models/main/main_checkpoint_latest.pt` -> `nn:main`; other specs unchanged. */
export function shortName(spec: string) {
    if (!spec.startsWith("nn:")) return spec
    const file = spec.slice(3).split("/").pop() ?? spec
    return `nn:${file.replace(/(_checkpoint)?(_latest)?\.pt$/, "")}`
}

/** Seconds as "45 s", "12 min" or "1 h 05 min". */
export function fmtDuration(seconds: number): string {
    if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s`
    const minutes = Math.round(seconds / 60)
    if (minutes < 60) return `${minutes} min`
    return `${Math.floor(minutes / 60)} h ${String(minutes % 60).padStart(2, "0")} min`
}

/** Elapsed time as m:ss or h:mm:ss, for a running clock. */
export function fmtClock(seconds: number): string {
    const s = Math.max(0, Math.floor(seconds))
    const hh = Math.floor(s / 3600), mm = Math.floor(s / 60) % 60, ss = s % 60
    const two = (n: number) => String(n).padStart(2, "0")
    return hh ? `${hh}:${two(mm)}:${two(ss)}` : `${mm}:${two(ss)}`
}

export function fmtValue(value: number | null | undefined, unit: string): string {
    if (value === null || value === undefined || Number.isNaN(value)) return "—"
    if (unit === "fraction") return `${(value * 100).toFixed(Math.abs(value) < 0.1 ? 2 : 1)}%`
    if (unit === "Elo") return `${value.toFixed(0)} Elo`
    if (unit === "ms") return `${value.toFixed(2)} ms`
    if (unit.endsWith("/s")) {
        const digits = Math.abs(value) < 100 ? 1 : 0
        return `${value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits })} ${unit}`
    }
    if (Number.isInteger(value)) return `${value} ${unit}`.trim()
    return `${value.toFixed(2)} ${unit}`.trim()
}

/** Round-numbered ticks; steps of 1, 2 or 5 × 10^n keep tick labels short. */
export function niceTicks(lo: number, hi: number, count = 4): { ticks: number[], step: number } {
    const span = hi - lo || Math.abs(hi) || 1
    const raw = span / count
    const mag = 10 ** Math.floor(Math.log10(raw))
    const step = [1, 2, 5, 10].map(m => m * mag).find(s => s >= raw) ?? raw
    const start = Math.ceil(lo / step) * step
    const ticks = []
    for (let i = 0; start + i * step <= hi + step * 1e-9; i++) ticks.push(start + i * step)
    return { ticks, step }
}

/** Decimals needed so values `step` apart print differently (e.g. step 0.2 -> 1). */
function decimalsFor(step: number): number {
    if (!(step > 0)) return 0
    return Math.min(3, Math.max(0, -Math.floor(Math.log10(step) + 1e-9)))
}

/** Short axis-tick form of a value; `step` is the spacing between ticks. */
export function fmtTick(value: number, unit: string, step: number): string {
    if (unit === "fraction") return `${(value * 100).toFixed(decimalsFor(step * 100))}%`
    const max = Math.abs(value)
    if (max >= 1e6) return `${(value / 1e6).toFixed(decimalsFor(step / 1e6))}M`
    if (max >= 1e3) return `${(value / 1e3).toFixed(decimalsFor(step / 1e3))}k`
    return value.toFixed(decimalsFor(step))
}

const pts = (x: number, digits = 1) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(digits)}`

export function fmtChange(row: CompareRow): string {
    if ((row.kind === "rate" || row.kind === "mean") && row.diff_ci && row.unit === "fraction") {
        const [lo, hi] = row.diff_ci
        const digits = row.kind === "mean" ? 2 : 1
        return `${pts(row.diff ?? (lo + hi) / 2, digits)} pts [${pts(lo, digits)}, ${pts(hi, digits)}]`
    }
    if (row.diff_ci) {
        const [lo, hi] = row.diff_ci
        const signed = (x: number) => `${x >= 0 ? "+" : ""}${x.toPrecision(3)}`
        return `${signed(row.diff ?? (lo + hi) / 2)} [${signed(lo)}, ${signed(hi)}]`
    }
    if (row.delta === null || row.delta === undefined) return ""
    if (row.kind === "timing" && row.delta_pct !== null && row.delta_pct !== undefined) {
        return `${row.delta_pct >= 0 ? "+" : ""}${(row.delta_pct * 100).toFixed(1)}%`
    }
    if (row.delta === 0) return "0"
    const d = Number.isInteger(row.delta) ? `${row.delta}` : row.delta.toPrecision(3)
    return row.delta > 0 ? `+${d}` : d
}

export function fmtDate(iso: string): string {
    const d = new Date(iso)
    return d.toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })
}

export function betterText(better: string): string {
    if (better === "higher") return "higher is better"
    if (better === "lower") return "lower is better"
    return "informational"
}
