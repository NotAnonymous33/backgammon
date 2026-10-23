import type { CompareRow } from "./types"

export function fmtValue(value: number | null | undefined, unit: string): string {
    if (value === null || value === undefined || Number.isNaN(value)) return "—"
    if (unit === "fraction") return `${(value * 100).toFixed(1)}%`
    if (unit === "ms") return `${value.toFixed(2)} ms`
    if (unit.endsWith("/s")) {
        const digits = Math.abs(value) < 100 ? 1 : 0
        return `${value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits })} ${unit}`
    }
    if (Number.isInteger(value)) return `${value} ${unit}`.trim()
    return `${value.toFixed(2)} ${unit}`.trim()
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

const pts = (x: number) => `${x >= 0 ? "+" : ""}${(x * 100).toFixed(1)}`

export function fmtChange(row: CompareRow): string {
    if (row.kind === "rate" && row.diff_ci) {
        const [lo, hi] = row.diff_ci
        return `${pts(row.diff ?? (lo + hi) / 2)} pts [${pts(lo)}, ${pts(hi)}]`
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
