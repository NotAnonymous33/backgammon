import { useLayoutEffect, useMemo, useRef, useState } from "react"
import type { RunSummary } from "./types"
import { betterText, fmtDate, fmtTick, fmtValue, niceTicks } from "./format"

interface Props {
    runs: RunSummary[]
    metricKey: string
    baseId?: string
    headId?: string
}

const HEIGHT = 240
const PAD = { top: 28, right: 24, bottom: 36, left: 64 }

/** One metric across every recorded run, oldest to newest. */
export default function TrendChart({ runs, metricKey, baseId, headId }: Props) {
    const wrapRef = useRef<HTMLElement>(null)
    const [width, setWidth] = useState(720)
    const [hover, setHover] = useState<number | null>(null)

    const points = useMemo(() => runs
        .map(run => ({ run, m: run.metrics[metricKey] }))
        .filter(p => p.m && p.m.value !== null && p.m.value !== undefined),
        [runs, metricKey])
    const hasPoints = points.length > 0

    // Measure synchronously first: ResizeObserver callbacks only arrive with
    // rendered frames, which background tabs skip.
    useLayoutEffect(() => {
        const el = wrapRef.current
        if (!el) return
        setWidth(Math.max(320, el.clientWidth))
        const ro = new ResizeObserver(entries => setWidth(Math.max(320, entries[0].contentRect.width)))
        ro.observe(el)
        return () => ro.disconnect()
    }, [hasPoints])

    if (points.length === 0) {
        return <div className="trend-empty">No recorded runs contain this metric.</div>
    }
    const meta = points[points.length - 1].m
    const unit = meta.unit

    const values = points.flatMap(p => [p.m.value as number, ...(p.m.ci ?? [])])
    let lo = Math.min(...values)
    let hi = Math.max(...values)
    if (unit === "fraction") { lo = Math.max(0, lo - 0.02); hi = Math.min(1, hi + 0.02) }
    const padY = (hi - lo) * 0.12 || Math.abs(hi) * 0.1 || 1
    if (unit !== "fraction") { lo = Math.max(0, lo - padY); hi = hi + padY }
    if (lo === hi) hi = lo + 1

    const innerW = width - PAD.left - PAD.right
    const innerH = HEIGHT - PAD.top - PAD.bottom
    const x = (i: number) => PAD.left + (points.length === 1 ? innerW / 2 : (i / (points.length - 1)) * innerW)
    const y = (v: number) => PAD.top + innerH - ((v - lo) / (hi - lo)) * innerH
    const { ticks, step: tickStep } = niceTicks(lo, hi)

    const line = points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.m.value as number)}`).join("")
    const withCi = points.every(p => p.m.ci)
    const band = withCi
        ? points.map((p, i) => `${i ? "L" : "M"}${x(i)},${y(p.m.ci![1])}`).join("") +
          [...points].reverse().map((p, j) => `L${x(points.length - 1 - j)},${y(p.m.ci![0])}`).join("") + "Z"
        : null

    // Thin x labels so they never collide (~70px each).
    const every = Math.max(1, Math.ceil(points.length / Math.max(1, Math.floor(innerW / 70))))

    const onMove = (e: React.MouseEvent<SVGSVGElement>) => {
        const rect = e.currentTarget.getBoundingClientRect()
        const px = e.clientX - rect.left
        let best = 0
        points.forEach((_, i) => { if (Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i })
        setHover(best)
    }
    const onKey = (e: React.KeyboardEvent) => {
        if (e.key === "ArrowLeft") setHover(h => Math.max(0, (h ?? points.length) - 1))
        if (e.key === "ArrowRight") setHover(h => Math.min(points.length - 1, (h ?? -1) + 1))
        if (e.key === "Escape") setHover(null)
    }

    const hp = hover !== null ? points[hover] : null
    const tipLeft = hover !== null ? Math.min(Math.max(x(hover) + 12, 0), width - 250) : 0

    return (
        <figure className="trend" ref={wrapRef}>
            <figcaption>
                <span className="trend-title">{meta.label}</span>
                <span className="trend-sub">{betterText(meta.better)} · {points.length} run{points.length === 1 ? "" : "s"}
                    {withCi ? " · band shows 95% CI" : ""}</span>
            </figcaption>
            <svg
                width={width} height={HEIGHT} role="img" tabIndex={0}
                aria-label={`${meta.label} across ${points.length} runs; use arrow keys to inspect`}
                onMouseMove={onMove} onMouseLeave={() => setHover(null)} onKeyDown={onKey}
            >
                {ticks.map(t => (
                    <g key={t}>
                        <line className="grid" x1={PAD.left} x2={width - PAD.right} y1={y(t)} y2={y(t)} />
                        <text className="tick" x={PAD.left - 8} y={y(t)} dy="0.32em" textAnchor="end">{fmtTick(t, unit, tickStep)}</text>
                    </g>
                ))}
                <line className="axis" x1={PAD.left} x2={width - PAD.right} y1={PAD.top + innerH} y2={PAD.top + innerH} />
                {points.map((p, i) => i % every === 0 || i === points.length - 1 ? (
                    <text key={p.run.id} className="tick" x={x(i)} y={HEIGHT - PAD.bottom + 18}
                        textAnchor={points.length > 1 && i === 0 ? "start" : points.length > 1 && i === points.length - 1 ? "end" : "middle"}>
                        {p.run.label || p.run.git.short || p.run.id.slice(9, 15)}
                    </text>
                ) : null)}
                {band && <path className="band" d={band} />}
                <path className="series" d={line} />
                {hp && hover !== null && (
                    <line className="crosshair" x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={PAD.top + innerH} />
                )}
                {points.map((p, i) => {
                    const role = p.run.id === headId ? "head" : p.run.id === baseId ? "base" : null
                    return (
                        <g key={p.run.id}>
                            <circle className={`dot${role ? " dot-key" : ""}${hover === i ? " dot-hover" : ""}`}
                                cx={x(i)} cy={y(p.m.value as number)} r={role || hover === i ? 5 : 4} />
                            {role && (
                                <text className="dot-label" x={x(i)} y={y(p.m.value as number) - 12} textAnchor="middle">{role}</text>
                            )}
                        </g>
                    )
                })}
            </svg>
            {hp && (
                <div className="trend-tip" style={{ left: tipLeft, top: PAD.top + 34 }}>
                    <div className="tip-value">{fmtValue(hp.m.value, unit)}</div>
                    {hp.m.ci && <div className="tip-row">95% CI {fmtValue(hp.m.ci[0], unit)} – {fmtValue(hp.m.ci[1], unit)}</div>}
                    <div className="tip-row">{hp.run.label ? <b>{hp.run.label} · </b> : null}{hp.run.git.short}{hp.run.git.dirty ? "*" : ""} · {hp.run.suite}</div>
                    <div className="tip-row muted">{hp.run.git.subject}</div>
                    <div className="tip-row muted">{fmtDate(hp.run.started_at)}</div>
                </div>
            )}
        </figure>
    )
}
