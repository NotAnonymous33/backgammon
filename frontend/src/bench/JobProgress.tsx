import { useEffect, useState } from "react"
import { fmtClock, fmtDuration, shortName } from "./format"
import { SECTION_TITLES } from "./types"
import type { Job, JobSection } from "./types"

const STATE_TEXT: Record<Job["state"], { icon: string, text: string, cls: string }> = {
    starting: { icon: "●", text: "starting", cls: "info" },
    running: { icon: "●", text: "running", cls: "info" },
    finished: { icon: "✓", text: "finished", cls: "improved" },
    failed: { icon: "✕", text: "failed", cls: "regressed" },
    cancelled: { icon: "−", text: "cancelled", cls: "unchanged" },
}

function jobTitle(job: Job) {
    const what = job.kind === "tournament"
        ? `Tournament: ${(job.pool ?? []).map(shortName).join(", ")}`
        : job.kind === "custom" ? `Custom benchmark (${job.suite === "custom" ? "changed settings" : `${job.suite}, some sections`})`
            : `${job.suite[0].toUpperCase()}${job.suite.slice(1)} benchmark`
    return job.label ? `${what} · ${job.label}` : what
}

/** Fraction of a section's work done: counted items where the section reports them, else time. */
function fraction(s: JobSection): number {
    if (s.state === "done") return 1
    if (s.state !== "running") return 0
    if (s.total) return Math.min(1, (s.done ?? 0) / s.total)
    if (s.estimate && s.elapsed !== null) return Math.min(0.95, s.elapsed / s.estimate)
    return 0
}

function sectionTime(s: JobSection, tick: number): string {
    if (s.state === "done") return fmtDuration(s.elapsed ?? 0)
    if (s.state === "running") {
        const elapsed = (s.elapsed ?? 0) + tick
        return s.estimate ? `${fmtClock(elapsed)} of ~${fmtDuration(s.estimate)}` : fmtClock(elapsed)
    }
    if (s.state === "skipped") return "not run"
    return s.estimate ? `~${fmtDuration(s.estimate)}` : "not estimated"
}

const SOURCE_TEXT: Record<string, string> = {
    "same settings": "from earlier runs with the same settings",
    "same suite": "from the last run of this suite (its settings differed)",
    "this run's progress": "from this run's progress so far",
    "game budget (upper bound)": "if every budgeted game is played; usually fewer are",
}

interface Props {
    job: Job
    /** When the job was last fetched (ms), so clocks keep moving between polls. */
    fetchedAt: number
    onCancel: () => void
    onDismiss: () => void
    onOpenRun: (runId: string) => void
}

/** A running (or just finished) benchmark: a meter with one segment per section, and time left. */
export default function JobProgress({ job, fetchedAt, onCancel, onDismiss, onOpenRun }: Props) {
    const [now, setNow] = useState(Date.now())
    const [hover, setHover] = useState<number | null>(null)
    const running = job.state === "running" || job.state === "starting"
    useEffect(() => {
        if (!running) return
        const id = setInterval(() => setNow(Date.now()), 1000)
        return () => clearInterval(id)
    }, [running])

    const tick = running ? Math.max(0, (now - fetchedAt) / 1000) : 0
    const sections = job.sections_detail ?? job.sections.map(name => ({
        name, state: "pending", estimate: job.estimate.sections[name]?.seconds ?? null, elapsed: null,
        source: job.estimate.sections[name]?.source ?? null, done: null, total: null, unit: null,
    }) as JobSection)
    const known = sections.map(s => s.estimate).filter((e): e is number => e !== null && e > 0)
    const fallback = known.length ? known.reduce((a, b) => a + b, 0) / known.length : 1
    const elapsed = (job.elapsed ?? 0) + tick
    const remaining = job.remaining === undefined ? null : Math.max(0, job.remaining - tick)
    const unknown = job.unknown ?? job.estimate.unknown
    const total = remaining === null ? null : elapsed + remaining
    const finishAt = remaining === null ? null : new Date(now + remaining * 1000)
    const st = STATE_TEXT[job.state]

    let timeLeft: string
    if (!running) timeLeft = "—"
    else if (remaining === null || (remaining === 0 && unknown.length)) timeLeft = "estimating…"
    else timeLeft = `${unknown.length ? "at least " : "≈ "}${fmtDuration(remaining)}`

    return (
        <section className="panel job" aria-live="polite">
            <div className="job-head">
                <div>
                    <h2>{jobTitle(job)}</h2>
                    <span className={`status status-${st.cls}`}><span className="status-icon" aria-hidden>{st.icon}</span>{st.text}</span>
                    {job.error && <p className="bench-error job-error">{job.error}</p>}
                </div>
                <div className="job-actions">
                    {running && <button type="button" onClick={onCancel}>Cancel</button>}
                    {job.state === "finished" && job.run_id && <button type="button" className="primary" onClick={() => onOpenRun(job.run_id!)}>View results</button>}
                    {!running && <button type="button" className="ghost" onClick={onDismiss}>Dismiss</button>}
                </div>
            </div>

            <div className="job-stats">
                <div className="stat">
                    <div className="stat-label">Time left</div>
                    <div className="stat-value">{timeLeft}</div>
                </div>
                <div className="stat">
                    <div className="stat-label">Elapsed</div>
                    <div className="stat-value">{fmtClock(elapsed)}</div>
                </div>
                <div className="stat">
                    <div className="stat-label">{running ? "Finishes around" : "Took"}</div>
                    <div className="stat-value">
                        {running
                            ? finishAt && !unknown.length && remaining ? finishAt.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" }) : "—"
                            : fmtDuration(elapsed)}
                    </div>
                </div>
            </div>

            <div className="meter" role="progressbar" aria-label="Benchmark progress"
                aria-valuemin={0} aria-valuemax={100}
                aria-valuenow={total ? Math.round(elapsed / total * 100) : undefined}
                aria-valuetext={running ? `${timeLeft} left` : st.text}
                onMouseLeave={() => setHover(null)}>
                {sections.map((s, i) => {
                    const f = fraction(s)
                    const left = s.state === "running" && s.estimate ? Math.max(0, s.estimate - (s.elapsed ?? 0) - tick) : null
                    return (
                        <div key={s.name} className={`segment segment-${s.state}`}
                            style={{ flexGrow: s.estimate && s.estimate > 0 ? s.estimate : fallback }}
                            tabIndex={0} onMouseEnter={() => setHover(i)} onFocus={() => setHover(i)} onBlur={() => setHover(null)}>
                            <div className="track"><div className="fill" style={{ width: `${f * 100}%` }} /></div>
                            <div className="segment-name">{SECTION_TITLES[s.name] ?? s.name}</div>
                            <div className="segment-time">{sectionTime(s, tick)}</div>
                            {hover === i && (
                                <div className="trend-tip segment-tip" role="tooltip">
                                    <div className="tip-value">{SECTION_TITLES[s.name] ?? s.name}</div>
                                    <div>{s.state === "running" ? "running" : s.state === "done" ? "done" : s.state === "skipped" ? "not run" : "waiting"}</div>
                                    {s.state === "running" && s.total ? <div>{s.done ?? 0} of {s.total} {s.unit}</div> : null}
                                    {s.state === "running" && left !== null && <div>~{fmtDuration(left)} left</div>}
                                    {s.state !== "done" && (
                                        <div className="tip-row muted">
                                            {s.estimate ? `Estimate ${s.source ? SOURCE_TEXT[s.source] ?? s.source : ""}` : "No earlier run with these settings; estimated once it starts."}
                                        </div>
                                    )}
                                </div>
                            )}
                        </div>
                    )
                })}
            </div>

            {running && unknown.length > 0 && (
                <p className="muted job-note">
                    Not estimated yet: {unknown.map(n => SECTION_TITLES[n] ?? n).join(", ")}. No earlier run used these settings, so
                    {unknown.length > 1 ? " each is" : " it is"} estimated from its own progress once it starts.
                </p>
            )}

            {job.log && job.log.length > 0 && (
                <details className="job-log" open={job.state === "failed"}>
                    <summary>Output</summary>
                    <pre>{job.log.join("\n")}</pre>
                </details>
            )}
        </section>
    )
}
