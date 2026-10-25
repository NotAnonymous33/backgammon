import { useEffect, useMemo, useRef, useState } from "react"
import { ROLL_MS, buildReplay, highlightMarks, moveNotation, stepPause } from "../game/replay"
import type { Landings, Mark } from "../game/replay"
import type { Preview, Role, Side, Snapshot } from "../game/types"
import type { Replay } from "./sessionReducer"

const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))
const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches

export type LastMoveHighlight = {
    /** e.g. "White played 24/18 18/13", ready to show. */
    text: string
    marks: Mark[]
}

export type MoveReplayView = {
    /** Someone else's move is being played back; the board is not ready for input. */
    playing: boolean
    /** What to show instead of the live position while playing. */
    preview: Preview | null
    landings: Landings | null
    highlight: LastMoveHighlight | null
    /** The opponent's dice are still tumbling. */
    diceSpinning: boolean
}

type Progress = { id: number; step: number; done: boolean }

/**
 * Plays the opponent's last move back one checker at a time, then marks it on the board.
 * The session already holds the final position; this only decides what to show on the way there.
 */
export function useMoveReplay(
    snapshot: Snapshot | null,
    replay: Replay | null,
    role: Role | null,
    turnSide: Side | null,
): MoveReplayView {
    const [progress, setProgress] = useState<Progress>({ id: -1, step: 0, done: false })
    const [diceSpinning, setDiceSpinning] = useState(false)
    const spinUntil = useRef(0)
    const spinTimer = useRef<ReturnType<typeof setTimeout>>()
    const wasRolled = useRef<boolean | null>(null)

    // Spin the dice for every roll, including the opponent's, before their move is replayed.
    useEffect(() => {
        const rolled = snapshot?.rolled ?? null
        const justRolled = wasRolled.current === false && rolled === true
        wasRolled.current = rolled
        if (!justRolled || role === null || role === turnSide || prefersReducedMotion()) return
        spinUntil.current = Date.now() + ROLL_MS
        setDiceSpinning(true)
        clearTimeout(spinTimer.current)
        spinTimer.current = setTimeout(() => setDiceSpinning(false), ROLL_MS)
    }, [snapshot, role, turnSide])
    useEffect(() => () => clearTimeout(spinTimer.current), [])

    const plan = useMemo(
        () => (replay ? buildReplay(replay.before, replay.last, String(replay.id)) : null),
        [replay],
    )
    const animated = replay?.animate === true && plan !== null && plan.steps.length > 0 && !prefersReducedMotion()

    useEffect(() => {
        if (!replay || !plan || !animated) return
        let cancelled = false
        void (async () => {
            const spinning = spinUntil.current - Date.now()
            if (spinning > 0) await sleep(spinning)
            for (const [i, step] of plan.steps.entries()) {
                if (cancelled) return
                setProgress({ id: replay.id, step: i + 1, done: false })
                await sleep(stepPause(step))
            }
            if (!cancelled) setProgress({ id: replay.id, step: plan.steps.length, done: true })
        })()
        return () => {
            cancelled = true
        }
    }, [replay, plan, animated])

    const current = replay !== null && progress.id === replay.id ? progress : null
    const playing = animated && !current?.done
    const step = current?.step ?? 0

    const highlight = useMemo((): LastMoveHighlight | null => {
        if (!replay || !plan || !snapshot || playing) return null
        const who = replay.last.player === 1 ? "White" : "Black"
        const text =
            plan.paths.length > 0
                ? `${who} played ${moveNotation(replay.last.player, plan.paths)}`
                : `${who} had no legal moves`
        return { text, marks: highlightMarks(snapshot, replay.last.player, plan.paths) }
    }, [replay, plan, snapshot, playing])

    return {
        playing,
        preview: playing && plan ? (step === 0 ? plan.start : plan.steps[step - 1].after) : null,
        landings: playing && plan && step > 0 ? plan.steps[step - 1].landings : null,
        highlight,
        diceSpinning,
    }
}
