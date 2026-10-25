import { applyMove } from "./draft"
import {
    SVG_HEIGHT,
    CHECKER_RADIUS,
    barCheckerXY,
    bearOffCheckerXY,
    pointCheckerXY,
} from "./geometry"
import type { XY } from "./geometry"
import { BAR, BLACK_OFF, WHITE_OFF, sideOfTurn } from "./types"
import type { LastMove, Move, Preview, Side, Snapshot } from "./types"

export const OPPONENT_STEP_MS = 450
export const LOCAL_STEP_MS = 250
export const DROP_STEP_MS = 120
export const HIT_MS = 350
export const STEP_GAP_MS = 150
export const ROLL_MS = 600

/** One checker's journey within a turn, e.g. 24 to 18 to 13. */
export type MovePath = {
    points: number[]
    hits: boolean[]
}

/** A checker that slides into its slot from (fromX, fromY). */
export type Landing = {
    fromX: number
    fromY: number
    delay: number
    duration: number
}

/** Landings keyed by checker slot. A new `id` restarts the CSS animation. */
export type Landings = {
    id: string
    bySlot: Record<string, Landing>
}

/** Drawn over the board to show where a checker left, where it went, and the route between. */
export type Mark = {
    ghost: XY
    arrow: string
    /** Ring around the checker that ended the path, when it is still there. */
    ring: XY | null
}

export type ReplayStep = {
    after: Preview
    hit: boolean
    landings: Landings
}

export type MoveReplay = {
    /** The position before the move, ready to display. */
    start: Preview
    steps: ReplayStep[]
    paths: MovePath[]
}

// Slot keys double as React keys in BoardView, which is how a landing finds its checker.
export const pointSlot = (index: number, stack: number) => `${index}-${stack}`
export const barSlot = (side: Side, stack: number) => `bar-${side}-${stack}`
export const offSlot = (side: Side, stack: number) => `off-${side}-${stack}`

const isOff = (to: number) => to === WHITE_OFF || to === BLACK_OFF
const barCount = (s: Snapshot, side: Side) => (side === "white" ? s.whiteBar : s.blackBar)
const offCount = (s: Snapshot, side: Side) => (side === "white" ? s.whiteOff : s.blackOff)
const otherSide = (side: Side): Side => (side === "white" ? "black" : "white")

/** Whether `move` lands on a lone opposing checker. */
export function isHit(position: Snapshot, [, to]: Move): boolean {
    return !isOff(to) && position.board[to] === -position.turn
}

/** Slide one checker (and the checker it hits, which then flies to the bar). */
export function stepLandings(
    prev: Snapshot,
    next: Snapshot,
    move: Move,
    hit: boolean,
    duration: number,
    id: string,
    from?: XY,
): Landings {
    const [start, to] = move
    const side = sideOfTurn(prev.turn)
    const source =
        from ??
        (start === BAR
            ? barCheckerXY(side, barCount(prev, side) - 1)
            : pointCheckerXY(start, Math.abs(prev.board[start]) - 1))
    const dest = isOff(to) ? offSlot(side, offCount(next, side) - 1) : pointSlot(to, Math.abs(next.board[to]) - 1)
    const bySlot: Record<string, Landing> = {
        [dest]: { fromX: source.x, fromY: source.y, delay: 0, duration },
    }
    if (hit) {
        // The blot waits for the hitting checker to arrive, then flies to the bar.
        const loser = otherSide(side)
        const blot = pointCheckerXY(to, 0)
        bySlot[barSlot(loser, barCount(next, loser) - 1)] = {
            fromX: blot.x,
            fromY: blot.y,
            delay: duration,
            duration: HIT_MS,
        }
    }
    return { id, bySlot }
}

/** Group steps into per-checker paths so 24/18 18/13 reads as 24/18/13. */
function extendPaths(paths: MovePath[], [from, to]: Move, hit: boolean) {
    const path = paths.find(p => p.points[p.points.length - 1] === from)
    if (path) {
        path.points.push(to)
        path.hits.push(hit)
    } else {
        paths.push({ points: [from, to], hits: [false, hit] })
    }
}

/**
 * Break the move that produced a board into single steps, each with the position and landings
 * to show. Without the earlier position there are no frames to play, only paths (without hits).
 */
export function buildReplay(before: Snapshot | null, last: LastMove, id: string): MoveReplay {
    const paths: MovePath[] = []
    const steps: ReplayStep[] = []
    const start: Preview | null = before && {
        ...before,
        turn: last.player,
        validMoves: [],
        usedDice: [],
    }
    let state = start
    last.sequence.forEach((move, i) => {
        const hit = state ? isHit(state, move) : false
        extendPaths(paths, move, hit)
        if (!state) return
        const after = applyMove(state, move)
        steps.push({ after, hit, landings: stepLandings(state, after, move, hit, OPPONENT_STEP_MS, `${id}-${i}`) })
        state = after
    })
    return { start: start as Preview, steps, paths }
}

/** Pause for a step: the slide itself, a hit's flight to the bar, and a beat before the next one. */
export const stepPause = (step: ReplayStep) => OPPONENT_STEP_MS + (step.hit ? HIT_MS : 0) + STEP_GAP_MS

/** Point number from the mover's perspective, as used in move notation. */
function pointLabel(index: number, player: number) {
    if (index === BAR) return "bar"
    if (isOff(index)) return "off"
    return String(player === 1 ? 24 - index : index + 1)
}

export function moveNotation(player: number, paths: MovePath[]): string {
    return paths
        .map(path => path.points.map((point, i) => pointLabel(point, player) + (path.hits[i] ? "*" : "")).join("/"))
        .join(" ")
}

/** Curved arrow between two checkers, bowing towards the middle of the board. */
function arrowPath(from: XY, to: XY): string {
    const mid = { x: (from.x + to.x) / 2, y: (from.y + to.y) / 2 }
    const control = { x: mid.x, y: mid.y + (SVG_HEIGHT / 2 - mid.y) * 0.6 }
    const towards = (p: XY, distance: number) => {
        const dx = control.x - p.x
        const dy = control.y - p.y
        const length = Math.hypot(dx, dy) || 1
        return { x: p.x + (dx / length) * distance, y: p.y + (dy / length) * distance }
    }
    const start = towards(from, CHECKER_RADIUS)
    const end = towards(to, CHECKER_RADIUS + 4)
    return `M ${start.x} ${start.y} Q ${control.x} ${control.y} ${end.x} ${end.y}`
}

/** The checker a path ended on, if it is still there. */
function landedAt(position: Snapshot, dest: number, arrived: number, side: Side, player: number): XY | null {
    if (isOff(dest)) {
        const stack = offCount(position, side) - 1 - arrived
        return stack >= 0 ? bearOffCheckerXY(side, stack) : null
    }
    if (position.board[dest] * player <= 0) return null
    const stack = Math.abs(position.board[dest]) - 1 - arrived
    return stack >= 0 ? pointCheckerXY(dest, stack) : null
}

/** Marks for a finished move: a ghost where each checker left, an arrow, and a ring where it landed. */
export function highlightMarks(after: Snapshot, player: number, paths: MovePath[]): Mark[] {
    const side = sideOfTurn(player)
    const departures: Record<number, number> = {}
    const arrivals: Record<number, number> = {}
    return paths.map(path => {
        const start = path.points[0]
        const dest = path.points[path.points.length - 1]
        const departed = departures[start] ?? 0
        departures[start] = departed + 1
        const arrived = arrivals[dest] ?? 0
        arrivals[dest] = arrived + 1

        const ghost =
            start === BAR
                ? barCheckerXY(side, barCount(after, side) + departed)
                : pointCheckerXY(start, Math.abs(after.board[start]) + departed)
        const ring = landedAt(after, dest, arrived, side, player)
        const end = ring ?? (isOff(dest) ? bearOffCheckerXY(side, 0) : pointCheckerXY(dest, 0))
        return { ghost, arrow: arrowPath(ghost, end), ring }
    })
}
