import { BAR, BLACK_OFF, WHITE_OFF } from "./types"
import type { Move, MoveSequence, Preview, Snapshot } from "./types"

const isOff = (to: number) => to === WHITE_OFF || to === BLACK_OFF

/** Which die a move consumes. Bearing off with a larger die than needed uses the highest die. */
function dieFor(dice: number[], turn: number, [from, to]: Move): number {
    if (from === BAR) return turn === 1 ? to + 1 : 24 - to
    if (isOff(to)) {
        const exact = to === WHITE_OFF ? 24 - from : from + 1
        return dice.includes(exact) ? exact : Math.max(...dice)
    }
    return Math.abs(to - from)
}

/** Apply one move to a preview. Pure: returns a new preview. */
export function applyMove(prev: Preview, move: Move): Preview {
    const [from, to] = move
    const board = [...prev.board]
    const { turn } = prev
    let { whiteBar, blackBar, whiteOff, blackOff } = prev

    if (to === WHITE_OFF) {
        board[from] -= 1
        whiteOff += 1
    } else if (to === BLACK_OFF) {
        board[from] += 1
        blackOff += 1
    } else {
        if (from === BAR) {
            if (turn === 1) whiteBar -= 1
            else blackBar -= 1
        } else {
            board[from] -= turn
        }
        if (board[to] === -turn) {
            // Hit a blot: the opponent's checker goes to the bar.
            board[to] = turn
            if (turn === 1) blackBar += 1
            else whiteBar += 1
        } else {
            board[to] += turn
        }
    }

    const dice = [...prev.dice]
    const die = dieFor(dice, turn, move)
    const at = dice.indexOf(die)
    const used = at >= 0 ? dice.splice(at, 1) : []

    return {
        ...prev,
        board,
        dice,
        whiteBar,
        blackBar,
        whiteOff,
        blackOff,
        usedDice: [...prev.usedDice, ...used],
        validMoves: prev.validMoves
            .filter(seq => seq.length > 0 && seq[0][0] === from && seq[0][1] === to)
            .map(seq => seq.slice(1)),
    }
}

/** The position the player sees: authoritative snapshot plus draft moves. */
export function previewOf(snapshot: Snapshot, draft: MoveSequence): Preview {
    return draft.reduce(applyMove, { ...snapshot, usedDice: [] } as Preview)
}

/** Whether `from -> to` is a legal next step of the draft. */
export function isLegalStep(preview: Preview, from: number, to: number): boolean {
    return preview.validMoves.some(seq => seq.length > 0 && seq[0][0] === from && seq[0][1] === to)
}

/** Every destination reachable from `from` as the next step. */
export function legalTargets(preview: Preview, from: number): Set<number> {
    const targets = new Set<number>()
    for (const seq of preview.validMoves) {
        if (seq.length > 0 && seq[0][0] === from) targets.add(seq[0][1])
    }
    return targets
}

/**
 * The server accepts only whole sequences. A draft is submittable when it equals one
 * of them, or when the player has no legal moves at all and passes with an empty draft.
 */
export function isComplete(preview: Preview): boolean {
    return preview.validMoves.length === 0 || preview.validMoves.some(seq => seq.length === 0)
}

/** Number of steps still needed on the shortest remaining valid sequence. */
export function movesRemaining(preview: Preview): number {
    if (preview.validMoves.length === 0) return 0
    return Math.min(...preview.validMoves.map(seq => seq.length))
}

export type ClickOutcome = { select: number | null; move?: Move }

/** What a click on `index` does, given the current selection. */
export function resolveClick(preview: Preview, selected: number | null, index: number): ClickOutcome {
    const owns = (i: number) =>
        i === BAR
            ? (preview.turn === 1 ? preview.whiteBar : preview.blackBar) > 0
            : i >= 0 && i < 24 && preview.board[i] * preview.turn > 0

    if (selected === null) return { select: owns(index) ? index : null }
    if (selected === index) return { select: null }
    if (isLegalStep(preview, selected, index)) return { select: null, move: [selected, index] }
    return { select: owns(index) ? index : null }
}
