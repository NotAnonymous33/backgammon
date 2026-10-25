import type { ServerBoard, ServerDice, Snapshot } from "./types"

/** The server sends dice as a string of digits ("34") from the database, or as a list after a roll. */
const toDice = (dice: string | number[] | undefined): number[] =>
    typeof dice === "string" ? Array.from(dice, Number) : (dice ?? []).map(Number)

export function snapshotFromBoard(data: ServerBoard): Snapshot {
    return {
        board: data.positions,
        turn: data.turn === -1 ? -1 : 1,
        dice: toDice(data.dice),
        invalidDice: toDice(data.invalid_dice),
        whiteBar: data.white_bar,
        blackBar: data.black_bar,
        whiteOff: data.white_off,
        blackOff: data.black_off,
        rolled: data.rolled,
        gameOver: Boolean(data.game_over) || data.white_off === 15 || data.black_off === 15,
        validMoves: data.valid_moves ?? [],
    }
}

export function withDice(snapshot: Snapshot, data: ServerDice): Snapshot {
    return {
        ...snapshot,
        dice: toDice(data.dice),
        invalidDice: toDice(data.invalidDice),
        validMoves: data.validMoves ?? [],
        rolled: data.rolled,
    }
}

/** Cheap structural identity, used to ignore re-sends of an unchanged position. */
export const snapshotKey = (s: Snapshot): string => JSON.stringify(s)

export function winnerOf(s: Snapshot): "White" | "Black" | null {
    if (s.whiteOff === 15) return "White"
    if (s.blackOff === 15) return "Black"
    return null
}
