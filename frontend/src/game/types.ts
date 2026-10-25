export type Side = "white" | "black"
export type Role = Side | "spectator"
export type AiAgent = "random" | "heuristic" | "mcts" | "neural"

/** [from, to]. `from` is a point index or -1 for the bar; `to` is a point index or +/-100 for bearing off. */
export type Move = [number, number]
export type MoveSequence = Move[]

export const BAR = -1
export const WHITE_OFF = 100
export const BLACK_OFF = -100

/** Wire format of the server's `update_board` event. */
export type ServerBoard = {
    positions: number[]
    turn: number
    dice: string | number[]
    invalid_dice: string | number[]
    white_bar: number
    black_bar: number
    white_off: number
    black_off: number
    rolled: boolean
    valid_moves: MoveSequence[]
    game_over?: boolean
}

/** Wire format of the server's `update_dice` event. */
export type ServerDice = {
    dice: number[]
    invalidDice: number[]
    validMoves: MoveSequence[]
    rolled: boolean
}

/** The board exactly as the server last described it. Never mutated by local play. */
export type Snapshot = {
    board: number[]
    turn: 1 | -1
    dice: number[]
    invalidDice: number[]
    whiteBar: number
    blackBar: number
    whiteOff: number
    blackOff: number
    rolled: boolean
    gameOver: boolean
    validMoves: MoveSequence[]
}

/** A snapshot with the player's draft moves applied on top. */
export type Preview = Snapshot & {
    /** Dice consumed by the draft, in order. */
    usedDice: number[]
}

export const sideOfTurn = (turn: number): Side => (turn === 1 ? "white" : "black")
