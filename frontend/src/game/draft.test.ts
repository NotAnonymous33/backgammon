import { describe, expect, it } from "vitest"
import { isComplete, legalTargets, movesRemaining, previewOf, resolveClick } from "./draft"
import type { Move, Snapshot } from "./types"

const emptyBoard = () => Array<number>(24).fill(0)

function snapshot(overrides: Partial<Snapshot> = {}): Snapshot {
    return {
        board: emptyBoard(),
        turn: 1,
        dice: [],
        invalidDice: [],
        whiteBar: 0,
        blackBar: 0,
        whiteOff: 0,
        blackOff: 0,
        rolled: true,
        gameOver: false,
        validMoves: [],
        ...overrides,
    }
}

describe("previewOf", () => {
    it("applies a move, consumes its die and narrows the valid sequences", () => {
        const board = emptyBoard()
        board[5] = 2
        const s = snapshot({
            board,
            dice: [3, 4],
            validMoves: [[[5, 8], [5, 9]], [[5, 9], [5, 8]]],
        })
        const p = previewOf(s, [[5, 8]])
        expect(p.board[5]).toBe(1)
        expect(p.board[8]).toBe(1)
        expect(p.dice).toEqual([4])
        expect(p.usedDice).toEqual([3])
        expect(p.validMoves).toEqual([[[5, 9]]])
        expect(s.board[5]).toBe(2) // the snapshot is untouched
    })

    it("sends a hit blot to the bar", () => {
        const board = emptyBoard()
        board[5] = 1
        board[8] = -1
        const p = previewOf(snapshot({ board, dice: [3] }), [[5, 8]])
        expect(p.board[8]).toBe(1)
        expect(p.blackBar).toBe(1)
    })

    it("enters from the bar with the matching die for each colour", () => {
        const white = previewOf(snapshot({ whiteBar: 1, dice: [3, 5] }), [[-1, 2]])
        expect(white.whiteBar).toBe(0)
        expect(white.board[2]).toBe(1)
        expect(white.dice).toEqual([5])

        const black = previewOf(snapshot({ turn: -1, blackBar: 1, dice: [3, 5] }), [[-1, 21]])
        expect(black.board[21]).toBe(-1)
        expect(black.dice).toEqual([5])
    })

    it("bears off with the exact die, else the highest", () => {
        const board = emptyBoard()
        board[20] = 2
        const exact = previewOf(snapshot({ board, dice: [4, 6] }), [[20, 100]])
        expect(exact.whiteOff).toBe(1)
        expect(exact.dice).toEqual([6])

        const over = previewOf(snapshot({ board, dice: [2, 6] }), [[20, 100]])
        expect(over.dice).toEqual([2])

        const blackBoard = emptyBoard()
        blackBoard[2] = -1
        const black = previewOf(snapshot({ turn: -1, board: blackBoard, dice: [3, 1] }), [[2, -100]])
        expect(black.blackOff).toBe(1)
        expect(black.dice).toEqual([1])
    })
})

describe("turn completion", () => {
    const board = emptyBoard()
    board[5] = 2
    const seq: Move[] = [[5, 8], [5, 9]]
    const s = snapshot({ board, dice: [3, 4], validMoves: [seq] })

    it("is incomplete until a whole valid sequence has been played", () => {
        expect(isComplete(previewOf(s, []))).toBe(false)
        expect(isComplete(previewOf(s, [seq[0]]))).toBe(false)
        expect(movesRemaining(previewOf(s, [seq[0]]))).toBe(1)
        expect(isComplete(previewOf(s, seq))).toBe(true)
    })

    it("lets a player with no legal moves pass", () => {
        expect(isComplete(previewOf(snapshot(), []))).toBe(true)
    })
})

describe("legalTargets and resolveClick", () => {
    const board = emptyBoard()
    board[5] = 1
    board[7] = 1
    const p = previewOf(snapshot({ board, dice: [3], validMoves: [[[5, 8]]] }), [])

    it("lists next-step destinations", () => {
        expect([...legalTargets(p, 5)]).toEqual([8])
        expect([...legalTargets(p, 7)]).toEqual([])
    })

    it("selects own checkers, moves on legal targets and deselects otherwise", () => {
        expect(resolveClick(p, null, 5)).toEqual({ select: 5 })
        expect(resolveClick(p, null, 8)).toEqual({ select: null })
        expect(resolveClick(p, 5, 5)).toEqual({ select: null })
        expect(resolveClick(p, 5, 8)).toEqual({ select: null, move: [5, 8] })
        expect(resolveClick(p, 5, 7)).toEqual({ select: 7 })
        expect(resolveClick(p, 5, 10)).toEqual({ select: null })
    })
})
