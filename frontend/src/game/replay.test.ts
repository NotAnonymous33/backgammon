import { describe, expect, it } from "vitest"
import { buildReplay, highlightMarks, moveNotation } from "./replay"
import type { Snapshot } from "./types"

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

describe("buildReplay", () => {
    it("steps through a move, flagging hits and joining steps into one path", () => {
        const board = emptyBoard()
        board[23] = 1
        board[17] = -1
        const before = snapshot({ board, dice: [6, 5] })
        const replay = buildReplay(before, { player: 1, sequence: [[23, 17], [17, 12]] }, "1")

        expect(replay.steps.map(s => s.hit)).toEqual([true, false])
        expect(replay.steps[0].after.blackBar).toBe(1)
        expect(replay.steps[1].after.board[12]).toBe(1)
        expect(replay.steps[1].after.usedDice).toEqual([6, 5])
        expect(replay.paths).toEqual([{ points: [23, 17, 12], hits: [false, true, false] }])
        expect(moveNotation(1, replay.paths)).toBe("1/7*/12")
    })

    it("lands the hit checker on the bar after the hitter arrives", () => {
        const board = emptyBoard()
        board[23] = 1
        board[17] = -1
        const replay = buildReplay(snapshot({ board, dice: [6] }), { player: 1, sequence: [[23, 17]] }, "1")
        const { bySlot } = replay.steps[0].landings

        expect(bySlot["17-0"].delay).toBe(0)
        expect(bySlot["bar-black-0"].delay).toBe(bySlot["17-0"].duration)
    })

    it("keeps paths but plays nothing when the earlier position is unknown", () => {
        const replay = buildReplay(null, { player: -1, sequence: [[0, 5]] }, "1")
        expect(replay.steps).toEqual([])
        expect(replay.paths).toHaveLength(1)
    })

    it("has no steps when the mover had no legal moves", () => {
        const replay = buildReplay(snapshot(), { player: 1, sequence: [] }, "1")
        expect(replay.steps).toEqual([])
        expect(replay.paths).toEqual([])
    })
})

describe("highlightMarks", () => {
    it("rings the checker that ended a path and omits the ring for a bear-off that left", () => {
        const board = emptyBoard()
        board[12] = 1
        const [onBoard] = highlightMarks(snapshot({ board }), 1, [{ points: [23, 12], hits: [false, false] }])
        expect(onBoard.ring).not.toBeNull()

        const [offBoard] = highlightMarks(snapshot({ board, whiteOff: 0 }), 1, [{ points: [3, 100], hits: [false, false] }])
        expect(offBoard.ring).toBeNull()
    })
})
