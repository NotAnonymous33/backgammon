import { describe, expect, it } from "vitest"
import { initialSession, sessionReducer } from "./sessionReducer"
import type { SessionAction, SessionState } from "./sessionReducer"
import type { ServerBoard } from "../game/types"

const serverBoard = (overrides: Partial<ServerBoard> = {}): ServerBoard => ({
    positions: Array<number>(24).fill(0),
    turn: 1,
    dice: "",
    invalid_dice: "",
    white_bar: 0,
    black_bar: 0,
    white_off: 0,
    black_off: 0,
    rolled: false,
    valid_moves: [],
    ...overrides,
})

const run = (actions: SessionAction[], from: SessionState = initialSession) =>
    actions.reduce(sessionReducer, from)

describe("sessionReducer", () => {
    it("keeps the draft when the same position is re-sent", () => {
        const s = run([
            { type: "board", data: serverBoard() },
            { type: "draft_add", move: [1, 2] },
            { type: "board", data: serverBoard() },
        ])
        expect(s.draft).toEqual([[1, 2]])
        expect(s.revision).toBe(1)
    })

    it("discards the draft when the authoritative position changes", () => {
        const s = run([
            { type: "board", data: serverBoard() },
            { type: "draft_add", move: [1, 2] },
            { type: "board", data: serverBoard({ turn: -1 }) },
        ])
        expect(s.draft).toEqual([])
        expect(s.revision).toBe(2)
    })

    it("parses dice sent as a digit string", () => {
        const s = run([{ type: "board", data: serverBoard({ dice: "34", rolled: true }) }])
        expect(s.snapshot?.dice).toEqual([3, 4])
    })

    it("keeps the draft and unlocks controls when a move is rejected", () => {
        const s = run([
            { type: "board", data: serverBoard({ rolled: true }) },
            { type: "role", role: "white" },
            { type: "draft_add", move: [1, 2] },
            { type: "request", pending: "move" },
            { type: "error", message: "Invalid move" },
        ])
        expect(s.draft).toEqual([[1, 2]])
        expect(s.pending).toBeNull()
        expect(s.error).toBe("Invalid move")
        expect(s.role).toBe("white")
    })

    it("clears pending when the move is accepted and the board changes", () => {
        const s = run([
            { type: "board", data: serverBoard({ rolled: true }) },
            { type: "request", pending: "move" },
            { type: "board", data: serverBoard({ turn: -1 }) },
        ])
        expect(s.pending).toBeNull()
    })

    it("drops the role when a seat claim is rejected", () => {
        const s = run([
            { type: "role", role: "white" },
            { type: "request", pending: "claim" },
            { type: "error", message: "white side already taken." },
        ])
        expect(s.role).toBeNull()
        expect(s.error).toMatch(/already taken/)
    })

    it("answers a roll with the dice update", () => {
        const s = run([
            { type: "board", data: serverBoard() },
            { type: "request", pending: "roll" },
            { type: "dice", data: { dice: [2, 5], invalidDice: [], validMoves: [], rolled: true } },
        ])
        expect(s.pending).toBeNull()
        expect(s.snapshot?.dice).toEqual([2, 5])
    })

    it("cancels in-flight requests on disconnect but keeps the draft", () => {
        const s = run([
            { type: "board", data: serverBoard() },
            { type: "draft_add", move: [1, 2] },
            { type: "request", pending: "move" },
            { type: "connection", status: "disconnected" },
        ])
        expect(s.pending).toBeNull()
        expect(s.draft).toEqual([[1, 2]])
    })

    describe("replay", () => {
        const played = { player: 1 as const, sequence: [[0, 3]] as [number, number][] }
        const moved = (extra: Partial<ServerBoard> = {}) =>
            serverBoard({ turn: -1, last_move: played, ...extra })

        it("animates the opponent's move from the position we last saw", () => {
            const s = run([
                { type: "role", role: "black" },
                { type: "board", data: serverBoard({ rolled: true }) },
                { type: "board", data: moved() },
            ])
            expect(s.replay).toMatchObject({ id: 2, animate: true, last: played })
            expect(s.replay?.before?.turn).toBe(1)
        })

        it("does not replay our own move", () => {
            const s = run([
                { type: "role", role: "white" },
                { type: "board", data: serverBoard({ rolled: true }) },
                { type: "board", data: moved() },
            ])
            expect(s.replay).toBeNull()
        })

        it("clears an earlier highlight once we move", () => {
            const s = run([
                { type: "role", role: "black" },
                { type: "board", data: serverBoard({ rolled: true }) },
                { type: "board", data: moved() },
                { type: "board", data: serverBoard({ turn: 1, last_move: { player: -1, sequence: [] } }) },
            ])
            expect(s.replay).toBeNull()
        })

        it("keeps the highlight across a dice update", () => {
            const s = run([
                { type: "role", role: "black" },
                { type: "board", data: serverBoard({ rolled: true }) },
                { type: "board", data: moved() },
                { type: "dice", data: { dice: [2, 3], invalidDice: [], validMoves: [], rolled: true } },
            ])
            expect(s.replay?.id).toBe(2)
        })

        it("shows a move without animating it when we have not seen the start", () => {
            const s = run([{ type: "role", role: "black" }, { type: "board", data: moved() }])
            expect(s.replay).toMatchObject({ animate: false, before: null })
        })
    })
})
