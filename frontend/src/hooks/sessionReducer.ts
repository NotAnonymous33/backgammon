import { snapshotFromBoard, snapshotKey, withDice } from "../game/snapshot"
import { sideOfTurn } from "../game/types"
import type { LastMove, Move, MoveSequence, Role, ServerBoard, ServerDice, Snapshot } from "../game/types"

/** The move behind the latest board update, kept so it can be replayed and highlighted. */
export type Replay = {
    /** Revision of the board this move produced. A new id means a new move. */
    id: number
    /** The position the move was played from, if we had seen it. */
    before: Snapshot | null
    last: LastMove
    /** Worth animating: someone else played it and we saw the position it started from. */
    animate: boolean
}

export type Connection = "connecting" | "connected" | "disconnected"

/** The one request we are currently waiting on the server to answer. */
export type Pending = "claim" | "roll" | "move" | null

export type SessionState = {
    connection: Connection
    /** The server has acknowledged that we are in the room. */
    inRoom: boolean
    /** Role confirmed by the server. Null until a seat is granted (or spectating begins). */
    role: Role | null
    /** Last authoritative board. Local play never mutates it. */
    snapshot: Snapshot | null
    /** Bumped each time the authoritative position changes; anything tied to the old one is stale. */
    revision: number
    /** The latest opponent move, or null after our own move. Dice updates leave it in place. */
    replay: Replay | null
    /** Moves the player has made this turn but not yet submitted. Applied on top of `snapshot`. */
    draft: MoveSequence
    pending: Pending
    /** Latest user-visible failure (rejected seat, rejected move, timeout...). */
    error: string | null
    /** Joined the room but no board ever arrived: the game does not exist or the server is not answering. */
    loadFailed: boolean
}

export const initialSession: SessionState = {
    connection: "connecting",
    inRoom: false,
    role: null,
    snapshot: null,
    revision: 0,
    replay: null,
    draft: [],
    pending: null,
    error: null,
    loadFailed: false,
}

export type SessionAction =
    | { type: "connection"; status: Connection }
    | { type: "room_joined" }
    | { type: "board"; data: ServerBoard }
    | { type: "dice"; data: ServerDice }
    | { type: "role"; role: Role }
    | { type: "request"; pending: Exclude<Pending, null> }
    | { type: "error"; message: string }
    | { type: "dismiss_error" }
    | { type: "load_failed" }
    | { type: "retry_load" }
    | { type: "draft_add"; move: Move }
    | { type: "draft_undo" }
    | { type: "draft_reset" }

/** Install a new authoritative snapshot, dropping the draft only when the position really changed. */
function replaceSnapshot(state: SessionState, next: Snapshot): SessionState {
    const unchanged = state.snapshot !== null && snapshotKey(state.snapshot) === snapshotKey(next)
    if (unchanged) return { ...state, loadFailed: false }

    return {
        ...state,
        snapshot: next,
        revision: state.revision + 1,
        draft: [],
        error: null,
        loadFailed: false,
        // A changed board answers a submitted move; a roll is answered once the dice are out.
        pending: state.pending === "move" || (state.pending === "roll" && next.rolled) ? null : state.pending,
    }
}

/** Attach the move that produced a freshly installed board, if the server said which one it was. */
function withReplay(before: SessionState, after: SessionState, last: LastMove | undefined): SessionState {
    if (!last || after.revision === before.revision) return after
    const mine = before.role !== null && before.role !== "spectator" && sideOfTurn(last.player) === before.role
    if (mine) return { ...after, replay: null }
    const seen = before.snapshot
    return {
        ...after,
        replay: {
            id: after.revision,
            before: seen,
            last,
            animate: before.role !== null && seen !== null && seen.turn === last.player,
        },
    }
}

export function sessionReducer(state: SessionState, action: SessionAction): SessionState {
    switch (action.type) {
        case "connection":
            return {
                ...state,
                connection: action.status,
                inRoom: action.status === "connected" ? state.inRoom : false,
                // Nothing in flight survives a dropped connection; the draft does.
                pending: action.status === "connected" ? state.pending : null,
            }
        case "room_joined":
            return { ...state, inRoom: true }
        case "board":
            return withReplay(state, replaceSnapshot(state, snapshotFromBoard(action.data)), action.data.last_move)
        case "dice":
            return state.snapshot ? replaceSnapshot(state, withDice(state.snapshot, action.data)) : state
        case "role":
            return { ...state, role: action.role, pending: state.pending === "claim" ? null : state.pending, error: null }
        case "request":
            return { ...state, pending: action.pending, error: null }
        case "error":
            return {
                ...state,
                error: action.message,
                // A rejected claim means we do not hold the seat after all.
                role: state.pending === "claim" ? null : state.role,
                pending: null,
            }
        case "dismiss_error":
            return { ...state, error: null }
        case "load_failed":
            return { ...state, loadFailed: true }
        case "retry_load":
            return { ...state, loadFailed: false }
        case "draft_add":
            return { ...state, draft: [...state.draft, action.move] }
        case "draft_undo":
            return { ...state, draft: state.draft.slice(0, -1) }
        case "draft_reset":
            return { ...state, draft: [] }
    }
}
