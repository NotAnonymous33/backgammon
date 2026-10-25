import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react"
import { socket } from "../socket"
import { isComplete, movesRemaining, previewOf } from "../game/draft"
import { sideOfTurn } from "../game/types"
import type { AiAgent, Move, Preview, ServerBoard, ServerDice, Side } from "../game/types"
import { initialSession, sessionReducer } from "./sessionReducer"
import type { SessionState } from "./sessionReducer"

/** How a player asks to take part in a game. */
export type JoinPlan =
    | { kind: "seat"; side: Side; opponent?: AiAgent }
    | { kind: "ai-vs-ai"; white: AiAgent; black: AiAgent }
    | { kind: "spectate" }

type JoinedRoom = { room_code: string; side: Side | null; playerType: "human" | "ai" }

const REQUEST_TIMEOUT_MS = 8000
const LOAD_TIMEOUT_MS = 4000

const TIMEOUT_MESSAGE = {
    claim: "The server did not answer the request to join. Try again.",
    roll: "The server did not answer the dice roll. Try rolling again.",
    move: "The server did not confirm your move. Your moves are kept; try confirming again.",
} as const

export type GameSession = SessionState & {
    /** The board the player sees: server snapshot plus draft moves. Null until the first board arrives. */
    preview: Preview | null
    /** Side whose turn it is, if the game is underway. */
    turnSide: Side | null
    /** It is this client's turn and nothing is in flight. */
    canAct: boolean
    canRoll: boolean
    canMove: boolean
    canConfirm: boolean
    /** Steps still missing from the shortest complete turn. */
    stepsLeft: number
    join: (plan: JoinPlan) => void
    roll: () => void
    confirm: () => void
    addMove: (move: Move) => void
    undo: () => void
    reset: () => void
    dismissError: () => void
    retryLoad: () => void
}

export function useGameSession(roomCode: string): GameSession {
    const [state, dispatch] = useReducer(sessionReducer, initialSession)
    const [attempt, setAttempt] = useState(0)

    // What we asked for, kept so it can be replayed after a reconnect or finished after the seat is granted.
    const planRef = useRef<JoinPlan | null>(null)
    const seatRef = useRef<Side | null>(null)

    const observe = useCallback(() => socket.emit("join_room", { room_code: roomCode }), [roomCode])

    useEffect(() => {
        const claimSeat = (side: Side) => {
            dispatch({ type: "request", pending: "claim" })
            socket.emit("join_room", { room_code: roomCode, side, playerType: "human" })
        }

        const onConnect = () => {
            dispatch({ type: "connection", status: "connected" })
            observe()
            // A server restart forgets seats while we still believe we hold one.
            if (seatRef.current) claimSeat(seatRef.current)
        }
        const onDisconnect = () => dispatch({ type: "connection", status: "disconnected" })

        const onJoined = (data: JoinedRoom) => {
            if (!data.side) {
                dispatch({ type: "room_joined" })
                return
            }
            const plan = planRef.current
            if (data.playerType === "human") {
                seatRef.current = data.side
                dispatch({ type: "role", role: data.side })
                // Only bring the AI in once our own seat is secured.
                if (plan?.kind === "seat" && plan.opponent) {
                    const aiSide: Side = data.side === "white" ? "black" : "white"
                    socket.emit("join_room", { room_code: roomCode, side: aiSide, playerType: "ai", aiModel: plan.opponent })
                    planRef.current = { kind: "seat", side: plan.side }
                }
            } else if (plan?.kind === "ai-vs-ai") {
                dispatch({ type: "role", role: "spectator" })
            }
        }

        const onBoard = (data: ServerBoard) => dispatch({ type: "board", data })
        const onDice = (data: ServerDice) => dispatch({ type: "dice", data })
        const onError = (data: { message?: string }) =>
            dispatch({ type: "error", message: data?.message ?? "Something went wrong." })

        socket.on("connect", onConnect)
        socket.on("disconnect", onDisconnect)
        socket.on("joined_room", onJoined)
        socket.on("update_board", onBoard)
        socket.on("update_dice", onDice)
        socket.on("error", onError)

        if (socket.connected) onConnect()
        else socket.connect()

        return () => {
            socket.off("connect", onConnect)
            socket.off("disconnect", onDisconnect)
            socket.off("joined_room", onJoined)
            socket.off("update_board", onBoard)
            socket.off("update_dice", onDice)
            socket.off("error", onError)
            // The shared connection (and our seat) outlives the page; only the room subscription goes.
            socket.emit("leave_room", { room_code: roomCode })
        }
    }, [roomCode, observe])

    // Give up waiting for the first board so the page can say the game was not found.
    const hasBoard = state.snapshot !== null
    useEffect(() => {
        if (state.connection !== "connected" || hasBoard) return
        const timer = setTimeout(() => dispatch({ type: "load_failed" }), LOAD_TIMEOUT_MS)
        return () => clearTimeout(timer)
    }, [state.connection, hasBoard, attempt])

    // Give up waiting for the answer to a request so controls never stay locked.
    const { pending } = state
    useEffect(() => {
        if (!pending) return
        const timer = setTimeout(() => dispatch({ type: "error", message: TIMEOUT_MESSAGE[pending] }), REQUEST_TIMEOUT_MS)
        return () => clearTimeout(timer)
    }, [pending])

    const preview = useMemo(
        () => (state.snapshot ? previewOf(state.snapshot, state.draft) : null),
        [state.snapshot, state.draft],
    )

    const turnSide = state.snapshot ? sideOfTurn(state.snapshot.turn) : null
    const myTurn = state.role !== null && state.role === turnSide
    const live = state.connection === "connected" && state.pending === null && !state.snapshot?.gameOver
    const canAct = myTurn && live
    const canRoll = canAct && state.snapshot?.rolled === false
    const canMove = canAct && state.snapshot?.rolled === true
    const canConfirm = canMove && preview !== null && isComplete(preview)
    const stepsLeft = preview && state.snapshot?.rolled ? movesRemaining(preview) : 0

    const join = useCallback((plan: JoinPlan) => {
        planRef.current = plan
        switch (plan.kind) {
            case "seat":
                dispatch({ type: "request", pending: "claim" })
                socket.emit("join_room", { room_code: roomCode, side: plan.side, playerType: "human" })
                break
            case "ai-vs-ai":
                dispatch({ type: "request", pending: "claim" })
                socket.emit("join_room", { room_code: roomCode, side: "white", playerType: "ai", aiModel: plan.white })
                socket.emit("join_room", { room_code: roomCode, side: "black", playerType: "ai", aiModel: plan.black })
                break
            case "spectate":
                dispatch({ type: "role", role: "spectator" })
                break
        }
    }, [roomCode])

    const roll = useCallback(() => {
        dispatch({ type: "request", pending: "roll" })
        socket.emit("roll_dice", { room_code: roomCode })
    }, [roomCode])

    const confirm = useCallback(() => {
        dispatch({ type: "request", pending: "move" })
        socket.emit("move", { moveSequence: state.draft, room_code: roomCode })
    }, [roomCode, state.draft])

    const retryLoad = useCallback(() => {
        dispatch({ type: "retry_load" })
        setAttempt(n => n + 1)
        observe()
    }, [observe])

    const addMove = useCallback((move: Move) => dispatch({ type: "draft_add", move }), [])
    const undo = useCallback(() => dispatch({ type: "draft_undo" }), [])
    const reset = useCallback(() => dispatch({ type: "draft_reset" }), [])
    const dismissError = useCallback(() => dispatch({ type: "dismiss_error" }), [])

    return {
        ...state,
        preview,
        turnSide,
        canAct,
        canRoll,
        canMove,
        canConfirm,
        stepsLeft,
        join,
        roll,
        confirm,
        retryLoad,
        addMove,
        undo,
        reset,
        dismissError,
    }
}
