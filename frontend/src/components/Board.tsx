import { Link, useParams } from "react-router-dom"
import { useWindowSize } from "react-use"
import Confetti from "react-confetti"
import "./Board.css"
import BoardView from "./BoardView"
import Dice from "./Dice"
import Lobby from "./Lobby"
import { winnerOf } from "../game/snapshot"
import { sideOfTurn } from "../game/types"
import { useGameSession } from "../hooks/useGameSession"
import type { GameSession } from "../hooks/useGameSession"
import { useMoveReplay } from "../hooks/useMoveReplay"

function statusLine(s: GameSession, replaying: boolean): string {
    if (!s.snapshot || !s.turnSide) return ""
    if (replaying) return "Watching the last move…"
    if (s.snapshot.gameOver) return "Game over"
    if (s.role === "spectator") return "Spectating"
    if (s.pending === "roll") return "Rolling…"
    if (s.pending === "move") return "Submitting your move…"
    if (s.role !== s.turnSide) return `Waiting for ${s.turnSide}…`
    if (!s.snapshot.rolled) return "Your turn — roll the dice"
    if (s.stepsLeft > 0) return `Move ${s.stepsLeft} more ${s.stepsLeft === 1 ? "checker step" : "checker steps"}`
    return s.snapshot.validMoves.length === 0 && s.draft.length === 0
        ? "No legal moves — confirm to pass"
        : "Turn complete — confirm your move"
}

function Notice({ tone, children, action }: { tone: "error" | "info"; children: React.ReactNode; action?: React.ReactNode }) {
    return (
        <div className={`notice notice-${tone}`} role={tone === "error" ? "alert" : "status"}>
            <span>{children}</span>
            {action}
        </div>
    )
}

export default function Board() {
    const { room_code: roomCode = "" } = useParams()
    const { width, height } = useWindowSize()
    const session = useGameSession(roomCode)
    const { snapshot, role } = session
    const replay = useMoveReplay(snapshot, session.replay, role, session.turnSide)
    const preview = replay.preview ?? session.preview

    const offline = session.connection !== "connected"

    if (!snapshot || !preview) {
        return (
            <div className="lobby-container">
                {offline ? (
                    <Notice tone="info">Connecting to the game server…</Notice>
                ) : session.loadFailed ? (
                    <Notice tone="error" action={<button className="btn" onClick={session.retryLoad}>Retry</button>}>
                        Game {roomCode} was not found, or the server is not responding.
                    </Notice>
                ) : (
                    <Notice tone="info">Loading game {roomCode}…</Notice>
                )}
                <p><Link to="/">Back to home</Link></p>
            </div>
        )
    }

    if (!role) {
        return <Lobby busy={session.pending === "claim"} error={session.error} onJoin={session.join} />
    }

    // The server's position already includes the move being replayed, so hold back the result.
    const busy = replay.playing
    const gameOver = snapshot.gameOver && !busy
    const winner = gameOver ? winnerOf(snapshot) : null
    const rolling = session.pending === "roll" || replay.diceSpinning
    const canRoll = session.canRoll && !busy
    const canConfirm = session.canConfirm && !busy
    const canUndo = session.canMove && !busy && session.draft.length > 0

    return (
        <>
            {gameOver && <Confetti width={width} height={height} />}

            <div className="turn-indicator">
                {gameOver
                    ? <h1><strong>{winner ? `${winner} wins!` : "Game over"}</strong></h1>
                    : <h1><strong>{sideOfTurn(preview.turn) === "white" ? "White" : "Black"}</strong> to move</h1>}
                <p className="status-line" role="status">{statusLine(session, busy)}</p>
                <p className="last-move-text">
                    {replay.highlight && (
                        <>
                            <span className="last-move-swatch" />
                            {replay.highlight.text}
                        </>
                    )}
                </p>
            </div>

            {offline && <Notice tone="info">Connection lost — reconnecting. Your unsent moves are kept.</Notice>}
            {session.error && (
                <Notice tone="error" action={<button className="btn" onClick={session.dismissError}>Dismiss</button>}>
                    {session.error}
                </Notice>
            )}

            <div className="game-layout">
                <BoardView
                    key={session.revision}
                    preview={preview}
                    interactive={session.canMove && !busy}
                    onMove={session.addMove}
                    landings={replay.landings}
                    marks={replay.highlight?.marks}
                />

                <div className="controls">
                    <button onClick={session.roll} className={`btn btn-roll${canRoll ? " pulse" : ""}`} disabled={!canRoll}>🎲 Roll Dice</button>
                    <button onClick={session.confirm} className="btn btn-confirm" disabled={!canConfirm}>✅ Confirm</button>
                    <button onClick={session.undo} className="btn btn-reset" disabled={!canUndo}>↩️ Undo</button>
                    <button onClick={session.reset} className="btn btn-reset" disabled={!canUndo}>🔄 Reset</button>
                </div>
            </div>

            <div className="dice-container">
                {preview.dice.map((value, i) => <Dice key={`d${i}`} value={value} rolling={rolling} />)}
                {preview.invalidDice.map((value, i) => <Dice key={`i${i}`} value={value} invalid rolling={rolling} />)}
                {preview.usedDice.map((value, i) => <Dice key={`u${i}`} value={value} used rolling={rolling} />)}
            </div>
        </>
    )
}
