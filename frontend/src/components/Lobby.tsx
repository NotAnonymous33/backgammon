import { useState } from "react"
import { AI_AGENTS } from "../game/agents"
import type { AiAgent } from "../game/types"
import type { JoinPlan } from "../hooks/useGameSession"

type AgentSelectProps = {
    label: string
    value: AiAgent
    onChange: (agent: AiAgent) => void
}

function AgentSelect({ label, value, onChange }: AgentSelectProps) {
    return (
        <div className="form-group">
            <label>
                {label}
                <select className="select" value={value} onChange={e => onChange(e.target.value as AiAgent)}>
                    {AI_AGENTS.map(agent => (
                        <option key={agent.value} value={agent.value}>{agent.label}</option>
                    ))}
                </select>
            </label>
        </div>
    )
}

type LobbyProps = {
    /** A request to join is in flight. */
    busy: boolean
    error: string | null
    onJoin: (plan: JoinPlan) => void
}

export default function Lobby({ busy, error, onJoin }: LobbyProps) {
    const [opponent, setOpponent] = useState<AiAgent>("random")
    const [whiteAgent, setWhiteAgent] = useState<AiAgent>("random")
    const [blackAgent, setBlackAgent] = useState<AiAgent>("random")

    return (
        <div className="lobby-container">
            {error && <p className="notice notice-error" role="alert">{error}</p>}
            {busy && <p className="notice" role="status">Joining…</p>}

            <section className="mode-section">
                <h2>Player vs Player</h2>
                <div className="button-group">
                    <button className="btn btn-white" disabled={busy} onClick={() => onJoin({ kind: "seat", side: "white" })}>Play as White</button>
                    <button className="btn btn-black" disabled={busy} onClick={() => onJoin({ kind: "seat", side: "black" })}>Play as Black</button>
                </div>
            </section>

            <section className="mode-section">
                <h2>Player vs AI</h2>
                <AgentSelect label="Choose AI Agent:" value={opponent} onChange={setOpponent} />
                <div className="button-group">
                    <button className="btn btn-white" disabled={busy} onClick={() => onJoin({ kind: "seat", side: "white", opponent })}>Play as White</button>
                    <button className="btn btn-black" disabled={busy} onClick={() => onJoin({ kind: "seat", side: "black", opponent })}>Play as Black</button>
                </div>
            </section>

            <section className="mode-section">
                <h2>AI vs AI</h2>
                <AgentSelect label="Choose White AI Agent:" value={whiteAgent} onChange={setWhiteAgent} />
                <AgentSelect label="Choose Black AI Agent:" value={blackAgent} onChange={setBlackAgent} />
                <div className="button-group">
                    <button className="btn" disabled={busy} onClick={() => onJoin({ kind: "ai-vs-ai", white: whiteAgent, black: blackAgent })}>Watch AI vs AI</button>
                </div>
            </section>

            <section className="mode-section">
                <h2>Spectate</h2>
                <div className="button-group">
                    <button className="btn" disabled={busy} onClick={() => onJoin({ kind: "spectate" })}>Watch this game</button>
                </div>
            </section>
        </div>
    )
}
