import { useState } from "react"
import { Link, useNavigate } from "react-router-dom"
import "./Home.css"
import { BACKEND_URL } from "../constants"

async function createGame(): Promise<string> {
    const response = await fetch(`${BACKEND_URL}/api/new_game`, { method: "POST" })
    if (!response.ok) throw new Error(`The server answered ${response.status}.`)
    const data: { room_code?: string } = await response.json()
    if (!data.room_code) throw new Error("The server did not return a game code.")
    return data.room_code
}

export default function Home() {
    const [roomCode, setRoomCode] = useState("")
    const [creating, setCreating] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const navigate = useNavigate()

    async function handleNewGame() {
        setCreating(true)
        setError(null)
        try {
            navigate(`/game/${await createGame()}`)
        } catch (err) {
            setError(`Could not create a game. ${err instanceof Error ? err.message : ""} Check that the server is running and try again.`)
            setCreating(false)
        }
    }

    function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
        event.preventDefault()
        const code = roomCode.trim().toUpperCase()
        if (code) navigate(`/game/${code}`)
    }

    return (
        <div className="home-container">
            <h1>Home</h1>
            <div className="button-div">
                <button onClick={handleNewGame} className="button-newgame" disabled={creating}>
                    {creating ? "Creating…" : error ? "Try Again" : "New Game"}
                </button>
                {error && <p className="home-error" role="alert">{error}</p>}
            </div>
            <form onSubmit={handleSubmit} className="join-form">
                <input
                    type="text"
                    placeholder="Enter Game ID"
                    onChange={event => setRoomCode(event.target.value)}
                    value={roomCode}
                    className="input-gameId"
                />
                <button type="submit" className="button-join" disabled={!roomCode.trim()}>Join Game</button>
            </form>
            <p><Link to="/bench">Benchmarks</Link></p>
        </div>
    )
}
