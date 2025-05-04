import { useEffect, useState } from "react"
import { socket } from "../socket"
import { useNavigate } from "react-router-dom"
import "./Home.css"
import { BACKEND_URL } from "../constants"

export default function Home() {
    const [roomCode, setRoomCode] = useState("")
    const navigate = useNavigate()

    useEffect(() => {
        socket.on("joined_room", (data) => {
            console.log("Joined room", data.room_code);
            navigate(`/game/${data.room_code}`);
        });

        socket.on("error", (data) => {
            console.error(data.message);
        })

        return () => {
            socket.off("joined_room");
            socket.off("error");
        }

    }, [navigate])


    const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setRoomCode(event.target.value)
    }

    const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault()
        socket.emit("join_room", { room_code: roomCode })
        navigate(`/game/${roomCode}`)
    }

    const handleNewGame = () => {
        const fetchData = async () => {
            const response = await fetch(`${BACKEND_URL}/api/new_game`, {
                method: "POST"
            })
            const data = await response.json()
            navigate(`/game/${data.room_code}`)
        }
        fetchData()

    }


    return (
        <div className="home-container">
            <h1>Home</h1>
            <div className="button-div">
                <button onClick={handleNewGame} className="button-newgame">New Game</button>
            </div>
            <form onSubmit={handleSubmit} className="join-form">
                <input
                    type="text"
                    placeholder="Enter Game ID"
                    onChange={handleChange}
                    value={roomCode}
                    className="input-gameId"
                />
                <button type="submit" className="button-join">Join Game</button>
            </form>
        </div>
    )
}