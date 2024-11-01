import { useEffect, useState } from "react"
import { socket } from "../socket"
import { useNavigate } from "react-router-dom"

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
        console.log("Joining room", roomCode)
        socket.emit("join_room", { room_code: roomCode })
        navigate(`/game/${roomCode}`)
    }

    const handleNewGame = () => {
        console.log("create room")
        const fetchData = async () => {
            const response = await fetch('http://localhost:5000/api/new_game', {
                method: "POST"
            })
            const data = await response.json()
            console.log("new game")
            navigate(`/game/${data.room_code}`)
        }
        fetchData()

    }


    return (
        <>
            <h1>Home</h1>
            <button onClick={() => handleNewGame()}>New Game</button>
            <form onSubmit={handleSubmit}>
                <input type="text" placeholder="Enter Game ID" onChange={handleChange} value={roomCode} />
                <button>Join Game</button>
            </form>
        </>
    )
}