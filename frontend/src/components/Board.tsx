import { useEffect, useState } from "react"
import '../Board.css'

export default function Board() {
    const [board, setBoard] = useState<number[]>([])
    const [activePiece, setActivePiece] = useState(-1)
    const [dice, setDice] = useState([0, 0])
    const [turn, setTurn] = useState(true)

    const rollDice = () => {
        const dice1 = Math.floor(Math.random() * 6) + 1
        const dice2 = Math.floor(Math.random() * 6) + 1
        setDice([dice1, dice2])
    }

    const get_board = () => {
        const fetchData = async () => {
            const response = await fetch('http://localhost:5000/api/get_board')
            const data = await response.json()
            setBoard(data["positions"])
            setTurn(Boolean(data["turn"]))
        }
        fetchData()
    }

    useEffect(() => {
        rollDice()
    }, [])

    useEffect(() => {
        get_board()
    }, [])


    const handleClick = (index: number) => {
        if (board[index] === 0) {
            return
        }
        setActivePiece(index)
    }

    return (
        <>
            <h1>Board</h1>
            <h2>{JSON.stringify(board)}</h2>
            <div className="board">
                {board && board.map((points, index) => (
                    <div className={
                        "point " +
                        ((index + Math.floor((index + 1) / 13)) % 2 === 0 ? "light " : "dark ") +
                        (index < 12 ? "top " : "bottom ")
                    }
                        onClick={() => { handleClick(index) }}
                        key={index}>
                        {Array.from({ length: Math.abs(points) }, (_, i) => (
                            <div className={"checker " + (points > 0 ? "white " : "black ") + (index === activePiece && i === Math.abs(points) - 1 && " active")} key={i}></div>
                        ))}
                    </div>
                ))}
            </div>
            <h2>{JSON.stringify(turn)}</h2>
            {activePiece}
            <h3>{dice}</h3>
            <button onClick={_ => rollDice()}>Roll dice</button>

        </>
    )

}