import { useEffect, useState } from "react"
import '../Board.css'

export default function Board() {
    const [board, setBoard] = useState<number[]>([])
    const [activePiece, setActivePiece] = useState(-1)
    const [dice, setDice] = useState([0, 0])
    const [turn, setTurn] = useState(1)

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
            setTurn(data["turn"])
        }
        fetchData()
    }

    const make_move = (current: number, next: number) => {
        const fetchData = async () => {
            const response = await fetch('http://localhost:5000/api/move', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ current, next })
            })
            const data = await response.json()
            setBoard(data["positions"])
            setTurn(data["turn"])
        }
        fetchData()
    }

    useEffect(() => {
        get_board()
        rollDice()
    }, [])


    const handleClick = (index: number) => {
        const prevActivePiece = activePiece

        // if click the same point, unselect
        if (prevActivePiece === index) {
            setActivePiece(-1)
            return
        }

        // if there is no intial position selected
        if (prevActivePiece === -1) {
            if (board[index] === 0) { // cant select empty position
                return
            }
            if (board[index] * turn < 0) { // cant select opposite colour
                return
            }
            setActivePiece(index)
            return
        }

        //// if activePiece != -1
        // if click on empty position, move it
        if (board[index] === 0 || board[index] * turn > 0) {
            make_move(prevActivePiece, index)
            setActivePiece(-1)
            return
        }
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
            <h2>turn: {turn}</h2>
            <h2>active piece: {activePiece}</h2>
            <h3>dice: {dice[0]} {dice[1]}</h3>
            <button onClick={_ => rollDice()}>Roll dice</button>

        </>
    )

}