import { useEffect, useState } from "react"
import '../Board.css'

type BoardType = {
    positions: number[],
    turn: number,
    dice: number[],
    white_bar: number,
    black_bar: number,
    rolled: boolean
}


export default function Board() {
    const [board, setBoard] = useState<number[]>([])
    const [activePiece, setActivePiece] = useState(-1)
    const [dice, setDice] = useState([0, 0])
    const [turn, setTurn] = useState(1)
    const [whiteBar, setWhiteBar] = useState(0)
    const [blackBar, setBlackBar] = useState(0)
    const [rolled, setRolled] = useState(false)

    const rollDice = () => {
        if (rolled) {
            return
        }
        const fetchData = async () => {
            const response = await fetch('http://localhost:5000/api/roll_dice', {
                method: 'POST'
            })
            const data = await response.json()
            setDice(data)
        }
        fetchData()
    }

    const setVals = (data: BoardType) => {
        setBoard(data["positions"])
        setTurn(data["turn"])
        setDice(data["dice"])
        setWhiteBar(data["white_bar"])
        setBlackBar(data["black_bar"])
        setRolled(data["rolled"])
    }

    const reset_board = () => {
        const fetchData = async () => {
            const response = await fetch('http://localhost:5000/api/reset_board', {
                method: 'POST',
            })
            const data = await response.json()
            setVals(data)
        }
        fetchData()
    }

    const make_move = (current: number, next: number) => {
        console.log(current, next)
        const fetchData = async () => {
            try {
                const response = await fetch('http://localhost:5000/api/move', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ current, next })
                })

                if (!response.ok) {
                    throw new Error(response.status.toString())
                }
                const data = await response.json()
                setVals(data)
            } catch (err: any) {
                console.error('Error:', err)
                if (err.message === "403") {
                    alert("Invalid move")
                }
            }
        }
        fetchData()
    }

    useEffect(() => {
        const fetchData = async () => {
            const response = await fetch('http://localhost:5000/api/get_board')
            const data = await response.json()
            setVals(data)
        }
        fetchData()
        rollDice()
    }, [])


    const handleClick = (index: number) => {

        const prevActivePiece = activePiece

        if (turn === 1 && whiteBar > 0) {
            console.log(index, dice)
            console.log("white bar")
            if (!dice.includes(index + 1)) {
                return
            }
            if (index > 5) {
                return
            }
            if (board[index] < -1) {
                return
            }
            make_move(-1, index)
            return

        }

        if (turn === -1 && blackBar > 0) {
            if (!dice.includes(24 - index)) {
                return
            }
            if (index < 17) {
                return
            }
            if (board[index] > 1) {
                return
            }
            make_move(-1, index)
            return
        }


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

        // if there is an initial position selected

        // has to match dice
        if (!dice.includes((index - prevActivePiece) * turn)) {
            return
        }

        // if click on empty position, move it
        if ((board[index] === 0 || board[index] * turn > 0)) {
            make_move(prevActivePiece, index)
            setActivePiece(-1)
            return
        }
        if (board[index] * turn < 0 && Math.abs(board[index]) === 1) { // can capture piece
            make_move(prevActivePiece, index)
            setActivePiece(-1)
            return
        }
        return

    }


    return (
        <>
            <h1>Board</h1>
            <button onClick={() => reset_board()}>Reset board</button>
            <h1>White bar: {whiteBar}</h1>
            <h1>Black bar: {blackBar}</h1>
            <div className="board">
                <div className="top-container">
                    {board && board.slice(0, 12).map((points, index) => (
                        <div className={
                            "point top-point " +
                            ((index + Math.floor((index + 1) / 13)) % 2 === 0 ? "light " : "dark ")
                        }
                            onClick={() => { handleClick(index) }}
                            key={index}>
                            <h2>{index}</h2>
                            {Array.from({ length: Math.abs(points) }, (_, i) => (
                                <div className={"checker " + (points > 0 ? "white " : "black ") + (index === activePiece && i === Math.abs(points) - 1 && " active")} key={i}></div>
                            ))}

                        </div>
                    ))}
                </div>
                <div className="bottom-container">
                    {board && board.slice(12, 24).map((points, index) => (
                        <div className={
                            "point bottom-point " +
                            ((index + 12 + Math.floor((index + 1) / 13)) % 2 === 0 ? "light " : "dark ")
                        }
                            onClick={() => { handleClick(index + 12) }}
                            key={index + 12}>
                            <h2>{index + 12}</h2>
                            {Array.from({ length: Math.abs(points) }, (_, i) => (
                                <div className={"checker " + (points > 0 ? "white " : "black ") + (index + 12 === activePiece && i === Math.abs(points) - 1 && " active")} key={i}></div>
                            ))}

                        </div>
                    ))}

                </div>
            </div>
            <h2>turn: {turn}</h2>
            <h2>active piece: {activePiece}</h2>
            <h3>dice: {dice}</h3>
            <button onClick={_ => rollDice()}>Roll dice</button>

        </>
    )

}