import { useEffect, useState } from "react"
import '../Board.css'

type BoardType = {
    positions: number[],
    turn: number,
    dice: number[],
    white_bar: number,
    black_bar: number,
    rolled: boolean
    white_off: number,
    black_off: number
}


export default function Board() {
    const [board, setBoard] = useState<number[]>([])
    const [activePiece, setActivePiece] = useState(-1)
    const [dice, setDice] = useState([0, 0])
    const [turn, setTurn] = useState(1)
    const [whiteBar, setWhiteBar] = useState(0)
    const [blackBar, setBlackBar] = useState(0)
    const [rolled, setRolled] = useState(false)
    const [whiteBearOff, setWhiteBearOff] = useState(0)
    const [blackBearOff, setBlackBearOff] = useState(0)

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
        console.log(board)
        setBoard(data["positions"])
        setTurn(data["turn"])
        setDice(Array.from(data["dice"]).map(Number))
        setWhiteBar(data["white_bar"])
        setBlackBar(data["black_bar"])
        setRolled(data["rolled"])
        setWhiteBearOff(data["white_off"])
        setBlackBearOff(data["black_off"])
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
        // TODO: check for win
        // TODO: add confirm move button
    }

    const canBearOff = (turn: Number) => {
        if (turn === 1) {
            if (whiteBar > 0) {
                return false
            }
            for (let i = 0; i < 18; i++) {
                if (board[i] > 0) {
                    return false
                }
            }
            return true
        }
        if (turn === -1) {
            if (blackBar > 0) {
                return false
            }
            for (let i = 6; i < 24; i++) {
                if (board[i] < 0) {
                    return false
                }
            }
            return true
        }
        console.log(`canBearOff was passed invalid value of ${turn}`)
        throw new Error("Invalid turn value")
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

        // can only move to starting pos to reenter white
        if (turn === 1 && whiteBar > 0) {
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

        // can only move to starting pos to reenter black
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
            if (index === 100 || index === -100) {
                // TODO: this thing
                // if can bear off and passed then pick one/find optimal
                return
            }
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
        if (index === 100 || index === -100) {
            if (!canBearOff(turn)) {
                return
            }
            make_move(prevActivePiece, index * turn)
            setActivePiece(-1)
            return
        }
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
            <h1>Can bear off: {canBearOff(turn) ? "Yes" : "No"}</h1>
            {canBearOff(1) && <div className="bearOffWhite" onClick={() => handleClick(100)}>Bear off white</div>}
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
            {canBearOff(-1) && <div className="bearOffBlack" onClick={() => handleClick(-100)}>Bear off black</div>}
            <h2>turn: {turn}</h2>
            <h2>active piece: {activePiece}</h2>
            <h3>dice: {dice}</h3>
            <button onClick={_ => rollDice()}>Roll dice</button>

        </>
    )

}