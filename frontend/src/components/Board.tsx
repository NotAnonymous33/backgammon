import { useEffect, useRef, useState } from "react"
import '../Board.css'
import { socket } from '../socket'
import { useParams } from "react-router-dom"
import { BACKEND_URL } from "../constants"

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
    const [message, setMessage] = useState(null)

    const params = useParams()

    const roomCode = params["room_code"]

    const pointWidth = 75
    const pointHeight = 200
    const pointPadding = 10
    const checkerRadius = 20
    const checkerPadding = 5
    const barWidth = 50
    const bearOffWidth = 50
    const boardPadding = 10

    const svgWidth = 2 * boardPadding + 12 * pointWidth + bearOffWidth + barWidth
    const svgHeight = 2 * boardPadding + 2.2 * pointHeight

    // colours
    const outerBoardColour = "#b39970"
    const innerBoardColour = "#f3d970"
    const darkPointColour = "#64230b"
    const lightPointColour = "#b96522"
    const whiteCheckerColour = "#f3f3f3"
    const blackCheckerColour = "#000000"


    useEffect(() => {
        socket.connect()
        console.log("socket useffect connected")


        socket.emit("join_room", { roomCode })


        socket.on('connect', () => {
            console.log("client connected")
        })

        socket.on('update_board', (data: BoardType) => {
            setVals(data)
        })

        socket.on("update_dice", (data: number[]) => {
            setDice(data)
        })

        socket.on('disconnect', (reason, details) => {
            console.log("client disconnected:", reason, details)
        })

        socket.on("message", (data) => {
            setMessage(data)
        })

        return () => {
            console.log("clean up sockets")
            socket.off('connect')
            socket.off('update_board')
            socket.off('update_dice')
            socket.off('disconnect')
            socket.off('message')
            socket.disconnect()
        }
    }, [])

    const rollDice = () => {
        if (rolled) {
            return
        }
        socket.emit("roll_dice", { roomCode })
    }

    const setVals = (data: BoardType) => {
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
        socket.emit("reset_board", { roomCode })
    }

    const make_move = (current: number, next: number) => {
        socket.emit('move', { current, next, roomCode });
        // TODO: check for win
        // TODO: add confirm move button
    }

    const testButton = () => {
        const fetchData = async () => {
            const response = await fetch(`${BACKEND_URL}/api/button_test`, {
                method: "GET"
            })
            const data = await response.json()
            alert(data.message)

        }
        fetchData()
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
        throw new Error("Invalid turn value")
    }


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

    const indexToCoords = (index: number) => {
        if (index < 0 || index > 23) {
            throw new Error("Invalid index")
        }
        if (index < 12) {
            index = 11 - index
        } else {
            index -= 12
        }
        return boardPadding + index * pointWidth + (index >= 6 ? barWidth : 0)
    }


    const triangleCoords: [number, number, number, number, number, number][] = [];
    for (let i = 0; i < 24; i++) {
        const x1 = indexToCoords(i)
        const x2 = indexToCoords(i) + pointWidth
        const x3 = indexToCoords(i) + pointWidth / 2
        let y1, y2, y3
        if (i < 12) {
            y1 = svgHeight - boardPadding - pointHeight
            y2 = svgHeight - boardPadding - pointHeight
            y3 = svgHeight - boardPadding
        } else {
            y1 = boardPadding
            y2 = boardPadding
            y3 = boardPadding + pointHeight
        }
        triangleCoords.push([x1, y1, x2, y2, x3, y3])
    }


    return (
        <>
            <h2>White bar: {whiteBar}</h2>
            <h2>Black bar: {blackBar}</h2>
            <h2>White bear off: {whiteBearOff}</h2>
            <h2>Black bear off: {blackBearOff}</h2>
            <h2>Can bear off: {canBearOff(turn) ? "Yes" : "No"}</h2>
            <h2>message: {JSON.stringify(message)}</h2>
            <button onClick={() => reset_board()}>Reset board</button>
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

            <svg width={svgWidth} height={svgHeight} xmlns="http://www.w3.org/2000/svg">
                // background
                <rect x={0} y={0} width={svgWidth} height={svgHeight} fill={outerBoardColour} />
                // left half
                <rect x={boardPadding} y={boardPadding} width={6 * pointWidth} height={svgHeight - 2 * boardPadding} fill={innerBoardColour} />
                // right half
                <rect x={6 * pointWidth + boardPadding + barWidth} y={boardPadding} width={6 * pointWidth} height={svgHeight - 2 * boardPadding} fill={innerBoardColour} />
                {triangleCoords.map((coords, index) => (
                    <polygon
                        points={coords.join(",")}
                        fill={(index % 2 === 0 ? darkPointColour : lightPointColour)}
                        transform={index < 12 ? `rotate(180, ${coords[0] + pointWidth / 2}, ${svgHeight - boardPadding - pointHeight / 2})` : undefined}
                        onClick={() => handleClick(index)} />
                ))}

                <g transform={"rotate(0,100,100)"}>
                    <rect x={100} y={100} width={200} height={50} fill="red" />
                </g>

            </svg>

            <h2>turn: {turn}</h2>
            <h2>active piece: {activePiece}</h2>
            <h3>dice: {dice}</h3>

            <button onClick={_ => rollDice()}>Roll dice</button>
            <button onClick={_ => testButton()}>test</button>

        </>
    )

}