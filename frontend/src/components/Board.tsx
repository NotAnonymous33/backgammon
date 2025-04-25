import { MouseEvent, useEffect, useRef, useState } from "react"
import "./Board.css"
import Dice from "./Dice"
import { socket } from "../socket"
import { useParams } from "react-router-dom"
import { BACKEND_URL } from "../constants"

//--- type sent from server
type BoardType = {
    positions: number[]
    turn: number
    dice: number[]
    invalid_dice: number[]
    white_bar: number
    black_bar: number
    rolled: boolean
    white_off: number
    black_off: number
    valid_moves: number[][][]
}

//--- game state type
type GameState = {
    board: number[]
    turn: number
    dice: number[]
    invalidDice: number[]
    whiteBar: number
    blackBar: number
    whiteBearOff: number
    blackBearOff: number
    validMoves: number[][][]
    usedDice: number[]
}

export default function Board() {
    const verbose = true
    const params = useParams()
    const roomCode = params["room_code"]

    const [selectedAiAgent, setSelectedAiAgent] = useState<string>("random")
    const [selectedAiAgent2, setSelectedAiAgent2] = useState<string>("random")

    const [playerSide, setPlayerSide] = useState<string | null>(null);
    const [playerType, setPlayerType] = useState<"human" | "ai">("human");

    //--- local game state
    const [gameState, setGameState] = useState<GameState>({
        board: [],
        turn: 1,
        dice: [],
        invalidDice: [],
        whiteBar: 0,
        blackBar: 0,
        whiteBearOff: 0,
        blackBearOff: 0,
        validMoves: [],
        usedDice: [],
    })

    //--- original true game state
    const originalGameStateRef = useRef<GameState | null>(null)

    const [selectedChecker, setSelectedChecker] = useState<number | null>(null)
    const [moveSequence, setMoveSequence] = useState<number[][]>([])
    const [message, setMessage] = useState<string | null>(null)

    //--- DRAG STATE (only used for visual dragging)
    const [dragPointIndex, setDragPointIndex] = useState<number | null>(null)
    const [dragColor, setDragColor] = useState<"white" | "black" | null>(null)
    const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
    const [dragPosition, setDragPosition] = useState({ x: 0, y: 0 })
    const [initialDragPosition, setInitialDragPosition] = useState({ x: 0, y: 0 })

    const [rolled, setRolled] = useState(false)
    const [isRollingAnimation, setIsRollingAnimation] = useState(false)

    const pointWidth = 75
    const pointHeight = 200
    const checkerRadius = 20
    const checkerPadding = 5
    const checkerBearOffPadding = -7
    const barWidth = 50
    const bearOffWidth = 50
    const boardPadding = 10
    const svgWidth = 2 * boardPadding + 12 * pointWidth + barWidth + bearOffWidth
    const svgHeight = 2 * boardPadding + 2.2 * pointHeight
    const outerBoardColour = "#b39970"
    const innerBoardColour = "#f3d970"
    const darkPointColour = "#64230b"
    const lightPointColour = "#b96522"
    const whiteCheckerColour = "#f3f3f3"
    const blackCheckerColour = "#000000"

    const boardRef = useRef<SVGSVGElement>(null)
    const turnLabel = gameState.turn === 1 ? "White" : "Black"


    //--- calculate bearing off area position and size
    const bearOffXStart = boardPadding + 12 * pointWidth + barWidth + 5
    const bearOffAreaHeight = svgHeight - 2 * boardPadding
    const blackBearOffHeight = bearOffAreaHeight / 2

    useEffect(() => {
        if (playerSide) {
            socket.emit("join_room", { room_code: roomCode, side: playerSide, playerType });
        }
    }, [playerSide, playerType, roomCode]);

    function setVals(data: BoardType) {
        //--- log setting values
        verbose && console.log("setVals")
        setRolled(data.rolled)
        console.log(data.rolled)
        const newGameState: GameState = {
            board: data.positions,
            turn: data.turn,
            dice: Array.from(data.dice).map(Number),
            invalidDice: data.invalid_dice,
            whiteBar: data.white_bar,
            blackBar: data.black_bar,
            whiteBearOff: data.white_off,
            blackBearOff: data.black_off,
            validMoves: data.valid_moves,
            usedDice: [],
        }
        setGameState(newGameState)
        console.log(data.valid_moves)
        originalGameStateRef.current = newGameState
    }

    useEffect(() => {
        //--- log move sequence 
        verbose && console.log(`moveSequence ${moveSequence}`)
    }, [moveSequence])

    useEffect(() => {
        //--- log game state update
        verbose && console.log(`gameState ${JSON.stringify(gameState)}`)
    }, [gameState])

    useEffect(() => {
        //--- connect socket and join room
        socket.connect()
        socket.emit("join_room", { room_code: roomCode })

        socket.on("connect", () => {
            verbose && console.log("socket:connect")
        })

        socket.on("disconnect", () => {
            verbose && console.log("socket:disconnect")
        })

        socket.on("update_board", (data: BoardType) => {
            verbose && console.log("socket:update_board")
            setVals(data)
        })

        socket.on("game_over", (data: { winner: number }) => {
            setMessage(`Game Over: ${data.winner}`)
            verbose && console.log("Game Over:", JSON.stringify(data))
        })

        socket.on("update_dice", (data: { dice: number[]; validMoves: number[][][]; invalidDice: number[]; rolled: boolean }) => {
            setRolled(data.rolled)
            setGameState((prevState) => {
                const newGameState: GameState = {
                    ...prevState,
                    dice: data.dice.map(Number),
                    validMoves: data.validMoves,
                    invalidDice: data.invalidDice.map(Number),
                    usedDice: [],
                }
                console.log(newGameState.validMoves)
                originalGameStateRef.current = newGameState
                verbose && console.log("Updated dice:", data.dice)
                return newGameState
            })
            setIsRollingAnimation(false)
        })

        socket.on("message", (data) => {
            setMessage(data)
        })

        return () => {
            //--- cleanup socket events
            socket.off("connect")
            socket.off("disconnect")
            socket.off("update_board")
            socket.off("update_dice")
            socket.off("message")
            socket.off("update_valid_moves")
            socket.off("game_over")
            socket.disconnect()
        }
    }, [roomCode])

    function confirmMove() {
        if (verbose) {
            console.log("ConfirmingMove move sequence:", moveSequence)
        }
        socket.emit("move", { moveSequence, room_code: roomCode })
        setMoveSequence([])
    }

    function makeLocalMove(current: number, next: number) {
        verbose && console.log(`makeLocalMove(${current}, ${next})`)
        setMoveSequence((prev) => [...prev, [current, next]])
        setGameState((prev) => {
            const newBoard = [...prev.board]
            let newWhiteBar = prev.whiteBar
            let newBlackBar = prev.blackBar
            let newWhiteBearOff = prev.whiteBearOff
            let newBlackBearOff = prev.blackBearOff

            if (next === 100) {
                newBoard[current] -= 1
                newWhiteBearOff += 1
            } else if (next === -100) {
                newBoard[current] += 1
                newBlackBearOff += 1
            } else {
                if (current === -1) {
                    if (prev.turn === 1) {
                        newWhiteBar -= 1
                    } else {
                        newBlackBar -= 1
                    }
                } else {
                    newBoard[current] = newBoard[current] - prev.turn
                }
                if (newBoard[next] === -prev.turn) {
                    newBoard[next] = prev.turn
                    if (prev.turn === 1) {
                        newBlackBar += 1
                    } else {
                        newWhiteBar += 1
                    }
                } else {
                    newBoard[next] = newBoard[next] + prev.turn
                }
            }

            let newDice = [...prev.dice]
            let diceIndex = newDice.indexOf(Math.abs(next - current))
            if (current === -1) diceIndex = prev.turn === 1 ? newDice.indexOf(next + 1) : newDice.indexOf(24 - next)
            if (next === 100 || next === -100) {
                diceIndex = next === 100 ? newDice.indexOf(24 - current) : newDice.indexOf(current + 1)
                if (diceIndex < 0) diceIndex = newDice.indexOf(Math.max(...newDice))
            }

            let removedDie: number | null = null
            if (diceIndex >= 0) {
                removedDie = newDice[diceIndex]
                newDice.splice(diceIndex, 1)
            }

            const newValidMoves = prev.validMoves
                .filter(seq => seq[0][0] === current && seq[0][1] === next)
                .map(seq => seq.slice(1))

            // update usedDice
            const newUsedDice = removedDie !== null ? [...prev.usedDice, removedDie] : prev.usedDice

            return {
                ...prev,
                board: newBoard,
                dice: newDice,
                validMoves: newValidMoves,
                whiteBar: newWhiteBar,
                blackBar: newBlackBar,
                whiteBearOff: newWhiteBearOff,
                blackBearOff: newBlackBearOff,
                usedDice: newUsedDice,
            }
        })
    }

    function resetBoard() {
        verbose && console.log("resetBoard")
        if (originalGameStateRef.current !== null) {
            setGameState(originalGameStateRef.current)
        }
        setSelectedChecker(null)
        setMoveSequence([])
    }

    function rollDice() {
        verbose && console.log("rollDice")
        verbose && console.log("rolled: ", rolled)
        if (rolled) return
        setIsRollingAnimation(true)
        verbose && console.log("emitting roll_dice")
        socket.emit("roll_dice", { room_code: roomCode })
    }

    function testButton() {
        verbose && console.log("testButton");
        (async () => {
            const response = await fetch(`${BACKEND_URL}/api/button_test`, {
                method: "GET",
            })
            const data = await response.json()
            alert(data.message)
        })()
    }

    function indexToCoords(index: number) {
        let realIndex = index < 12 ? 11 - index : index - 12
        const skipBar = realIndex >= 6 ? barWidth : 0
        return boardPadding + realIndex * pointWidth + skipBar
    }

    const triangleCoords: [number, number, number, number, number, number][] = []
    for (let i = 0; i < 24; i++) {
        const x1 = indexToCoords(i)
        const x2 = x1 + pointWidth
        const x3 = x1 + pointWidth / 2
        let y1, y2, y3
        if (i < 12) {
            y1 = boardPadding
            y2 = boardPadding
            y3 = boardPadding + pointHeight
        } else {
            y1 = svgHeight - boardPadding - pointHeight
            y2 = svgHeight - boardPadding - pointHeight
            y3 = svgHeight - boardPadding
        }
        triangleCoords.push([x1, y1, x2, y2, x3, y3])
    }

    function moveIsValid(current: number, next: number) {
        for (let i = 0; i < gameState.validMoves.length; i++) {
            if (gameState.validMoves[i].length === 0) {
                return false
            }
            if (
                gameState.validMoves[i][0][0] === current &&
                gameState.validMoves[i][0][1] === next
            ) {
                return true
            }
        }
        return false
    }

    function handleClick(index: number, barColor: number = 0) {
        verbose && console.log(`handleClick(${index}, ${barColor})`)
        verbose && console.log(`selectedChecker: ${selectedChecker}`)
        if (selectedChecker === null) {
            if (index === -1) {
                if (barColor === gameState.turn) {
                    setSelectedChecker(index)
                }
                return
            }
            if (gameState.board[index] * gameState.turn >= 0) {
                setSelectedChecker(index)
            }
            return
        }
        if (selectedChecker === index) {
            setSelectedChecker(null)
            return
        }
        if (moveIsValid(selectedChecker, index)) {
            makeLocalMove(selectedChecker, index)
            setSelectedChecker(null)
        } else if (gameState.board[index] * gameState.turn > 0) {
            setSelectedChecker(index)
        } else {
            setSelectedChecker(null)
        }
    }

    function handleCheckerMouseDown(
        e: MouseEvent<SVGCircleElement>,
        pointIndex: number,
        stackIndex: number,
        barNum: number = 0
    ) {
        verbose && console.log("handleCheckerMouseDown")
        let val: number
        if (pointIndex === -1) {
            val = gameState.turn === 1 ? gameState.whiteBar : gameState.blackBar
        } else {
            val = gameState.board[pointIndex]
        }
        if (val === 0) return
        if (barNum === 0) {
            if (val * gameState.turn < 0) return
        } else {
            if (barNum !== gameState.turn) return
        }
        e.stopPropagation()

        const totalCheckersAtPoint = Math.abs(val)
        const topmostIndex = totalCheckersAtPoint - 1
        if (stackIndex !== topmostIndex) return

        let color: "white" | "black"
        if (pointIndex === -1) {
            color = barNum === 1 ? "white" : "black"
        } else {
            color = val > 0 ? "white" : "black"
        }

        const { x, y } = computeCheckerXY(pointIndex, stackIndex, gameState.turn)
        if (!boardRef.current) return
        const svgRect = boardRef.current.getBoundingClientRect()
        const mouseX = e.clientX - svgRect.left
        const mouseY = e.clientY - svgRect.top

        setDragPointIndex(pointIndex)
        setDragColor(color)
        setInitialDragPosition({ x, y })
        setDragOffset({ x: mouseX - x, y: mouseY - y })
        setDragPosition({ x, y })
    }

    function handleMouseMove(e: MouseEvent<SVGSVGElement>) {
        if (dragPointIndex === null || !dragColor) return
        if (!boardRef.current) return
        const svgRect = boardRef.current.getBoundingClientRect()
        const mouseX = e.clientX - svgRect.left
        const mouseY = e.clientY - svgRect.top
        setDragPosition({ x: mouseX - dragOffset.x, y: mouseY - dragOffset.y })
    }

    function handleMouseUp() {
        verbose && console.log("handleMouseUp")
        if (dragPointIndex === null || !dragColor) return
        const { x, y } = dragPosition
        const { x: initialX, y: initialY } = initialDragPosition
        if (Math.abs(x - initialX) < 5 && Math.abs(y - initialY) < 5) {
            if (dragPointIndex !== -1) {
                handleClick(dragPointIndex)
            } else {
                handleClick(-1, dragColor === "white" ? 1 : -1)
            }
            setDragPointIndex(null)
            setDragColor(null)
            return
        }
        if (x >= bearOffXStart && x <= bearOffXStart + bearOffWidth) {
            if (dragColor === "black" && y >= boardPadding && y <= boardPadding + blackBearOffHeight) {
                if (moveIsValid(dragPointIndex, -100)) {
                    makeLocalMove(dragPointIndex, -100)
                }
            } else if (dragColor === "white" && y >= boardPadding + blackBearOffHeight && y <= boardPadding + bearOffAreaHeight) {
                if (moveIsValid(dragPointIndex, 100)) {
                    makeLocalMove(dragPointIndex, 100)
                }
            }
            setDragPointIndex(null)
            setDragColor(null)
            return
        }
        let bestPoint = -1
        let bestDist = Number.POSITIVE_INFINITY
        for (let i = 0; i < 24; i++) {
            const px = indexToCoords(i) + pointWidth / 2
            const py = i < 12
                ? boardPadding + pointHeight / 2
                : svgHeight - boardPadding - pointHeight / 2
            const dx = x - px
            const dy = y - py
            const dist = dx * dx + dy * dy
            if (dist < bestDist) {
                bestDist = dist
                bestPoint = i
            }
        }
        if (moveIsValid(dragPointIndex, bestPoint)) {
            makeLocalMove(dragPointIndex, bestPoint)
        }
        setDragPointIndex(null)
        setDragColor(null)
    }

    function computeCheckerXY(
        pointIndex: number,
        checkerStackIndex: number,
        color: Number = 0
    ) {
        if (pointIndex === -1) {
            const x = 6 * pointWidth + boardPadding + barWidth / 2
            let sign: 1 | -1
            let yBase: number
            if (color === 1) {
                sign = 1
                yBase = boardPadding
            } else {
                sign = -1
                yBase = svgHeight - boardPadding
            }
            const y = yBase + sign * (checkerRadius + checkerPadding) * (checkerStackIndex + 0.5)
            return { x, y }
        }
        const x = indexToCoords(pointIndex) + pointWidth / 2
        let yBase: number
        let direction: 1 | -1
        if (pointIndex < 12) {
            yBase = boardPadding
            direction = 1
        } else {
            yBase = svgHeight - boardPadding
            direction = -1
        }
        const y = yBase + direction * (checkerRadius + checkerPadding) * (checkerStackIndex + 0.5)
        return { x, y }
    }

    const [origin, setOrigin] = useState<number | null>(null)
    useEffect(() => {
        setOrigin(selectedChecker !== null ? selectedChecker : (dragPointIndex !== null ? dragPointIndex : null));
    }, [selectedChecker, dragPointIndex]);

    // --- Pre-game selection UI with a dropdown for AI agent selection
    if (!playerSide) {
        return (
            <div style={{ textAlign: "center", padding: "2rem" }}>
                <h2>Player vs Player</h2>
                <button onClick={() => setPlayerSide("white")}>Play as White</button>
                <button onClick={() => setPlayerSide("black")}>Play as Black</button>
                <h2>Player vs AI</h2>
                <div>
                    <label>
                        Choose AI Agent:{" "}
                        <select
                            value={selectedAiAgent}
                            onChange={e => setSelectedAiAgent(e.target.value)}
                        >
                            <option value="random">Random</option>
                            <option value="first">First Move</option>
                            <option value="mcts">Monte Carlo Tree Search</option>
                            <option value="neural">Neural Network</option>
                        </select>
                    </label>
                </div>
                <button
                    onClick={() => {
                        setPlayerSide("white");
                        setPlayerType("human");
                        // Also join the opposite side as AI with chosen model.
                        socket.emit("join_room", { room_code: roomCode, side: "black", playerType: "ai", aiModel: selectedAiAgent });
                    }}
                >
                    Play as White
                </button>
                <button
                    onClick={() => {
                        setPlayerSide("black");
                        setPlayerType("human");
                        socket.emit("join_room", { room_code: roomCode, side: "white", playerType: "ai", aiModel: selectedAiAgent });
                    }}
                >
                    Play as Black
                </button>
                <h2>AI vs AI (under construction)</h2>
                <div>
                    <label>
                        Choose White AI Agent:{" "}
                        <select
                            value={selectedAiAgent}
                            onChange={e => setSelectedAiAgent(e.target.value)}
                        >
                            <option value="random">Random</option>
                            <option value="first">First Move</option>
                            <option value="mcts">Monte Carlo Tree Search</option>
                            <option value="neural">Neural Network</option>
                        </select>
                    </label>
                </div>
                <div>
                    <label>
                        Choose Black AI Agent:{" "}
                        <select
                            value={selectedAiAgent2}
                            onChange={e => setSelectedAiAgent2(e.target.value)}
                        >
                            <option value="random">Random</option>
                            <option value="first">First Move</option>
                            <option value="mcts">Monte Carlo Tree Search</option>
                            <option value="neural">Neural Network</option>
                        </select>
                    </label>
                </div>

                <button
                    onClick={() => {
                        setPlayerSide("spectator");
                        socket.emit("join_room", { room_code: roomCode, side: "white", playerType: "ai", aiModel: selectedAiAgent });
                        socket.emit("join_room", { room_code: roomCode, side: "black", playerType: "ai", aiModel: selectedAiAgent2 });
                    }}
                >
                    Watch AI vs AI
                </button>
            </div>
        );
    }

    return (
        <>
            <h2>message: {JSON.stringify(message)}</h2>

            <div className="turn-indicator">
                <h1><strong>{turnLabel}</strong> to move</h1>
            </div>

            <div style={{ display: "flex" }}>
                <svg
                    ref={boardRef}
                    width={svgWidth}
                    height={svgHeight}
                    xmlns="http://www.w3.org/2000/svg"
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    style={{ border: "1px solid #333", background: outerBoardColour }}
                >
                    <rect
                        x={boardPadding}
                        y={boardPadding}
                        width={6 * pointWidth}
                        height={svgHeight - 2 * boardPadding}
                        fill={innerBoardColour}
                        stroke="#000"
                        strokeWidth={1}
                    />
                    <rect
                        x={boardPadding + 6 * pointWidth + barWidth}
                        y={boardPadding}
                        width={6 * pointWidth}
                        height={svgHeight - 2 * boardPadding}
                        fill={innerBoardColour}
                        stroke="#000"
                        strokeWidth={1}
                    />
                    <rect
                        x={bearOffXStart}
                        y={boardPadding + blackBearOffHeight}
                        width={bearOffWidth}
                        height={blackBearOffHeight}
                        fill={innerBoardColour}
                        stroke={(origin !== null && moveIsValid(origin, 100)) ? "#00FF00" : "#000"}
                        strokeWidth={(origin !== null && moveIsValid(origin, 100)) ? 3 : 1}
                        onClick={() => handleClick(100)}
                    />
                    <rect
                        x={bearOffXStart}
                        y={boardPadding}
                        width={bearOffWidth}
                        height={blackBearOffHeight}
                        fill={innerBoardColour}
                        stroke={(origin !== null && moveIsValid(origin, -100)) ? "#00FF00" : "#000"}
                        strokeWidth={(origin !== null && moveIsValid(origin, -100)) ? 3 : 1}
                        onClick={() => handleClick(-100)}
                    />

                    {triangleCoords.map((coords, index) => {
                        const [x1, y1, x2, y2, x3, y3] = coords
                        const fill = index % 2 === 0 ? darkPointColour : lightPointColour
                        let transform: string | undefined
                        if (index >= 12) {
                            const midX = x1 + pointWidth / 2
                            const midY = y1 + pointHeight / 2
                            transform = `rotate(180, ${midX}, ${midY})`
                        }
                        let strokeColor = "#000"
                        let strokeWidthValue = 1

                        if (origin !== null && moveIsValid(origin, index)) {
                            strokeColor = index % 2 === 0 ? "#00CC00" : "#00FF00"
                            strokeWidthValue = 3
                        }
                        return (
                            <polygon
                                key={index}
                                points={`${x1},${y1} ${x2},${y2} ${x3},${y3}`}
                                fill={fill}
                                stroke={strokeColor}
                                strokeWidth={strokeWidthValue}
                                transform={transform}
                                onClick={() => handleClick(index)}
                            />
                        )
                    })}

                    {gameState.board.map((pointVal, pointIndex) => {
                        if (pointVal === 0) return null
                        const color = pointVal > 0 ? "white" : "black"
                        const count = Math.abs(pointVal)
                        return Array.from({ length: count }).map((_, stackIndex) => {
                            if (dragPointIndex === pointIndex && dragColor === color && stackIndex === count - 1) {
                                return null
                            }
                            const { x, y } = computeCheckerXY(pointIndex, stackIndex, gameState.turn)
                            const isSelected = selectedChecker === pointIndex && stackIndex === count - 1
                            return (
                                <circle
                                    key={`${pointIndex}-${stackIndex}`}
                                    cx={x}
                                    cy={y}
                                    r={checkerRadius}
                                    fill={color === "white" ? whiteCheckerColour : blackCheckerColour}
                                    stroke="#444"
                                    strokeWidth={2}
                                    style={{ cursor: "grab", outline: isSelected ? "3px solid red" : "none" }}
                                    onMouseDown={e => handleCheckerMouseDown(e, pointIndex, stackIndex)}
                                />
                            )
                        })
                    })}

                    {Array.from({ length: gameState.whiteBar }).map((_, index) => {
                        if (dragPointIndex === -1 && dragColor === "white" && index === gameState.whiteBar - 1) {
                            return null
                        }
                        const { x, y } = computeCheckerXY(-1, index, 1)
                        return (
                            <circle
                                key={`${-1}-${index}-w`}
                                cx={x}
                                cy={y}
                                r={checkerRadius}
                                fill={whiteCheckerColour}
                                stroke="#444"
                                strokeWidth={2}
                                style={{ cursor: "grab", outline: selectedChecker === -1 ? "3px solid red" : "none" }}
                                onMouseDown={e => handleCheckerMouseDown(e, -1, index, 1)}
                            />
                        )
                    })}

                    {Array.from({ length: gameState.blackBar }).map((_, index) => {
                        if (dragPointIndex === -1 && dragColor === "black" && index === gameState.blackBar - 1) {
                            return null
                        }
                        const { x, y } = computeCheckerXY(-1, index, -1)
                        return (
                            <circle
                                key={`${-1}-${index}-b`}
                                cx={x}
                                cy={y}
                                r={checkerRadius}
                                fill={blackCheckerColour}
                                stroke="#444"
                                strokeWidth={2}
                                style={{ cursor: "grab", outline: selectedChecker === -1 ? "3px solid red" : "none" }}
                                onMouseDown={e => handleCheckerMouseDown(e, -1, index, -1)}
                            />
                        )
                    })}

                    {Array.from({ length: gameState.blackBearOff }).map((_, index) => {
                        const x = bearOffXStart + bearOffWidth / 2
                        const y = boardPadding + (checkerRadius + checkerBearOffPadding) * (index + 1.5)
                        return (
                            <circle
                                key={`blackBearOff-${index}`}
                                cx={x}
                                cy={y}
                                r={checkerRadius}
                                fill={blackCheckerColour}
                                stroke="#444"
                                strokeWidth={2}
                            />
                        )
                    })}

                    {Array.from({ length: gameState.whiteBearOff }).map((_, index) => {
                        const x = bearOffXStart + bearOffWidth / 2
                        const bottomY = boardPadding + blackBearOffHeight + blackBearOffHeight
                        const y = bottomY - (checkerRadius + checkerBearOffPadding) * (index + 1.5)
                        return (
                            <circle
                                key={`whiteBearOff-${index}`}
                                cx={x}
                                cy={y}
                                r={checkerRadius}
                                fill={whiteCheckerColour}
                                stroke="#444"
                                strokeWidth={2}
                            />
                        )
                    })}

                    {dragPointIndex !== null && dragColor && (
                        <circle
                            cx={dragPosition.x}
                            cy={dragPosition.y}
                            r={checkerRadius}
                            fill={dragColor === "white" ? whiteCheckerColour : blackCheckerColour}
                            stroke="#444"
                            strokeWidth={2}
                            style={{ pointerEvents: "none" }}
                        />
                    )}
                </svg>

                <div style={{ display: "flex", flexDirection: "column" }}>
                    <button onClick={rollDice} className={`btn btn-roll ${!rolled && playerSide && ((playerSide === "white" && gameState.turn === 1) || (playerSide === "black" && gameState.turn === -1)) ? " pulse" : ""}`} disabled={rolled}>🎲 Roll Dice</button>
                    <button onClick={confirmMove} className="btn btn-confirm" disabled={rolled ? (gameState.validMoves.length > 0 && moveSequence.length == 0) : true}>✅ Confirm</button>
                    <button onClick={resetBoard} className="btn btn-reset">🔄 Reset</button>
                </div>
            </div>


            <div className="dice-container">
                {gameState.dice && gameState.dice.map((value, idx) => {
                    return <Dice key={idx} value={value} rolling={isRollingAnimation} />
                })}
                {Array.isArray(gameState.invalidDice) && gameState.invalidDice && gameState.invalidDice.map((value, idx) => {
                    return <Dice key={idx} value={value} invalid rolling={isRollingAnimation} />
                })}
                {gameState.usedDice && gameState.usedDice.map((value, idx) => {
                    return <Dice key={idx} value={value} used rolling={isRollingAnimation} />
                })}

            </div>
        </>
    )
}
