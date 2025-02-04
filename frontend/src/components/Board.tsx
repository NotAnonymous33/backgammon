import { MouseEvent, useEffect, useRef, useState } from "react"
import "../Board.css"
import { socket } from "../socket"
import { useParams } from "react-router-dom"
import { BACKEND_URL } from "../constants"

// type sent from server
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

// game state type
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
}

export default function Board() {
    const verbose = true
    const params = useParams()
    const roomCode = params["room_code"]

    // local game state
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
    })

    // original true game state
    const originalGameStateRef = useRef<GameState | null>(null)

    const [selectedChecker, setSelectedChecker] = useState<number | null>(null)
    const [moveSequence, setMoveSequence] = useState<number[][]>([])
    const [message, setMessage] = useState<string | null>(null)

    const [dragPointIndex, setDragPointIndex] = useState<number | null>(null)
    const [dragColor, setDragColor] = useState<"white" | "black" | null>(null)
    const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 })
    const [dragPosition, setDragPosition] = useState({ x: 0, y: 0 })
    const [initialDragPosition, setInitialDragPosition] = useState({ x: 0, y: 0 })

    const [rolled, setRolled] = useState(false)

    const pointWidth = 75
    const pointHeight = 200
    const checkerRadius = 20
    const checkerPadding = 5
    const barWidth = 50
    const bearOffWidth = 50
    const boardPadding = 10
    const svgWidth = 2 * boardPadding + 12 * pointWidth + bearOffWidth + barWidth
    const svgHeight = 2 * boardPadding + 2.2 * pointHeight
    const outerBoardColour = "#b39970"
    const innerBoardColour = "#f3d970"
    const darkPointColour = "#64230b"
    const lightPointColour = "#b96522"
    const whiteCheckerColour = "#f3f3f3"
    const blackCheckerColour = "#000000"

    const boardRef = useRef<SVGSVGElement>(null)

    function setVals(data: BoardType) {
        verbose && console.log("setVals")
        setRolled(data.rolled)
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
        }
        setGameState(newGameState)
        verbose && console.log("setting newGameState")
        originalGameStateRef.current = newGameState
    }

    // logging useEffect
    useEffect(() => {
        verbose && console.log(`moveSequence ${moveSequence}`)
    }, [moveSequence])

    useEffect(() => {
        verbose && console.log(`gameState ${JSON.stringify(gameState)}`)
    }, [gameState])

    useEffect(() => {
        socket.connect()
        socket.emit("join_room", { roomCode })

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

        socket.on("update_dice", (data: { dice: number[], validMoves: number[][][], invalidDice: number[] }) => {
            setGameState((prev) => ({
                ...prev,
                dice: data.dice.map(Number),
                validMoves: data.validMoves,
                invalidDice: data.invalidDice.map(Number),
            }))
        })


        socket.on("message", (data) => {
            setMessage(data)
        })


        return () => {
            socket.off("connect")
            socket.off("disconnect")
            socket.off("update_board")
            socket.off("update_dice")
            socket.off("message")
            socket.off("update_valid_moves")
            socket.disconnect()
        }
    }, [roomCode])

    function confirmMove() {
        verbose && console.log("confirmMove")
        socket.emit("move", { moveSequence, roomCode })
    }

    // When a local move is made, update gameState locally
    function makeLocalMove(current: number, next: number) {
        verbose && console.log(`makeLocalMove(${current}, ${next})`)
        setMoveSequence((prev) => [...prev, [current, next]])
        setGameState((prev) => {
            const newBoard = [...prev.board]
            let newWhiteBar = prev.whiteBar
            let newBlackBar = prev.blackBar

            // moving from the bar
            if (current === -1) {
                if (prev.turn === 1) { // white move
                    newWhiteBar -= 1
                } else { // black move
                    newBlackBar -= 1
                }
            } else {
                newBoard[current] = newBoard[current] - prev.turn
            }
            // If the destination has a single opposing checker, capture it:
            if (newBoard[next] === -prev.turn) {
                newBoard[next] = prev.turn
                // update bar count
                if (prev.turn === 1) {
                    // captured black checker
                    newBlackBar += 1
                } else {
                    newWhiteBar += 1
                }
            } else {
                newBoard[next] = newBoard[next] + prev.turn
            }

            // remove the used die
            let newDice = [...prev.dice]
            let diceIndex = newDice.indexOf(Math.abs(next - current))
            if (current === -1) {
                diceIndex =
                    prev.turn === 1
                        ? newDice.indexOf(next + 1)
                        : newDice.indexOf(24 - next)
            }
            if (diceIndex >= 0) {
                newDice.splice(diceIndex, 1)
            }
            // Also update the valid moves: filter out sequences that don’t match,
            // and remove the first move of matching sequences.
            const newValidMoves = prev.validMoves
                .filter(
                    (sequence) =>
                        sequence[0][0] === current && sequence[0][1] === next
                )
                .map((sequence) => sequence.slice(1))

            return {
                ...prev,
                board: newBoard,
                dice: newDice,
                validMoves: newValidMoves,
                whiteBar: newWhiteBar,
                blackBar: newBlackBar,
            }
        })
    }

    // Reset our local changes by reverting to the original game state
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
        if (rolled) return
        socket.emit("roll_dice", { roomCode })
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

    // Example of a backgammon check using our gameState:
    function canBearOff(turn: number) {
        if (turn === 1) {
            if (gameState.whiteBar > 0) return false
            for (let i = 0; i < 18; i++) {
                if (gameState.board[i] > 0) return false
            }
            return true
        }
        if (turn === -1) {
            if (gameState.blackBar > 0) return false
            for (let i = 6; i < 24; i++) {
                if (gameState.board[i] < 0) return false
            }
            return true
        }
        throw new Error("Invalid turn value")
    }

    // Helper: convert point index to x coordinate
    function indexToCoords(index: number) {
        let realIndex = index < 12 ? 11 - index : index - 12
        const skipBar = realIndex >= 6 ? barWidth : 0
        return boardPadding + realIndex * pointWidth + skipBar
    }

    // Pre-calculate triangle coordinates
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

    // moveIsValid checks against the valid moves in gameState
    function moveIsValid(current: number, next: number) {
        verbose && console.log(`moveIsValid(${current}, ${next})`)
        for (let i = 0; i < gameState.validMoves.length; i++) {
            if (
                gameState.validMoves[i][0][0] === current &&
                gameState.validMoves[i][0][1] === next
            ) {
                verbose && console.log("Move is valid")
                return true
            }
        }
        verbose && console.log("Move is invalid")
        return false
    }

    // handleClick and drag handlers remain mostly the same,
    // but they now reference gameState (instead of individual state vars).
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

    // --- DRAG AND DROP HANDLERS (mostly unchanged) ---
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

        // Remove the checker locally
        if (pointIndex !== -1) {
            setGameState((prev) => {
                const newBoard = [...prev.board]
                newBoard[pointIndex] =
                    color === "white" ? newBoard[pointIndex] - 1 : newBoard[pointIndex] + 1
                return { ...prev, board: newBoard }
            })
        } else {
            if (color === "white") {
                setGameState((prev) => ({ ...prev, whiteBar: prev.whiteBar - 1 }))
            } else {
                setGameState((prev) => ({ ...prev, blackBar: prev.blackBar - 1 }))
            }
        }
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
            // Restore the dropped checker if it was just a click:
            if (dragPointIndex === -1) {
                if (dragColor === "white") {
                    setGameState((prev) => ({ ...prev, whiteBar: prev.whiteBar + 1 }))
                } else {
                    setGameState((prev) => ({ ...prev, blackBar: prev.blackBar + 1 }))
                }
            } else {
                setGameState((prev) => {
                    const newBoard = [...prev.board]
                    newBoard[dragPointIndex] =
                        dragColor === "white"
                            ? newBoard[dragPointIndex] + 1
                            : newBoard[dragPointIndex] - 1
                    return { ...prev, board: newBoard }
                })
            }
            setDragPointIndex(null)
            setDragColor(null)
            return
        }

        // Find the closest point on the board
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
        } else {
            // If the move is invalid, restore the checker at its original point.
            if (dragPointIndex === -1) {
                if (dragColor === "white") {
                    setGameState((prev) => ({ ...prev, whiteBar: prev.whiteBar + 1 }))
                } else {
                    setGameState((prev) => ({ ...prev, blackBar: prev.blackBar + 1 }))
                }
            } else {
                setGameState((prev) => {
                    const newBoard = [...prev.board]
                    newBoard[dragPointIndex] =
                        dragColor === "white"
                            ? newBoard[dragPointIndex] + 1
                            : newBoard[dragPointIndex] - 1
                    return { ...prev, board: newBoard }
                })
            }
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

    return (
        <>
            <h2>White bar: {gameState.whiteBar}</h2>
            <h2>Black bar: {gameState.blackBar}</h2>
            <h2>White bear off: {gameState.whiteBearOff}</h2>
            <h2>Black bear off: {gameState.blackBearOff}</h2>
            <h2>Can bear off: {canBearOff(gameState.turn) ? "Yes" : "No"}</h2>
            <h2>message: {JSON.stringify(message)}</h2>

            {canBearOff(1) && (
                <div className="bearOffWhite" onClick={() => handleClick(100)}>
                    Bear off white
                </div>
            )}

            {canBearOff(-1) && (
                <div className="bearOffBlack" onClick={() => handleClick(-100)}>
                    Bear off black
                </div>
            )}

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
                    {/* Inner board */}
                    <rect
                        x={boardPadding}
                        y={boardPadding}
                        width={6 * pointWidth}
                        height={svgHeight - 2 * boardPadding}
                        fill={innerBoardColour}
                    />
                    <rect
                        x={6 * pointWidth + boardPadding + barWidth}
                        y={boardPadding}
                        width={6 * pointWidth}
                        height={svgHeight - 2 * boardPadding}
                        fill={innerBoardColour}
                    />

                    {/* Triangles */}
                    {triangleCoords.map((coords, index) => {
                        const [x1, y1, x2, y2, x3, y3] = coords
                        const fill = index % 2 === 0 ? darkPointColour : lightPointColour
                        let transform: string | undefined
                        if (index >= 12) {
                            const midX = x1 + pointWidth / 2
                            const midY = y1 + pointHeight / 2
                            transform = `rotate(180, ${midX}, ${midY})`
                        }
                        return (
                            <polygon
                                key={index}
                                points={`${x1},${y1} ${x2},${y2} ${x3},${y3}`}
                                fill={fill}
                                stroke="#000"
                                strokeWidth={1}
                                transform={transform}
                                onClick={() => handleClick(index)}
                            />
                        )
                    })}

                    {/* Checkers on board */}
                    {gameState.board.map((pointVal, pointIndex) => {
                        if (pointVal === 0) return null
                        const color = pointVal > 0 ? "white" : "black"
                        const count = Math.abs(pointVal)
                        return Array.from({ length: count }).map((_, stackIndex) => {
                            const { x, y } = computeCheckerXY(pointIndex, stackIndex, gameState.turn)
                            // Do not show the checker if it’s being dragged
                            let isSelected = pointIndex === selectedChecker && stackIndex === count - 1
                            if (dragPointIndex === pointIndex && dragColor === color) {
                                isSelected = false
                            }
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
                                    onMouseDown={(e) => handleCheckerMouseDown(e, pointIndex, stackIndex)}
                                />
                            )
                        })
                    })}

                    {/* Checkers on white bar */}
                    {Array.from({ length: gameState.whiteBar }).map((_, index) => {
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
                                onMouseDown={(e) => handleCheckerMouseDown(e, -1, index, 1)}
                            />
                        )
                    })}

                    {/* Checkers on black bar */}
                    {Array.from({ length: gameState.blackBar }).map((_, index) => {
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
                                onMouseDown={(e) => handleCheckerMouseDown(e, -1, index, -1)}
                            />
                        )
                    })}

                    {/* Render dragged checker */}
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
                    <button onClick={resetBoard}>Reset</button>
                    <button onClick={confirmMove}>
                        Confirm
                    </button>
                    <button onClick={rollDice}>Roll dice</button>
                    <button onClick={testButton}>test</button>

                </div>
            </div>

            <h2>turn: {gameState.turn}</h2>
            <h2>active piece: {selectedChecker}</h2>
            <h3>dice: {gameState.dice.join(", ")}</h3>


        </>
    )
}
