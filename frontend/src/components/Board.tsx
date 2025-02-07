import { MouseEvent, useEffect, useRef, useState } from "react"
import "../Board.css"
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
}

export default function Board() {
    const verbose = true
    const params = useParams()
    const roomCode = params["room_code"]

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

    const pointWidth = 75
    const pointHeight = 200
    const checkerRadius = 20
    const checkerPadding = 5
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

    //--- calculate bearing off area position and size
    const bearOffXStart = boardPadding + 12 * pointWidth + barWidth + 5
    const bearOffAreaHeight = svgHeight - 2 * boardPadding
    const blackBearOffHeight = bearOffAreaHeight / 2

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
        }
        setGameState(newGameState)
        console.log(data.valid_moves)
        originalGameStateRef.current = newGameState
    }

    useEffect(() => {
        //--- log move sequence update
        verbose && console.log(`moveSequence ${moveSequence}`)
    }, [moveSequence])

    useEffect(() => {
        //--- log game state update
        verbose && console.log(`gameState ${JSON.stringify(gameState)}`)
    }, [gameState])

    useEffect(() => {
        //--- connect socket and join room
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

        socket.on("update_dice", (data: { dice: number[]; validMoves: number[][][]; invalidDice: number[]; rolled: boolean }) => {
            const newGameState: GameState = {
                ...gameState,
                dice: data.dice.map(Number),
                validMoves: data.validMoves,
                invalidDice: data.invalidDice.map(Number),
            }
            setGameState(newGameState)
            originalGameStateRef.current = newGameState
            verbose && console.log("Updated dice:", data.dice)
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
            socket.disconnect()
        }
    }, [roomCode])

    function confirmMove() {
        //--- confirm move by emitting to server
        verbose && console.log("confirmMove")
        socket.emit("move", { moveSequence, roomCode })
        setMoveSequence([])
    }

    //--- update local game state when a move is made
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
                //--- bear off white
                newBoard[current] -= 1
                newWhiteBearOff += 1
            } else if (next === -100) {
                //--- bear off black
                newBoard[current] += 1
                newBlackBearOff += 1
            } else {
                //--- moving from the bar
                if (current === -1) {
                    if (prev.turn === 1) {
                        newWhiteBar -= 1
                    } else {
                        newBlackBar -= 1
                    }
                } else {
                    newBoard[current] = newBoard[current] - prev.turn
                }
                //--- capture opposing checker if present
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

            //--- remove used die from dice
            let newDice = [...prev.dice]
            let diceIndex = newDice.indexOf(Math.abs(next - current))
            if (current === -1) {
                diceIndex =
                    prev.turn === 1 ? newDice.indexOf(next + 1) : newDice.indexOf(24 - next)
            }
            if (next === 100 || next === -100) {
                diceIndex = newDice.indexOf(Math.max(...newDice))
            }
            if (diceIndex >= 0) {
                newDice.splice(diceIndex, 1)
            }
            //--- update valid moves by filtering sequences
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
                whiteBearOff: newWhiteBearOff,
                blackBearOff: newBlackBearOff,
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

    // TODO: if i move, roll dice, then reset, valid_moves is empty (no confirm). shit is bugged the fuck out dawg idk man

    function rollDice() {
        verbose && console.log("rollDice")
        verbose && console.log("rolled: ", rolled)
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

    //--- check if bearing off is allowed for the current turn
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

    //--- helper to convert point index to x coordinate
    function indexToCoords(index: number) {
        let realIndex = index < 12 ? 11 - index : index - 12
        const skipBar = realIndex >= 6 ? barWidth : 0
        return boardPadding + realIndex * pointWidth + skipBar
    }

    //--- pre-calculate triangle coordinates
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

    //--- check move validity against valid moves in gameState
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

    //--- DRAG AND DROP HANDLERS (modified to support bearing off)
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
        //--- do not remove the checker from state on drag start
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
        //--- if movement is minimal, treat as a click
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
        //--- check if dropped in bearing off area
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
        //--- otherwise, find the closest triangle on the board
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

    return (
        <>
            <h2>White bar: {gameState.whiteBar}</h2>
            <h2>Black bar: {gameState.blackBar}</h2>
            <h2>White bear off: {gameState.whiteBearOff}</h2>
            <h2>Black bear off: {gameState.blackBearOff}</h2>
            <h2>Can bear off: {canBearOff(gameState.turn) ? "Yes" : "No"}</h2>
            <h2>message: {JSON.stringify(message)}</h2>

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
                    {/*//--- inner board left half */}
                    <rect
                        x={boardPadding}
                        y={boardPadding}
                        width={6 * pointWidth}
                        height={svgHeight - 2 * boardPadding}
                        fill={innerBoardColour}
                        stroke="#000"
                        strokeWidth={1}
                    />
                    {/*//--- inner board right half */}
                    <rect
                        x={boardPadding + 6 * pointWidth + barWidth}
                        y={boardPadding}
                        width={6 * pointWidth}
                        height={svgHeight - 2 * boardPadding}
                        fill={innerBoardColour}
                        stroke="#000"
                        strokeWidth={1}
                    />
                    {/*//--- white bearing off area */}
                    <rect
                        x={bearOffXStart}
                        y={boardPadding + blackBearOffHeight}
                        width={bearOffWidth}
                        height={blackBearOffHeight}
                        fill={innerBoardColour}
                        stroke="#000"
                        strokeWidth={1}
                        onClick={() => handleClick(100)}
                    />
                    {/*//--- black bearing off area */}
                    <rect
                        x={bearOffXStart}
                        y={boardPadding}
                        width={bearOffWidth}
                        height={blackBearOffHeight}
                        fill={innerBoardColour}
                        stroke="#000"
                        strokeWidth={1}
                        onClick={() => handleClick(-100)}
                    />

                    {/*//--- triangles */}
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

                    {/*//--- checkers on board */}
                    {gameState.board.map((pointVal, pointIndex) => {
                        if (pointVal === 0) return null
                        const color = pointVal > 0 ? "white" : "black"
                        const count = Math.abs(pointVal)
                        return Array.from({ length: count }).map((_, stackIndex) => {
                            //--- hide the checker if it is currently being dragged from this point
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
                                    onMouseDown={(e) => handleCheckerMouseDown(e, pointIndex, stackIndex)}
                                />
                            )
                        })
                    })}

                    {/*//--- checkers on white bar */}
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
                                onMouseDown={(e) => handleCheckerMouseDown(e, -1, index, 1)}
                            />
                        )
                    })}

                    {/*//--- checkers on black bar */}
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
                                onMouseDown={(e) => handleCheckerMouseDown(e, -1, index, -1)}
                            />
                        )
                    })}

                    {/*//--- render bear off checkers for black */}
                    {Array.from({ length: gameState.blackBearOff }).map((_, index) => {
                        const x = bearOffXStart + bearOffWidth / 2
                        const y = boardPadding + (checkerRadius + checkerPadding) * (index + 0.5)
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

                    {/*//--- render bear off checkers for white */}
                    {Array.from({ length: gameState.whiteBearOff }).map((_, index) => {
                        const x = bearOffXStart + bearOffWidth / 2
                        const bottomY = boardPadding + blackBearOffHeight + blackBearOffHeight
                        const y = bottomY - (checkerRadius + checkerPadding) * (index + 0.5)

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

                    {/*//--- render dragged checker */}
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
                    <button onClick={confirmMove}>Confirm</button>
                    <button onClick={rollDice}>Roll dice</button>
                    <button onClick={testButton}>test</button>
                </div>
            </div>

            <h2>turn: {gameState.turn}</h2>
            <h2>active piece: {selectedChecker}</h2>
            <h3>dice: {gameState.dice.join(", ")}</h3>
            <h3>invalid dice:{gameState.invalidDice}</h3>
        </>
    )
}
