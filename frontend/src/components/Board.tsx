import { CSSProperties, MouseEvent, ReactElement, SVGProps, useEffect, useRef, useState } from "react"
import "./Board.css"
import Dice from "./Dice"
import { socket } from "../socket"
import { useParams } from "react-router-dom"
import { useWindowSize } from 'react-use'
import Confetti from 'react-confetti'

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
    last_move?: LastMove
}

//--- move that produced an update_board, sent by the server
type LastMove = {
    player: number
    sequence: [number, number][]
}

//--- one checker's journey within a turn, e.g. 24/18*/13
type MovePath = {
    points: number[]
    hits: boolean[]
}

type MoveHighlight = {
    player: number
    paths: MovePath[]
}

//--- where a rendered checker sits
type Slot =
    | { kind: "point"; index: number; stack: number }
    | { kind: "bar" | "off"; color: number; stack: number }

//--- checker that slides into its slot from (fromX, fromY)
type Landing = {
    fromX: number
    fromY: number
    delay: number
    duration: number
}

type Landings = {
    id: number
    bySlot: Record<string, Landing>
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

const OPPONENT_STEP_MS = 450
const LOCAL_STEP_MS = 250
const DROP_STEP_MS = 120
const HIT_MS = 350
const STEP_GAP_MS = 150
const ROLL_MS = 600
const highlightColour = "#1e90ff"

const sleep = (ms: number) => new Promise<void>(resolve => setTimeout(resolve, ms))
const prefersReducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches

function sideToPlayer(side: string | null) {
    return side === "white" ? 1 : side === "black" ? -1 : 0
}

function slotKey(slot: Slot) {
    return slot.kind === "point" ? `p${slot.index}-${slot.stack}` : `${slot.kind}${slot.color}-${slot.stack}`
}

function barCount(state: GameState, player: number) {
    return player === 1 ? state.whiteBar : state.blackBar
}

function offCount(state: GameState, player: number) {
    return player === 1 ? state.whiteBearOff : state.blackBearOff
}

function isBearOff(next: number) {
    return next === 100 || next === -100
}

//--- point number from the mover's perspective, as used in move notation
function pointLabel(index: number, player: number) {
    if (index === -1) return "bar"
    if (isBearOff(index)) return "off"
    return String(player === 1 ? 24 - index : index + 1)
}

function moveNotation(highlight: MoveHighlight) {
    return highlight.paths
        .map(path => path.points.map((point, i) => pointLabel(point, highlight.player) + (path.hits[i] ? "*" : "")).join("/"))
        .join(" ")
}

//--- move one checker of `player` from current to next
function applyStep(prev: GameState, player: number, current: number, next: number) {
    const newBoard = [...prev.board]
    let newWhiteBar = prev.whiteBar
    let newBlackBar = prev.blackBar
    let newWhiteBearOff = prev.whiteBearOff
    let newBlackBearOff = prev.blackBearOff
    let hit = false

    if (next === 100) {
        newBoard[current] -= 1
        newWhiteBearOff += 1
    } else if (next === -100) {
        newBoard[current] += 1
        newBlackBearOff += 1
    } else {
        if (current === -1) {
            if (player === 1) {
                newWhiteBar -= 1
            } else {
                newBlackBar -= 1
            }
        } else {
            newBoard[current] = newBoard[current] - player
        }
        if (newBoard[next] === -player) {
            hit = true
            newBoard[next] = player
            if (player === 1) {
                newBlackBar += 1
            } else {
                newWhiteBar += 1
            }
        } else {
            newBoard[next] = newBoard[next] + player
        }
    }

    const state: GameState = {
        ...prev,
        board: newBoard,
        whiteBar: newWhiteBar,
        blackBar: newBlackBar,
        whiteBearOff: newWhiteBearOff,
        blackBearOff: newBlackBearOff,
    }
    return { state, hit }
}

//--- remove the die that a move from current to next uses
function consumeDie(prev: GameState, player: number, current: number, next: number): GameState {
    const newDice = [...prev.dice]
    let diceIndex = newDice.indexOf(Math.abs(next - current))
    if (current === -1) diceIndex = player === 1 ? newDice.indexOf(next + 1) : newDice.indexOf(24 - next)
    if (isBearOff(next)) {
        diceIndex = next === 100 ? newDice.indexOf(24 - current) : newDice.indexOf(current + 1)
        if (diceIndex < 0) diceIndex = newDice.indexOf(Math.max(...newDice))
    }

    if (diceIndex < 0) return prev
    const removedDie = newDice[diceIndex]
    newDice.splice(diceIndex, 1)
    return { ...prev, dice: newDice, usedDice: [...prev.usedDice, removedDie] }
}

//--- group steps into per-checker paths so 24/18 18/13 reads as 24/18/13
function extendPaths(paths: MovePath[], current: number, next: number, hit: boolean) {
    const path = paths.find(p => p.points[p.points.length - 1] === current)
    if (path) {
        path.points.push(next)
        path.hits.push(hit)
    } else {
        paths.push({ points: [current, next], hits: [false, hit] })
    }
}

export default function Board() {
    const verbose = true
    const params = useParams()
    const roomCode = params["room_code"]
    const { width, height } = useWindowSize()

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

    //--- move animation / highlighting
    const [landings, setLandings] = useState<Landings>({ id: 0, bySlot: {} })
    const landingIdRef = useRef(0)
    const [animating, setAnimating] = useState(false)
    const [lastMove, setLastMove] = useState<MoveHighlight | null>(null)
    // Socket events are applied one at a time so a replay finishes before the next update lands.
    const eventQueueRef = useRef<Promise<void>>(Promise.resolve())
    const playerSideRef = useRef<string | null>(null)

    const [rolled, setRolled] = useState(false)
    const [isRollingAnimation, setIsRollingAnimation] = useState(false)
    const [gameOver, setGameOver] = useState(false)

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
    playerSideRef.current = playerSide
    const turnLabel = gameState.turn === 1 ? "White" : "Black"


    //--- calculate bearing off area position and size
    const bearOffXStart = boardPadding + 12 * pointWidth + barWidth + 5
    const bearOffAreaHeight = svgHeight - 2 * boardPadding
    const blackBearOffHeight = bearOffAreaHeight / 2

    useEffect(() => {
        if (!playerSide) return

        const rejoinAsPlayer = () => {
            socket.emit("join_room", { room_code: roomCode, side: playerSide, playerType });
        }

        // A server reload forgets player assignments while React keeps the side.
        socket.on("connect", rejoinAsPlayer)
        if (socket.connected) rejoinAsPlayer()

        return () => {
            socket.off("connect", rejoinAsPlayer)
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
        setLandings({ id: landingIdRef.current, bySlot: {} })
        console.log(data.valid_moves)
        originalGameStateRef.current = newGameState
    }

    function enqueue(task: () => void | Promise<void>) {
        eventQueueRef.current = eventQueueRef.current.then(task).catch(err => console.error(err))
    }

    function stepLandings(
        prev: GameState,
        next: GameState,
        player: number,
        current: number,
        target: number,
        hit: boolean,
        duration: number,
        from?: { x: number; y: number }
    ): Landings {
        const source = from ?? (current === -1
            ? computeCheckerXY(-1, barCount(prev, player) - 1, player)
            : computeCheckerXY(current, Math.abs(prev.board[current]) - 1))
        const destSlot: Slot = isBearOff(target)
            ? { kind: "off", color: player, stack: offCount(next, player) - 1 }
            : { kind: "point", index: target, stack: Math.abs(next.board[target]) - 1 }
        const bySlot: Record<string, Landing> = {
            [slotKey(destSlot)]: { fromX: source.x, fromY: source.y, delay: 0, duration },
        }
        if (hit) {
            // The blot waits for the hitting checker to arrive, then flies to the bar.
            const blot = computeCheckerXY(target, 0)
            const barSlot: Slot = { kind: "bar", color: -player, stack: barCount(next, -player) - 1 }
            bySlot[slotKey(barSlot)] = { fromX: blot.x, fromY: blot.y, delay: duration, duration: HIT_MS }
        }
        landingIdRef.current += 1
        return { id: landingIdRef.current, bySlot }
    }

    //--- apply a board update, replaying the opponent's move checker by checker
    async function receiveBoard(data: BoardType) {
        const move = data.last_move
        const base = originalGameStateRef.current
        if (!move || !base) {
            setVals(data)
            return
        }
        if (move.player === sideToPlayer(playerSideRef.current)) {
            // Our own move: it is already on screen, and the opponent's highlight is now stale.
            setLastMove(null)
            setVals(data)
            return
        }

        const animate = playerSideRef.current !== null && !prefersReducedMotion()
        if (animate) {
            setAnimating(true)
            setSelectedChecker(null)
            setDragPointIndex(null)
            setDragColor(null)
            setMoveSequence([])
            setLastMove(null)
        }
        let state: GameState = { ...base, validMoves: [] }
        const paths: MovePath[] = []
        for (const [current, next] of move.sequence) {
            const { state: moved, hit } = applyStep(state, move.player, current, next)
            const nextState = consumeDie(moved, move.player, current, next)
            extendPaths(paths, current, next, hit)
            if (animate) {
                setLandings(stepLandings(state, nextState, move.player, current, next, hit, OPPONENT_STEP_MS))
                setGameState(nextState)
                await sleep(OPPONENT_STEP_MS + (hit ? HIT_MS : 0) + STEP_GAP_MS)
            }
            state = nextState
        }
        if (animate) setAnimating(false)
        setVals(data)
        setLastMove({ player: move.player, paths })
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

        const handleConnect = () => {
            verbose && console.log("socket:connect")
        }
        socket.on("connect", handleConnect)

        socket.on("disconnect", () => {
            verbose && console.log("socket:disconnect")
        })

        socket.on("update_board", (data: BoardType) => {
            verbose && console.log("socket:update_board")
            enqueue(() => receiveBoard(data))
        })

        socket.on("game_over", (data: { winner: number }) => enqueue(() => {
            setMessage(`${data.winner} wins!`)
            setGameOver(true)

            verbose && console.log("Game Over:", JSON.stringify(data))
        }))

        socket.on("update_dice", (data: { dice: number[]; validMoves: number[][][]; invalidDice: number[]; rolled: boolean }) => enqueue(async () => {
            const base = originalGameStateRef.current
            if (!base) return
            setRolled(data.rolled)
            const newGameState: GameState = {
                ...base,
                dice: data.dice.map(Number),
                validMoves: data.validMoves,
                invalidDice: data.invalidDice.map(Number),
                usedDice: [],
            }
            console.log(newGameState.validMoves)
            originalGameStateRef.current = newGameState
            setGameState(newGameState)
            verbose && console.log("Updated dice:", data.dice)
            if (!prefersReducedMotion()) {
                // Spin the dice for every roll, including the opponent's, before moves are replayed.
                setIsRollingAnimation(true)
                await sleep(ROLL_MS)
            }
            setIsRollingAnimation(false)
        }))

        socket.on("message", (data) => {
            setMessage(data)
        })

        return () => {
            //--- cleanup socket events
            socket.off("connect", handleConnect)
            socket.off("disconnect")
            socket.off("update_board")
            socket.off("update_dice")
            socket.off("message")
            socket.off("update_valid_moves")
            socket.off("game_over")
            // Keep the shared connection (and player SID) across hot reloads.
            socket.emit("leave_room", { room_code: roomCode })
        }
    }, [roomCode])

    function confirmMove() {
        if (animating) return
        if (verbose) {
            console.log("ConfirmingMove move sequence:", moveSequence)
        }
        socket.emit("move", { moveSequence, room_code: roomCode })
        setMoveSequence([])
    }

    function makeLocalMove(current: number, next: number, dropPosition?: { x: number; y: number }) {
        verbose && console.log(`makeLocalMove(${current}, ${next})`)
        const prev = gameState
        const { state: moved, hit } = applyStep(prev, prev.turn, current, next)
        const newState: GameState = {
            ...consumeDie(moved, prev.turn, current, next),
            validMoves: prev.validMoves
                .filter(seq => seq[0][0] === current && seq[0][1] === next)
                .map(seq => seq.slice(1)),
        }
        const duration = dropPosition ? DROP_STEP_MS : LOCAL_STEP_MS
        setLandings(stepLandings(prev, newState, prev.turn, current, next, hit, duration, dropPosition))
        setMoveSequence((seq) => [...seq, [current, next]])
        setGameState(newState)
    }

    function resetBoard() {
        verbose && console.log("resetBoard")
        if (animating) return
        if (originalGameStateRef.current !== null) {
            setGameState(originalGameStateRef.current)
        }
        setLandings({ id: landingIdRef.current, bySlot: {} })
        setSelectedChecker(null)
        setMoveSequence([])
    }

    function rollDice() {
        verbose && console.log("rollDice")
        verbose && console.log("rolled: ", rolled)
        if (rolled || animating) return
        setIsRollingAnimation(true)
        verbose && console.log("emitting roll_dice")
        socket.emit("roll_dice", { room_code: roomCode })
        console.log("turn: ", gameState.turn)
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
        if (animating) return
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
        if (animating) return
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
                    makeLocalMove(dragPointIndex, -100, dragPosition)
                }
            } else if (dragColor === "white" && y >= boardPadding + blackBearOffHeight && y <= boardPadding + bearOffAreaHeight) {
                if (moveIsValid(dragPointIndex, 100)) {
                    makeLocalMove(dragPointIndex, 100, dragPosition)
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
            makeLocalMove(dragPointIndex, bestPoint, dragPosition)
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

    function bearOffXY(color: number, index: number) {
        const x = bearOffXStart + bearOffWidth / 2
        if (color === -1) {
            return { x, y: boardPadding + (checkerRadius + checkerBearOffPadding) * (index + 1.5) }
        }
        const bottomY = boardPadding + blackBearOffHeight + blackBearOffHeight
        return { x, y: bottomY - (checkerRadius + checkerBearOffPadding) * (index + 1.5) }
    }

    function slotXY(slot: Slot) {
        if (slot.kind === "point") return computeCheckerXY(slot.index, slot.stack)
        if (slot.kind === "bar") return computeCheckerXY(-1, slot.stack, slot.color)
        return bearOffXY(slot.color, slot.stack)
    }

    //--- the checker a path ended on, if it is still there
    function landedSlot(dest: number, arrived: number, player: number): Slot | null {
        if (isBearOff(dest)) {
            const stack = offCount(gameState, player) - 1 - arrived
            return stack >= 0 ? { kind: "off", color: player, stack } : null
        }
        if (gameState.board[dest] * player <= 0) return null
        const stack = Math.abs(gameState.board[dest]) - 1 - arrived
        return stack >= 0 ? { kind: "point", index: dest, stack } : null
    }

    //--- curved arrow between two checkers, bowing towards the middle of the board
    function arrowPath(from: { x: number; y: number }, to: { x: number; y: number }) {
        const midX = (from.x + to.x) / 2
        const midY = (from.y + to.y) / 2
        const control = { x: midX, y: midY + (svgHeight / 2 - midY) * 0.6 }
        const towards = (p: { x: number; y: number }, distance: number) => {
            const dx = control.x - p.x
            const dy = control.y - p.y
            const length = Math.hypot(dx, dy) || 1
            return { x: p.x + dx / length * distance, y: p.y + dy / length * distance }
        }
        const start = towards(from, checkerRadius)
        const end = towards(to, checkerRadius + 4)
        return `M ${start.x} ${start.y} Q ${control.x} ${control.y} ${end.x} ${end.y}`
    }

    const [origin, setOrigin] = useState<number | null>(null)
    useEffect(() => {
        setOrigin(selectedChecker !== null ? selectedChecker : (dragPointIndex !== null ? dragPointIndex : null));
    }, [selectedChecker, dragPointIndex]);

    // --- Pre-game selection UI with a dropdown for AI agent selection
    if (!playerSide) {
        return (
            <div className="lobby-container">
                <section className="mode-section">
                    <h2>Player vs Player</h2>
                    <div className="button-group">
                        <button className="btn btn-white" onClick={() => setPlayerSide("white")}>Play as White</button>
                        <button className="btn btn-black" onClick={() => setPlayerSide("black")}>Play as Black</button>
                    </div>
                </section>

                <section className="mode-section">
                    <h2>Player vs AI</h2>
                    <div className="form-group">
                        <label>
                            Choose AI Agent:
                            <select className="select" value={selectedAiAgent} onChange={e => setSelectedAiAgent(e.target.value)}>
                                <option value="random">Random</option>
                                <option value="heuristic">Heuristic</option>
                                <option value="mcts">Monte Carlo Tree Search</option>
                                <option value="neural">Neural Network</option>
                            </select>
                        </label>
                    </div>
                    <div className="button-group">
                        <button className="btn btn-white" onClick={() => {
                            setPlayerSide("white");
                            setPlayerType("human");
                            socket.emit("join_room", { room_code: roomCode, side: "black", playerType: "ai", aiModel: selectedAiAgent });
                        }}>Play as White</button>
                        <button className="btn btn-black" onClick={() => {
                            setPlayerSide("black");
                            setPlayerType("human");
                            socket.emit("join_room", { room_code: roomCode, side: "white", playerType: "ai", aiModel: selectedAiAgent });
                        }}>Play as Black</button>
                    </div>
                </section>

                <section className="mode-section">
                    <h2>AI vs AI</h2>
                    <div className="form-group">
                        <label>
                            Choose White AI Agent:
                            <select className="select" value={selectedAiAgent} onChange={e => setSelectedAiAgent(e.target.value)}>
                                <option value="random">Random</option>
                                <option value="heuristic">Heuristic</option>
                                <option value="mcts">Monte Carlo Tree Search</option>
                                <option value="neural">Neural Network</option>
                            </select>
                        </label>
                    </div>
                    <div className="form-group">
                        <label>
                            Choose Black AI Agent:
                            <select className="select" value={selectedAiAgent2} onChange={e => setSelectedAiAgent2(e.target.value)}>
                                <option value="random">Random</option>
                                <option value="heuristic">Heuristic</option>
                                <option value="mcts">Monte Carlo Tree Search</option>
                                <option value="neural">Neural Network</option>
                            </select>
                        </label>
                    </div>
                    <div className="button-group">
                        <button className="btn" onClick={() => {
                            setPlayerSide("spectator");
                            socket.emit("join_room", { room_code: roomCode, side: "white", playerType: "ai", aiModel: selectedAiAgent });
                            socket.emit("join_room", { room_code: roomCode, side: "black", playerType: "ai", aiModel: selectedAiAgent2 });
                        }}>Watch AI vs AI</button>
                    </div>
                </section>
            </div>
        );
    }

    //--- sliding checkers are drawn last so they pass over the others
    const staticCheckers: ReactElement[] = []
    const slidingCheckers: ReactElement[] = []
    function drawChecker(slot: Slot, colour: number, props: SVGProps<SVGCircleElement> = {}) {
        const { x, y } = slotXY(slot)
        const key = slotKey(slot)
        const landing = landings.bySlot[key]
        const style = landing ? {
            ...props.style,
            "--dx": `${landing.fromX - x}px`,
            "--dy": `${landing.fromY - y}px`,
            "--delay": `${landing.delay}ms`,
            "--duration": `${landing.duration}ms`,
        } as CSSProperties : props.style
        const checker = (
            <circle
                key={landing ? `${key}-landing${landings.id}` : key}
                cx={x}
                cy={y}
                r={checkerRadius}
                fill={colour === 1 ? whiteCheckerColour : blackCheckerColour}
                stroke="#444"
                strokeWidth={2}
                {...props}
                className={landing ? "checker-landing" : undefined}
                style={style}
            />
        )
        if (landing) {
            slidingCheckers.push(checker)
        } else {
            staticCheckers.push(checker)
        }
    }

    gameState.board.forEach((pointVal, pointIndex) => {
        const colour = pointVal > 0 ? 1 : -1
        const count = Math.abs(pointVal)
        for (let stackIndex = 0; stackIndex < count; stackIndex++) {
            if (dragPointIndex === pointIndex && dragColor === (colour === 1 ? "white" : "black") && stackIndex === count - 1) {
                continue
            }
            const isSelected = selectedChecker === pointIndex && stackIndex === count - 1
            drawChecker({ kind: "point", index: pointIndex, stack: stackIndex }, colour, {
                style: { cursor: "grab", outline: isSelected ? "3px solid red" : "none" },
                onMouseDown: e => handleCheckerMouseDown(e, pointIndex, stackIndex),
            })
        }
    })

    for (const colour of [1, -1]) {
        const count = barCount(gameState, colour)
        for (let index = 0; index < count; index++) {
            if (dragPointIndex === -1 && dragColor === (colour === 1 ? "white" : "black") && index === count - 1) {
                continue
            }
            drawChecker({ kind: "bar", color: colour, stack: index }, colour, {
                style: { cursor: "grab", outline: selectedChecker === -1 ? "3px solid red" : "none" },
                onMouseDown: e => handleCheckerMouseDown(e, -1, index, colour),
            })
        }
    }

    for (const colour of [-1, 1]) {
        for (let index = 0; index < offCount(gameState, colour); index++) {
            drawChecker({ kind: "off", color: colour, stack: index }, colour)
        }
    }

    //--- opponent's last move: ghost where each checker left, an arrow, and a ring where it landed
    const lastMoveArrows: ReactElement[] = []
    const lastMoveRings: ReactElement[] = []
    if (lastMove) {
        const player = lastMove.player
        const departures: Record<number, number> = {}
        const arrivals: Record<number, number> = {}
        lastMove.paths.forEach((path, i) => {
            const start = path.points[0]
            const dest = path.points[path.points.length - 1]
            const departed = departures[start] ?? 0
            departures[start] = departed + 1
            const arrived = arrivals[dest] ?? 0
            arrivals[dest] = arrived + 1

            const ghost = start === -1
                ? computeCheckerXY(-1, barCount(gameState, player) + departed, player)
                : computeCheckerXY(start, Math.abs(gameState.board[start]) + departed)
            const landed = landedSlot(dest, arrived, player)
            const end = landed ? slotXY(landed) : isBearOff(dest) ? bearOffXY(player, 0) : computeCheckerXY(dest, 0)

            lastMoveArrows.push(
                <circle
                    key={`ghost-${i}`}
                    className="last-move-mark"
                    cx={ghost.x}
                    cy={ghost.y}
                    r={checkerRadius}
                    fill={highlightColour}
                    fillOpacity={0.15}
                    stroke={highlightColour}
                    strokeWidth={2}
                    strokeDasharray="5 4"
                />,
                <path
                    key={`arrow-${i}`}
                    className="last-move-mark"
                    d={arrowPath(ghost, end)}
                    fill="none"
                    stroke={highlightColour}
                    strokeOpacity={0.85}
                    strokeWidth={4}
                    strokeLinecap="round"
                    markerEnd="url(#last-move-arrowhead)"
                />
            )
            if (landed) {
                lastMoveRings.push(
                    <circle
                        key={`ring-${i}`}
                        className="last-move-mark"
                        cx={end.x}
                        cy={end.y}
                        r={checkerRadius + 3}
                        fill="none"
                        stroke={highlightColour}
                        strokeWidth={3}
                    />
                )
            }
        })
    }

    return (
        <>
            {gameOver && <Confetti width={width} height={height} />}

            <div className="turn-indicator">
                {
                    gameOver ? (<h1><strong>{message}</strong></h1>) : (<h1><strong>{turnLabel}</strong> to move</h1>)
                }
                <p className="last-move-text">
                    {lastMove && (
                        <>
                            <span className="last-move-swatch" style={{ background: highlightColour }} />
                            {lastMove.player === 1 ? "White" : "Black"}{" "}
                            {lastMove.paths.length > 0 ? <>played <strong>{moveNotation(lastMove)}</strong></> : "had no legal moves"}
                        </>
                    )}
                </p>

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
                    <defs>
                        <marker
                            id="last-move-arrowhead"
                            viewBox="0 0 10 10"
                            refX="7"
                            refY="5"
                            markerWidth="3"
                            markerHeight="3"
                            orient="auto"
                        >
                            <path d="M 0 0 L 10 5 L 0 10 z" fill={highlightColour} fillOpacity={0.85} />
                        </marker>
                    </defs>
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

                    {staticCheckers}
                    <g className="last-move-layer">
                        {lastMoveArrows}
                        {lastMoveRings}
                    </g>
                    {slidingCheckers}

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
                    <button onClick={rollDice} className={`btn btn-roll ${!rolled && !animating && playerSide && ((playerSide === "white" && gameState.turn === 1) || (playerSide === "black" && gameState.turn === -1)) ? " pulse" : ""}`} disabled={rolled || animating}>🎲 Roll Dice</button>
                    <button onClick={confirmMove} className="btn btn-confirm" disabled={animating || (rolled ? (gameState.validMoves.length > 0 && moveSequence.length == 0) : true)}>✅ Confirm</button>
                    <button onClick={resetBoard} className="btn btn-reset" disabled={animating}>🔄 Reset</button>
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
