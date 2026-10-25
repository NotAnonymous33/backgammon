import { useMemo, useRef, useState } from "react"
import type { CSSProperties, PointerEvent, ReactNode } from "react"
import { applyMove, isLegalStep, legalTargets, resolveClick } from "../game/draft"
import {
    CHECKER_RADIUS,
    SVG_HEIGHT,
    SVG_WIDTH,
    barCheckerXY,
    bearOffCheckerXY,
    bearOffRect,
    dropTarget,
    innerBoardRect,
    pointCheckerXY,
    pointTriangle,
} from "../game/geometry"
import type { XY } from "../game/geometry"
import {
    DROP_STEP_MS,
    LOCAL_STEP_MS,
    barSlot,
    isHit,
    offSlot,
    pointSlot,
    stepLandings,
} from "../game/replay"
import type { Landing, Landings, Mark } from "../game/replay"
import { BAR, BLACK_OFF, WHITE_OFF, sideOfTurn } from "../game/types"
import type { Move, Preview, Side } from "../game/types"

const COLOURS = {
    outer: "#b39970",
    inner: "#f3d970",
    darkPoint: "#64230b",
    lightPoint: "#b96522",
    white: "#f3f3f3",
    black: "#000000",
    legal: "#00ff00",
    selected: "#e33",
    highlight: "#1e90ff",
}

/** Pointer travel (in board units) below which a press counts as a click, not a drag. */
const DRAG_THRESHOLD = 5

type Drag = {
    pointerId: number
    from: number
    side: Side
    /** Offset from the checker's centre to where it was grabbed. */
    grab: XY
    start: XY
    at: XY
}

type CheckerProps = XY & {
    side: Side
    selected?: boolean
    grabbable?: boolean
    /** Slide in from where the checker used to be. */
    landing?: Landing
    onPointerDown?: (e: PointerEvent<SVGCircleElement>) => void
}

function Checker({ x, y, side, selected = false, grabbable = false, landing, onPointerDown }: CheckerProps) {
    // The slide runs in CSS, starting from this offset back to the checker's own slot.
    const slide = landing && {
        "--dx": `${landing.fromX - x}px`,
        "--dy": `${landing.fromY - y}px`,
        "--delay": `${landing.delay}ms`,
        "--duration": `${landing.duration}ms`,
    }
    return (
        <circle
            cx={x}
            cy={y}
            r={CHECKER_RADIUS}
            fill={side === "white" ? COLOURS.white : COLOURS.black}
            stroke={selected ? COLOURS.selected : "#444"}
            strokeWidth={selected ? 4 : 2}
            className={landing ? "checker-landing" : undefined}
            style={{ cursor: grabbable ? "grab" : "default", pointerEvents: grabbable ? "auto" : "none", ...slide } as CSSProperties}
            onPointerDown={onPointerDown}
        />
    )
}

type BoardViewProps = {
    /** What the player should see: the server position with their draft applied. */
    preview: Preview
    /** The player may select and move checkers right now. */
    interactive: boolean
    onMove: (move: Move) => void
    /** Checkers to slide into place, e.g. while an opponent's move is played back. */
    landings?: Landings | null
    /** Marks showing the opponent's last move. */
    marks?: Mark[]
}

/**
 * Renders the board and turns pointer input into moves. Selection and drag state live here;
 * the parent remounts this component (via `key`) whenever the authoritative position changes.
 */
export default function BoardView({ preview, interactive, onMove, landings: replayLandings = null, marks = [] }: BoardViewProps) {
    const svgRef = useRef<SVGSVGElement>(null)
    const [selected, setSelected] = useState<number | null>(null)
    const [drag, setDrag] = useState<Drag | null>(null)
    const [localLandings, setLocalLandings] = useState<Landings | null>(null)
    const landingCount = useRef(0)
    const landings = replayLandings ?? localLandings

    const mover = sideOfTurn(preview.turn)
    const origin = drag?.from ?? selected
    const targets = useMemo(
        () => (interactive && origin !== null ? legalTargets(preview, origin) : new Set<number>()),
        [interactive, preview, origin],
    )

    function toBoardCoords(e: { clientX: number; clientY: number }): XY | null {
        const svg = svgRef.current
        const matrix = svg?.getScreenCTM()
        if (!svg || !matrix) return null
        const point = svg.createSVGPoint()
        point.x = e.clientX
        point.y = e.clientY
        const { x, y } = point.matrixTransform(matrix.inverse())
        return { x, y }
    }

    /** Make a move, sliding the checker from `from` (default: where it stood) into place. */
    function play(move: Move, duration: number, from?: XY) {
        landingCount.current += 1
        const id = `local${landingCount.current}`
        setLocalLandings(stepLandings(preview, applyMove(preview, move), move, isHit(preview, move), duration, id, from))
        onMove(move)
    }

    function click(index: number) {
        if (!interactive) return
        const outcome = resolveClick(preview, selected, index)
        setSelected(outcome.select)
        if (outcome.move) play(outcome.move, LOCAL_STEP_MS)
    }

    function startDrag(e: PointerEvent<SVGCircleElement>, from: number, center: XY) {
        if (!interactive || e.button !== 0) return
        const pointer = toBoardCoords(e)
        if (!pointer) return
        try {
            svgRef.current?.setPointerCapture(e.pointerId)
        } catch {
            // Capture is a nicety (keeps the drag alive outside the board); dragging still works without it.
        }
        setDrag({
            pointerId: e.pointerId,
            from,
            side: mover,
            grab: { x: pointer.x - center.x, y: pointer.y - center.y },
            start: center,
            at: center,
        })
    }

    function moveDrag(e: PointerEvent<SVGSVGElement>) {
        if (!drag || e.pointerId !== drag.pointerId) return
        const pointer = toBoardCoords(e)
        if (pointer) setDrag({ ...drag, at: { x: pointer.x - drag.grab.x, y: pointer.y - drag.grab.y } })
    }

    function endDrag(e: PointerEvent<SVGSVGElement>) {
        if (!drag || e.pointerId !== drag.pointerId) return
        setDrag(null)
        if (svgRef.current?.hasPointerCapture(e.pointerId)) svgRef.current.releasePointerCapture(e.pointerId)

        // Use the release position itself rather than the last rendered one, which may lag a frame.
        const pointer = toBoardCoords(e)
        const at = pointer ? { x: pointer.x - drag.grab.x, y: pointer.y - drag.grab.y } : drag.at
        if (Math.hypot(at.x - drag.start.x, at.y - drag.start.y) < DRAG_THRESHOLD) {
            click(drag.from)
            return
        }
        const target = dropTarget(at, drag.side)
        setSelected(null)
        if (target !== null && isLegalStep(preview, drag.from, target)) play([drag.from, target], DROP_STEP_MS, at)
    }

    function cancelDrag(e: PointerEvent<SVGSVGElement>) {
        if (drag && e.pointerId === drag.pointerId) setDrag(null)
    }

    const legalStroke = (target: number, base = "#000") => (targets.has(target) ? COLOURS.legal : base)
    const legalWidth = (target: number) => (targets.has(target) ? 3 : 1)
    const barCount = (side: Side) => (side === "white" ? preview.whiteBar : preview.blackBar)
    const isDragged = (from: number, stack: number, count: number) =>
        drag?.from === from && drag.side === mover && stack === count - 1

    // Sliding checkers are drawn last so they pass over the others.
    const still: ReactNode[] = []
    const sliding: ReactNode[] = []
    function place(slot: string, props: CheckerProps) {
        const landing = landings?.bySlot[slot]
        const checker = <Checker key={landing ? `${slot}-landing${landings?.id}` : slot} {...props} landing={landing} />
        ;(landing ? sliding : still).push(checker)
    }

    preview.board.forEach((value, index) => {
        if (value === 0) return
        const side: Side = value > 0 ? "white" : "black"
        const count = Math.abs(value)
        const ownsPoint = interactive && side === mover
        for (let stack = 0; stack < count; stack++) {
            if (isDragged(index, stack, count)) continue
            const center = pointCheckerXY(index, stack)
            const top = stack === count - 1
            place(pointSlot(index, stack), {
                ...center,
                side,
                selected: top && selected === index,
                grabbable: ownsPoint && top,
                onPointerDown: e => startDrag(e, index, center),
            })
        }
    })

    for (const side of ["white", "black"] as const) {
        const onBar = barCount(side)
        for (let stack = 0; stack < onBar; stack++) {
            if (isDragged(BAR, stack, onBar) && side === mover) continue
            const center = barCheckerXY(side, stack)
            const top = stack === onBar - 1
            place(barSlot(side, stack), {
                ...center,
                side,
                selected: top && selected === BAR && side === mover,
                grabbable: interactive && side === mover && top,
                onPointerDown: e => startDrag(e, BAR, center),
            })
        }
    }

    for (const side of ["white", "black"] as const) {
        const borneOff = side === "white" ? preview.whiteOff : preview.blackOff
        for (let stack = 0; stack < borneOff; stack++) {
            place(offSlot(side, stack), { ...bearOffCheckerXY(side, stack), side })
        }
    }

    return (
        <svg
            ref={svgRef}
            viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
            style={{ width: "100%", maxWidth: SVG_WIDTH, height: "auto", border: "1px solid #333", background: COLOURS.outer, touchAction: "none" }}
            onPointerMove={moveDrag}
            onPointerUp={endDrag}
            onPointerCancel={cancelDrag}
        >
            {[0, 1].map(half => {
                const r = innerBoardRect(half as 0 | 1)
                return <rect key={half} {...r} fill={COLOURS.inner} stroke="#000" strokeWidth={1} />
            })}

            {(["white", "black"] as const).map(side => {
                const target = side === "white" ? WHITE_OFF : BLACK_OFF
                return (
                    <rect
                        key={side}
                        {...bearOffRect(side)}
                        fill={COLOURS.inner}
                        stroke={legalStroke(target)}
                        strokeWidth={legalWidth(target)}
                        onClick={() => click(target)}
                    />
                )
            })}

            {Array.from({ length: 24 }, (_, index) => (
                <polygon
                    key={index}
                    points={pointTriangle(index)}
                    fill={index % 2 === 0 ? COLOURS.darkPoint : COLOURS.lightPoint}
                    stroke={legalStroke(index)}
                    strokeWidth={legalWidth(index)}
                    onClick={() => click(index)}
                />
            ))}

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
                    <path d="M 0 0 L 10 5 L 0 10 z" fill={COLOURS.highlight} fillOpacity={0.85} />
                </marker>
            </defs>

            {still}

            <g className="last-move-layer">
                {marks.map((mark, i) => (
                    <g key={`mark-${i}`}>
                        <circle
                            className="last-move-mark"
                            cx={mark.ghost.x}
                            cy={mark.ghost.y}
                            r={CHECKER_RADIUS}
                            fill={COLOURS.highlight}
                            fillOpacity={0.15}
                            stroke={COLOURS.highlight}
                            strokeWidth={2}
                            strokeDasharray="5 4"
                        />
                        <path
                            className="last-move-mark"
                            d={mark.arrow}
                            fill="none"
                            stroke={COLOURS.highlight}
                            strokeOpacity={0.85}
                            strokeWidth={4}
                            strokeLinecap="round"
                            markerEnd="url(#last-move-arrowhead)"
                        />
                        {mark.ring && (
                            <circle
                                className="last-move-mark"
                                cx={mark.ring.x}
                                cy={mark.ring.y}
                                r={CHECKER_RADIUS + 3}
                                fill="none"
                                stroke={COLOURS.highlight}
                                strokeWidth={3}
                            />
                        )}
                    </g>
                ))}
            </g>

            {sliding}

            {drag && <Checker {...drag.at} side={drag.side} />}
        </svg>
    )
}
