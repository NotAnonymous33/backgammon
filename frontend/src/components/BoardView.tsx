import { useMemo, useRef, useState } from "react"
import type { PointerEvent } from "react"
import { isLegalStep, legalTargets, resolveClick } from "../game/draft"
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
    onPointerDown?: (e: PointerEvent<SVGCircleElement>) => void
}

function Checker({ x, y, side, selected = false, grabbable = false, onPointerDown }: CheckerProps) {
    return (
        <circle
            cx={x}
            cy={y}
            r={CHECKER_RADIUS}
            fill={side === "white" ? COLOURS.white : COLOURS.black}
            stroke={selected ? COLOURS.selected : "#444"}
            strokeWidth={selected ? 4 : 2}
            style={{ cursor: grabbable ? "grab" : "default", pointerEvents: grabbable ? "auto" : "none" }}
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
}

/**
 * Renders the board and turns pointer input into moves. Selection and drag state live here;
 * the parent remounts this component (via `key`) whenever the authoritative position changes.
 */
export default function BoardView({ preview, interactive, onMove }: BoardViewProps) {
    const svgRef = useRef<SVGSVGElement>(null)
    const [selected, setSelected] = useState<number | null>(null)
    const [drag, setDrag] = useState<Drag | null>(null)

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

    function click(index: number) {
        if (!interactive) return
        const outcome = resolveClick(preview, selected, index)
        setSelected(outcome.select)
        if (outcome.move) onMove(outcome.move)
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
        if (target !== null && isLegalStep(preview, drag.from, target)) onMove([drag.from, target])
    }

    function cancelDrag(e: PointerEvent<SVGSVGElement>) {
        if (drag && e.pointerId === drag.pointerId) setDrag(null)
    }

    const legalStroke = (target: number, base = "#000") => (targets.has(target) ? COLOURS.legal : base)
    const legalWidth = (target: number) => (targets.has(target) ? 3 : 1)
    const barCount = (side: Side) => (side === "white" ? preview.whiteBar : preview.blackBar)
    const isDragged = (from: number, stack: number, count: number) =>
        drag?.from === from && drag.side === mover && stack === count - 1

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

            {preview.board.map((value, index) => {
                if (value === 0) return null
                const side: Side = value > 0 ? "white" : "black"
                const count = Math.abs(value)
                const ownsPoint = interactive && side === mover
                return Array.from({ length: count }, (_, stack) => {
                    if (isDragged(index, stack, count)) return null
                    const center = pointCheckerXY(index, stack)
                    const top = stack === count - 1
                    return (
                        <Checker
                            key={`${index}-${stack}`}
                            {...center}
                            side={side}
                            selected={top && selected === index}
                            grabbable={ownsPoint && top}
                            onPointerDown={e => startDrag(e, index, center)}
                        />
                    )
                })
            })}

            {(["white", "black"] as const).map(side =>
                Array.from({ length: barCount(side) }, (_, stack) => {
                    if (isDragged(BAR, stack, barCount(side)) && side === mover) return null
                    const center = barCheckerXY(side, stack)
                    const top = stack === barCount(side) - 1
                    return (
                        <Checker
                            key={`bar-${side}-${stack}`}
                            {...center}
                            side={side}
                            selected={top && selected === BAR && side === mover}
                            grabbable={interactive && side === mover && top}
                            onPointerDown={e => startDrag(e, BAR, center)}
                        />
                    )
                }),
            )}

            {(["white", "black"] as const).map(side =>
                Array.from({ length: side === "white" ? preview.whiteOff : preview.blackOff }, (_, stack) => (
                    <Checker key={`off-${side}-${stack}`} {...bearOffCheckerXY(side, stack)} side={side} />
                )),
            )}

            {drag && <Checker {...drag.at} side={drag.side} />}
        </svg>
    )
}
