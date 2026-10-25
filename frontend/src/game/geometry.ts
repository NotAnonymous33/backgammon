import { BLACK_OFF, WHITE_OFF } from "./types"
import type { Side } from "./types"

export const POINT_WIDTH = 75
export const POINT_HEIGHT = 200
export const CHECKER_RADIUS = 20
export const CHECKER_PADDING = 5
export const BEAR_OFF_STACK_STEP = CHECKER_RADIUS - 7
export const BAR_WIDTH = 50
export const BEAR_OFF_WIDTH = 50
export const BOARD_PADDING = 10

export const SVG_WIDTH = 2 * BOARD_PADDING + 12 * POINT_WIDTH + BAR_WIDTH + BEAR_OFF_WIDTH
export const SVG_HEIGHT = 2 * BOARD_PADDING + 2.2 * POINT_HEIGHT

const CHECKER_STEP = CHECKER_RADIUS + CHECKER_PADDING
const isTopRow = (index: number) => index < 12

export type XY = { x: number; y: number }
export type Rect = { x: number; y: number; width: number; height: number }

/** Left edge of a point. Points 0-11 run right to left on top, 12-23 left to right on the bottom. */
function pointLeft(index: number): number {
    const column = isTopRow(index) ? 11 - index : index - 12
    return BOARD_PADDING + column * POINT_WIDTH + (column >= 6 ? BAR_WIDTH : 0)
}

const pointCenterX = (index: number) => pointLeft(index) + POINT_WIDTH / 2

/** SVG `points` attribute for a point's triangle. */
export function pointTriangle(index: number): string {
    const x1 = pointLeft(index)
    const x2 = x1 + POINT_WIDTH
    const x3 = x1 + POINT_WIDTH / 2
    const [baseY, apexY] = isTopRow(index)
        ? [BOARD_PADDING, BOARD_PADDING + POINT_HEIGHT]
        : [SVG_HEIGHT - BOARD_PADDING, SVG_HEIGHT - BOARD_PADDING - POINT_HEIGHT]
    return `${x1},${baseY} ${x2},${baseY} ${x3},${apexY}`
}

/** Centre of the `stack`-th checker on a point (0 = nearest the board edge). */
export function pointCheckerXY(index: number, stack: number): XY {
    const offset = CHECKER_STEP * (stack + 0.5)
    return {
        x: pointCenterX(index),
        y: isTopRow(index) ? BOARD_PADDING + offset : SVG_HEIGHT - BOARD_PADDING - offset,
    }
}

export function barCheckerXY(side: Side, stack: number): XY {
    const offset = CHECKER_STEP * (stack + 0.5)
    return {
        x: 6 * POINT_WIDTH + BOARD_PADDING + BAR_WIDTH / 2,
        y: side === "white" ? BOARD_PADDING + offset : SVG_HEIGHT - BOARD_PADDING - offset,
    }
}

const BEAR_OFF_X = BOARD_PADDING + 12 * POINT_WIDTH + BAR_WIDTH + 5
const BEAR_OFF_HEIGHT = (SVG_HEIGHT - 2 * BOARD_PADDING) / 2

/** White bears off into the bottom tray, black into the top tray. */
export function bearOffRect(side: Side): Rect {
    return {
        x: BEAR_OFF_X,
        y: side === "white" ? BOARD_PADDING + BEAR_OFF_HEIGHT : BOARD_PADDING,
        width: BEAR_OFF_WIDTH,
        height: BEAR_OFF_HEIGHT,
    }
}

export function bearOffCheckerXY(side: Side, stack: number): XY {
    const offset = BEAR_OFF_STACK_STEP * (stack + 1.5)
    const { y, height } = bearOffRect(side)
    return {
        x: BEAR_OFF_X + BEAR_OFF_WIDTH / 2,
        y: side === "white" ? y + height - offset : y + offset,
    }
}

export const innerBoardRect = (half: 0 | 1): Rect => ({
    x: BOARD_PADDING + half * (6 * POINT_WIDTH + BAR_WIDTH),
    y: BOARD_PADDING,
    width: 6 * POINT_WIDTH,
    height: SVG_HEIGHT - 2 * BOARD_PADDING,
})

const inside = ({ x, y }: XY, r: Rect) =>
    x >= r.x && x <= r.x + r.width && y >= r.y && y <= r.y + r.height

/** Where a dragged checker of `side` was dropped: a bear-off marker or the nearest point. */
export function dropTarget(at: XY, side: Side): number | null {
    if (inside(at, bearOffRect(side))) return side === "white" ? WHITE_OFF : BLACK_OFF
    if (inside(at, bearOffRect(side === "white" ? "black" : "white"))) return null

    let best = 0
    let bestDist = Number.POSITIVE_INFINITY
    for (let i = 0; i < 24; i++) {
        const px = pointCenterX(i)
        const py = isTopRow(i) ? BOARD_PADDING + POINT_HEIGHT / 2 : SVG_HEIGHT - BOARD_PADDING - POINT_HEIGHT / 2
        const dist = (at.x - px) ** 2 + (at.y - py) ** 2
        if (dist < bestDist) {
            bestDist = dist
            best = i
        }
    }
    return best
}
