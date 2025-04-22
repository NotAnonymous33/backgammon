import "./Dice.css";

type DiceProps = {
    value: number;
    invalid?: boolean;
    rolling?: boolean;
    used?: boolean;
}

const pipMap: Record<number, [number, number][]> = {
    1: [[2, 2]],
    2: [[1, 1], [3, 3]],
    3: [[1, 1], [2, 2], [3, 3]],
    4: [[1, 1], [1, 3], [3, 1], [3, 3]],
    5: [[1, 1], [1, 3], [2, 2], [3, 1], [3, 3]],
    6: [[1, 1], [1, 3], [2, 1], [2, 3], [3, 1], [3, 3]],
}

export default function Dice({ value, invalid = false, rolling = false, used = false }: DiceProps) {
    const positions = pipMap[value] || [];
    return (
        <div className={`die${invalid ? " invalid" : ""}${rolling ? " rolling" : ""}${used ? " used" : ""}`}>
            <div className="pip-grid">
                {positions.map(([r, c], idx) => (
                    <div
                        key={idx}
                        className="pip"
                        style={{ gridRow: r, gridColumn: c }}
                    />
                ))}
            </div>
        </div>
    )
}