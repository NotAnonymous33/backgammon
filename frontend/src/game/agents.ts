import type { AiAgent } from "./types"

export const AI_AGENTS: { value: AiAgent; label: string }[] = [
    { value: "random", label: "Random" },
    { value: "heuristic", label: "Heuristic" },
    { value: "mcts", label: "Monte Carlo Tree Search" },
    { value: "neural", label: "Neural Network" },
]
