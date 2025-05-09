import multiprocessing as mp
import time
import random
import math

from board_cpp import Board # type: ignore
from .agents.RandomAgent import RandomAgent
from .agents.CMCTS2 import MCTSAgent2
from .agents.HeuristicAgent import HeuristicAgent
from .agents.NNAgent import FinalNNAgent


# Initialize log file and helper
log_file = open('tournament_log.txt', 'w')
def log(msg: str):
    print(msg, flush=True)
    log_file.write(msg + '\n')
    log_file.flush()

# nn = FinalNNAgent(checkpoint_path='models/main/backgammon_main_checkpoint_latest.pt')
# Map agent names to their classes and default parameters
AGENTS = {
    'Rand': (RandomAgent, {}),
    # 'M1s': (MCTSAgent2, {'time_budget': 1}),
    # 'M2s': (MCTSAgent2, {'time_budget': 2}),
    # 'M10s': (MCTSAgent2, {'time_budget': 10}),
    # 'MNN80': (MCTSAgent2, {'mode': 'neural', 'eval_depth':80, 'nn_agent': nn, 'time_budget': 1}),
    # 'MH80': (MCTSAgent2, {'mode': 'heuristic', 'eval_depth': 80, 'time_budget': 1}),
    # 'MNN40': (MCTSAgent2, {'mode': 'neural', 'eval_depth':40, 'nn_agent': nn, 'time_budget': 1}),
    # 'MH40': (MCTSAgent2, {'mode': 'heuristic', 'eval_depth': 40, 'time_budget': 1}),
    # 'MNN10': (MCTSAgent2, {'mode': 'neural', 'eval_depth':10, 'nn_agent': nn, 'time_budget': 1}),
    # 'MH10': (MCTSAgent2, {'mode': 'heuristic', 'eval_depth': 10, 'time_budget': 1}),
    'HDef': (HeuristicAgent, {}),
    'HOne': (HeuristicAgent, {'weights': [1]*10}),
    'NN': (FinalNNAgent, {'checkpoint_path': 'models/main/main_checkpoint_latest.pt'}),
    'NN25': (FinalNNAgent, {'checkpoint_path': 'models/main/checkpoint_epoch_main65.pt'}),
    'NN50': (FinalNNAgent, {'checkpoint_path': 'models/main/checkpoint_epoch_main130.pt'}),
    'NN75': (FinalNNAgent, {'checkpoint_path': 'models/main/checkpoint_epoch_main195.pt'}),
    'NN10': (FinalNNAgent, {'checkpoint_path': 'models/main/checkpoint_epoch_main25.pt'}),
}


def play_game(white_name: str, black_name: str):
    """
    Play a single game between two agents by name.
    Returns a tuple: (white_name, black_name, winner_name, move_count).
    """
    WhiteCls, white_kwargs = AGENTS[white_name]
    BlackCls, black_kwargs = AGENTS[black_name]
    white = WhiteCls(**white_kwargs)
    black = BlackCls(**black_kwargs)

    board = Board()
    move_count = 0
    while not board.game_over:
        agent = white if board.turn == 1 else black
        dice, invdice, moves = board.roll_dice()
        move = agent.select_move(board)
        board.move_from_sequence(move)
        move_count += 1

    winner = white_name if board.white_off == 15 else black_name
    return white_name, black_name, winner, move_count


if __name__ == '__main__':
    GAMES_PER_PAIR = 5000

    agent_names = list(AGENTS.keys())
    pairings = []
    for i in range(len(agent_names)):
        for j in range(i + 1, len(agent_names)):
            a, b = agent_names[i], agent_names[j]
            half = GAMES_PER_PAIR // 2
            pairings.extend([(a, b)] * half)
            pairings.extend([(b, a)] * half)
    random.shuffle(pairings)
    print(len(pairings), "games to play")

    # Tracking
    results = {name: {opp: 0 for opp in agent_names if opp != name} for name in agent_names}
    totals = {name: {opp: 0 for opp in agent_names if opp != name} for name in agent_names}
    win_counts = {name: 0 for name in agent_names}
    game_counts = {name: 0 for name in agent_names}
    lengths_by_agent = {name: [] for name in agent_names}
    game_lengths = []

    def on_result(res):
        white, black, winner, length = res
        # Update counts and records
        for name in (white, black):
            game_counts[name] += 1
            lengths_by_agent[name].append(length)
        win_counts[winner] += 1
        loser = black if winner == white else white
        results[winner][loser] += 1
        totals[white][black] += 1
        totals[black][white] += 1
        game_lengths.append(length)

        # Log
        game_num = sum(game_counts.values()) // 2
        w_rate = win_counts[white] / game_counts[white] * 100
        b_rate = win_counts[black] / game_counts[black] * 100
        if game_num % 10 != 0:
            return
        print(
            f"Game {game_num}: {white}(white) vs {black}(black) -> "
            f"Winner: {winner} in {length} moves. "
            f"Win rates: {white} {win_counts[white]}/{game_counts[white]} ({w_rate:.1f}%), "
            f"{black} {win_counts[black]}/{game_counts[black]} ({b_rate:.1f}%)",
            flush=True
        )

    start = time.perf_counter()
    print(mp.cpu_count())
    pool = mp.Pool(mp.cpu_count())
    for white, black in pairings:
        pool.apply_async(play_game, args=(white, black), callback=on_result)
    pool.close()
    pool.join()
    elapsed = time.perf_counter() - start

    log("\nTournament Results (wins/games):")
    header = "Agent\t" + "\t".join(agent_names)
    log(header)
    for a in agent_names:
        row = [a] + ["--" if a == b else f"{results[a][b]}/{totals[a][b]}" for b in agent_names]
        log("\t".join(row))

    overall_avg = sum(game_lengths) / len(game_lengths) if game_lengths else 0
    log(f"\nTotal time: {elapsed:.2f}s")
    log(f"Overall average game length: {overall_avg:.2f} moves")

    # Per-agent stats with 95% CI
    log("\nAverage game length per agent (95% CI):")
    for a in agent_names:
        lengths = lengths_by_agent[a]
        n = len(lengths)
        mean = sum(lengths) / n if n else 0
        sd = math.sqrt(sum((x - mean) ** 2 for x in lengths) / (n - 1)) if n > 1 else 0
        se = sd / math.sqrt(n) if n else 0
        margin = 1.96 * se
        lower, upper = mean - margin, mean + margin
        log(f"{a}: {mean:.2f} ± {margin:.2f} moves (95% CI: {lower:.2f}–{upper:.2f}), n={n}")

    # Total winrates per agent (descending)
    log("\nTotal winrates per agent (descending):")
    sorted_agents = sorted(
        agent_names,
        key=lambda a: (win_counts[a] / game_counts[a] * 100) if game_counts[a] else 0,
        reverse=True
    )
    for a in sorted_agents:
        rate = (win_counts[a] / game_counts[a] * 100) if game_counts[a] else 0
        log(f"{a}: {win_counts[a]}/{game_counts[a]} wins ({rate:.1f}%)")

    # Close log
    log_file.close()
