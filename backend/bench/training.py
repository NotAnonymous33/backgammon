"""Neural-network training: how fast it runs and how much a fixed budget learns.

Everything here drives the project's own training code in
classes/agents/NNAgent.py (BackgammonNN, TDLambda, BackgammonTrainer), so
changes to that code show up directly. The model configuration mirrors
``NNAgent.main()`` and lives in the suite definition; networks use the
``canonical-34`` feature schema that ``main()`` trains.

Throughput (timing metrics)
    self-play      BackgammonTrainer.play_game, positions/s in one process
    TD update      TDLambda.update over recorded games, positions/s
    epoch          BackgammonTrainer.train_epoch end to end (worker pool,
                   self-play and serial updates), games/s

Learning (rate metrics)
    Several independent runs, each from its own seed: initialise a network,
    train it on N self-play games exactly as train_epoch does (one TD update
    per finished game), then play the trained network against fixed
    opponents on the benchmark's seeded dice. With the native dice seeded,
    identical code reproduces identical networks and outcomes, so runs can
    be compared game by game.
"""
import multiprocessing as mp
import os
import random
import time

from . import progress
from .seeding import derive_seed
from .speed import timing_metric
from .stats import percentile, wilson
from .store import metric
from .strength import _init_worker, outcome_for, play_agents, rate_metric, score

SELFPLAY_FLOOR = 0.20
UPDATE_FLOOR = 0.15
EPOCH_FLOOR = 0.35  # includes pool start-up and inter-process transfer


def _quiet():
    os.environ["TQDM_DISABLE"] = "1"  # train_epoch draws a progress bar


def build(model_cfg, seed, epoch_games=32, epoch_workers=1):
    """Seeded network, TD(λ) learner and trainer configured like NNAgent.main()."""
    import torch
    from classes import board_cpp  # type: ignore
    from classes.agents.NNAgent import BackgammonNN, BackgammonTrainer, TDLambda
    from classes.agents.features import CANONICAL

    seed32 = seed & 0xFFFFFFFF
    torch.manual_seed(seed32)
    random.seed(seed)
    seeded = hasattr(board_cpp, "seed_dice")
    if seeded:
        board_cpp.seed_dice(seed32)
    model = BackgammonNN(CANONICAL.size, hidden_sizes=list(model_cfg["hidden"]))
    td = TDLambda(model, learning_rate=model_cfg["lr"], lambda_param=model_cfg["lambda"])
    trainer = BackgammonTrainer(model, CANONICAL, td, games_per_epoch=epoch_games,
                                eval_games=0, num_workers=epoch_workers)
    return model, td, trainer, seeded


def throughput(cfg, log=print):
    """Run the throughput benchmarks in a fresh process (see _throughput_task)."""
    from concurrent.futures import ProcessPoolExecutor

    # ProcessPoolExecutor workers are not daemonic, so train_epoch may start its own pool.
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(_throughput_task, cfg).result()


def _random_trajectories(n, seed):
    """Episodes from random play: shaped like training data, no network involved."""
    import numpy as np
    from classes.board_cpp import Board  # type: ignore
    from classes.agents.features import CANONICAL
    from classes.agents.NNAgent import Episode
    from .seeding import DiceStream

    rng = random.Random(seed)
    out = []
    for g in range(n):
        dice = DiceStream(derive_seed(seed, g))
        board = Board()
        states, players = [], []
        while not board.game_over and len(states) < 200:
            board.set_dice(dice.next())
            states.append(CANONICAL.features(board))
            players.append(board.turn)
            board.move_from_sequence(rng.choice(board.valid_moves) if board.valid_moves else [])
        winner = (1 if board.white_off == 15 else -1) if board.game_over else None
        out.append(Episode(np.asarray(states, dtype=np.float32), players, winner))
    return out


def _throughput_task(cfg):
    """Training throughput, ordered so the epoch's fork stays safe.

    Batched inference (as in play_game) starts PyTorch's intra-op thread pool,
    and a process forked after that deadlocks inside train_epoch's worker pool.
    The real training loop never runs inference in the parent, so this mirrors
    it: time epochs first, run self-play in this process last.
    """
    _quiet()
    log = print
    tc = cfg["training"]
    repeats = tc["speed_repeats"]
    min_time = tc["speed_min_time"]
    _, td, trainer, _ = build(tc["model"], derive_seed("training-speed"), tc["epoch_games"], tc["epoch_workers"])
    results = []

    # The first update pays one-off start-up costs; report it separately.
    warm = _random_trajectories(1, derive_seed("training-warmup"))[0]
    start = time.perf_counter()
    td.update(warm.states, warm.players, warm.winner, 0)
    first = time.perf_counter() - start
    results.append(metric("training.first_update_s", "First TD update (one-off start-up)", first, unit="s"))

    samples = []
    for _ in range(tc["epoch_repeats"]):
        start = time.perf_counter()
        trainer.train_epoch(0)
        samples.append(tc["epoch_games"] / (time.perf_counter() - start))
    m = timing_metric("training.epoch", f"Training epoch ({tc['epoch_games']} games, "
                      f"{tc['epoch_workers']} workers)", "games/s", samples,
                      better="higher", noise_floor=EPOCH_FLOOR)
    results.append(m)
    log(f"  train_epoch: {m['value']:.1f} games/s (first TD update {first:.1f}s)", flush=True)

    # No forking from here on.
    games = []

    def selfplay():
        episode = trainer.play_game()
        games.append(episode)
        return len(episode.players)

    samples = throughput_samples(selfplay, repeats, min_time)
    m = timing_metric("training.selfplay", "Self-play (play_game, one process)", "pos/s", samples,
                      better="higher", noise_floor=SELFPLAY_FLOOR)
    m["detail"] = {"games": len(games), "mean_positions_per_game":
                   round(sum(len(e.players) for e in games) / len(games), 1)}
    results.append(m)
    log(f"  self-play: {m['value']:,.0f} positions/s", flush=True)

    recorded = games[:20]
    cursor = [0]

    def update():
        episode = recorded[cursor[0] % len(recorded)]
        cursor[0] += 1
        td.update(episode.states, episode.players, episode.winner, 0)
        return len(episode.players)

    samples = throughput_samples(update, repeats, min_time)
    m = timing_metric("training.td_update", "TD(λ) update (TDLambda.update)", "pos/s", samples,
                      better="higher", noise_floor=UPDATE_FLOOR)
    results.append(m)
    log(f"  TD update: {m['value']:,.0f} positions/s", flush=True)
    return results


def throughput_samples(fn, repeats, min_time, warmup=True):
    """Like speed.throughput, but fn returns work units (positions) per call."""
    if warmup:
        fn()
    samples = []
    for _ in range(repeats):
        units = 0
        start = time.perf_counter()
        while True:
            units += fn()
            elapsed = time.perf_counter() - start
            if elapsed >= min_time:
                break
        samples.append(units / elapsed)
    return samples


def _evaluate(agent, opponents, pairs, base_seed, move_limit):
    from . import agents as agent_specs

    out = {}
    for spec in opponents:
        opponent = agent_specs.make_agent(spec, {})
        ordered = []
        for pair in range(pairs):
            pair_seed = derive_seed(base_seed, "pair", pair)
            for a_white in (True, False):
                random.seed(derive_seed(pair_seed, "agents", "trained", spec))
                white, black = (agent, opponent) if a_white else (opponent, agent)
                winner, _, _ = play_agents(white, black, pair_seed, move_limit)
                ordered.append(outcome_for(winner, a_white))
        out[spec] = ordered
    return out


def _learn_task(args):
    index, tc, base_seed, move_limit = args
    _quiet()
    import torch

    torch.set_num_threads(1)
    from classes.agents.NNAgent import NNAgent

    model, td, trainer, seeded = build(tc["model"], derive_seed(base_seed, "train", index))
    agent = NNAgent(model, trainer.schema, exploration_rate=0.0)
    before = _evaluate(agent, tc["learn_opponents"], tc["learn_pairs"], base_seed, move_limit)

    # Re-seed so training does not depend on how much randomness evaluation used.
    build_seed = derive_seed(base_seed, "train", index, "games")
    random.seed(build_seed)
    from classes import board_cpp  # type: ignore
    if seeded:
        board_cpp.seed_dice(build_seed & 0xFFFFFFFF)

    start = time.perf_counter()
    white_wins = 0
    positions = 0
    for _ in range(tc["learn_games"]):
        episode = trainer.play_game()
        # What train_epoch does for each finished game.
        td.update(episode.states, episode.players, episode.winner, 0)
        white_wins += episode.winner == 1
        positions += len(episode.players)
    train_s = time.perf_counter() - start

    after = _evaluate(agent, tc["learn_opponents"], tc["learn_pairs"], base_seed, move_limit)
    return index, before, after, train_s, white_wins / tc["learn_games"], positions, seeded


def learning(cfg, log=print):
    tc = cfg["training"]
    seeds = tc["learn_seeds"]
    if tc["learn_games"] <= 0 or seeds <= 0:
        log("  learning check skipped in this suite (enable with --training-games N)")
        return []
    tasks = [(i, tc, cfg["seed"], cfg["move_limit"]) for i in range(seeds)]
    workers = max(1, min(cfg["workers"], seeds))
    log(f"  learning: {seeds} run(s) × {tc['learn_games']} self-play games, "
        f"evaluated over {tc['learn_pairs']} pairs per opponent, on {workers} worker(s)")

    steps = 1 + seeds
    results = []
    if workers == 1:
        for t in tasks:
            results.append(_learn_task(t))
            progress.update(1 + len(results), steps, "steps")
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker) as pool:
            for r in pool.imap_unordered(_learn_task, tasks):
                results.append(r)
                progress.update(1 + len(results), steps, "steps")
    results.sort(key=lambda r: r[0])

    seeded = all(r[6] for r in results)
    model = tc["model"]
    pairing = (f"{cfg['seed']}:{seeds}x{tc['learn_games']}:{tc['learn_pairs']}:"
               f"{model['hidden']}:{model['lr']}:{model['lambda']}") if seeded else None
    budget = f"{tc['learn_games']} games"
    metrics = []
    for spec in tc["learn_opponents"]:
        after = [o for r in results for o in r[2][spec]]
        before = [o for r in results for o in r[1][spec]]
        per_run = [sum(score(r[2][spec])) / len(score(r[2][spec])) for r in results]
        base = f"training.learn[{budget} vs {spec}]"
        m = rate_metric(f"{base}.score", f"After {budget}: score vs {spec}",
                        after, pairing,
                        detail={"per_run": per_run, "runs": seeds, "seeded": seeded})
        metrics.append(m)
        before_scores = score(before)
        untrained = sum(before_scores) / len(before_scores)
        metrics.append(metric(f"{base}.untrained", f"Untrained network: score vs {spec}",
                              untrained, unit="fraction", ci=list(wilson(sum(before_scores), len(before_scores)))))
        log(f"  vs {spec}: {untrained:.1%} untrained → {m['value']:.1%} after {budget} "
            f"(95% CI {m['ci'][0]:.1%}–{m['ci'][1]:.1%})")

    train_s = [r[3] for r in results]
    metrics.append(metric(f"training.learn[{budget}].seconds", f"Wall time to train {budget} (median run)",
                          percentile(train_s, 50), unit="s"))
    metrics.append(metric(f"training.learn[{budget}].selfplay_white", "Self-play games won by white during training",
                          sum(r[4] for r in results) / len(results), unit="fraction"))
    if not seeded:
        log("  warning: native dice cannot be seeded (rebuild board_cpp); learning results are not reproducible")
    return metrics


def run(cfg, log=print):
    tc = cfg["training"]
    steps = 1 + (tc["learn_seeds"] if tc["learn_games"] > 0 else 0)
    progress.update(0, steps, "steps")
    metrics = throughput(cfg, log)
    progress.update(1, steps, "steps")
    return metrics + learning(cfg, log)
