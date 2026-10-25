# Benchmarks

Reproducible measurements of **rules correctness**, **engine speed**, **MCTS
search throughput**, **neural-network training**, **playing strength**,
**decision quality** and **Elo ratings**, recorded per run so any change can be
compared with any earlier state of the code.

Run everything from `backend/` with the virtual environment active. The
native extensions are compiled for the venv's Python 3.11, and other
interpreters cannot load them.

```bash
source venv/bin/activate                 # Windows: .\venv\Scripts\Activate.ps1
python -m bench run --label baseline     # record a run (quick suite, ~1 min)
# ...make a change, rebuild native extensions if you touched them...
python -m bench run                      # records, then compares with the previous run
python -m bench compare baseline latest  # compare any two runs
python -m bench list                     # recorded runs
python -m bench show latest              # every metric in one run
```

The dashboard at http://localhost:5173/bench (with `python app.py` and the
frontend running) shows the same comparisons, plus each metric's trend across
all runs. Click a metric to chart it. When the head run has a `skill` section,
a decision-quality panel compares its agents side by side. It plots
winning chance lost per decision against mean decision time, with MCTS
iteration budgets joined into a ladder, so you can see what extra search buys.
A table breaks the loss down by game phase. The dashboard only reads results;
start runs from the command line.

## What is measured

| Section | Metrics | Nature |
|---|---|---|
| `rules` | legal turns that change with dice order; turns mixing sequence lengths; listed moves the board refuses; moves that create or lose checkers | exact counts over the fixed position suite: 0 is correct, any change is real |
| `speed` | move generation, board cloning, applying sequences, full random games, median and p95 decision time per agent | throughput/latency; judged against noise floors |
| `search` | MCTS iterations per second and per decision, for each MCTS configuration | throughput; judged against noise floors |
| `training` | self-play, TD(λ) update and full `train_epoch` throughput; time of the first (compiling) update; score of networks trained for a fixed number of games | throughput, plus seeded learning runs with 95% confidence intervals |
| `strength` | score of agent A against B, game length, move-limit truncations, illegal moves, crashes | seeded, colour-swapped game pairs with 95% confidence intervals |
| `skill` | winning chance each agent gives up per decision, overall and per game phase; best-play and blunder rates; for networks, value-estimate error | fixed labelled positions, so no dice luck; exact in the bearoff, rollout estimates elsewhere; paired across runs |
| `ratings` | Elo rating of every agent in a pool, on one scale anchored at `heuristic` = 1500 | adaptively scheduled game pairs; bootstrap 95% confidence intervals |

`rules.dice_order_mismatch` already detects the known move-generation bug
(improvements.md #1). The suite includes that reproduction as
`known-dice-order-16-19`. It currently also finds two other affected
positions, listed under `detail.examples` in the run file.

### MCTS iterations

Each MCTS configuration searches every non-forced contact position in the
suite from a fresh tree. MCTSAgent2 skips search for forced moves and race
positions. One iteration is one selection, expansion, rollout and
backpropagation; `mcts.sim_count` counts them. The headline metric is the
median over positions of iterations per second of measured search time,
which doesn't depend on the time budget. Iterations per decision at the
configured budget are reported alongside. Rollout dice are seeded per
position, so every run simulates the same games. The quick suite times the
default (pip count), heuristic and neural rollout evaluations at 0.1 s.

### Neural-network training

Everything drives the project's own `BackgammonNN`, `TDLambda` and
`BackgammonTrainer`, with the configuration `NNAgent.main()` uses (hidden
layer of 40, learning rate 0.1, λ 0.7, `canonical-34` features; see
`suites.py`). Changes to the training code therefore show up directly.

- **Throughput** has three parts. *Self-play* runs `play_game` in one process
  (positions/s, so game length doesn't matter). *TD update* runs
  `TDLambda.update` over recorded games (positions/s). *Epoch* runs a whole
  `train_epoch`, including its worker pool and the serial updates (games/s).
  The first update is reported separately, since it pays one-off start-up
  costs.
- **Learning** trains several fresh networks from fixed seeds. Each plays N
  self-play games and gets one TD update per finished game, as
  `train_epoch` does. Each network is then played, before and after
  training, against `random` and `heuristic` on the benchmark's seeded dice.
  The native board's `seed_dice()` makes training fully reproducible:
  identical code yields identical networks and identical games, so changes
  to targets, features or learning rates can be compared game by game.

Learning only shows after thousands of games, so the quick suite measures
throughput only. The standard budget is 8 runs × 3,000 games:

| Training code | vs `random` | vs `heuristic` (95% CI) |
|---|---|---|
| Original: white's-view features, ±1 targets on a sigmoid, terminal reward one step early | 57% → 68% | 9.8% → 24.3% (22.3–26.5%) |
| Current: `canonical-34`, targets in [0, 1], every state trained (see "Network value convention") | 42% → 99.8% | 3.0% → 67.5% (65.2–69.8%) |

The shipped main network (about 130,000 games of the original training)
scores about 69% against `heuristic`, so the current code reaches roughly
the same strength in 3,000 games.

Throughput runs in a fresh process, in an order that never runs batched
inference before `train_epoch` forks. Once a process has run batched
inference (as in `play_game`), the worker pool `train_epoch` forks next
deadlocks. The real training loop never runs inference in its parent, but
any new code that calls `train_epoch` should keep this in mind.

### Network value convention

New networks use the `canonical-34` feature schema (`classes/agents/features.py`):

- **Inputs from the side to roll.** The board is mirrored for black, so the
  side about to roll always sees its own checkers as positive, moving
  towards point 23. Being on roll is part of every input.
- **Output is P(side to roll wins).** To choose a move, the mover evaluates
  each reachable position with the opponent on roll and picks the lowest
  value. A move that ends the game is always taken.
- **TD targets lie in [0, 1].** Between turns, the target for V(s_t) is
  1 − V(s_{t+1}). The last state of a finished game is trained on the
  outcome, 1 for the side that bore off. A game cut off by the move limit
  gets no final target. The eligibility trace flips sign with the point of
  view, which makes this exactly TD(λ) on P(white wins).
  `tests/test_network.py` checks it against an independent white-frame
  implementation.

The original `white-34` networks remain loadable. Their features are always
from white's side and their output is P(white wins), with no input saying
whose turn it is. Checkpoints now store `feature_schema` (name, version,
inputs, perspective, value meaning, feature names) and `architecture`.
`FinalNNAgent`, the benchmark and neural MCTS read both, and files without a
schema are treated as `white-34`. Unknown schemas fail to load rather than
being misread.

Results on the first canonical networks (2026-10-25), each trained on 100,000
self-play games with `main()`'s settings and evaluated over 500 game pairs:

| Network | vs shipped `nn` (white-34, 128×128, ~130k games) | vs `heuristic` | Loss per decision | Value error, exact / rollout |
|---|---|---|---|---|
| shipped `nn` | — | 69.0% | 1.50% | 22.1% / 8.3% |
| `models/canonical/canonical_40_100k.pt` (34-40-1) | 52.5% (49.4–55.6%) | 69.4% | 1.54% | 8.9% / 6.3% |
| `models/canonical/canonical_128x128_100k.pt` | **60.4% (57.3–63.4%)** | 75.0% | 1.36% | 8.1% / 5.7% |

The on-roll bias is gone too. In exact bearoffs with white on roll, the
shipped network's P(white wins) is 31 points low; both canonical networks
are within 5 points. Loss per decision differs little between the networks.
The table's skill columns use the labels rebuilt on 2026-10-25 from the
128×128 network's own play (see "Measuring skill"), which may favour it
slightly; the previous labels, from the shipped network, gave 1.52%, 1.60%
and 1.41% in the same order.

## Measuring skill

The `strength` matches answer "does A beat B?", which is not enough to judge
the network or MCTS agents:

- Scores saturate. The network beats `random` in 100% of games, which says
  nothing about how much better it is, or whether a change helped.
- The `heuristic` baseline is itself flawed (improvements.md #4), so a score
  against it is hard to interpret.
- MCTS at the server's 5 s budget takes minutes per game, so enough games for
  a tight interval take hours. A time budget also makes the result depend on
  how busy the machine was.

Two sections address this. `skill` scores single decisions against known
answers, which is fast and precise and shows *where* an agent goes wrong.
`ratings` places agents on one Elo scale from games, which is the final word
on strength but costs more.

### Decision quality (`skill`)

`data/reference.json` holds 355 positions with dice: the 15 opening rolls,
120 contact, 60 bar, 60 race and 100 exact-bearoff positions, sampled from
the network's self-play with some random moves. For every distinct position
each one can lead to, it stores white's chance of winning from there:

- **Exact.** Once both sides are bearing off with at most 6 checkers, the
  chance comes from a two-sided bearoff database (`bearoff.py`) solved
  exactly by dynamic programming. It is independent of every agent and of
  the native move generator, and a test checks that its moves agree with the
  native board's on thousands of rolls.
- **Rollouts.** Everywhere else, the game is played out from each position
  with greedy play by the pinned network (the default `nn`; `valuenet.py`, a NumPy copy
  of `FinalNNAgent` that is tested to choose the same moves). All plays from
  one position share their dice (common random numbers), rollouts stop
  exactly once they reach the bearoff database, and plays that could still
  be best get 640 rollouts instead of 128.

An agent's **loss** on a decision is the winning chance of the best play minus
that of its play, from its own side's point of view. A perfect player scores
0, and a 5-point loss is counted as a blunder. Results are reported overall
and per phase (`opening`, `contact`, `bar`, `race`, `exact`).

Reading the numbers:

- **`exact` is ground truth.** The other phases are measured against what
  rollouts with the pinned network say, so an agent that plays better than
  the network can look slightly worse there. That makes them a strong
  reference for this project's agents, not an absolute one.
- **Rollout labels are noisy.** `skill.noise_floor` estimates how much loss
  noise alone produces, by choosing with half of each position's rollouts
  and scoring with the other half. It is a rough upper bound. Comparisons
  between runs are paired by position, so the floor cancels out of them.
- **Value accuracy.** For network agents, `value_error` compares the
  network's own estimate of every labelled position with the label. It
  checks the value function directly, not just the moves it leads to.
  `value_bias.white_on_roll` and `value_bias.black_on_roll` split the error
  in P(white wins) by who is on roll. A network that cannot see the turn is
  biased in opposite directions for the two: the original main network
  underrates white by about 31 points in exact bearoffs when white is on
  roll.
- **Repeats.** Random and MCTS agents play each position `--skill-repeats`
  times with different seeds. Their per-position loss is the mean.

The labels depend only on the positions, the pinned network and the rules,
so they are committed. They were last rebuilt on 2026-10-25 with the default
`nn` (`canonical_128x128_100k.pt`) as the pinned network, replacing the
`white-34` main network that pinned the first version. Rebuilding them (`python -m bench reference --build
--force`, about 30 minutes on 16 threads) changes every skill result and
makes comparisons with earlier runs unpaired. Do it deliberately, for
example after fixing the move generator. `python -m bench reference` shows
what the current file contains.

### MCTS: skill per iteration and per second

`mcts:iters=N` stops each search after exactly N iterations
(`max_iterations` in `CMCTS2.pyx`). Searches are then reproducible, since
rollout dice are seeded too, and independent of machine speed. MCTS skill
splits into two measurable parts:

- **Search quality:** `skill` at fixed budgets (`iters=100, 200, 400, ...`).
  The budgets double, so each step shows what twice the speed would be worth.
- **Speed:** `search.iterations_per_s`, measured as before.

A speed-up therefore converts directly into skill. If MCTS runs twice as fast,
its loss at the server's time budget becomes its loss at twice the
iterations, which the iteration ladder already shows, so no slow
time-budget matches are needed. `mcts:time=1.0` is scored too, as a
check on the real configuration.

Judge scaling against fixed opponents or the reference, never MCTS against
itself. On the first runs (2026-10-24), 1,600-iteration MCTS beat
100-iteration MCTS in 79% of games. Yet on identical dice against
`heuristic` it scored 7.0% against 6.2% (paired difference +0.8 points,
95% CI −5.6 to +7.1). Against `nn` it scored 1.6% against 0%, and its loss
per decision did not fall (+0.40 points, CI −0.05 to +0.85). Deeper search
only learned to exploit shallower search. With the current random-rollout
evaluation and chance handling (improvements.md #3), more speed does not
make MCTS stronger. Fix the search first, then the iteration ladder shows
what speed is worth.

### Elo ratings (`ratings`)

Every pair of agents in the pool plays seeded, colour-swapped game pairs.
Ratings come from a Bradley-Terry fit on the Elo scale, with the anchor
agent fixed at 1500 (`--anchor`, default `heuristic`). Then:

- **Adaptive scheduling.** While any rating is less precise than the target,
  more pairs go to the closest matchups of the least precise agents.
  Lopsided matchups carry almost no information about a rating gap, so they
  are not extended. Scheduling depends only on outcomes, so identical code
  replays identical games.
- **Bootstrap intervals.** The intervals resample whole game pairs, which
  keeps the colour-swap pairing that cancels luck.
- **Bounds.** An agent that won every game it played has only a lower bound,
  and is labelled as such. Add an opponent of similar strength to the pool to
  pin it down.

`python -m bench show` prints the score table behind the ratings. The
model assumes strength is transitive, which agents of one family can
violate (see the MCTS example above). If the table disagrees with the
ratings, trust the scores against fixed opponents.

## Suites

| Suite | Strength games per match (95% CI near 70%) | MCTS positions | Training learning check | Skill agents | Rating pool | Time on 16 threads |
|---|---|---|---|---|---|---|
| `quick` (default) | 200 (±6 points) | 20 | off | heuristic, nn, `mcts:iters=100` | random, first, heuristic, nn | ~1.5 min |
| `standard` | 2,000 (±2 points), plus nn vs `mcts:iters=400` | 60 | 8 runs × 3,000 games | + canonical and tdgammon networks, MCTS at 100–800 iterations, heuristic-rollout MCTS, `mcts:time=1.0` | + canonical, tdgammon, `mcts:iters=100` | ~15 min |
| `full` | 6,000 (±1.2 points), plus MCTS matches | 120 | 16 runs × 10,000 games | + momentum network, MCTS up to 1,600 iterations, neural-rollout MCTS | + momentum and earlier main checkpoints, `mcts:iters=400`, `mcts:iters=1600` | several hours (not timed) |

Comparisons between runs with the same seed are **paired** (see below). They
cancel shared dice luck, so they detect smaller changes than these
single-run intervals suggest.

Customise any suite:

```bash
python -m bench run --only strength --match "nn:models/main/checkpoint_epoch_main130.pt vs heuristic" --pairs 500
python -m bench run --only speed --latency-agent "mcts:time=0.1"
python -m bench run --only search --mcts-agent "mcts:time=0.5,mode=neural,depth=20"
python -m bench run --only training --training-games 2000   # learning check in any suite
python -m bench run --only skill --skill-agent nn:models/main/checkpoint_epoch_main130.pt --skill-agent nn
python -m bench run --only skill --skill-agent mcts:iters=400 --skill-agent mcts:iters=400,c=2.0 --skill-repeats 3
python -m bench run --only ratings --pool nn --pool mcts:iters=400 --pool heuristic --max-games 2000
```

### Agent specs

| Spec | Agent |
|---|---|
| `random`, `first`, `heuristic` | the simple agents |
| `nn` / `nn:path/to/checkpoint.pt` | FinalNNAgent (default `models/canonical/canonical_128x128_100k.pt`, also the server's network) |
| `mcts:time=0.5,mode=heuristic,depth=40` | MCTSAgent2 (keys: `time`, `iters`, `mode`, `depth`, `sim`, `c`, `nn`) |
| `mcts:iters=400` | MCTSAgent2 stopping after exactly 400 iterations per move (reproducible) |
| `py:package.module:Class,key=value` | any class with `select_move(board)` |

Each run records a SHA-256 digest of every checkpoint it used, so retraining
over the same file name remains visible. Checkpoints of any `BackgammonNN`
architecture load: the layer sizes are read from the file, and loading is
strict, so a mismatched file fails instead of playing with random weights.

## How results are kept honest

- **Seeded dice.** The native board seeds itself from `std::random_device`,
  so the benchmark never calls `roll_dice`. Dice come from a seeded Python
  stream applied with `set_dice`. Agents' own randomness is seeded
  separately, so it cannot shift the dice.
- **Colour-swapped pairs.** Each pair plays the same dice twice with seats
  swapped, which cancels most of the luck.
- **Paired comparisons across runs.** Pair seeds depend only on the base seed,
  so two runs with the default seed play identical dice. Per-game outcomes are
  stored, and the comparison tests per-game differences: identical code gives
  identical outcomes, and any difference comes from decisions. Keep the default
  `--seed` to retain this.
- **Fixed position suite.** `data/positions.json` is committed. Regenerating
  it (`python -m bench positions --generate --force`) makes rules and speed
  results incomparable with earlier runs, so only do it deliberately.
- **Stale-build guard.** A run refuses to start if `board_cpp.cpp` or
  `CMCTS2.pyx` is newer than its compiled extension. That would silently
  benchmark old code. Rebuild, or pass `--allow-stale`.
- **Recorded context.** Each run stores the commit, uncommitted changes, host,
  library versions, and CPU utilisation before each timed section. Runs from a dirty
  tree or a busy machine carry warnings.

### Speed noise

Speed differs between runs on a desktop even when nothing changes. Identical
code measured repeatedly on this project's development machine varied by:

- engine throughput: up to about 10%
- median decision time: about 7%
- p95 decision time: about 8%, taken over each position's fastest repeat, which discards scheduler hiccups

Changes smaller than the noise floors (15%, 15% and 20% respectively), or
than a run's own sample spread, are reported as unchanged. MCTS iteration
rates varied by about 3% and use a 15% floor. Training throughput varied
more: self-play by up to 17%, because its batched inference uses PyTorch's
multithreading, and whole epochs by up to 24%, because they include starting
a worker pool. Their floors are 20% (self-play), 15% (TD updates) and 35%
(epochs). Speed comparisons
between different machines are shown but never judged. For smaller effects,
close other heavy programs and use `--suite standard`.

## Exit codes and automation

`run` and `compare` accept `--fail-on-regression`, which exits with status 1
if any metric regressed. `compare --json` and `show --json` print
machine-readable output.

## Files

Runs are written to `bench/results/<timestamp>-<commit>[-dirty].json`.
The bearoff database is cached in `bench/cache/` (git-ignored, rebuilt in
about 10 seconds when missing). `bench/data/reference.json` holds the skill
labels and is committed.
This directory is git-ignored: results are machine-specific, and keeping them
outside git means checking out an older commit to benchmark it leaves earlier
runs in place. The commit must already contain `bench/`.

Unit tests: `python -m unittest tests.test_bench tests.test_skill`.
