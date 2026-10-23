# Benchmarks

Reproducible measurements of **rules correctness**, **engine speed**, **MCTS
search throughput**, **neural-network training** and **playing strength**,
recorded per run so any change can be compared with any earlier state of the
code.

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
all runs. Click a metric to chart it. It only reads results; start runs from
the command line.

## What is measured

| Section | Metrics | Nature |
|---|---|---|
| `rules` | legal turns that change with dice order; turns mixing sequence lengths; listed moves the board refuses; moves that create or lose checkers | exact counts over the fixed position suite: 0 is correct, any change is real |
| `speed` | move generation, board cloning, applying sequences, full random games, median and p95 decision time per agent | throughput/latency; judged against noise floors |
| `search` | MCTS iterations per second and per decision, for each MCTS configuration | throughput; judged against noise floors |
| `training` | self-play, TD(λ) update and full `train_epoch` throughput; time of the first (compiling) update; score of networks trained for a fixed number of games | throughput, plus seeded learning runs with 95% confidence intervals |
| `strength` | score of agent A against B, game length, move-limit truncations, illegal moves, crashes | seeded, colour-swapped game pairs with 95% confidence intervals |

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
layer of 40, learning rate 0.1, λ 0.7; see `suites.py`). Changes to the
training code therefore show up directly.

- **Throughput** has three parts. *Self-play* runs `play_game` in one process
  (positions/s, so game length doesn't matter). *TD update* runs
  `TDLambda.update` over recorded games (positions/s). *Epoch* runs a whole
  `train_epoch`, including its worker pool and the serial updates (games/s).
  The first update's `torch.compile` time is reported separately, since it
  depends on PyTorch's on-disk cache.
- **Learning** trains several fresh networks from fixed seeds. Each plays N
  self-play games and gets one TD update per finished game, as
  `train_epoch` does. Each network is then played, before and after
  training, against `random` and `heuristic` on the benchmark's seeded dice.
  The native board's `seed_dice()` makes training fully reproducible:
  identical code yields identical networks and identical games, so changes
  to targets, features or learning rates can be compared game by game.

Learning only shows after thousands of games. In the project's own logged
training (`eval_results/results_main.txt`, 500 games per epoch), the score
against the heuristic agent stayed around 16–23% for the first 15 epochs.
It only climbed past 50% after roughly 9,000 games. The quick suite
therefore measures throughput only. The standard budget of 8 runs × 3,000
games does show learning: on the current code, the score against
`heuristic` rose from 9.8% untrained to 24.3% (95% CI 22.3–26.5%), and
against `random` from 57% to 68%. That took about 6.5 minutes on 16 cores.

Throughput runs in a fresh process, in an order that never runs batched
inference before `train_epoch` forks. Once a process has run batched
inference (as in `play_game`), the worker pool `train_epoch` forks next
deadlocks. The real training loop never runs inference in its parent, but
any new code that calls `train_epoch` should keep this in mind.

## Suites

| Suite | Strength games per match (95% CI near 70%) | MCTS positions | Training learning check | Time on 16 cores |
|---|---|---|---|---|
| `quick` (default) | 200 (±6 points) | 20 | off | ~1 min |
| `standard` | 2,000 (±2 points) | 60 | 8 runs × 3,000 games | ~10 min |
| `full` | 6,000 (±1.2 points), plus an MCTS match | 120 | 16 runs × 10,000 games | ~30 min |

Comparisons between runs with the same seed are **paired** (see below). They
cancel shared dice luck, so they detect smaller changes than these
single-run intervals suggest.

Customise any suite:

```bash
python -m bench run --only strength --match "nn:models/main/checkpoint_epoch_main130.pt vs heuristic" --pairs 500
python -m bench run --only speed --latency-agent "mcts:time=0.1"
python -m bench run --only search --mcts-agent "mcts:time=0.5,mode=neural,depth=20"
python -m bench run --only training --training-games 2000   # learning check in any suite
```

### Agent specs

| Spec | Agent |
|---|---|
| `random`, `first`, `heuristic` | the simple agents |
| `nn` / `nn:path/to/checkpoint.pt` | FinalNNAgent (default `models/main/main_checkpoint_latest.pt`) |
| `mcts:time=0.5,mode=heuristic,depth=40` | MCTSAgent2 (keys: `time`, `mode`, `depth`, `sim`, `c`, `nn`) |
| `py:package.module:Class,key=value` | any class with `select_move(board)` |

Each run records a SHA-256 digest of every checkpoint it used, so retraining
over the same file name remains visible.

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
This directory is git-ignored: results are machine-specific, and keeping them
outside git means checking out an older commit to benchmark it leaves earlier
runs in place. The commit must already contain `bench/`.

Unit tests: `python -m unittest tests.test_bench`.
