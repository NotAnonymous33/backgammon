# Performance improvements

The web framework is not a bottleneck. Almost all CPU time goes into **move generation, the Python ↔ C++ boundary, and MCTS rollouts**. The largest cheap win is to stop re-reading native vectors inside Python loops: that alone makes the heuristic evaluator about **6× faster** (re-measured; see #1).

A full-system profile on 2026-10-24 ([Measured profile](#measured-profile-2026-10-24)) confirmed this picture and added items 13–20. The biggest new findings are:
- **Training workers:** `train_epoch` with 4 workers is slower than a single process, because each game's states come back as ~80 tensors through shared memory. Returning one array per game makes it **7.8× faster** (#13).
- **MCTS rollouts:** a prototype C++ rollout that picks one die at a time is **43× faster** than the current Python-driven one (#3).
- **MCTS garbage collection:** Python's cyclic garbage collector takes **12–17%** of MCTS time today, and half of it once rollouts are fast (#14).

This list is ranked by expected gain for the effort required. It overlaps with some items in [improvements.md](improvements.md), which are linked rather than repeated. Items marked **measured** were timed locally (Python 3.11, the repo's `venv`, current native builds). The rest are estimates and should be verified with the benchmark suite:

```sh
cd backend
python -m bench run --suite quick --label baseline     # before
python -m bench run --suite quick --label <change>     # after; compares to the previous run
```

Fix correctness in the move generator first ([improvements.md #1](improvements.md)). Several items below rewrite the same code, and speeding up a generator that is wrong only makes it wrong faster.

## Measured profile (2026-10-24)

**How it was measured:**
- Tools: the benchmark suite (`python -m bench run`, quick suite), py-spy sampling with native frames, cProfile, and targeted micro-benchmarks over the committed position suite (`bench/data/positions.json`). Each micro-benchmark reports the best of several repeats.
- Prototypes: these were built in scratch copies under `/tmp` and never touched the repo. They include a C++ rollout, a hand-written TD(λ) update, and NumPy inference.
- Machine load: a browser kept the 16-core machine 15–28% busy. Absolute numbers are therefore 15–25% below the 2026-10-23 baseline run, but ratios are unaffected.

**Benchmark suite (quick), current code:**

| Area | Metric | Value |
|---|---|---|
| Engine | move generation (clone + `set_dice`) | 184k positions/s |
| Engine | random-vs-random full games | 853 games/s |
| Agents | median decision time: heuristic / NN | 0.85 ms / 0.97 ms |
| MCTS | iterations/s: default / heuristic / neural rollouts | 1,367 / 1,490 / 1,265 |
| Training | self-play, TD(λ) update | 963 positions/s, 1,941 positions/s |
| Training | full `train_epoch` | 3.5 games/s |

**Where the time goes:**

| Path | Breakdown | Items |
|---|---|---|
| MCTS iteration (~786 µs) | Rollout ≈ 665 µs and tree work ≈ 121 µs. With rollouts disabled, the tree alone runs 8,200 iterations/s, and 47% of that time is the cyclic garbage collector. In a roll, 60% of the time is `roll_dice`, and half of that converts move lists into Python objects. | #3, #14, #15 |
| `set_dice`/`roll_dice` (9.6 µs) | `dfs` 6.2 µs, `get_invalid_dice` 2.6 µs, converting the returned `valid_moves` 1.9 µs | #4, #5, #15 |
| `FinalNNAgent.select_move` (730 µs, ~10 sequences) | features 366 µs; batched forward pass 113 µs; one tensor per row + `stack` 54 µs; `deepcopy` 37 µs | #1, #17 |
| Self-play (`play_game`) | `extract_features` ≈ 63% of the time (cProfile) | #1 |
| `train_epoch` | Returning each game as a list of tensors costs 51 ms per game in the parent. 4 workers manage 4.2 games/s against 7.5 games/s for one process. | #13 |
| TD(λ) update | Anomaly detection makes it 2.75× slower, and `torch.compile` 20% slower than eager | #11, #16 |
| Socket handler (4.9 ms) | SQLite commit 3.75 ms (fsync), query 0.36 ms; board rebuild and `convert` < 0.01 ms | #12, #19 |
| Idle dev server | 3.2% of a core: Werkzeug's reloader checks 2,301 files every second | #20 |
| Frontend | 85 KB gzip in total: React 46 KB, router 14, socket.io-client 13, bench dashboard 6.3, confetti 3.2 | #10 |

**Suggested order after this audit**, combining old and new items by gain for effort:
1. #13 training transfer (tiny change, 7.8× epochs).
2. #11 training flags (anomaly detection, `torch.compile`; 1.9×).
3. #1 cached vectors (~6×).
4. #14 garbage collection during search.
5. #8 sleeps.
6. #20 reloader.
7. #19 SQLite.
8. #17 NumPy inference.
9. #3 + #15 native rollout and leaner tree (the big MCTS win).
10. #16 manual TD gradients.
11. Then #4/#5/#18 move generation, and the rest.

## Summary

| # | Improvement | Where | Gain | Effort |
|---|---|---|---|---|
| 1 | Cache `board.positions` / `valid_moves` in local variables | Heuristic, NN features | **5.8×** heuristic eval, **6×** features (re-measured) | Tiny |
| 2 | Run AI in worker processes | `app.py` | No more cross-game interference; MCTS strength stays stable under load | Medium |
| 3 | Move MCTS rollouts into C++ | `CMCTS2.pyx`, `board_cpp.cpp` | Rollout 2× (loop only) to **43×** (pick one die at a time); MCTS ~5–12× (prototype measured) | Medium |
| 4 | Stop searching the move tree twice on every roll | `board_cpp.cpp` | ≤1.3× `roll_dice` (measured: the second search is 27% of a roll) | Small–medium |
| 5 | Allocation-free move generation | `board_cpp.cpp` | ~2–4× movegen (estimate) | Medium |
| 6 | Make MCTS tree reuse actually work | `app.py`, `CMCTS2.pyx` | More effective sims per decision | Small–medium |
| 7 | Deduplicate resulting positions | all agents | ~10% fewer evaluations (measured) | Small |
| 8 | Remove artificial sleeps | `app.py` | −2 s per AI turn | Tiny |
| 9 | Lazy and lighter imports | `NNAgent.py` | Server startup 3.2 s → ~1.5 s (estimate) | Small |
| 10 | Frontend render hygiene | `Board.tsx` | Smoother dragging; less console work | Small |
| 11 | Training pipeline fixes | `NNAgent.py` | TD updates **1.9×** (no anomaly detection, eager instead of `torch.compile`; measured) | Small–medium |
| 12 | Smaller items | various | Minor | Tiny |
| 13 | Return one array per game from training workers | `NNAgent.py` | **`train_epoch` 4.2 → 32.7 games/s** (7.8×, measured) | Tiny |
| 14 | Pause the cyclic GC during MCTS search | `CMCTS2.pyx` | 12–17% more iterations now; 2–2.4× tree throughput (measured) | Tiny |
| 15 | Cut MCTS tree overhead (121 µs/iteration) | `CMCTS2.pyx`, `board_cpp.cpp` | Needed for #3 to pay off; ~2× tree throughput (estimate) | Small–medium |
| 16 | Hand-written TD(λ) gradients | `NNAgent.py` | **4×** over eager, **7.5×** over the shipped config; identical weights (measured) | Small |
| 17 | NumPy inference for the small MLP | `NNAgent.py`, `CMCTS2.pyx` | Batch-1 forward 45 → 9 µs; NN decision ~730 → ~150 µs with #1 (measured parts) | Small |
| 18 | Don't copy `valid_moves` when cloning | `board_cpp.cpp` | Clone 5.7 → 0.5 µs on 150-sequence rolls; per-decision cost stops being quadratic (measured) | Small |
| 19 | SQLite WAL + `synchronous=NORMAL` | `app.py` | Commit 3.75 → 0.79 ms; socket handler ~4.9 → ~2 ms (measured) | Tiny |
| 20 | Turn off the polling reloader | `app.py` | Idle dev server 3.2% CPU → ~0; one process instead of two (measured) | Tiny |

---

## 1. Cache native vectors before looping over them

**Measured · effort: tiny · highest return in the repo**

`positions`, `dice` and `valid_moves` are exposed through `def_readwrite` ([board_cpp.cpp:1285](backend/classes/board_cpp.cpp#L1285)). Every attribute access converts the whole `std::vector` into a **new Python list**: `b.positions is b.positions` is `False`. Code like `board.positions[i]` inside a `for i in range(24)` loop therefore rebuilds a 24-element list on every iteration, often several times per iteration.

| Function | Now | With `pos = board.positions` once | Speedup |
|---|---|---|---|
| `HeuristicAgent.evaluate_board` | 145 µs | 6.5 µs | **22×** |
| `extract_features` (NN) | 28 µs | 4.6 µs | **6×** |

**Re-measured 2026-10-24:**
- **Method:** I generated a local-variable copy of `evaluate_board`/`longest_prime` from the source and asserted that it returns identical scores on all 336 positions in the suite.
- **`evaluate_board`:** 47–49 µs → 8–9 µs on every category (contact, opening, all), so **5.4–5.8×**.
- **`extract_features`:** 36.3 → 5.9 µs (**6.1×**).
- **The 22× figure didn't reproduce.** The baseline is about 48 µs, not 145. Each `.positions` read costs 0.62 µs, and the function reads it about 70 times. Plan on ~6×.

It matters more than the per-call numbers suggest. `extract_features` is about 63% of training self-play time, and 366 of the 730 µs in a `FinalNNAgent` decision.

Where it happens:
- [HeuristicAgent.py:42](backend/classes/agents/HeuristicAgent.py#L42): the main loop reads `board.positions[i]` up to six times per point. `longest_prime` at [HeuristicAgent.py:105](backend/classes/agents/HeuristicAgent.py#L105) and the anchor loops do the same.
- [NNAgent.py:31](backend/classes/agents/NNAgent.py#L31): `extract_features` and `longest_prime` at [NNAgent.py:730](backend/classes/agents/NNAgent.py#L730).
- `FinalNNAgent.select_move` and `NNAgent.select_move` read `board.valid_moves` three times per call, converting the entire move list each time.

**Fix:** read `pos = board.positions` (and `vm = board.valid_moves`) once at the top of each function and pass the list to helpers. The heuristic is used inside MCTS (`mode="heuristic"`), in tournaments, and as the benchmark baseline, so this speeds up all of them.

**Longer term:** use `def_property_readonly` returning a NumPy view or tuple, or add native `board.features()` / `board.afterstate_features()` that return one contiguous array for all candidates ([improvements.md #10](improvements.md)).

## 2. Run AI moves in worker processes

**Effort: medium · biggest win for concurrent games**

Flask-SocketIO runs in threading mode here (no eventlet/gevent; `simple-websocket` is installed). MCTS and agent code is Python/Cython that holds the GIL, so concurrent AI searches share **one core**. MCTS uses a **time** budget (`time_budget=5`, [app.py:242](backend/app.py#L242)), so N simultaneous MCTS games don't get slower. Instead, each search gets roughly 1/N of the simulations and **the AI silently gets weaker under load**. Human move and roll handling in every other room slows down as well.

**Fix:** a `concurrent.futures.ProcessPoolExecutor` (or a few long-lived workers), each with the NN model and agents created once. Send the board as `board.convert()` and return the chosen sequence. Discard stale results using a game revision ([improvements.md #6, #8](improvements.md)).

Correction to an earlier assumption: reloading the NN checkpoint per turn ([app.py:244](backend/app.py#L244)) costs only **~2.5 ms** warm (measured). Caching it is still worthwhile in a worker, but it is not the big cost.

## 3. Run MCTS rollouts in C++

**Partly measured · effort: medium**

MCTS currently reaches **~1,660 simulations/s** in default mode and **~1,280/s** in heuristic mode, single core, from the opening (measured). A full random game takes about 1 ms, roughly 16 µs per move, and each move crosses the Python/C++ boundary several times ([CMCTS2.pyx:245](backend/classes/agents/CMCTS2.pyx#L245)):

- `state.roll_dice()` builds and returns a Python tuple that includes the **converted `valid_moves`**, which the rollout throws away.
- `state.valid_moves` converts the full list again.
- `random.choice` is Python-level.
- `move_from_sequence` linearly re-validates the sequence against `valid_moves`.

Native move generation is about 10 µs of each 16 µs step (`roll_dice` measured at 10.1 µs). The rest is boundary overhead.

**Fix:** add `double Board::rollout(int max_depth, int perspective)` in C++. It should loop roll → pick a random legal sequence → apply it with no validation, and return the pip-count or D²/S value. Use the existing `dice_rng()` so seeded benchmarks stay reproducible. Combined with #4 and #5, expect several times more simulations per second. A random playout doesn't need the full sequence list: picking a random legal die-by-die move is cheaper, as long as the max-dice and higher-die rules are respected.

**Prototype measured (2026-10-24).** I added two methods to a scratch copy of `board_cpp.cpp`, built with the same `-O3` flags, and timed them against the current Python-driven loop. All three run from the opening, with depth ≤ 50 and the same mean playout length (~48 moves):

| Rollout | Time | Speedup |
|---|---|---|
| Python loop, as in `CMCTS2.simulate` | 693–697 µs | 1× |
| C++ loop, same policy (enumerate every sequence, pick one) | 345–355 µs | **2.0×** |
| C++ loop, one die at a time (random legal single move per die) | 16 µs | **43×** |

- **Moving the loop alone gains only 2×.** Enumerating every sequence each ply (~7 µs) then dominates, so the die-by-die picker is what makes this item worthwhile.
- **The 43× is an upper bound.** The prototype ignores the "use both dice / use the higher die" rule. Enforcing it adds a check only when one die blocks the other, so expect most of the gain to survive.
- **End-to-end MCTS:**
  - Today an iteration is ~786 µs: ~665 µs rollout plus ~121 µs tree work, measured by running with rollouts disabled.
  - With a 16 µs rollout, the tree becomes the limit: ~137 µs per iteration, so **~7,300 iterations/s (5.7×)**.
  - With #14 and #15 the tree work drops below ~50 µs, giving **~12×**.

Smaller items in the same file:
- `node.untried_moves.remove(move)` is an O(n) list-equality scan ([CMCTS2.pyx:189](backend/classes/agents/CMCTS2.pyx#L189)). Pick a random index and swap-pop instead.
- `best_child` → `value()` recomputes `math.log(parent.N)` for every child. Compute it once per parent and use `libc.math` instead of Python `math`.
- `lookup_win_probability` scans 50 Python tuples linearly. Use `bisect`, or precompute a C array.
- `mode="neural"` runs a **batch-1** torch forward per leaf ([CMCTS2.pyx:278](backend/classes/agents/CMCTS2.pyx#L278)), about 30 µs plus tensor creation. Batch leaves, or evaluate the small MLP with NumPy. Re-measured: 45.6 µs (torch) versus 9.2 µs (NumPy), plus 6.3 µs tensor creation. See #17.

Fixing the chance-node design ([improvements.md #3](improvements.md)) changes the tree shape. Do it before or together with this item.

## 4. Stop searching the move tree twice on every roll

**Effort: small–medium**

Every `roll_dice()`, `set_dice()` and DB-backed constructor runs two independent recursive searches over the same move tree:

1. `get_invalid_dice()` → `verify_permutation(...)` ([board_cpp.cpp:520](backend/classes/board_cpp.cpp#L520)), to find which dice are unplayable.
2. `set_valid_moves()` → `dfs(...)` ([board_cpp.cpp:677](backend/classes/board_cpp.cpp#L677)), to enumerate the sequences.

**Fix:** do one enumeration over both dice orders, track the maximum number of dice used and the highest die, and filter sequences to those meeting the rules. Derive `invalid_dice` from that result. This also fixes the dice-order bug in [improvements.md #1](improvements.md), so one change addresses correctness and speed.

**Measured 2026-10-24.** These are means over 1,005 rolls of suite positions; each roll has 9.1 legal sequences on average, a median of 4 and a maximum of 159.

| Part of `set_dice` | Time |
|---|---|
| Whole call (minus the clone) | 9.6 µs |
| `dfs` (enumerate sequences) | 6.2 µs |
| `get_invalid_dice` | 2.6 µs |
| Converting the returned `valid_moves` to Python | 1.9 µs |

`verify_permutation` stops at the first sequence that uses every die, so the duplicate search costs only 27% of a roll. Removing it is worth **at most ~1.3×**, not 1.5–2×. The correctness fix remains the main reason to do it. Converting the returned tuple costs another 20%, wasted whenever the caller ignores the return value (#15).

Move generation also runs for every socket event, because `Board(board_db=...)` is rebuilt from the DB in each handler ([board_cpp.cpp:84](backend/classes/board_cpp.cpp#L84)). That is cheap per event (~10 µs) but is wasted work. See #12.

## 5. Allocation-free move generation

**Effort: medium**

Move generation allocates heavily:

- `positions`, `dice` and `invalid_dice` are `std::vector`s ([board_cpp.cpp:38](backend/classes/board_cpp.cpp#L38)), so every `Board board_copy = board` is several heap allocations. Copies also include the whole `valid_moves` vector-of-vectors, which matters for `clone()` in Python agents ([improvements.md #9](improvements.md)). The cost is measured in #18.
- `dfs` takes `prev_moves` **by value** and returns a fresh vector of vectors at every level, then concatenates them ([board_cpp.cpp:634](backend/classes/board_cpp.cpp#L634)).
- `get_single_moves` returns a `std::set` (one tree node allocation per move) ([board_cpp.cpp:549](backend/classes/board_cpp.cpp#L549)).
- `verify_permutation` copies `used_dice` on entry and builds `new_remaining`/`new_used` vectors per branch.
- `is_valid` calls `can_bearoff()` (an 18-point scan) for each candidate bear-off.

**Fix:**
- Use `std::array<int8_t, 24>` for points, a fixed `int dice[4]; int n_dice`, and keep move lists out of the copied state.
- Write `dfs` with one shared path buffer (push/pop) and an output vector passed by reference.
- Make `get_single_moves` fill a small fixed-capacity array.
- Cache "furthest-back checker" and checker counts so `can_bearoff()` is O(1).

Benchmark against make/unmake (apply a move, recurse, undo) instead of copying.

Build flags: Linux already builds with `-O3` (from Python's `sysconfig`). The Windows flags in [setup.py:18](backend/classes/setup.py#L18) have `/Ox` commented out. The comment's "optimizations slower" result came from end-to-end random games, which are dominated by Python overhead. Re-measure with `python -m bench run --only speed` once native code dominates.

## 6. Make MCTS tree reuse actually work

**Effort: small–medium**

The agent tries to reuse the previous search tree ([CMCTS2.pyx:445](backend/classes/agents/CMCTS2.pyx#L445)), but in practice this almost never happens:

- **The server builds a new `MCTSAgent` every AI turn** ([app.py:242](backend/app.py#L242)), so there is never a previous tree to reuse.
- Even within one agent, matching uses a board hash that **includes dice**. Child nodes pre-roll a random dice value in `Node.__init__` ([CMCTS2.pyx:31](backend/classes/agents/CMCTS2.pyx#L31)). A match requires the opponent's real roll and our next real roll to both equal the pre-rolled values, plus the opponent's actual reply to have been expanded. That is well under 1% of turns.

**Fix:** keep one agent per room/side (inside the AI worker from #2). Once chance nodes exist ([improvements.md #3](improvements.md)), descend the tree by (our move → opponent roll → opponent move → our roll) instead of hash-matching children. Every simulation from the previous turn that lands in that subtree is then reused.

## 7. Evaluate each resulting position once

**Measured · effort: small**

Different move sequences often reach the same position. Across 30 random games, **8,074 legal sequences produced 7,312 distinct positions (≈9.4% duplicates)**. On specific rolls it is much higher: 6-1 from the opening gives 12 sequences and 10 positions ([improvements.md #10](improvements.md) found 21 → 15 for 1-2).

**Fix:** deduplicate by resulting position before evaluating, in C++ (preferred: return one representative sequence per afterstate) or in the agents. This benefits the heuristic and NN agents and MCTS expansion (fewer duplicate children, so the visit budget is spread over genuinely different moves).

## 8. Remove artificial server sleeps

**Effort: tiny**

`socketio.sleep(1)` appears at [app.py:112](backend/app.py#L112), [200](backend/app.py#L200), [248](backend/app.py#L248), [259](backend/app.py#L259), [272](backend/app.py#L272) and [317](backend/app.py#L317). Each AI turn waits **2 s** on top of the search: 1 s before selecting a move and 1 s after broadcasting. The others block the human's handler thread for 1 s after it has already finished. AI-vs-AI games run at ≥2 s per turn regardless of agent speed.

**Fix:** remove them and pace animation on the client (it already has a dice-roll animation state).

## 9. Lighter imports and faster startup

**Measured · effort: small**

Importing `app` takes **3.2 s**, of which **2.9 s** is `NNAgent` ([NNAgent.py:10](backend/classes/agents/NNAgent.py#L10)):

- `torch` itself (~1.1 s) is required.
- `import torch._dynamo` plus `dynamo.config` changes pull in `torch._inductor` (~0.7 s) **at import time in the web server**, only to support training's `@torch.compile`.
- `matplotlib`, `pyplot`, `tqdm` and `multiprocessing` are training-only.
- `CMCTS2.pyx` imports `NNAgent` and `torch` even for non-neural MCTS.

**Fix:** split `NNAgent.py` into `nn_model.py` (network, features, `FinalNNAgent`) and `nn_training.py` (trainer, evaluator, plotting, dynamo config). Import plotting inside the plot functions. This also matters for every multiprocessing worker spawned by the benchmarks, tournaments and training, since each one pays the import cost.

Measured and **not** worth doing: `torch.set_num_threads(1)` made no difference for batch-1 inference with this model (≈30 µs either way). It may still help in multiprocess training (#11) to avoid oversubscription.

## 10. Frontend rendering

**Effort: small**

[Board.tsx](frontend/src/components/Board.tsx) is one component, so any state change re-renders the whole board.

- **Debug logging is always on.** `const verbose = true` ([Board.tsx:38](frontend/src/components/Board.tsx#L38)), and an effect runs `JSON.stringify(gameState)` on every state change, including the full `validMoves` list ([Board.tsx:152](frontend/src/components/Board.tsx#L152)). There are also unconditional `console.log`s in `setVals` and `update_dice`. Gate them on `import.meta.env.DEV`.
- **Dragging re-renders everything.** Every `mousemove` calls `setDragPosition`, which re-renders all 24 points, every checker, and about 28 `moveIsValid` scans over `validMoves`. Move the dragged checker into its own small component (or update its `transform` via a ref), memoize the static board layer, and precompute legal destinations for the selected origin once with `useMemo` ([improvements.md, smaller items](improvements.md)).
- **`useWindowSize()` re-renders the whole board on every resize** ([Board.tsx:41](frontend/src/components/Board.tsx#L41)), but only `<Confetti>` uses it. Move it into a small wrapper rendered only when `gameOver`, and load `react-confetti` with `React.lazy`.
- `triangleCoords` is rebuilt every render but never changes, so hoist it to module scope. `origin` is derived state set in a `useEffect`, which costs an extra render per selection; compute it inline instead.
- **Bundle size:** `BenchDashboard` is imported eagerly in `App.tsx`, so every player downloads the benchmark dashboard. Use `React.lazy` for the `/bench` route. Remove unused dependencies (`framer-motion`, `@million/lint`, and the `socket.io` *server* package).
  - **Measured 2026-10-24.** I split a production build by library: 260 KB / **85 KB gzip** in total. React is 46 KB, react-router 14 KB, socket.io-client 13 KB, the bench dashboard **6.3 KB** (JS + CSS), app code 5 KB and confetti 3.2 KB.
  - Lazy-loading `/bench` therefore saves about 7% of the download. Lazy-loading confetti saves another 4%.
  - The unused dependencies don't appear in the bundle (tree-shaken), so removing them only shrinks `node_modules` and install time.
  - The bundle is already small. The render items above matter more than its size.

## 11. Training pipeline

*Status (2026-10-25): partly done.* `torch.compile` and anomaly detection are removed, self-play copies boards with `clone()`, and a game's states travel as one NumPy array (#13). In the benchmark, self-play went from 1,228 to 3,995 positions/s, TD updates from 2,668 to 3,943, and `train_epoch` from 6.2 to 9.2 games/s. Persistent pools and hand-written gradients (#16) remain.

**Effort: small–medium · affects offline training only**

- `torch.autograd.set_detect_anomaly(True)` is enabled for normal training ([NNAgent.py:807](backend/classes/agents/NNAgent.py#L807)). It records stack traces for every op and commonly slows backward passes several times. Remove it or put it behind a flag.
- `@torch.compile(mode="reduce-overhead")` on `TDLambda.update` ([NNAgent.py:126](backend/classes/agents/NNAgent.py#L126)) wraps Python loops, list comprehensions, `.item()` calls and variable-length episodes. Expect graph breaks and recompiles; it is probably slower than eager. Benchmark with and without it.
- `V_t.backward(retain_graph=True)` keeps graphs alive unnecessarily, and `delta.item()` forces a sync per step.
- `pool.imap_unordered(self.simulate_one_game, ...)` pickles the **entire trainer** (model, optimizer, traces) for every game, because the default `chunksize` is 1. `Evaluator` does the same with the agent. Send a `state_dict` once via a pool `initializer`, use a larger `chunksize`, keep the pool alive across epochs, and call `torch.set_num_threads(1)` in workers ([improvements.md #14](improvements.md)).
- Self-play `NNAgent.select_move` uses `deepcopy(board)` (2.5 µs) where `board.clone()` (0.9 µs) does the same thing (measured). Item #1 applies here as well.
- For a bigger change, compute λ-returns for a whole episode in one batched forward/backward pass instead of one forward/backward per step. That changes online TD(λ) into an offline variant, so treat it as an algorithm change and compare strength.

**Measured 2026-10-24.** `TDLambda.update` over 20 recorded self-play games (2,492 positions), with the `main()` model (hidden 40):

| Configuration | Positions/s | vs shipped |
|---|---|---|
| Compiled + `set_detect_anomaly(True)` (what `main()` runs) | 1,232 | 1× |
| Compiled, no anomaly detection | 1,884 | 1.5× |
| Eager (`torch.compile` removed), with anomaly detection | 852 | 0.7× |
| **Eager, no anomaly detection** | **2,345** | **1.9×** |
| Hand-written gradients, no autograd (#16) | 9,270 | 7.5× |

- **Anomaly detection** costs 2.75× in eager mode.
- **`torch.compile`** is 20% *slower* than eager, and its first call takes 3.6–3.8 s to compile. Remove both.
- **Pickling the trainer:** this costs only ~1 ms and 2.8 KB per task, because torch passes the parameters as shared-memory handles. It isn't the epoch bottleneck. The returned states are (#13).
- **`deepcopy` vs `clone()`:** re-measured per decision at 37 µs versus 18 µs (~10 copies each).

## 12. Smaller items

- **DB work per event.** Every handler queries `Game` and runs `.order_by(Game.id.desc())` even though `room_code` is unique ([app.py:64](backend/app.py#L64)). `update_board_db` then queries **again** instead of updating the row it already loaded. Pass the row through. Consider keeping the live `Board` in memory per room and persisting after each accepted turn.
- **`PickleType` for positions** ([models.py:8](backend/models.py#L8)) pickles on every read and write. A JSON column or a 24-char string is cheaper and portable.
- **SQLite:** enable WAL mode (`PRAGMA journal_mode=WAL`) so commits don't block readers. It matters once AI workers and request handlers write concurrently. It also cuts latency for a single game today; see #19.
- **Disconnect** scans every room ([app.py:128](backend/app.py#L128)). Keep a `sid → (room, side)` map.
- **Bench API:** `store.load_all()` re-reads and parses every result JSON on every request ([bench/store.py:55](backend/bench/store.py#L55)). Cache it keyed on the directory listing and file mtimes.
- **Single-move shortcut:** MCTS returns early when there is one legal sequence, but the heuristic and NN agents still clone and evaluate. Add the same shortcut there.

---

Items 13–20 were found by the 2026-10-24 profile. Every number below was measured locally. Where a fix was prototyped, the prototype lived in a scratch copy and the repo is unchanged.

## 13. Return one array per game from training workers

**Measured · effort: tiny · biggest training win**

**The problem:**
- `play_game` returns `states` as a list of ~80 separate `1×34` tensors ([NNAgent.py:361](backend/classes/agents/NNAgent.py#L361)).
- `train_epoch` sends them back through `pool.imap_unordered` ([NNAgent.py:321](backend/classes/agents/NNAgent.py#L321)).
- Importing torch registers its multiprocessing reducers. Every tensor that crosses the pool is therefore moved into its **own shared-memory segment**, and its file descriptor is passed over a socket.
- The parent receives these one at a time, so the pool is serialised on it.

Returning one game's states from a 4-worker pool:

| Returned as | Cost per game | Pool limit |
|---|---|---|
| list of 80 tensors (current) | **50.8 ms** | 20 games/s |
| one `80×34` tensor | 0.75 ms | 1,300 games/s |
| nested Python lists | 0.17 ms | 6,000 games/s |
| one NumPy array | **0.06 ms** | 16,000 games/s |

End to end (TD update in eager mode, so compile time isn't counted):

| Configuration | Games/s |
|---|---|
| Single process, no pool | 6.0 (8 torch threads), 7.5 (1 thread) |
| `train_epoch`, 4 workers (current) | **4.2** |
| `train_epoch`, 4 workers, `torch.set_num_threads(1)` | 5.7 |
| `train_epoch`, 16 workers, 1 thread | 7.9 |
| **`train_epoch`, 4 workers, 1 thread, one NumPy array per game** | **32.7** |
| `train_epoch`, 16 workers, 1 thread, one NumPy array per game | 22.7 |

So the pool currently makes training *slower* than a plain loop.

**Fix:**
- Return `torch.cat(states).numpy()` (or the feature lists) from `play_game`. In `update`, rebuild the tensors with `torch.from_numpy(arr)`, or better, index rows of one tensor.
- Call `torch.set_num_threads(1)` in the workers. Oversubscription (workers × 8 intra-op threads) costs 25–40% on its own.

With 16 workers, the serial TD update in the parent becomes the limit. That's why 16 workers are slower than 4 here; #11 and #16 raise that ceiling.

The benchmark's `training.epoch` metric (3.5 games/s on the quick suite) will show the change directly.

## 14. Pause Python's cyclic garbage collector during MCTS search

**Measured · effort: tiny**

**The problem:**
- Every expansion and rollout step allocates thousands of short-lived tuples and lists, mostly from converting `valid_moves`. These keep triggering Python's cyclic GC.
- The search tree is a large graph of `Node` objects with parent ↔ children cycles. Each older-generation collection re-traverses all of it.
- With rollouts disabled, py-spy attributes **47%** of search time to `gc_collect_main`.

MCTS iterations/s on 20 suite positions, GC on versus off:

| Configuration | GC on | GC off | Gain |
|---|---|---|---|
| Rollout depth 50, 0.2 s budget | 1,441 | 1,463 | 1.02× |
| Rollout depth 50, 2 s budget (`MCTSAgent2` default) | 1,235 | 1,441 | **1.17×** |
| Rollout depth 50, 5 s budget (server setting) | 1,186 | 1,331 | **1.12×** |
| No rollout (tree only), 0.2 s | 8,542 | 20,227 | 2.37× |
| No rollout (tree only), 2 s | 8,699 | 17,615 | 2.02× |

The cost grows with tree size, so it hurts longer searches most. Once rollouts are fast (#3), it becomes half of all MCTS time.

**Fix:**
- In `MCTSBackgammonAgent.search` ([CMCTS2.pyx:170](backend/classes/agents/CMCTS2.pyx#L170)), call `gc.disable()` before the loop and `gc.enable()` in a `finally`.
- Or call `gc.freeze()` after building the root.
- Discarded subtrees still need a collection, so run `gc.collect(0)` or `gc.collect(1)` between moves, not during the search.
- Longer term, remove the cycles: hold `parent` as a weak reference, or store the tree in flat arrays. Then reference counting alone frees nodes.

## 15. Cut MCTS tree overhead

**Measured cost · effort: small–medium · required for #3 to pay off**

With `simulation_depth=0` (no rollout), `MCTSAgent2` manages only **8,200 iterations/s**, which is about **121 µs of tree work per iteration**. Today that's hidden behind a 665 µs rollout. After #3, it would be 88% of the iteration.

The native profile of that tree work:
- **`Node.__init__` takes 66%** ([CMCTS2.pyx:31](backend/classes/agents/CMCTS2.pyx#L31)):
  - It calls `state.roll_dice()`, which converts the whole `valid_moves` list into the tuple it returns. The tuple is discarded.
  - It then reads `state.valid_moves`, which **converts the list a second time**.
  - pybind11 list/tuple casters alone are 52% of tree time.
- **Selection (`best_child` → `value`) takes 16%.** That's Python-level calls and `math.log`/`math.sqrt` per child (see the smaller items in #3).
- **`untried_moves.remove(move)`** compares lists element by element.

**Fix:**
- Add a native `roll()` that returns nothing, or make `roll_dice` return `None` when the caller doesn't need the tuple. Use it in `Node.__init__`, rollouts, `play_game`, `Evaluator` and `tournament.py`. That alone saves ~20% of every roll (1.9 of 9.6 µs, #4).
- Store untried moves as a count, and fetch a sequence by index from the board (`board.sequence(i)`), instead of copying the whole list into every node.
- Pick a random index and swap-pop it instead of calling `remove`.
- Combined with #14, the tree should reach roughly 20k iterations/s. The GC-off figure above is 17.6–20.2k before any of these changes.

## 16. Hand-written TD(λ) gradients

**Measured · effort: small**

The network `main()` trains is a 34 → 40 → 1 MLP. Its gradients are a few lines of closed-form maths, so autograd's per-step graph construction, `backward()` and parameter iteration are almost all overhead.

I wrote a sketch of the same online TD(λ) update. It uses the same targets, traces, learning-rate decay and update order, and computes gradients by hand under `torch.no_grad()`.
- **Speed:** 9,270 positions/s, against 2,345 for eager autograd (**4×**) and 1,232 for the shipped compiled + anomaly-detection configuration (**7.5×**).
- **Correctness:** I played 5 games from the same initial weights. The largest difference in any parameter was **4.5e-8**, against parameter changes of 3.6e-2, so the results are identical up to float rounding.

**Fix:** implement `TDLambda.update` for the `BackgammonNN` layer layout with explicit gradients. Keep the autograd path as a fallback for other architectures, and add a unit test comparing the two. Most of the remaining time is per-step `.item()` and small-tensor calls. Doing the same arithmetic in NumPy, or batching the forward passes for the next-state targets, would gain a further 2–3×.

## 17. NumPy inference for the small MLP

**Measured · effort: small**

For networks this small, PyTorch's per-call dispatch overhead (~45 µs) is far larger than the arithmetic.

| Operation | PyTorch | NumPy / alternative |
|---|---|---|
| Batch-1 forward (MCTS neural leaf) | 45.6 µs (+6.3 µs to build the tensor) | **9.2 µs** |
| Batched forward in `FinalNNAgent.select_move` (~10 rows, incl. `model.eval()`, `squeeze`, `argmax`, `item`) | 112.7 µs; `inference_mode` 77.3 µs | **40.8 µs** |
| Building the input: one tensor per row + `torch.stack` | 54.4 µs | 19.0 µs as one `torch.tensor(list_of_lists)`; ~0 with NumPy |
| Copying candidate boards: `deepcopy` | 37.1 µs | 18.0 µs with `clone()` |

Torch's thread count makes no difference at batch ≤ 10 (58.5 µs with either 1 or 8 threads). Extra threads only help from about batch 160 (105 µs with 8 threads, 165 µs with 1).

**Fix:**
- After loading the checkpoint in `FinalNNAgent.__init__`, extract the weight matrices as NumPy arrays.
- Evaluate with `x @ W + b` and `np.maximum`.
- Build features directly into one `float32` array.
- Drop the per-call `self.model.eval()` ([NNAgent.py:797](backend/classes/agents/NNAgent.py#L797)).
- Use the same evaluator for the MCTS neural leaf ([CMCTS2.pyx:278](backend/classes/agents/CMCTS2.pyx#L278)).

Together with #1, a `FinalNNAgent` decision should drop from ~730 µs to roughly 150 µs (estimated from the measured parts). That benefits the server, tournaments and the strength benchmarks. Training self-play (`NNAgent.select_move`) has the same structure, but its weights change each game, so refresh the arrays per game.

## 18. Don't copy `valid_moves` when cloning

**Measured · effort: small**

`clone()` ([board_cpp.cpp:1273](backend/classes/board_cpp.cpp#L1273)) copies the entire `valid_moves` vector-of-vectors, even though every caller immediately applies a move. Applying a move then re-validates against that list (`move_from_sequence`), which is another linear scan.

| Stored sequences | `clone()` | Clone with an empty move list |
|---|---|---|
| 0–24 (93% of rolls) | 1.06 µs | 0.76 µs |
| 50–74 | 2.77 µs | 0.66 µs |
| 150–174 | 5.72 µs | 0.48 µs |

Every agent clones once per legal sequence. The per-decision cost is therefore quadratic in the number of sequences: clone + apply for every sequence averages 17 µs, but the worst roll in the suite (159 sequences) takes **0.99 ms**.

**Fix:**
- Make `clone()` leave `valid_moves` empty; the clone recomputes it on its next roll anyway.
- Add `apply_unchecked(seq)` for sequences that came from the board itself.
- Best of all, add a native `afterstates()` that returns one representative board (or feature row) per distinct resulting position. That also covers #7.

## 19. SQLite: WAL and `synchronous=NORMAL`

**Measured · effort: tiny**

I drove the Socket.IO handlers in-process with Flask-SocketIO's test client, against a scratch SQLite file on the same ext4/NVMe disk as `instance/test.db`. Over a 64-turn human-vs-human game, `roll_dice` and `move` each took a median of **4.9 ms**:

| Part | Time |
|---|---|
| Commit of the changed row (default rollback journal, `fsync`) | **3.75 ms** |
| `Game` query | 0.36 ms |
| `Board(board_db=…)` | 0.007 ms |
| `convert()` | 0.003 ms |
| Same commit with `journal_mode=WAL` + `synchronous=NORMAL` | **0.79 ms** |

**Fix:** set both pragmas on every connection with an SQLAlchemy `connect` event listener. Handlers then drop to about 2 ms. Combined with passing the loaded row to `update_board_db` instead of re-querying (#12), they drop to about 1.5 ms. With `synchronous=NORMAL` in WAL mode, a power cut can lose the last few commits but never corrupts the database, which is fine for game state.

## 20. Turn off the polling reloader in the dev server

**Measured · effort: tiny**

`socketio.run(..., debug=True)` ([app.py:343](backend/app.py#L343)) enables Werkzeug's reloader. `watchdog` isn't installed, so it uses the **stat reloader**:
- **CPU:** once a second it stats every loaded module file. With torch imported that's **2,301 files and 26 ms per scan**. The running dev server used **3.2% of a core while idle**, measured over 20 s.
- **Memory:** the reloader keeps a second parent process (94 MB RSS) next to the serving process (369 MB).
- **Restarts:** every restart re-imports torch (~3–4 s), and edits anywhere under the watched paths trigger one.

**Fix:** use `use_reloader=False` (keeping `debug=True` for tracebacks if wanted), or `pip install watchdog` so the reloader uses inotify. Take `debug` from an environment variable so a deployment never runs the debugger. #9's import split also makes each restart cheaper.
