# New feature ideas

These features connect **playing games, reviewing decisions, and improving the AI**. The highest-value additions are a historical-game archive with AI analysis and a reproducible neural-network benchmark based on decision quality.

## 1. A searchable archive of historical games

Save completed and unfinished games with their players, AI agents/checkpoints, date, result, rules, dice, and moves. Let users filter by opponent, model, outcome, and analysis status.

Each game could show its result, average decision error, biggest mistake, and whether AI analysis is complete. Support bookmarks, annotations, sharing, and importing/exporting compatible games.

This requires recording turns: the current [Game model](backend/models.py) primarily stores the latest board snapshot, so it cannot reconstruct previously played games by itself.

## 2. Interactive replay with AI evaluation

Opening an archived game would show the board, move timeline, and an evaluation graph. Users could step forward/backward, autoplay, or jump directly to important decisions.

For each turn, show:

- The dice and move played.
- The AI's preferred alternatives, with board previews.
- Estimated winning chances and the value lost by the chosen move.
- Separate changes caused by dice luck from changes caused by decisions.
- The evaluator's version, analysis settings, and uncertainty.

Offer quick analysis first and deeper analysis on request. Preserve previous analyses so users can compare how different model versions assess the same game.

## 3. A neural-network evaluation workspace based on decision quality

The current evaluator mainly reports wins and game length. Add a **fixed position benchmark** that tests every checkpoint on the same held-out decisions.

Build a diverse collection covering contact play, races, bearing off, bar entry, hitting, blocking, and difficult tradeoffs. Establish reference move values using an independent strong engine, deeper rollouts, or exact endgame calculations.

The main metric would be:

```text
Decision loss =
    reference value of the best available move
    − reference value of the model's chosen move
```

For the current win-only rules, express this in estimated winning-probability points. For example, choosing a move valued at 57% when the best reference move is valued at 61% incurs an estimated loss of **4 percentage points**.

Evaluating decisions through average error has precedent in [GNU Backgammon's analysis methodology](https://www.gnu.org/software/gnubg/manual/html_node/How-is-the-error-rate-determined_003f.html).

The dashboard should report several complementary measures:

| Measure | What it reveals |
|---|---|
| Average decision loss | Overall move quality |
| Serious-error frequency | How often the model makes costly mistakes |
| Error by position category | Specific strengths and weaknesses |
| Prediction error and calibration | Whether its probability estimates are reliable |
| Median/tail latency, throughput, memory | Computational performance |
| Paired match results with uncertainty | Performance across complete games |

Count statistically indistinguishable reference moves as ties. Exclude forced moves from the decision-quality average and report them separately.

Keep two evaluation tracks: **the network itself**, using consistent inputs and selection rules; and **the complete agent**, including search, at equal decision-time budgets. That separates better learned judgment from additional computation.

Keep the final test set separate from training and practice positions. Use a separate diagnostic set for inspecting individual failures. Calibration tests should use documented continuation policies, since winning chances depend on subsequent play.

## 4. Checkpoint comparisons with concrete examples

Select two checkpoints and see which improved, where it regressed, and how much more computation it requires.

Clicking a category such as "race positions" would open diagnostic examples where their choices differ. Users could also reanalyze an archived game with both versions.

This would make statements such as "the newer model wins more often" much more informative: perhaps it improved contact play while becoming worse at bearing off.

## 5. Post-game reports

Generate a readable report containing the largest mistakes, strongest decisions, missed opportunities, and separate summaries of decision quality and dice luck.

Every finding should link to the relevant board position. Explain recommendations using observable evidence, such as exposed blots, blocked points, pip counts, or rollout comparisons, and mark uncertain judgments accordingly.

## 6. Practice puzzles generated from mistakes

Turn reviewed positions into exercises: choose a move, commit to an answer, then compare alternatives.

Add topic filters, difficulty levels, spaced repetition, and personal progress tracking. Historical mistakes could become a personalized practice collection, using positions kept separate from the final model benchmark.

## 7. A position editor and analysis sandbox

Let users arrange checkers, set bars and borne-off counts, choose the player and dice, and ask different agents to analyze the position.

From any replay position, users could branch into a hypothetical continuation or play the position themselves. Keep that variation separate from the original game record.

## 8. An AI spectator mode

Watch two agents with pause, single-turn stepping, replay speed controls, and visible model versions.

An optional analysis panel could show candidate moves, evaluation changes, search effort, and decision time. Completed AI games would automatically enter the archive.

## 9. A tournament and benchmark center

Configure round-robin tournaments across agents or checkpoints, with controlled dice seeds, balanced colors, consistent compute budgets, and confidence intervals.

Show standings alongside decision quality, runtime, and incomplete or failed games. Link results to their individual game replays and analysis.

## 10. A training experiment dashboard

Track training runs, configurations, checkpoints, learning curves, elapsed time, and benchmark results in one place.

Plot decision quality against training time or games generated. Automatically evaluate checkpoints and flag regressions. Promoting a checkpoint to the default opponent should depend on reproducible evaluation evidence.

## 11. Persistent matches and player profiles

Support resuming interrupted games, reconnecting to the same seat, rematches, personal game history, and progress over time.

Profiles could show recurring mistake categories and practice improvements alongside match results.

## 12. Scored matches and doubling-cube play

Add multi-game matches, gammons/backgammons, doubling decisions, and match-score context.

Extend replay analysis and model benchmarks to evaluate cube decisions and expected points or match-winning chances. These features would require corresponding changes to the rules engine and neural-network objectives.
