# Benchmark protocol and reported results

RT-SAFE evaluates navigation and safety while the world continues during inference. The agent receives RGB observations, navigation instructions, recent action frames, and seven movement markers. It selects from 16 discrete move, turn, or wait actions.

| Axis | Reported setting |
|---|---|
| Routes | 36 over five city maps |
| Difficulty | Easy, medium, hard; 60%, 80%, 100% authored actor activation |
| Timing | Real-time world evolution during planning; static world paused during inference |
| Observation | 720 × 640 RGB, 100° horizontal field of view; previous-action frames at approximately 0.5 simulated seconds |
| Completion | Success rate and success weighted by path length (SPL) |
| Safety | Physical contacts, trip/oil/water events, traffic-rule violations |
| Timing attribution | Active events during execution; passive events during inference |
| Safe success | Successful completion with zero recorded safety events across the evaluated categories |

The detailed implementation protocol is in [benchmark/DEFAULT_SETTINGS.md](../benchmark/DEFAULT_SETTINGS.md). Presets are explicit configuration snapshots. `hard` is called `default` in parts of the legacy low-level runner.

## Results in the website

`website/app/data.json` and `website/public/data/results.csv` are extracted from the supplied manuscript archive dated 2026-10-01. The matched hard comparison contains 36 routes per model per timing mode, with eight models (288 paired episodes). All-difficulty summaries contain 108 episodes per model. Do not mix these denominators.

For the matched hard comparison, success is 91.3% static and 94.1% real-time; safe success is 19.8% and 0.7%. Mean contacts per episode rise from 3.31 to 40.68. These are manuscript results, not fresh measurements made during repository preparation.

## Interpretation limits

- Static and real-time prompts differ in timing advice, so their comparison is not a pure timing-only intervention.
- Repeated contact callbacks may count as multiple collisions. Collision count is not a count of distinct accidents.
- Hazard evaluation uses endpoint overlap; water is recorded without a movement perturbation in the reported version.
- CLI, direct API and routing-provider transports may add different model-visible context. Keep access surfaces labeled.
- The separate offline BC/RL study uses a fixed three-second delay and 16 held-out tasks. Its results are included, but its training implementation, dataset and checkpoints have not been located in the supplied benchmark tree.
- The current source snapshot contains research changes made after some campaigns. Exact paper reproduction requires the recorded runtime, serving versions, per-run resolved configuration and campaign source snapshot.

Raw rollouts contain more metadata than a public result table needs. The repository includes only curated decision examples and aggregate manuscript tables, not private campaign logs or account-usage records.
