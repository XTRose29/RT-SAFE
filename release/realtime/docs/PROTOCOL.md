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

## Example evidence

This package includes model-free fresh-install examples and three curated
historical decisions from the original maps. See [examples](../examples/README.md).
The historical manuscript tables are outside this source distribution.

## Interpretation limits

- Static and real-time prompts differ in timing advice, so their comparison is not a pure timing-only intervention.
- Repeated contact callbacks may count as multiple collisions. Collision count is not a count of distinct accidents.
- Hazard evaluation uses endpoint overlap; water is recorded without a movement perturbation in the reported version.
- CLI, direct API and routing-provider transports may add different model-visible context. Keep access surfaces labeled.
- The separate offline BC/RL study uses a fixed three-second delay and 16 held-out tasks. Its training implementation, dataset and checkpoints are not supplied here.
- The current source snapshot contains research changes made after some campaigns. Exact paper reproduction requires the recorded runtime, serving versions, per-run resolved configuration and campaign source snapshot.

Raw rollouts contain more metadata than a public result table needs. This distribution includes curated decision examples, not private campaign logs or account-usage records.
