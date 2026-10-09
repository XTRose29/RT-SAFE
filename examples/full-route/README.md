# Complete greedy route example

The fresh installation completed RT10 task 0 with the deterministic greedy
policy in both modes. Each condition used a fresh engine, seed 0, easy
difficulty, and one second of simulated thinking per decision.

| Observed outcome | Static | Real-time |
|---|---:|---:|
| Route completed | Yes | Yes |
| Decisions | 30 | 30 |
| Reported simulation time | 81 s | 111 s |
| Model tokens | 0 | 0 |
| Oil events | 2 | 2 |

The 30-second difference matches the 30 added thinking intervals. See the
[complete configuration and metrics](results.json). These are fresh code-policy
results, not historical model scores; completion with oil events is not safe
success. Physics/contact outcomes can vary between runs.

To repeat a condition, start a fresh RT10 engine and apply the environment
exports from [the runtime guide](../../docs/REALTIME_ADDON.md), then run:

```bash
python evaluation/run_code_baselines.py \
  --task-file data/map1_10roads/tasks.json --task-index 0 --seed 0 \
  --baselines greedy --env-modes static --time-modes sync \
  --difficulties easy --max-steps 120 --internal-latency-seconds 1 \
  --max-rounds 1 --ue-port 19091 --suite-name greedy-full-static \
  --no-record-per-step
```

Restart Unreal before the second condition. Change `--env-modes` to `realtime`
and use a new suite name such as `greedy-full-realtime`. Check `summary.jsonl`
for `status: completed` and inspect the referenced result. The 120-decision
limit permits route completion, unlike the deliberately short installation
example.
