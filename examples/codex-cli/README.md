# Live Codex CLI example

The fresh runtime was also exercised with GPT-6 Astra through Codex CLI 0.153.4,
medium reasoning, RT10 task 0, easy difficulty, seed 0, and a two-decision cap.
Both runs returned two valid actions without rollout errors.

| Observed in this pilot | Static | Real-time |
|---|---:|---:|
| Decisions | 2 | 2 |
| Reported simulation time | 3.00 s | 23.33 s |
| Concurrent world evolution during inference | No | Yes |
| Contacts | 0 | 0 |
| Route completed | No (step cap) | No (step cap) |

See [results and per-decision timings](results.json), the
[static observation](static-first-observation.png), and the
[real-time observation](realtime-first-observation.png). Both observations
contain all seven policy waypoints. Real-time planning exposure includes
context/control overhead as well as the model response interval.

This is a transport and runtime check, not a historical score reproduction.
CLI transport adds product/runtime context. Model actions, tokens and latency
are not deterministic targets and should not be compared directly with an
API-only paper result.

## Repeat the pilot

Use an installed, signed-in Codex CLI. Start RT10 with the manual command and
common environment exports in [the runtime guide](../../docs/REALTIME_ADDON.md),
wait for engine initialization, then run from the source root:

```bash
python evaluation/run_openai_benchmark.py \
  --provider codex-cli --api-mode codex_cli \
  --model gpt-6-astra --reasoning-effort medium --max-output-tokens none \
  --task-file data/map1_10roads/tasks.json --task-index 0 --seed 0 \
  --difficulty easy --max-steps 2 --ue-port 19091 \
  --record-output-images --results-dir results/codex-realtime
```

For static mode, restart Unreal, add `--static-thinking`, and select a new
results directory. This entry point returns exit code **2** when the agent
does not complete the route, including these intentionally short pilots.
Check the result JSON: `decision_count: 2`, `termination_reason: "max_steps"`,
`rollout_error: null`, and `parse_success_rate: 1.0` establish the expected
pilot outcome. The automatic model-free example instead exits zero when its
reference comparison passes.
