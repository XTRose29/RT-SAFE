# Reproducible real-time examples

After [installing Python dependencies](../docs/INSTALL.md) and the
[pinned runtime](../docs/REALTIME_ADDON.md), run from the source root:

```bash
python tools/run_realtime_examples.py \
  --launcher /absolute/path/to/Linux/SimWorld.sh \
  --gpu 0 --port 19091 --output results/realtime-examples
```

The script launches and stops a fresh engine for each condition, runs the
actual greedy benchmark policy, saves raw results and logs, and checks them
against [expected.json](realtime/expected.json). It exits nonzero on an engine,
runner, or comparison failure. The output directory must be new; add
`--repeats 3` to run three independent pairs.

| RT10 task 0, easy, seed 0 | Static | Real-time |
|---|---:|---:|
| Greedy decisions | 2 | 2 |
| Added thinking time per decision | 1 s | 1 s |
| Reported simulation time | 4 s | 6 s |
| Model tokens | 0 | 0 |

Static mode excludes the thinking intervals from simulated time. Real-time
mode includes both one-second intervals. `sim_time` is the runner's accounted
duration, not an independent measurement of every engine physics tick.
The short runs deliberately stop after two decisions, so `success: false`
is expected; it does not mean installation failed.

All six runs in three independent fresh-process pairs matched these expected
fields, with no rollout, parse or signal errors. Each also recorded one oil
event and zero contacts. The [fresh comparison record](../validation/realtime-fresh-examples.json)
contains the observed values, configuration, and source-result hashes.

These examples check timing/configuration invariants. Physics trajectories,
pixel hashes and hazard/contact counts can vary with scheduling and are
recorded without requiring exact equality. They do not reproduce historical
paper scores or establish minimum hardware requirements. No model API key is
needed. A separate full five-map check is available in
`tools/check_runtime_addon.py`.

[Fresh runtime screenshots and five-map checks](runtime-checks/README.md) are also
included, so the expected map appearance can be inspected without running
Unreal.

A [live Codex CLI pilot](codex-cli/README.md) includes real model actions,
observations and timing records from both modes, plus commands to repeat it.
The [complete greedy route example](full-route/README.md) also reached the goal
in both modes: 30 decisions, 81 seconds static and 111 seconds real-time.

The first authored task on each of RT12, RT15, RT18 and RT20 also completed a
two-decision greedy scene check, including actor generation and policy
observations. See [the full fresh benchmark check record](../validation/realtime-fresh-campaign.json).

## Recorded campaign examples

These are three previously curated decisions from original benchmark maps,
included so readers can inspect observations, actions and timing without
installing Unreal. They are qualitative excerpts, not new measurements.

| Recorded policy | Map/task | Action | Response latency | Inference exposure |
|---|---|---|---:|---:|
| Gemini | RT18 / 25 | Move 2 m | 23.35 s | 25.98 s |
| Sonnet | RT18 / 31 | Move 2 m | 3.96 s | 6.16 s |
| Fable | RT15 / 19 | Wait 1 s | 4.41 s | 5.83 s |

Each record includes its observation and resulting image:
[Gemini](recorded/case-1.json), [Sonnet](recorded/case-6.json),
[Fable](recorded/case-7.json). The [provenance manifest](recorded/provenance.json)
identifies the source snapshot and image hashes. These use hard difficulty,
provider-default effort and seed 0. Exposure includes more than response
latency; it should not be substituted for the API response time.
