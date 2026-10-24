# New-model discovery and RT-SAFE jobs

`tools/model_watch.py` consumes the shared model-release feed, freezes an RT-SAFE
job, and files completed results to a private ledger. `tools/model_watch_slack.py`
adds an owner-controlled `/rtsafe` menu, Start/resume and Stop buttons, release
notices, and result summaries. Discovery alone makes no inference calls. This
matches the handover's human-approved Start workflow; unattended execution and
public leaderboard publication are not enabled.

## Deployment

Copy [the example configuration](../tools/model_watch_config.example.json) to a
private deployment directory and replace the paths. Use a dedicated checkout of
`fix/rt-safe-v4` and a **new state directory**. Do not repoint a started historical
job to a different checkout. Install benchmark dependencies and, in a separate
bot environment if desired, [the Slack requirement](../tools/requirements-model-watch-slack.txt).

Supply a frozen selection manifest with seed 0, hard difficulty, realtime timing,
and ten or twenty distinct tasks, five per map. Each task needs `map`, `task`,
and `length_m`; planning verifies these against the source task geometry. The
serv6 v4 deployment uses the existing matched twenty-task set, including task 12.
Changing or excluding tasks creates a different benchmark configuration.

```bash
python tools/model_watch.py --config /private/deployment/config.json scan
python tools/model_watch.py --config /private/deployment/config.json status
python tools/model_watch.py --config /private/deployment/config.json plan JOB_ID
# A deliberate start is required; it consumes quota or API budget:
python tools/model_watch.py --config /private/deployment/config.json start JOB_ID
python tools/model_watch.py --config /private/deployment/config.json stop JOB_ID
```

Schedule only `scan` hourly, at minute 35 after the shared feed's minute-27 export.
`scan` locks itself, is idempotent, rejects stale feeds, and records release notices
even when the project's provider catalogue is unavailable. The initial historical
feed is baselined when Slack first connects, avoiding a backlog of release posts.
New notices after that are posted to the configured channel. Completed jobs are
not restarted by another Start click. Source drift blocks old jobs; changed source
or task selections produce new job IDs. `refresh-job JOB_ID` only replaces an
unstarted plan and returns the new ID when its source changed.

## Accounts and execution

The shared feed supplies model metadata, **not credentials or inference access**.
Use this project's authorized account gateway or this project's own OpenRouter
key. The v4 account adapter explicitly pins medium reasoning for:

- `antigravity/gemini-4-argon-medium`
- `antigravity/claude-sonnet-5-5-medium`
- `codex/gpt-6-astra`

A route must be in the authenticated catalogue, support Chat Completions, and
explicitly report `billable: false`. Every request disables paid fallback and
checks the returned model identity. New model names are discovery notices, not
permission to guess a route or reasoning level; add and test a verified profile
before enabling them. Catalogue presence does not prove that image inference is
healthy: Claude's earlier image preflight failed, so it has no completed native
pilot evidence. The separately validated local Codex CLI pilot is not the same
transport as this gateway adapter.

Start with one worker. Jobs freeze source/data, task geometry, effort, transport,
prices and startup flags. The default example retains the two startup flags used
by native validation and the default rendering threads. Shared GPU allocation
must be agreed before starting; a process lock serializes this deployment only.
There is no global reservation of another project's GPUs.

Completed episodes are reused. Infrastructure failures get isolated bounded
attempts; failed navigation is a valid result and remains in the denominator.
This simulator does **not** restore a partially completed episode or conversation.
The complete expected task/effort set and matching `execution_alignment_v4`
metadata are required before filing. Keep original failure attempts and transcripts
private. The source hashes prevent silent replacement of a queued experiment.

The gateway's payload limit can require full-resolution JPEG recompression;
per-decision metadata records it. Missing token usage stays unknown. Subscription
jobs do not claim to be free: API-dollar estimates are null and account quotas
still apply. OpenRouter estimates are planning estimates, not enforced budget
limits; set a provider-side cap before enabling that transport.

## Slack handover for Koe

1. Create/install an app using [the manifest](../tools/model_watch_slack_manifest.yaml).
   It uses Socket Mode, `/rtsafe`, `commands` and `chat:write`.
2. Invite the bot into the chosen channel (previous proposal: `#proj-safety`).
3. Privately deliver `SLACK_BOT_TOKEN` (`xoxb-`) and `SLACK_APP_TOKEN` (`xapp-`,
   `connections:write`). Save them as `NAME=value` lines in `slack.token_file`, mode 600.
4. Set `slack.channel_id` and `slack.owner_user_id` to real IDs. There is no
   first-user ownership claim. Only that owner in that channel can start or stop.
5. Run `python tools/model_watch_slack.py --config /private/deployment/config.json`
   using the Slack environment. After connection and authorized channel testing,
   supervise it with cron or a service. The bot has its own process lock.

No Slack messages can be sent before these credentials and IDs are configured.

## Three-benchmark handover and publication

BuildingBench already owns its runner. This repository implements the RT-SAFE
adapter only. To connect Code4Scene, its maintainer must provide a repository and
pinned version, a noninteractive run/score command, fixed cases/seeds/limits,
result schema, required assets/runtime, resource needs, and account routing.
Use the shared canonical model name to correlate the three jobs, but keep their
scores, protocols and failure states separate. Post a benchmark result when its
own complete run is validated; a missing provider result must be reported as
unavailable rather than assigned a zero score.

The team must decide channel and owner IDs, eligible model families and reasoning
policy, manual Start versus pre-approved automatic runs, per-model/monthly budget,
GPU ownership, and publication destination/approver. Each public update can carry
an RT-SAFE result, a Code4Scene result and a BuildingBench result once all three
maintainers supply their adapters/results.

`state/ledger/` and `state/board.json` are **private local outputs**, with source and
transport provenance. They can contain gateway URLs. They are not served by the
public website. An approved publisher must export only public aggregate fields
and publish a separate v4 leaderboard; never overwrite historical v3/manuscript
scores with this pilot. There is no automatic GitHub Pages push in this setup.
