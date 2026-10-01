#!/usr/bin/env python3
"""Drive the Claude Code subscription arms to completion across plan resets.

Work is split into lanes that run concurrently. Each lane runs its phases in
order; a phase is one arm at one effort, difficulty and environment mode,
stored under ``<safety-root>/<phase root>/<model slug>``.

Lane syntax: ``NAME=PHASES@GPU[,GPU...]@PORT[,PORT...]`` with one UE worker
per GPU entry. A GPU may appear more than once and lanes may share GPUs; base
ports must be disjoint and >=300 apart. ``PHASES`` is a comma list of

* ``arm:effort[:difficulty[:env_mode]]`` (defaults ``--difficulty`` and
  ``realtime``), stored under ``<difficulty>_<env_mode>_<effort>_seed0``; or
* ``main_table:ARM[:PREFIX[+PREFIX...]]``, which expands to the order of the
  other models' main table campaign (optionally only the phase roots starting
  with the given prefixes, e.g. ``main_table:sonnet_5:01``):
  ``01_hard_realtime_default``, ``04_hard_static_default``,
  ``05_hard_realtime_efforts`` at low effort, ``03_easy_realtime_default``,
  ``06_hard_realtime_upper`` at max effort, then
  ``02_medium_realtime_default``. Claude Code applies
  effort ``high`` when ``--effort`` is omitted, so "default" phases run an
  explicit, per-decision verified ``--effort high``; the additional effort is
  ``low`` and the highest supported explicit effort is ``max`` (both arms
  accept low, medium, high, xhigh and max).

For each idle lane the supervisor launches ``run_claude_code_campaign.py
--mode run`` for the first incomplete phase that plan usage allows. A lane
blocked only by its model-scoped window (Fable weekly) sleeps until that
reset while the other lanes keep running; account windows (5-hour, weekly)
pause every lane until they reset.

``--adopt PID=LANE:ARM:EFFORT`` takes over guarding an already-running
coordinator process group (one started by an earlier campaign process): the
supervisor enforces the same usage/auth guards on it and interrupts it at the
limits, then continues that lane normally once it exits.

Stop files written by the usage guards are archived when usage allows
running again; any other stop reason (model/effort/auth mismatch) halts that
lane for good. Enabled usage credits or credit spend halt everything. Creating
``<safety-root>/SUPERVISOR_STOP`` stops all lanes and the supervisor. It never
enables usage credits or raises a spend limit.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation import run_claude_code_campaign as campaign  # noqa: E402

CAMPAIGN_RUNNER = REPO_ROOT / "evaluation" / "run_claude_code_campaign.py"
DEFAULT_SAFETY_ROOT = Path("results/claude-code")
# Stop reasons that only mean "plan usage ran low" (or a deliberate stop);
# everything else halts the lane.
USAGE_STOP_REASONS = {
    "account_guard", "account_guard_before_start", "model_scope_guard",
    "usage_guard_before_arm", "usage_guard_unavailable", "ClaudeCodeUsageLimit",
    "global_stop_file", "supervisor_stop",
}
RESET_MARGIN = timedelta(minutes=10)
MAX_ATTEMPTS_CAP = 8
STOP_GRACE = timedelta(minutes=3)
# Campaign guard thresholds the supervisor may override (None keeps the campaign default).
GUARD_THRESHOLDS = ("min_five_hour_remaining", "min_weekly_remaining", "min_model_scoped_remaining")
# Claude Code 2.1.270 applies this effort to both arms when --effort is
# omitted (checked with a get_settings control request; no model turn).
CLI_DEFAULT_EFFORT = "high"
# Run order requested 2026-09-15: prioritize the hard default/static/effort
# comparisons, then easy, highest-supported effort, and medium.
MAIN_TABLE_PIPELINE = (
    ("01_hard_realtime_default", "hard", "realtime", (CLI_DEFAULT_EFFORT,)),
    ("04_hard_static_default", "hard", "static", (CLI_DEFAULT_EFFORT,)),
    ("05_hard_realtime_efforts", "hard", "realtime", ("low",)),
    ("03_easy_realtime_default", "easy", "realtime", (CLI_DEFAULT_EFFORT,)),
    ("06_hard_realtime_upper", "hard", "realtime", ("max",)),
    ("02_medium_realtime_default", "medium", "realtime", (CLI_DEFAULT_EFFORT,)),
)


class Phase(NamedTuple):
    arm: str
    effort: str
    difficulty: str
    env_mode: str
    root: str

    @property
    def condition(self) -> str:
        return f"{self.difficulty}_{self.env_mode}_instructional_{self.effort}"

    @property
    def label(self) -> str:
        return f"{self.arm}:{self.effort}:{self.difficulty}:{self.env_mode}"


def make_phase(arm: str, effort: str, difficulty: str = "easy", env_mode: str = "realtime",
               root: str | None = None) -> Phase:
    return Phase(arm, effort, difficulty, env_mode, root or f"{difficulty}_{env_mode}_{effort}_seed0")


def main_table_phases(arm: str) -> list[Phase]:
    return [make_phase(arm, effort, difficulty, env_mode, root)
            for root, difficulty, env_mode, efforts in MAIN_TABLE_PIPELINE for effort in efforts]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, default=str) + "\n")


def arm_by_key(key: str) -> dict[str, Any]:
    return next(arm for arm in campaign.ARMS if arm["key"] == key)


def phase_root(safety_root: Path, phase: Phase) -> Path:
    return safety_root / phase.root


def phase_suite(safety_root: Path, phase: Phase) -> Path:
    return phase_root(safety_root, phase) / campaign.model_slug(arm_by_key(phase.arm)["model"])


def phase_progress(safety_root: Path, phase: Phase, expected: int) -> dict[str, int]:
    counts = {"expected": expected, "completed": 0, "needs_attention": 0, "other": 0}
    for status_path in phase_suite(safety_root, phase).glob(f"cells/{phase.condition}/*/task_*/status.json"):
        try:
            state = json.loads(status_path.read_text()).get("state")
        except (OSError, json.JSONDecodeError):
            state = None
        key = state if state in ("completed", "needs_attention") else "other"
        counts[key] += 1
    return counts


def stop_files(safety_root: Path, phases: list[Phase]) -> list[Path]:
    paths = [phase_root(safety_root, phase) / "CLAUDE_CODE_STOP.json" for phase in phases]
    paths += [phase_suite(safety_root, phase) / "CLAUDE_CODE_STOP.json" for phase in phases]
    return sorted({path for path in paths if path.exists()})


def stop_reason(path: Path) -> str:
    try:
        return str(json.loads(path.read_text()).get("reason"))
    except (OSError, json.JSONDecodeError, AttributeError):
        return "unreadable"


def write_stop(path: Path, reason: str, detail: Any) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"time": utc_now().isoformat(), "reason": reason,
                                    "detail": detail, "pid": os.getpid()}, default=str) + "\n")


def _reset(value: str | None, now: datetime) -> datetime:
    if not value:
        return now + timedelta(hours=1)
    return datetime.fromisoformat(value) + RESET_MARGIN


def decide(
    phases: list[Phase],
    progress: dict[Phase, dict[str, int]],
    reasons: list[str],
    usage: dict[str, Any],
    cargs: argparse.Namespace,
    baseline_used_minor: int,
    now: datetime,
) -> dict[str, Any]:
    """Pure policy for one lane: what to do next given progress, stop files, and usage."""
    remaining = [p for p in phases if progress[p]["completed"] < progress[p]["expected"]]
    if not remaining:
        return {"action": "done"}
    halting = [reason for reason in reasons if reason not in USAGE_STOP_REASONS]
    if halting:
        return {"action": "halt", "reason": f"non-usage stop file(s): {halting}"}
    credits = usage["usage_credits"]
    if credits["enabled"] and not cargs.acknowledge_enabled_usage_credits:
        return {"action": "halt", "reason": "usage credits are enabled on the account"}
    if credits["spend_limit_reached"] or credits["used_minor"] > baseline_used_minor:
        return {"action": "halt", "reason": "usage credits were spent"}
    if usage.get("subscription_type") not in cargs.allowed_subscriptions:
        return {"action": "halt", "reason": f"subscription_type={usage.get('subscription_type')!r}"}
    waits = []
    if 100.0 - usage["five_hour"]["utilization"] <= cargs.min_five_hour_remaining:
        waits.append(("five_hour", _reset(usage["five_hour"].get("resets_at"), now)))
    if 100.0 - usage["seven_day"]["utilization"] <= cargs.min_weekly_remaining:
        waits.append(("seven_day", _reset(usage["seven_day"].get("resets_at"), now)))
    if waits:
        return {"action": "sleep", "until": max(when for _, when in waits),
                "reason": f"account windows low: {[name for name, _ in waits]}"}
    runnable = []
    scoped_waits = []
    for phase in remaining:
        scope = arm_by_key(phase.arm)["usage_scope"]
        if campaign.scope_violations(usage, cargs, scope):
            entry = next((e for e in usage["model_scoped"] if e["display_name"] == scope), {})
            scoped_waits.append(_reset(entry.get("resets_at"), now))
        else:
            runnable.append(phase)
    if not runnable:
        return {"action": "sleep", "until": min(scoped_waits), "reason": "model-scoped windows low"}
    return {"action": "run", "phase": runnable[0]}


@dataclass
class Lane:
    name: str
    phases: list[Phase]
    gpus: list[int]
    ports: list[int]
    process: subprocess.Popen | None = None
    running_phase: Phase | None = None
    launch_tag: str | None = None
    events_offset: int = 0
    completed_at_launch: int = 0
    adopted: dict[str, Any] | None = None
    next_check: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=timezone.utc))
    launches: dict[Phase, int] = field(default_factory=dict)
    stalls: int = 0
    finished: str | None = None   # "done" or "halted"
    last_decision: str | None = None

    @property
    def busy(self) -> bool:
        return self.process is not None or self.adopted is not None


def campaign_command(args: argparse.Namespace, lane: Lane, phase: Phase, max_attempts: int) -> list[str]:
    gpus, ports = lane.gpus, lane.ports
    later_workers = getattr(args, "later_phase_workers", None)
    if later_workers and phase != lane.phases[0]:
        # The lane's first phase keeps every GPU; later phases hand the tail
        # GPUs to other campaigns.
        gpus, ports = gpus[:later_workers], ports[:later_workers]
    command = [
        sys.executable, "-u", str(CAMPAIGN_RUNNER), "--mode", "run", "--confirm-claude-code-run",
        "--results-root", str(phase_root(args.safety_root, phase)),
        "--arms", phase.arm, "--efforts", phase.effort, "--env-modes", phase.env_mode,
        "--difficulties", phase.difficulty,
        "--workers", str(len(gpus)), "--gpus", *map(str, gpus), "--ports", *map(str, ports),
        "--max-attempts", str(max_attempts), "--claude-executable", args.claude_executable,
    ]
    for name in GUARD_THRESHOLDS:
        value = getattr(args, name, None)
        if value is not None:
            command += [f"--{name.replace('_', '-')}", str(value)]
    if args.task_ids:
        command += ["--task-ids", *map(str, args.task_ids)]
    return command


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _alive(pid: int) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rsplit(")", 1)[-1].split()[0] != "Z"


class Supervisor:
    def __init__(self, args: argparse.Namespace, *,
                 read_usage: Callable[[], dict[str, Any]] | None = None,
                 auth_ok: Callable[[], bool] | None = None,
                 launcher: Callable[..., subprocess.Popen] | None = None,
                 now: Callable[[], datetime] = utc_now) -> None:
        self.args = args
        self.cargs = campaign.parse_args(["--gpus", "0", "--ports", "1", "--workers", "1"])
        for name in GUARD_THRESHOLDS:
            if getattr(args, name, None) is not None:
                setattr(self.cargs, name, getattr(args, name))
        self.expected = len(args.task_ids) if args.task_ids else campaign.TASK_COUNT
        self.events = args.safety_root / "supervisor_events.jsonl"
        self.halt_file = args.safety_root / "SUPERVISOR_STOP"
        self.lanes = args.lanes
        self.read_usage = read_usage or (lambda: campaign.read_plan_usage(args.claude_executable))
        self.auth_ok = auth_ok or (lambda: "auth" in campaign.cli_preflight(
            args.claude_executable, tuple(self.cargs.allowed_subscriptions)))
        self.launcher = launcher or self._launch
        self.now = now
        self.baseline: int | None = None
        self.usage_failures = 0

    def log(self, event: str, **fields: Any) -> None:
        append_jsonl(self.events, {"time": self.now(), "event": event, **fields})

    # ----------------------------------------------------------- lane pieces
    def progress(self, lane: Lane) -> dict[Phase, dict[str, int]]:
        return {phase: phase_progress(self.args.safety_root, phase, self.expected) for phase in lane.phases}

    def _launch(self, command: list[str], log_path: Path) -> subprocess.Popen:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stream = log_path.open("ab", buffering=0)
        return subprocess.Popen(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                start_new_session=True)

    def guard_adopted(self, lane: Lane, usage: dict[str, Any] | None, auth_ok: bool) -> None:
        adopted = lane.adopted
        assert adopted is not None
        if not _alive(adopted["pid"]):
            self.log("adopted_exit", lane=lane.name, pid=adopted["pid"])
            lane.adopted = None
            return
        if adopted.get("stopping_since") is None:
            phase = adopted["phase"]
            if usage is None:
                problems = ["plan usage unavailable"]
                reason = "usage_guard_unavailable"
            else:
                scope = arm_by_key(phase.arm)["usage_scope"]
                account = campaign.account_violations(usage, self.cargs, self.baseline or 0)
                scoped = campaign.scope_violations(usage, self.cargs, scope)
                problems = account + scoped + ([] if auth_ok else ["auth route check failed"])
                reason = "model_scope_guard" if scoped and not account and auth_ok else "account_guard"
            if problems:
                write_stop(phase_suite(self.args.safety_root, phase) / "CLAUDE_CODE_STOP.json", reason, problems)
                _signal_group(adopted["pid"], signal.SIGINT)
                adopted["stopping_since"] = self.now()
                self.log("adopted_guard_stop", lane=lane.name, pid=adopted["pid"], reason=reason, problems=problems)
            return
        waited = self.now() - adopted["stopping_since"]
        if waited > STOP_GRACE + timedelta(seconds=30):
            _signal_group(adopted["pid"], signal.SIGKILL)
        elif waited > STOP_GRACE:
            _signal_group(adopted["pid"], signal.SIGTERM)

    def reap(self, lane: Lane) -> None:
        assert lane.process is not None and lane.running_phase is not None
        code = lane.process.poll()
        if code is None:
            return
        phase = lane.running_phase
        started = False
        events_file = phase_root(self.args.safety_root, phase) / "campaign_events.jsonl"
        if events_file.exists():
            with events_file.open(encoding="utf-8") as stream:
                stream.seek(lane.events_offset)
                for line in stream:
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("event") == "campaign_start" and event.get("tag") == lane.launch_tag:
                        started = True
        completed = self.progress(lane)[phase]["completed"]
        progressed = completed > lane.completed_at_launch
        self.log("campaign_exit", lane=lane.name, phase=phase.label, return_code=code,
                 campaign_started=started, completed=completed)
        # A guard stop (75) is expected; repeated exits with no new completed
        # cells for any other reason indicate a problem that retries won't fix.
        lane.stalls = 0 if (progressed or code == 75) else lane.stalls + 1
        lane.process = lane.running_phase = lane.launch_tag = None
        if lane.stalls >= 3:
            lane.finished = "halted"
            self.log("halt", lane=lane.name, reason=f"{phase.label} exited {lane.stalls} times without progress")

    def step_lane(self, lane: Lane, usage: dict[str, Any]) -> None:
        if self.now() < lane.next_check:
            return
        progress = self.progress(lane)
        stops = stop_files(self.args.safety_root, lane.phases)
        # A shared root's global stop file may belong to a campaign in another
        # lane that is still shutting down; never archive while one is busy.
        decision = decide(lane.phases, progress, [stop_reason(p) for p in stops], usage,
                          self.cargs, self.baseline or 0, self.now())
        key = json.dumps(decision, default=str, sort_keys=True)
        if key != lane.last_decision:
            lane.last_decision = key
            self.log("decision", lane=lane.name,
                     decision={**decision, "phase": decision["phase"].label} if "phase" in decision else decision,
                     progress={phase.label: v["completed"] for phase, v in progress.items()},
                     usage={"five_hour": usage["five_hour"]["utilization"],
                            "seven_day": usage["seven_day"]["utilization"],
                            "scoped": {e["display_name"]: e["utilization"] for e in usage["model_scoped"]}})
        action = decision["action"]
        if action == "done":
            lane.finished = "done"
            self.log("lane_done", lane=lane.name)
            return
        if action == "halt":
            lane.finished = "halted"
            self.log("halt", lane=lane.name, reason=decision["reason"])
            return
        if action == "sleep":
            lane.next_check = decision["until"]
            return
        phase = decision["phase"]
        root = phase_root(self.args.safety_root, phase)
        sharing = [other for other in self.lanes if other is not lane and other.busy and any(
            phase_root(self.args.safety_root, p) == root for p in (
                [other.running_phase] if other.running_phase else [other.adopted["phase"]] if other.adopted else []))]
        for path in stops:
            if path.parent == root and sharing:
                continue
            archived = path.with_name(f"{path.stem}.archived_{self.now():%Y%m%dT%H%M%S}.json")
            path.rename(archived)
            self.log("stop_file_archived", lane=lane.name, path=str(archived))
        if any(path.exists() for path in stop_files(self.args.safety_root, [phase])):
            lane.next_check = self.now() + timedelta(minutes=5)   # wait for the other lane to settle
            return
        lane.launches[phase] = lane.launches.get(phase, 0) + 1
        max_attempts = min(MAX_ATTEMPTS_CAP, self.args.base_max_attempts + 2 * (lane.launches[phase] - 1))
        events_file = root / "campaign_events.jsonl"
        lane.events_offset = events_file.stat().st_size if events_file.exists() else 0
        lane.completed_at_launch = progress[phase]["completed"]
        lane.launch_tag = campaign.campaign_tag([arm_by_key(phase.arm)], [phase.effort], [phase.difficulty])
        command = campaign_command(self.args, lane, phase, max_attempts)
        lane.process = self.launcher(command, root / f"supervisor_{lane.name}_campaign.log")
        lane.running_phase = phase
        self.log("launch", lane=lane.name, phase=phase.label, root=phase.root, pid=lane.process.pid,
                 max_attempts=max_attempts, command=command)

    def stop_everything(self) -> None:
        for lane in self.lanes:
            if lane.process is not None and lane.running_phase is not None:
                write_stop(phase_root(self.args.safety_root, lane.running_phase) / "CLAUDE_CODE_STOP.json",
                           "supervisor_stop", "SUPERVISOR_STOP present")
            if lane.adopted is not None:
                write_stop(phase_suite(self.args.safety_root, lane.adopted["phase"]) / "CLAUDE_CODE_STOP.json",
                           "supervisor_stop", "SUPERVISOR_STOP present")
                _signal_group(lane.adopted["pid"], signal.SIGINT)

    # ------------------------------------------------------------------ loop
    def cycle(self) -> bool:
        """One supervision pass; returns False when the supervisor should exit."""
        if self.halt_file.exists():
            self.log("halt", reason="SUPERVISOR_STOP")
            self.stop_everything()
            return False
        try:
            usage = self.read_usage()
            self.usage_failures = 0
        except Exception as exc:  # never run blind
            self.usage_failures += 1
            self.log("usage_unavailable", error=str(exc), consecutive=self.usage_failures)
            usage = None
        if usage is not None and self.baseline is None:
            self.baseline = usage["usage_credits"]["used_minor"]
        auth_ok = True
        if any(lane.adopted for lane in self.lanes):
            try:
                auth_ok = self.auth_ok()
            except Exception:
                auth_ok = False
        for lane in self.lanes:
            if lane.adopted is not None:
                self.guard_adopted(lane, usage if self.usage_failures < 3 else None, auth_ok)
            if lane.process is not None:
                self.reap(lane)
            if lane.busy or lane.finished or usage is None:
                continue
            self.step_lane(lane, usage)
        if all(lane.finished and not lane.busy for lane in self.lanes):
            self.log("supervisor_exit", lanes={lane.name: lane.finished for lane in self.lanes})
            return False
        return True

    def run(self) -> int:
        self.log("supervisor_start", pid=os.getpid(), lanes={
            lane.name: {"phases": [f"{p.root}/{p.label}" for p in lane.phases], "gpus": lane.gpus,
                        "ports": lane.ports,
                        "adopted": lane.adopted and {"pid": lane.adopted["pid"],
                                                     "phase": lane.adopted["phase"].label}}
            for lane in self.lanes})
        while self.cycle():
            time.sleep(self.args.poll_seconds)
        while any(lane.busy for lane in self.lanes):   # let stopped campaigns wind down
            for lane in self.lanes:
                if lane.process is not None and lane.process.poll() is not None:
                    lane.process = None
                if lane.adopted is not None and not _alive(lane.adopted["pid"]):
                    lane.adopted = None
            time.sleep(5)
        return 0 if all(lane.finished == "done" for lane in self.lanes) else 3


def parse_lane(spec: str, default_difficulty: str = "easy") -> Lane:
    """NAME=PHASES@gpu[,gpu...]@port[,port...]; see the module docstring for PHASES."""
    name, _, rest = spec.partition("=")
    phases_text, gpus_text, ports_text = rest.split("@")
    arm_keys = {arm["key"] for arm in campaign.ARMS}
    phases: list[Phase] = []
    for item in phases_text.split(","):
        parts = item.split(":")
        if parts[0] == "main_table":
            if not 2 <= len(parts) <= 3 or parts[1] not in arm_keys:
                raise ValueError(f"bad pipeline {item!r}")
            selected = main_table_phases(parts[1])
            if len(parts) == 3:
                prefixes = parts[2].split("+")
                selected = [phase for phase in selected if phase.root.startswith(tuple(prefixes))]
                if not selected:
                    raise ValueError(f"no main-table phase root starts with {parts[2]!r}")
            phases += selected
            continue
        if not 2 <= len(parts) <= 4:
            raise ValueError(f"bad phase {item!r}")
        arm, effort = parts[0], parts[1]
        difficulty = parts[2] if len(parts) > 2 else default_difficulty
        env_mode = parts[3] if len(parts) > 3 else "realtime"
        if arm not in arm_keys or effort not in campaign.ALL_EFFORTS or difficulty not in campaign.ALL_DIFFICULTIES:
            raise ValueError(f"bad phase {item!r}")
        phases.append(make_phase(arm, effort, difficulty, env_mode))
    for phase in phases:
        if phase.env_mode not in arm_by_key(phase.arm)["env_modes"]:
            raise ValueError(f"{phase.arm} has no {phase.env_mode} mode")
    if len(set(phases)) != len(phases):
        raise ValueError(f"lane {name}: duplicate phase")
    gpus = [int(value) for value in gpus_text.split(",")]
    ports = [int(value) for value in ports_text.split(",")]
    if len(gpus) != len(ports):
        raise ValueError(f"lane {name}: one base port per UE worker (GPU entry) is required")
    return Lane(name=name, phases=phases, gpus=gpus, ports=ports)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--safety-root", type=Path, default=DEFAULT_SAFETY_ROOT)
    parser.add_argument("--lane", action="append", dest="lane_specs",
                        help="NAME=PHASES@gpus@ports (repeatable; phases run in the given order)")
    parser.add_argument("--adopt", action="append", default=[],
                        help="PID=LANE:ARM:EFFORT: guard an already-running coordinator group until it exits")
    parser.add_argument("--task-ids", nargs="+", type=int)
    parser.add_argument("--difficulty", choices=campaign.ALL_DIFFICULTIES, default="easy",
                        help="Difficulty for arm:effort phases that do not name one.")
    parser.add_argument("--base-max-attempts", type=int, default=2)
    parser.add_argument("--later-phase-workers", type=int, default=None,
                        help="Run every phase after a lane's first phase on only the lane's first N GPUs/ports, "
                             "freeing the rest for other campaigns once the first phase is done.")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    for name in GUARD_THRESHOLDS:
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=None,
                            help="Override the campaign guard threshold (percent remaining) for decisions and launches.")
    parser.add_argument("--claude-executable", default=campaign.DEFAULT_EXECUTABLE)
    parser.add_argument("--confirm-claude-code-run", action="store_true")
    parser.add_argument("--plan", action="store_true", help="Print each lane's next decision and exit.")
    args = parser.parse_args(argv)
    args.safety_root = args.safety_root.resolve()
    specs = args.lane_specs or ["fable=fable_5_1:low@2@20000", "sonnet=sonnet_5:low,sonnet_5:high@1,5@21000,21500"]
    try:
        args.lanes = [parse_lane(spec, args.difficulty) for spec in specs]
    except ValueError as exc:
        parser.error(str(exc))
    ports = [port for lane in args.lanes for port in lane.ports]
    if len(set(ports)) != len(ports):
        parser.error("base ports must be disjoint across UE workers")
    if min(abs(a - b) for a in ports for b in ports if a != b) < 300 if len(ports) > 1 else False:
        parser.error("base ports must be >=300 apart (each UE restart advances the port)")
    phases = [phase for lane in args.lanes for phase in lane.phases]
    if len(set(phases)) != len(phases):
        parser.error("a phase may belong to only one lane")
    by_name = {lane.name: lane for lane in args.lanes}
    for spec in args.adopt:
        pid_text, _, target = spec.partition("=")
        lane_name, arm, effort = target.split(":")
        match = next((phase for phase in by_name[lane_name].phases
                      if (phase.arm, phase.effort) == (arm, effort)), None) if lane_name in by_name else None
        if match is None:
            parser.error(f"--adopt {spec}: unknown lane or phase")
        by_name[lane_name].adopted = {"pid": int(pid_text), "phase": match, "stopping_since": None}
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    supervisor = Supervisor(args)
    if args.plan:
        usage = supervisor.read_usage()
        supervisor.baseline = usage["usage_credits"]["used_minor"]
        report = {}
        for lane in args.lanes:
            progress = supervisor.progress(lane)
            stops = stop_files(args.safety_root, lane.phases)
            decision = decide(lane.phases, progress, [stop_reason(p) for p in stops], usage,
                              supervisor.cargs, supervisor.baseline, utc_now())
            if "phase" in decision:
                decision = {**decision, "phase": f"{decision['phase'].root}/{decision['phase'].label}"}
            report[lane.name] = {
                "gpus": lane.gpus, "ports": lane.ports,
                "adopted": lane.adopted and {"pid": lane.adopted["pid"], "alive": _alive(lane.adopted["pid"])},
                "progress": {f"{p.root}/{p.label}": v for p, v in progress.items()},
                "stop_files": {str(p): stop_reason(p) for p in stops},
                "decision": decision,
            }
        print(json.dumps({"usage": {"five_hour": usage["five_hour"], "seven_day": usage["seven_day"],
                                    "model_scoped": usage["model_scoped"]}, "lanes": report}, indent=2, default=str))
        return 0
    if not args.confirm_claude_code_run:
        print("Refusing to supervise: pass --confirm-claude-code-run.", file=sys.stderr)
        return 2
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
