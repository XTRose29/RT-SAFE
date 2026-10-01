"""Tests for the Claude Code lane supervisor (fake campaigns; no model calls)."""

from __future__ import annotations

import json
import signal
import subprocess
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from evaluation import run_claude_code_supervisor as supervisor

NOW = datetime(2026, 9, 14, 8, 0, tzinfo=timezone.utc)
WEEKLY_RESET = "2026-09-15T10:00:00+00:00"
FIVE_HOUR_RESET = "2026-09-14T10:10:00+00:00"
PHASES = [supervisor.make_phase("fable_5_1", "low"), supervisor.make_phase("sonnet_5", "low"),
          supervisor.make_phase("sonnet_5", "high")]


def usage(five=40.0, weekly=80.0, fable=85.0, credits=False, spent=0):
    return {
        "subscription_type": "max",
        "five_hour": {"utilization": five, "resets_at": FIVE_HOUR_RESET},
        "seven_day": {"utilization": weekly, "resets_at": WEEKLY_RESET},
        "model_scoped": [{"display_name": "Fable", "utilization": fable, "resets_at": WEEKLY_RESET}],
        "usage_credits": {"enabled": credits, "spend_limit_reached": False, "used_minor": spent,
                          "exponent": 2, "user_disabled": not credits},
    }


def progress(phases=PHASES, **completed):
    return {phase: {"expected": 36, "completed": completed.get(f"{phase.arm}_{phase.effort}", 0),
                    "needs_attention": 0, "other": 0} for phase in phases}


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.cargs = supervisor.campaign.parse_args(["--gpus", "2", "--ports", "20000", "--workers", "1"])

    def decide(self, prog, reasons=(), use=None, phases=PHASES):
        return supervisor.decide(phases, prog, list(reasons), use or usage(), self.cargs, 0, NOW)

    def test_runs_highest_priority_incomplete_phase(self):
        self.assertEqual(self.decide(progress()), {"action": "run", "phase": PHASES[0]})
        self.assertEqual(self.decide(progress(fable_5_1_low=36))["phase"], PHASES[1])
        done = progress(fable_5_1_low=36, sonnet_5_low=36, sonnet_5_high=36)
        self.assertEqual(self.decide(done), {"action": "done"})

    def test_fable_scope_limit_skips_to_sonnet(self):
        self.assertEqual(self.decide(progress(), use=usage(fable=96.0))["phase"], PHASES[1])
        decision = self.decide(progress(), use=usage(fable=96.0), phases=PHASES[:1])
        self.assertEqual(decision["action"], "sleep")
        self.assertEqual(decision["until"], datetime.fromisoformat(WEEKLY_RESET) + supervisor.RESET_MARGIN)

    def test_account_windows_sleep_until_their_reset(self):
        weekly = self.decide(progress(), use=usage(weekly=91.0))
        self.assertEqual(weekly["until"], datetime.fromisoformat(WEEKLY_RESET) + supervisor.RESET_MARGIN)
        five = self.decide(progress(), use=usage(five=85.0))
        self.assertEqual(five["until"], datetime.fromisoformat(FIVE_HOUR_RESET) + supervisor.RESET_MARGIN)

    def test_usage_stops_resume_but_safety_stops_halt(self):
        for reason in ("model_scope_guard", "account_guard", "global_stop_file", "supervisor_stop", "ClaudeCodeUsageLimit"):
            self.assertEqual(self.decide(progress(), reasons=[reason])["action"], "run", reason)
        for reason in ("ClaudeCodeModelMismatch", "ClaudeCodeEffortMismatch", "ClaudeCodeAuthError", "unreadable", "None"):
            self.assertEqual(self.decide(progress(), reasons=[reason])["action"], "halt", reason)

    def test_usage_credits_halt_instead_of_waiting(self):
        self.assertEqual(self.decide(progress(), use=usage(credits=True))["action"], "halt")
        self.assertEqual(self.decide(progress(), use=usage(spent=1))["action"], "halt")


class MainTablePipelineTests(unittest.TestCase):
    def test_main_table_expands_in_the_other_models_order(self):
        lane = supervisor.parse_lane("fable=main_table:fable_5_1@3,7@22000,22500")
        self.assertEqual([(p.root, p.effort, p.difficulty, p.env_mode) for p in lane.phases], [
            ("01_hard_realtime_default", "high", "hard", "realtime"),
            ("04_hard_static_default", "high", "hard", "static"),
            ("05_hard_realtime_efforts", "low", "hard", "realtime"),
            ("03_easy_realtime_default", "high", "easy", "realtime"),
            ("06_hard_realtime_upper", "max", "hard", "realtime"),
            ("02_medium_realtime_default", "high", "medium", "realtime"),
        ])
        self.assertEqual((lane.gpus, lane.ports), ([3, 7], [22000, 22500]))
        self.assertEqual(lane.phases[1].condition, "hard_static_instructional_high")

    def test_campaign_command_carries_the_phase_selection(self):
        args = supervisor.parse_args(["--safety-root", "/tmp/claude_main_table",
                                      "--lane", "fable=main_table:fable_5_1@3,7@22000,22500",
                                      "--lane", "sonnet=main_table:sonnet_5@3,5@23000,23500"])
        fable = args.lanes[0]
        command = supervisor.campaign_command(args, fable, fable.phases[1], 2)

        def flag(name):
            return command[command.index(name) + 1]

        self.assertTrue(flag("--results-root").endswith("04_hard_static_default"))
        self.assertEqual((flag("--arms"), flag("--efforts"), flag("--env-modes"), flag("--difficulties")),
                         ("fable_5_1", "high", "static", "hard"))
        self.assertEqual(command[command.index("--gpus") + 1:command.index("--gpus") + 3], ["3", "7"])
        self.assertEqual(flag("--workers"), "2")

    def test_later_phases_release_tail_gpus(self):
        args = supervisor.parse_args(["--lane", "fable=main_table:fable_5_1@3,7,2@22000,22500,24500",
                                      "--later-phase-workers", "2"])
        lane = args.lanes[0]

        def placement(phase):
            command = supervisor.campaign_command(args, lane, phase, 2)
            start = command.index("--gpus")
            return command[command.index("--workers") + 1], command[start + 1:command.index("--ports")]

        self.assertEqual(placement(lane.phases[0]), ("3", ["3", "7", "2"]))
        self.assertEqual(placement(lane.phases[1]), ("2", ["3", "7"]))

    def test_lanes_share_phase_roots_but_not_suites(self):
        fable = supervisor.main_table_phases("fable_5_1")[0]
        sonnet = supervisor.main_table_phases("sonnet_5")[0]
        root = Path("/tmp/claude_main_table")
        self.assertEqual(supervisor.phase_root(root, fable), supervisor.phase_root(root, sonnet))
        self.assertNotEqual(supervisor.phase_suite(root, fable), supervisor.phase_suite(root, sonnet))

    def test_main_table_subset_selects_phase_root_prefixes(self):
        only_01 = supervisor.parse_lane("s=main_table:sonnet_5:01@1@23000")
        self.assertEqual([(p.root, p.effort) for p in only_01.phases], [("01_hard_realtime_default", "high")])
        subset = supervisor.parse_lane("s=main_table:sonnet_5:01+05@1@23000")
        self.assertEqual([(p.root, p.effort) for p in subset.phases], [
            ("01_hard_realtime_default", "high"), ("05_hard_realtime_efforts", "low")])
        with self.assertRaises(ValueError):
            supervisor.parse_lane("s=main_table:sonnet_5:09@1@23000")

    def test_guard_thresholds_reach_campaign_commands(self):
        args = supervisor.parse_args(["--lane", "s=main_table:sonnet_5:01@1@23000", "--min-weekly-remaining", "4"])
        command = supervisor.campaign_command(args, args.lanes[0], args.lanes[0].phases[0], 2)
        self.assertEqual(command[command.index("--min-weekly-remaining") + 1], "4.0")
        self.assertNotIn("--min-five-hour-remaining", command)


class FakeProcess:
    _next_pid = 900000

    def __init__(self):
        FakeProcess._next_pid += 1
        self.pid = FakeProcess._next_pid
        self.code = None

    def poll(self):
        return self.code


class SupervisorCycleTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary.name)
        self.usage = usage()
        self.clock = NOW
        self.launched = []
        self.children = []

    def tearDown(self):
        for child in self.children:
            if child.poll() is None:
                child.kill()
                child.wait()
        self._temporary.cleanup()

    def make(self, *extra, lanes=("fable=fable_5_1:low@2@20000", "sonnet=sonnet_5:low,sonnet_5:high@1,5@21000,21500")):
        lane_args = [value for spec in lanes for value in ("--lane", spec)]
        args = supervisor.parse_args(["--safety-root", str(self.root), "--task-ids", "35", *lane_args, *extra])

        def launcher(command, log_path):
            process = FakeProcess()
            self.launched.append((command, process))
            return process

        return supervisor.Supervisor(args, read_usage=lambda: self.usage, auth_ok=lambda: True,
                                     launcher=launcher, now=lambda: self.clock)

    def complete(self, phase):
        path = supervisor.phase_suite(self.root, phase) / f"cells/{phase.condition}/map5_20roads/task_035/status.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"state": "completed"}))

    def flag(self, command, name):
        return command[command.index(name) + 1]

    def events(self):
        path = self.root / "supervisor_events.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_fable_paused_by_scope_while_sonnet_low_then_high_run(self):
        self.usage = usage(fable=96.0)
        sup = self.make()
        self.assertTrue(sup.cycle())
        self.assertEqual(len(self.launched), 1)
        command, process = self.launched[0]
        self.assertEqual((self.flag(command, "--arms"), self.flag(command, "--efforts")), ("sonnet_5", "low"))
        self.assertEqual(command[command.index("--gpus") + 1:command.index("--gpus") + 3], ["1", "5"])
        self.assertTrue(self.flag(command, "--results-root").endswith("easy_realtime_low_seed0"))
        fable = sup.lanes[0]
        self.assertEqual(fable.next_check, datetime.fromisoformat(WEEKLY_RESET) + supervisor.RESET_MARGIN)
        # Sonnet low finishes -> Sonnet high launches on the same lane GPUs.
        self.complete(PHASES[1])
        process.code = 0
        self.assertTrue(sup.cycle())
        command, process = self.launched[1]
        self.assertEqual((self.flag(command, "--arms"), self.flag(command, "--efforts")), ("sonnet_5", "high"))
        self.assertTrue(self.flag(command, "--results-root").endswith("easy_realtime_high_seed0"))
        # After the Fable reset, the Fable lane resumes on its own GPU.
        self.usage = usage(fable=1.0, weekly=5.0)
        self.clock = fable.next_check + timedelta(minutes=1)
        self.assertTrue(sup.cycle())
        command, _ = self.launched[2]
        self.assertEqual((self.flag(command, "--arms"), self.flag(command, "--gpus")), ("fable_5_1", "2"))

    def test_main_table_lanes_start_with_hard_realtime_default(self):
        self.usage = usage(fable=96.0)
        sup = self.make(lanes=("fable=main_table:fable_5_1@3,7@22000,22500",
                               "sonnet=main_table:sonnet_5@2,3@23000,23500"))
        sup.cycle()
        self.assertEqual(len(self.launched), 1, "Fable is scope-limited, so only Sonnet launches")
        command, process = self.launched[0]
        self.assertTrue(self.flag(command, "--results-root").endswith("01_hard_realtime_default"))
        self.assertEqual((self.flag(command, "--arms"), self.flag(command, "--efforts"),
                          self.flag(command, "--difficulties"), self.flag(command, "--env-modes")),
                         ("sonnet_5", "high", "hard", "realtime"))
        # Phase 01 done -> phase 04 (hard static, default effort) follows on the same lane.
        self.complete(sup.lanes[1].phases[0])
        process.code = 0
        sup.cycle()
        command, _ = self.launched[1]
        self.assertTrue(self.flag(command, "--results-root").endswith("04_hard_static_default"))
        self.assertEqual((self.flag(command, "--env-modes"), self.flag(command, "--difficulties")), ("static", "hard"))

    def test_lower_weekly_buffer_lets_lanes_use_the_rest_of_the_week(self):
        self.usage = usage(weekly=91.0, fable=98.0)
        self.make().cycle()
        self.assertEqual(self.launched, [], "default 10% weekly buffer waits for the reset")
        sup = self.make("--min-weekly-remaining", "4")
        self.assertEqual(sup.cargs.min_weekly_remaining, 4.0)
        sup.cycle()
        self.assertEqual([self.flag(c, "--arms") for c, _ in self.launched], ["sonnet_5"],
                         "Sonnet runs on the remaining weekly allowance; Fable waits for its own window")
        self.assertEqual(self.flag(self.launched[0][0], "--min-weekly-remaining"), "4.0")

    def test_weekly_limit_pauses_every_lane(self):
        self.usage = usage(weekly=95.0)
        sup = self.make()
        sup.cycle()
        self.assertEqual(self.launched, [])
        self.assertTrue(all(lane.next_check > NOW for lane in sup.lanes))

    def test_safety_stop_halts_only_its_lane(self):
        stop = supervisor.phase_suite(self.root, PHASES[1]) / "CLAUDE_CODE_STOP.json"
        stop.parent.mkdir(parents=True)
        stop.write_text(json.dumps({"reason": "ClaudeCodeModelMismatch"}))
        sup = self.make()
        sup.cycle()
        self.assertEqual(sup.lanes[1].finished, "halted")
        self.assertEqual([self.flag(c, "--arms") for c, _ in self.launched], ["fable_5_1"])
        self.assertTrue(stop.exists(), "safety stop files are never archived")

    def test_usage_stop_files_are_archived_before_relaunch(self):
        stop = supervisor.phase_suite(self.root, PHASES[0]) / "CLAUDE_CODE_STOP.json"
        stop.parent.mkdir(parents=True)
        stop.write_text(json.dumps({"reason": "model_scope_guard"}))
        self.make().cycle()
        self.assertFalse(stop.exists())
        self.assertTrue(list(stop.parent.glob("CLAUDE_CODE_STOP.archived_*.json")))

    def test_adopted_coordinator_is_interrupted_at_the_fable_limit(self):
        child = subprocess.Popen(["sleep", "60"], start_new_session=True)
        self.children.append(child)
        sup = self.make("--adopt", f"{child.pid}=fable:fable_5_1:low")
        sup.cycle()   # within limits: keep running, Sonnet lane launches
        self.assertIsNone(child.poll())
        self.assertEqual([self.flag(c, "--arms") for c, _ in self.launched], ["sonnet_5"])
        self.usage = usage(fable=96.0)
        sup.cycle()
        stop = supervisor.phase_suite(self.root, PHASES[0]) / "CLAUDE_CODE_STOP.json"
        self.assertEqual(json.loads(stop.read_text())["reason"], "model_scope_guard")
        self.assertEqual(child.wait(timeout=5), -signal.SIGINT)
        sup.cycle()
        self.assertIsNone(sup.lanes[0].adopted)
        self.assertIn("adopted_exit", [e["event"] for e in self.events()])

    def test_supervisor_stop_file_stops_running_campaigns(self):
        sup = self.make()
        sup.cycle()
        (self.root / "SUPERVISOR_STOP").write_text("stop")
        self.assertFalse(sup.cycle())
        global_stop = self.root / "easy_realtime_low_seed0" / "CLAUDE_CODE_STOP.json"
        self.assertEqual(json.loads(global_stop.read_text())["reason"], "supervisor_stop")

    def test_repeated_exits_without_progress_halt_the_lane(self):
        sup = self.make()
        for _ in range(3):
            sup.cycle()
            for _, process in self.launched:
                process.code = 1
        sup.cycle()
        self.assertEqual(sup.lanes[0].finished, "halted")

    def test_lane_arguments_are_validated(self):
        # Lanes may share a GPU (several UE workers per GPU) as long as ports are apart.
        shared = supervisor.parse_args(["--lane", "a=fable_5_1:low@2@20000", "--lane", "b=sonnet_5:low@2@21000"])
        self.assertEqual([lane.gpus for lane in shared.lanes], [[2], [2]])
        with self.assertRaises(SystemExit):
            supervisor.parse_args(["--lane", "a=fable_5_1:low@2@20000", "--lane", "b=sonnet_5:low@1@20100"])
        with self.assertRaises(SystemExit):
            supervisor.parse_args(["--lane", "a=fable_5_1:static@2@20000"])
        with self.assertRaises(SystemExit):
            supervisor.parse_args(["--lane", "a=fable_5_1:high:extreme@2@20000"])
        with self.assertRaises(SystemExit):
            supervisor.parse_args(["--lane", "a=main_table:fable_5_1@2@20000", "--lane", "b=main_table:fable_5_1@3@21000"])
        static = supervisor.parse_args(["--lane", "a=fable_5_1:high:hard:static@2@20000"])
        self.assertEqual(static.lanes[0].phases[0].root, "hard_static_high_seed0")


if __name__ == "__main__":
    unittest.main()
