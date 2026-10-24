"""No model calls: exercise queueing, real planning, ownership and result filing."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import model_watch as watch
import model_watch_slack as slack
from evaluation import run_openai_frontier_matrix as campaign


def model():
    return {"id": "google/gemini-3.8-flash", "name": "Gemini 3.8 Flash",
            "architecture": {"input_modalities": ["text", "image"]},
            "pricing": {"prompt": "0.000002", "completion": "0.00001"},
            "reasoning": {"supported_efforts": ["low", "medium", "high"]}}


def record(task, effort, success):
    return dict(artifact_schema_version="execution_alignment_v4", task_id=task, model=model()["id"], reasoning_effort=effort,
                success=success, collision_count=0, passive_collision_count=0,
                fall_count=0, oil_count=0, water_count=0, red_light_violations_count=0,
                illegal_crossing_violations_count=0, decision_count=10, avg_response_time=2,
                benchmark_metrics=dict(success_indicator=int(success), collision_count=0,
                    passive_collision_count=0, active_collision_count=0,
                    safe_success_indicator=int(success), spl=.8 if success else 0))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        selected = []
        for name in ("map2_12roads", "map5_20roads"):
            selected.extend(dict(map=t.map_name, task=t.task_id, length_m=t.route_length_m)
                            for t in [t for t in campaign.all_tasks() if t.map_name == name][:5])
        manifest = dict(id="test-balanced", difficulty="hard", timing="realtime", seed=0, tasks=selected)
        watch.write(self.root / "selection.json", manifest)
        self.config = dict(repo=str(ROOT), state_dir=str(self.root / "state"),
            selection_manifest=str(self.root / "selection.json"), python=sys.executable,
            key_file=str(self.root / "not-read-secret"), ue_launcher=str(self.root / "not-started-engine"),
            workers=1, gpus=[6], ports=[19660], max_attempts=3,
            feed_url="https://feed.invalid", catalog_url="https://catalog.invalid",
            model_prefixes=["google/gemini-"], efforts=["low", "medium", "high"])
        self.feed = {"generated": watch.now(), "models": [{"name": "gemini-3-8-flash", "sources": {"openrouter": model()["id"]}}]}

    def fetch(self, url):
        return self.feed if url == self.config["feed_url"] else {"data": [model()]}

    def queue(self):
        with patch.object(watch, "fetch", side_effect=self.fetch):
            result = watch.scan(self.config)
        return watch.job_path(self.config, result["added"][0])

    def test_discovery_is_idempotent_and_real_plan_selects_only_ten_tasks(self):
        path = self.queue()
        plan = watch.read(path.parent / "plan.json")
        self.assertEqual(plan["rollout_count"], 30)
        self.assertEqual(len(plan["tasks"]), 10)
        self.assertEqual(plan["model"], model()["id"])
        self.assertEqual({c["effort"] for c in plan["conditions"]}, {"low", "medium", "high"})
        self.assertEqual(plan["planning"]["input_price_usd_per_million"], 2)
        self.assertFalse((path.parent / "suite").exists())
        with patch.object(watch, "fetch", side_effect=self.fetch):
            self.assertEqual(watch.scan(self.config)["added"], [])
        self.assertEqual(len(watch.status(self.config)), 1)
        self.assertEqual(watch.status(self.config)[0]["state"], "queued")

    def test_rules_exclude_batch_nonvision_and_unknown_reasoning(self):
        for change, expected in [({"id": "google/gemini-4-argon:batch"}, "skip"),
                                 ({"architecture": {"input_modalities": ["text"]}}, "skip"),
                                 ({"id": "google/gemini-4-next"}, "needs_profile"),
                                 ({"reasoning": {"supported_efforts": ["high"]}}, "needs_profile"),
                                 ({"pricing": {"prompt": "nan", "completion": "0"}}, "needs_profile")]:
            self.assertEqual(watch.decision(model() | change, self.config)[0], expected)

    def test_stale_feed_does_not_queue(self):
        self.feed["generated"] = "2000-01-01T00:00:00+00:00"
        with patch.object(watch, "fetch", side_effect=self.fetch), self.assertRaises(ValueError):
            watch.scan(self.config)
        self.assertEqual(watch.status(self.config), [])

    def test_concurrent_start_cannot_launch_and_completed_start_is_noop(self):
        path = self.queue()
        with watch.lock(Path(self.config["state_dir"]) / "run.lock"), patch.object(watch.subprocess, "Popen") as spawn:
            with self.assertRaises(BlockingIOError):
                watch.start(self.config, path.parent.name)
            spawn.assert_not_called()
        watch.write(path.parent / "status.json", {"state": "completed"})
        with patch.object(watch.subprocess, "Popen") as spawn:
            self.assertEqual(watch.start(self.config, path.parent.name)["state"], "completed")
            spawn.assert_not_called()

    def test_source_drift_and_started_refresh_are_rejected(self):
        path = self.queue()
        with patch.object(watch, "source_digest", return_value="changed"), self.assertRaisesRegex(ValueError, "source changed"):
            watch.check_job(self.config, path)
        (path.parent / "suite").mkdir()
        with self.assertRaisesRegex(ValueError, "immutable"):
            watch.refresh_job(self.config, path.parent.name)

    def test_incomplete_campaign_never_writes_ledger(self):
        path = self.queue()
        with patch.object(watch.subprocess, "check_output", return_value=json.dumps({"completed": 29, "fresh_validation_failures": []})), self.assertRaisesRegex(ValueError, "incomplete"):
            watch.file_results(self.config, path)
        self.assertFalse((Path(self.config["state_dir"]) / "ledger").exists())

    def test_filing_keeps_navigation_failures_and_is_idempotent(self):
        path = self.queue()
        job = watch.read(path)
        for effort in job["efforts"]:
            for i, task in enumerate(job["selection"]["tasks"]):
                cell = path.parent / "suite/cells" / f"hard_realtime_instructional_{effort}" / task["map"] / f"task_{task['task']:03d}"
                result = cell / "attempt_001/result.json"
                watch.write(result, record(task["task"], effort, i < 5))
                watch.write(cell / "status.json", {"state": "completed", "result_path": str(result)})
        with patch.object(watch.subprocess, "check_output", return_value=json.dumps({"completed": 30, "fresh_validation_failures": []})):
            result = watch.file_results(self.config, path)
            watch.file_results(self.config, path)
        self.assertEqual([r["success"] for r in result["effort"]], [50, 50, 50])
        self.assertEqual(len(watch.read(Path(self.config["state_dir"]) / "board.json")["models"]), 1)

    def test_slack_requires_both_owner_and_channel(self):
        settings = {"owner_user_id": "U_OWNER", "channel_id": "C_SAFETY"}
        self.assertTrue(slack.authorized({"user_id": "U_OWNER", "channel_id": "C_SAFETY"}, settings))
        self.assertFalse(slack.authorized({"user_id": "U_OTHER", "channel_id": "C_SAFETY"}, settings))
        self.assertFalse(slack.authorized({"user_id": "U_OWNER", "channel_id": "C_OTHER"}, settings))
        self.assertFalse(slack.authorized({}, {}))

    def pool_config(self):
        self.config.update(provider="account-pool", base_url="http://pool.invalid:9443/v1",
                           model_prefixes=["antigravity/gemini-4-"], efforts=["medium"])
        return [{"id": f"antigravity/gemini-4-argon-{effort}", "capabilities": ["chat"], "billable": False}
                for effort in self.config["efforts"]]

    def test_pool_discovery_uses_exact_routes_and_subscription_plans(self):
        models = self.pool_config()
        self.feed["models"][0]["name"] = "gemini-4-argon"
        with patch.object(watch, "fetch", side_effect=self.fetch), patch.object(watch, "catalog_data", return_value={"data": models}):
            result = watch.scan(self.config)
            self.assertEqual(watch.scan(self.config)["added"], [])
        self.assertEqual(len(result["added"]), 1)
        for job_id in result["added"]:
            path = watch.job_path(self.config, job_id)
            job = watch.read(path)
            plan = watch.read(path.parent / "plan.json")
            self.assertEqual(plan["rollout_count"], 10)
            self.assertEqual(plan["provider"], "account-pool")
            self.assertFalse(plan["allow_paid"])
            self.assertIsNone(plan["planning"]["all_caps_estimate_usd"])
            self.assertEqual(plan["planning"]["billing_mode"], "account_pool_subscription")
            self.assertTrue(job["model"].endswith("-" + job["efforts"][0]))
            with patch.dict(watch.os.environ, {"OPENAI_API_KEY": "other", "OPENROUTER_API_KEY": "other", "POOL_KEY": "other"}):
                _, environment = watch.campaign(self.config, path, "plan")
                self.assertFalse(any(k in environment for k in ["POOL_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"]))
        message = slack.blocks(watch.status(self.config)[0])
        self.assertIn("subscription quota", message[0]["text"]["text"])

    def test_all_three_pinned_pool_families_and_unknown_release(self):
        from llm.account_pool import PINNED_ROUTES
        self.pool_config()
        self.config['model_prefixes'] = ['antigravity/gemini-', 'antigravity/claude-', 'codex/gpt-']
        models = [{'id': route, 'capabilities': ['chat'], 'billable': False} for route in PINNED_ROUTES]
        models.append({'id': 'codex/gpt-new-model', 'capabilities': ['chat'], 'billable': False})
        self.feed['models'] = [{'name': name, 'sources': {}} for name in
                              ['gemini-4-argon', 'claude-sonnet-5-5', 'gpt-6-astra', 'gpt-new-model']]
        with patch.object(watch, 'fetch', return_value=self.feed), patch.object(watch, 'catalog_data', return_value={'data': models}):
            result = watch.scan(self.config)
        self.assertEqual(len(result['added']), 3)
        self.assertEqual(watch.read(Path(self.config['state_dir']) / 'seen.json')['codex/gpt-new-model']['state'], 'needs_profile')

    def test_twenty_task_selection_is_planned_and_filed_without_dropping_failures(self):
        manifest = watch.read(self.config['selection_manifest'])
        for name in ('map3_15roads', 'map4_18roads'):
            manifest['tasks'].extend(dict(map=t.map_name, task=t.task_id, length_m=t.route_length_m)
                                    for t in [t for t in campaign.all_tasks() if t.map_name == name][:5])
        watch.write(self.config['selection_manifest'], manifest)
        self.config['efforts'] = ['medium']
        path = self.queue()
        self.assertEqual(watch.read(path.parent / 'plan.json')['rollout_count'], 20)
        for i, task in enumerate(manifest['tasks']):
            cell = path.parent / 'suite/cells/hard_realtime_instructional_medium' / task['map'] / f"task_{task['task']:03d}"
            output = cell / 'attempt_001/result.json'
            watch.write(output, record(task['task'], 'medium', i < 15))
            watch.write(cell / 'status.json', {'state': 'completed', 'result_path': str(output)})
        with patch.object(watch.subprocess, 'check_output', return_value=json.dumps({'completed': 20, 'fresh_validation_failures': []})):
            result = watch.file_results(self.config, path)
            self.assertEqual(result['taskCount'], 20)
            self.assertEqual(result['effort'][0]['success'], 75)
            bad = watch.read(output)
            bad['artifact_schema_version'] = 'execution_alignment_v3'
            watch.write(output, bad)
            with self.assertRaisesRegex(ValueError, 'protocol'):
                watch.file_results(self.config, path)

    def test_feed_discovery_survives_provider_catalogue_outage(self):
        with patch.object(watch, 'fetch', return_value=self.feed), patch.object(watch, 'catalog_data', side_effect=OSError('offline')):
            with self.assertRaises(OSError):
                watch.scan(self.config)
        self.assertIn('gemini-3-8-flash', watch.read(Path(self.config['state_dir']) / 'releases.json'))
        self.assertEqual(watch.status(self.config), [])

    def test_frozen_startup_flags_override_ambient_and_source_changes_create_new_jobs(self):
        self.config['environment'] = {'SIMWORLD_SKIP_INITIAL_SETRES': '1'}
        path = self.queue()
        with patch.dict(watch.os.environ, {'SIMWORLD_UE_EXTRA_ARGS': '-norhithread', 'SIMWORLD_SKIP_INITIAL_SETRES': '0'}):
            _, env = watch.campaign(self.config, path, 'plan')
        self.assertNotIn('SIMWORLD_UE_EXTRA_ARGS', env)
        self.assertEqual(env['SIMWORLD_SKIP_INITIAL_SETRES'], '1')
        with patch.object(watch, 'fetch', side_effect=self.fetch), patch.object(watch, 'source_digest', return_value='new-source'):
            updated = watch.scan(self.config)
        self.assertEqual(len(updated['added']), 1)
        self.assertNotEqual(updated['added'][0], path.parent.name)

    def test_pool_rejects_paid_and_unavailable_protocols(self):
        candidate = self.pool_config()[0]
        self.assertEqual(watch.decision(candidate, self.config)[0], "queued")
        for change in [{"billable": True}, {"billable": None}, {"capabilities": ["task"]}]:
            self.assertEqual(watch.decision(candidate | change, self.config)[0], "skip")

    def test_retired_openrouter_job_cannot_start_after_pool_switch(self):
        path = self.queue()
        self.pool_config()
        with self.assertRaisesRegex(ValueError, "retired transport"):
            watch.check_job(self.config, path)
        watch.write(path.parent / "status.json", {"state": "superseded"})
        with patch.object(watch.subprocess, "Popen") as spawn, self.assertRaisesRegex(ValueError, "superseded"):
            watch.start(self.config, path.parent.name)
        spawn.assert_not_called()

    def test_pool_request_forces_unpaid_exact_model_and_transmits_image(self):
        from evaluation import run_openai_benchmark as runner
        from llm.rt_llm import RTLLM
        from PIL import Image
        route = "antigravity/gemini-4-argon-medium"
        flags = ["--provider", "account-pool", "--model", route, "--api-mode", "chat_completions",
                 "--reasoning-effort", "medium", "--key-file", "pool.key", "--base-url", "http://pool.invalid/v1"]
        args = runner.parse_args(flags)
        config = runner.build_llm_config(args)["llm"][0]
        self.assertEqual(config["extra_body"], {"allow_paid": False})
        self.assertIsNone(runner.build_cost_estimate(args)["estimated_cost_usd"])
        with patch.dict(watch.os.environ, {"POOL_KEY": "test-pool-key"}), patch("llm.rt_llm.openai.OpenAI") as client:
            llm = RTLLM(route, url=args.base_url, provider="account-pool", reasoning_effort="medium",
                        extra_body={"allow_paid": True}, raise_on_api_error=True)
            response = SimpleNamespace(model=route, usage=None,
                choices=[SimpleNamespace(message=SimpleNamespace(content="Action: wait\nParam: 1"))])
            client.return_value.chat.completions.create.return_value = response
            llm.generate_response_openai("system", "user", [Image.new("RGB", (2, 2))])
            request = client.return_value.chat.completions.create.call_args.kwargs
            self.assertEqual(request["model"], route)
            self.assertFalse(request["extra_body"]["allow_paid"])
            self.assertTrue(request["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
            self.assertNotIn("reasoning_effort", request)
            self.assertNotIn("temperature", request)
            response.model = "antigravity/gemini-4-argon-high"
            with self.assertRaisesRegex(ValueError, "different model"):
                llm.generate_response_openai("system", "user")
        with self.assertRaises(SystemExit):
            runner.parse_args(flags + ["--reasoning-effort", "high"])
        with self.assertRaises(SystemExit):
            runner.parse_args(flags + ["--model", "auto"])

    def test_pool_validation_accepts_unknown_reasoning_but_checks_route_and_usage(self):
        route = "antigravity/gemini-4-argon-medium"
        task = SimpleNamespace(task_id=0, max_steps=10)
        condition = campaign.Condition(0, "hard", "realtime", "instructional", "instructional", "medium")
        attempt = self.root / "attempt"
        watch.write(attempt / "task_0.json", dict(task_id=0, model=route, provider="account-pool",
            api_mode="chat_completions", difficulty="default", seed=0, reasoning_effort="medium",
            prompt_style="instructional", traffic_policy="visual_only", realtime_thinking=True,
            use_action_frames=True, image_detail="low", static_signal_vehicles=True,
            max_steps=10, max_llm_tokens=None, total_prompt_tokens=2, total_completion_tokens=1,
            total_reasoning_tokens=0, decision_count=1))
        manifest_path = attempt / "task_0_steps/step_000/step_manifest.json"
        step = dict(prompt={"system": "environment continues to evolve during both your reasoning\nReasoning: [ONE sentence, <=30 words]",
                            "user": "continuing to the far\n  curb is still legal"},
                    feedback="Waited", model_output={"parsed_action": "wait"},
                    observation_geometry={"all_waypoint_markers_visible": True},
                    timing={"input_render": {"resolution_pixels": [720, 640]}},
                    image_retention={"policy": "first_last_ring_v1", "first_steps": 0, "last_steps": 0, "retained": False},
                    model_input_image_count=1, metrics={"prompt_tokens": 2, "completion_tokens": 1,
                        "total_tokens": 3, "reasoning_tokens": None, "provider_usage": {
                            "served_model": route, "account_pool": {"requested_model": route, "allow_paid": False}}})
        watch.write(manifest_path, step)
        with patch.multiple(campaign, PROVIDER="account-pool", MODEL=route, API_MODE="chat_completions"):
            self.assertEqual(campaign.validate_attempt(attempt, task, condition)[1], [])
            step["metrics"]["provider_usage"]["account_pool"]["allow_paid"] = True
            watch.write(manifest_path, step)
            self.assertIn("step 0 account-pool paid fallback was not disabled", campaign.validate_attempt(attempt, task, condition)[1])
            step["metrics"]["prompt_tokens"] = None
            watch.write(manifest_path, step)
            self.assertIn("step 0 token field prompt_tokens absent", campaign.validate_attempt(attempt, task, condition)[1])

    def test_gemini_reasoning_is_transmitted_and_none_is_rejected(self):
        from evaluation import run_openai_benchmark as runner
        from llm.rt_llm import RTLLM
        for effort in ["low", "medium", "high"]:
            args = runner.parse_args(["--provider", "openrouter", "--model", model()["id"],
                                      "--api-mode", "chat_completions", "--reasoning-effort", effort])
            config = runner.build_llm_config(args)["llm"][0]
            instance = RTLLM.__new__(RTLLM)
            instance.model_name = model()["id"]
            instance.reasoning = config["reasoning"]
            instance.reasoning_effort = config["reasoning_effort"]
            self.assertEqual(instance._effective_reasoning_effort(), effort)
        with self.assertRaises(SystemExit):
            runner.parse_args(["--provider", "openrouter", "--model", model()["id"],
                               "--api-mode", "chat_completions", "--reasoning-effort", "none"])


if __name__ == "__main__":
    unittest.main()
