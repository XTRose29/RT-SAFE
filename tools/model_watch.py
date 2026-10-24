#!/usr/bin/env python3
"""RT-SAFE model discovery, frozen jobs, manual starts, and local result filing."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
from statistics import mean
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm.reasoning_capabilities import openrouter_reasoning_profile


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


@contextmanager
def lock(path, blocking=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield stream


def event(config, kind, **data):
    with (Path(config["state_dir"]) / "events.jsonl").open("a") as stream:
        stream.write(json.dumps({"time": now(), "event": kind, **data}) + "\n")


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "RT-SAFE-model-watch/1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def provider(config):
    return config.get("provider", "openrouter")


def catalog_data(config):
    if provider(config) != "account-pool":
        return fetch(config["catalog_url"])
    # The configured pool credential is never sent to the public catalogue.
    from evaluation.run_openai_benchmark import load_api_key
    from llm.account_pool import validate_route
    validate_route("antigravity/gemini-4-argon-medium", "medium", config["base_url"])
    key_path = Path(config["key_file"])
    if key_path.stat().st_mode & 0o077:
        raise ValueError("Pool key file requires chmod 600")
    key = load_api_key(key_path, "POOL_KEY")
    request = urllib.request.Request(config["base_url"].rstrip("/") + "/models",
                                     headers={"Authorization": "Bearer " + key})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def feed_candidates(feed, catalog, config):
    if provider(config) == "account-pool":
        # Discovery may recognize a family; only validate_route can authorize a run.
        def canonical(value):
            name = value.rsplit("/", 1)[-1].replace(".", "-")
            return re.sub(r"-(low|medium|high)$", "", name)
        names = {canonical(value) for entry in feed["models"]
                 for value in [entry["name"], *entry.get("sources", {}).values()]}
        return sorted(name for name in catalog if canonical(name) in names)
    return [entry["sources"]["openrouter"] for entry in feed["models"]
            if entry.get("sources", {}).get("openrouter")]


def source_digest(repo):
    """Freeze benchmark source/data, including local uncommitted additions."""
    hasher = hashlib.sha256()
    repo = Path(repo)
    paths = set()
    for folder in ("base", "manager", "llm", "utils", "sample", "evaluation", "data", "SimWorld/simworld"):
        for ext in ("*.py", "*.json", "*.yaml"):
            paths.update((repo / folder).rglob(ext))
    paths.update([repo / "tools/model_watch.py", repo / "benchmark/release_sources.json",
                  repo / "website/scripts/build-subset-results.py"])
    for path in sorted(paths):
        if "__pycache__" not in path.parts:
            hasher.update(str(path.relative_to(repo)).encode())
            hasher.update(path.read_bytes())
    return hasher.hexdigest()


def selection(config):
    path = Path(config["selection_manifest"])
    value = read(path)
    tasks = value["tasks"]
    counts = Counter(t["map"] for t in tasks)
    if (len(tasks) not in {10, 20} or len({(t["map"], t["task"]) for t in tasks}) != len(tasks)
            or len(counts) != len(tasks) // 5 or set(counts.values()) != {5}
            or value["seed"] != 0 or value["difficulty"] != "hard" or value["timing"] != "realtime"):
        raise ValueError("Expected a frozen balanced hard/realtime selection: five tasks per map, ten or twenty tasks")
    return value, hashlib.sha256(path.read_bytes()).hexdigest()


def decision(model, config):
    name = model["id"]
    if ":" in name or not any(name.startswith(p) for p in config["model_prefixes"]):
        return "skip", "Outside configured model families or a routing variant", []
    if provider(config) == "account-pool":
        from llm.account_pool import PINNED_ROUTES, validate_route
        effort = PINNED_ROUTES.get(name)
        try:
            validate_route(name, effort, config["base_url"])
        except ValueError:
            return "needs_profile", "Review image support and pin exact model route and reasoning controls", []
        if effort not in config["efforts"] or "chat" not in model.get("capabilities", []):
            return "skip", "Effort or Chat Completions transport unavailable", []
        if model.get("billable") is not False:
            return "skip", "Only explicitly non-billable subscription routes are allowed", []
        return "queued", "Authorized subscription model; paid fallback disabled", [effort]
    if "image" not in model.get("architecture", {}).get("input_modalities", []):
        return "skip", "RT-SAFE needs image input", []
    profile = openrouter_reasoning_profile(name)
    efforts = config["efforts"]
    catalog_efforts = model.get("reasoning", {}).get("supported_efforts", [])
    if not profile or not set(efforts) <= set(profile["efforts"]) or not set(efforts) <= set(catalog_efforts):
        return "needs_profile", "Review and pin this model's reasoning controls before running", []
    for key in ("prompt", "completion"):
        try:
            price = float(model["pricing"][key])
        except (KeyError, ValueError, TypeError):
            return "needs_profile", "Missing token prices", []
        if not math.isfinite(price) or price < 0:
            return "needs_profile", "Invalid token prices", []
    return "queued", "Vision model with verified reasoning controls", efforts


def campaign(config, job_path, mode):
    job = read(job_path)
    settings = job["execution"]
    command = [settings["python"], str(Path(job["repo"]) / "evaluation/run_model_watch_campaign.py"),
               "--mode", mode, "--suite-dir", str(job_path.parent / "suite"),
               "--key-file", settings["key_file"], "--ue-launcher", settings["ue_launcher"],
               "--workers", str(settings["workers"]), "--worker-count", str(settings["workers"]),
               "--gpus", *map(str, settings["gpus"]), "--ports", *map(str, settings["ports"]),
               "--max-attempts", str(settings["max_attempts"])]
    environment = os.environ.copy()
    # An unrelated interactive campaign must not change this job's transport.
    for key in list(environment):
        if key.startswith("SIMWORLD_CAMPAIGN_"):
            del environment[key]
    environment.pop("OPENROUTER_API_KEY", None)
    environment.pop("OPENAI_API_KEY", None)
    environment.pop("POOL_KEY", None)
    environment["RTSAFE_WATCH_JOB"] = str(job_path)
    environment["SIMWORLD_CAMPAIGN_PROVIDER"] = job.get("provider", "openrouter")
    # Freeze startup flags rather than inheriting another interactive run's flags.
    for key in list(environment):
        if key.startswith("SIMWORLD_UE_") or key.startswith("SIMWORLD_SKIP_"):
            del environment[key]
    environment.update(settings.get("environment", {}))
    return command, environment


def scan(config):
    state = Path(config["state_dir"])
    state.mkdir(parents=True, exist_ok=True)
    with lock(state / "scan.lock", blocking=False):
        feed = fetch(config["feed_url"])
        generated = datetime.fromisoformat(feed["generated"])
        age = (datetime.now(timezone.utc) - generated).total_seconds()
        if age > config.get("max_feed_age_hours", 6) * 3600 or age < -300:
            raise ValueError("Shared feed timestamp is stale or in the future")
        chosen, manifest_hash = selection(config)
        fingerprint = source_digest(config["repo"])
        release = read(Path(config["repo"]) / "benchmark/release_sources.json")
        # Keep all release notices, even if this project's catalogue cannot run them.
        releases = read(state / "releases.json") if (state / "releases.json").exists() else {}
        for entry in feed["models"]:
            if entry["name"] not in releases:
                releases[entry["name"]] = {"name": entry["name"], "first_seen": entry.get("first_seen"),
                    "status": "review_required", "reason": "Check project account, vision support and reasoning profile"}
                event(config, "release_detected", model=entry["name"])
        write(state / "releases.json", releases)
        catalog = {m["id"]: m for m in catalog_data(config)["data"]}
        seen = read(state / "seen.json") if (state / "seen.json").exists() else {}
        added = []
        for name in feed_candidates(feed, catalog, config):
            model = catalog.get(name)
            if model is None:
                result, reason, efforts = "unavailable", "No longer in provider catalogue", []
            else:
                result, reason, efforts = decision(model, config)
            snapshot = {"state": result, "reason": reason}
            if seen.get(name) != snapshot:
                event(config, "discovery", model=name, **snapshot)
                seen[name] = snapshot
            if result != "queued":
                continue
            # One model/protocol/effort job; recurring polls never requeue a completed job.
            identity = {"model": name, "selection": manifest_hash, "efforts": efforts,
                        "source": fingerprint, "protocol": release["protocol"]}
            if provider(config) == "account-pool":
                identity.update(provider=provider(config), base_url=config["base_url"])
            job_id = re.sub(r"[^a-zA-Z0-9_-]", "_", name) + "-" + digest(identity)[:10]
            job_path = state / "jobs" / job_id / "job.json"
            if job_path.exists():
                continue
            settings = {k: config[k] for k in ("python", "key_file", "ue_launcher", "workers", "gpus", "ports", "max_attempts")}
            settings["environment"] = config.get("environment", {})
            job = {"protocol": release["protocol"], "id": job_id, "created": now(), "model": name, "label": model.get("name", name),
                   "provider": provider(config), "base_url": config.get("base_url"),
                   "efforts": efforts, "selection": chosen, "selection_sha256": manifest_hash,
                   "repo": config["repo"], "source_sha256": fingerprint, "execution": settings,
                   "catalog": model, "feed_generated": feed["generated"],
                   "pricing_per_M": (None if provider(config) == "account-pool" else
                                     {"input": float(model["pricing"]["prompt"]) * 1e6,
                                      "output": float(model["pricing"]["completion"]) * 1e6})}
            write(job_path, job)
            try:
                command, environment = campaign(config, job_path, "plan")
                plan = json.loads(subprocess.check_output(command, env=environment, cwd=job["repo"], text=True, timeout=60))
                write(job_path.parent / "plan.json", plan)
                write(job_path.parent / "status.json", {"state": "queued", "updated": now()})
            except Exception:
                job_path.unlink()
                raise
            event(config, "queued", job=job_id, model=name, episodes=plan["rollout_count"])
            added.append(job_id)
        write(state / "seen.json", seen)
        write(state / "last_scan.json", {"time": now(), "feed_generated": feed["generated"], "feed_models": len(feed["models"]), "added": added})
    return {"added": added, "feed_models": len(feed["models"])}


def job_path(config, job_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", job_id):
        raise ValueError("Invalid job ID")
    path = Path(config["state_dir"]) / "jobs" / job_id / "job.json"
    if not path.is_file():
        raise ValueError("Unknown job ID")
    return path


def status(config):
    result = []
    for path in sorted((Path(config["state_dir"]) / "jobs").glob("*/job.json")):
        job = read(path)
        current = read(path.parent / "status.json") if (path.parent / "status.json").exists() else {"state": "preparing"}
        plan = read(path.parent / "plan.json") if (path.parent / "plan.json").exists() else {}
        result.append({"id": job["id"], "model": job["model"], **current,
                       "provider": job.get("provider", "openrouter"),
                       "episodes": plan.get("rollout_count"),
                       "planning_estimate_usd": plan.get("planning", {}).get("all_caps_estimate_usd")})
    return result


def check_job(config, path):
    job = read(path)
    if job.get("provider", "openrouter") != provider(config) or job.get("base_url") != config.get("base_url"):
        raise ValueError("Job uses a retired transport; use a job from the current queue")
    if source_digest(job["repo"]) != job["source_sha256"]:
        raise ValueError("Benchmark source changed since queueing; review and use refresh-job before starting")
    key = Path(job["execution"]["key_file"])
    if not key.is_file() or key.stat().st_mode & 0o077:
        raise ValueError("API key file must exist and be readable only by its owner (chmod 600)")
    if not Path(job["execution"]["ue_launcher"]).is_file():
        raise ValueError("Unreal launcher is missing")
    models = {m["id"]: m for m in catalog_data(config)["data"]}
    current = models.get(job["model"])
    if not current or decision(current, {**config, "efforts": job["efforts"]})[0] != "queued":
        raise ValueError("Model/effort availability changed; review job before starting")
    if provider(config) == "account-pool":
        return job
    prices = {"input": float(current["pricing"]["prompt"]) * 1e6,
              "output": float(current["pricing"]["completion"]) * 1e6}
    if prices != job["pricing_per_M"]:
        raise ValueError("Model prices changed; review and refresh the unstarted job")
    return job


def file_results(config, path):
    """Use existing strict safety metrics; require every frozen cell, including failures."""
    job = read(path)
    if source_digest(job["repo"]) != job["source_sha256"]:
        raise ValueError("Benchmark source changed during the campaign; review provenance before filing")
    command, env = campaign(config, path, "summary")
    summary = json.loads(subprocess.check_output(command, env=env, cwd=job["repo"], text=True, timeout=120))
    expected = len(job["selection"]["tasks"]) * len(job["efforts"])
    if summary["completed"] != expected or summary["fresh_validation_failures"]:
        raise ValueError(f"Campaign incomplete: {summary['completed']}/{expected}")
    spec = importlib.util.spec_from_file_location("subset_metrics", Path(job["repo"]) / "website/scripts/build-subset-results.py")
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    rows, audit = [], []
    for effort in job["efforts"]:
        records = []
        for task in job["selection"]["tasks"]:
            cell = path.parent / "suite/cells" / f"hard_realtime_instructional_{effort}" / task["map"] / f"task_{task['task']:03d}"
            current = read(cell / "status.json")
            if current["state"] != "completed":
                raise ValueError("Unfinished cell cannot be ranked")
            result_path = Path(current["result_path"])
            record = read(result_path)
            if record.get("artifact_schema_version") != job["protocol"]:
                raise ValueError("Episode artifact protocol does not match the frozen job")
            if record["task_id"] != task["task"] or record["model"] != job["model"] or record["reasoning_effort"] != effort:
                raise ValueError("Cell identity mismatch")
            records.append(record)
            audit.append({"effort": effort, "map": task["map"], "task": task["task"],
                          "result": str(result_path), "sha256": hashlib.sha256(result_path.read_bytes()).hexdigest()})
        episodes = [metrics.episode_metrics(record) for record in records]
        rows.append({"label": effort, **{key: round(mean(e[key] for e in episodes), 6)
                                        for key in episodes[0]}})
    result = {"job": job["id"], "model": job["model"], "name": job["label"], "effort": rows,
              "protocol": job["protocol"], "taskCount": len(job["selection"]["tasks"]), "manifestSha256": job["selection_sha256"],
              "sourceSha256": job["source_sha256"],
              "transport": job.get("provider", "openrouter") + "/chat_completions",
              "baseUrl": job.get("base_url"), "filed": now()}
    write(path.parent / "result.json", result)
    write(path.parent / "audit.json", audit)
    # One immutable file per job avoids duplicate ledger entries on resume.
    write(Path(config["state_dir"]) / "ledger" / (job["id"] + ".json"), result)
    board = [read(p) for p in sorted((Path(config["state_dir"]) / "ledger").glob("*.json"))]
    write(Path(config["state_dir"]) / "board.json", {"models": board, "updated": now()})
    return result


def start(config, job_id):
    state = Path(config["state_dir"])
    path = job_path(config, job_id)
    with lock(state / "run.lock", blocking=False) as run_lock:
        current = read(path.parent / "status.json")
        if current["state"] == "completed":
            return {"state": "completed", "message": "Already filed; no new calls"}
        if current["state"] == "superseded":
            raise ValueError("Job was superseded; use a job from the current queue")
        try:
            job = check_job(config, path)
        except Exception as exc:
            write(path.parent / "status.json", {"state": "needs_attention", "updated": now(), "reason": str(exc)})
            event(config, "preflight_failed", job=job_id, reason=str(exc))
            raise
        command, environment = campaign(config, path, "coordinator")
        write(path.parent / "status.json", {"state": "running", "updated": now()})
        event(config, "started", job=job_id)
        with (path.parent / "run.log").open("a") as output:
            process = subprocess.Popen(command, env=environment, cwd=job["repo"], stdout=output, stderr=subprocess.STDOUT,
                                       start_new_session=True, pass_fds=(run_lock.fileno(),))
            write(path.parent / "status.json", {"state": "running", "pid": process.pid, "updated": now()})
            previous = {}
            def stop(signum, frame):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
            for sig in (signal.SIGINT, signal.SIGTERM):
                previous[sig] = signal.signal(sig, stop)
            try:
                code = process.wait()
                if code:
                    raise RuntimeError(f"Campaign exited {code}; see {path.parent / 'run.log'}")
                result = file_results(config, path)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGINT)
                    process.wait(timeout=180)
                write(path.parent / "status.json", {"state": "needs_attention", "updated": now()})
                event(config, "needs_attention", job=job_id)
                raise
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
        write(path.parent / "status.json", {"state": "completed", "updated": now()})
        event(config, "completed", job=job_id)
        return result


def refresh_job(config, job_id):
    path = job_path(config, job_id)
    if read(path).get("provider", "openrouter") != provider(config):
        raise ValueError("Cannot refresh a job from a retired transport")
    with lock(Path(config["state_dir"]) / "run.lock", blocking=False), lock(Path(config["state_dir"]) / "scan.lock"):
        if (path.parent / "suite").exists():
            raise ValueError("Started jobs are immutable; preserve their original source and settings")
        archived = read(path)
        write(path.parent / "previous_job.json", archived)
        path.unlink()
    result = scan(config)
    return {"previous": job_id, **result}


def stop_job(config, job_id):
    path = job_path(config, job_id)
    current = read(path.parent / "status.json")
    if current["state"] != "running" or not current.get("pid"):
        raise ValueError("Job is not running")
    pid = current["pid"]
    proc = Path("/proc") / str(pid)
    # Guard against a stale/reused PID; never signal someone else's process.
    if proc.stat().st_uid != os.getuid() or (b"RTSAFE_WATCH_JOB=" + str(path).encode()) not in (proc / "environ").read_bytes().split(b"\0"):
        raise ValueError("Recorded process does not own this job")
    os.killpg(pid, signal.SIGINT)
    return {"job": job_id, "state": "stop_requested"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("scan", "status"):
        commands.add_parser(name)
    for name in ("start", "stop", "plan", "refresh-job", "file"):
        commands.add_parser(name).add_argument("job")
    args = parser.parse_args(argv)
    config = read(args.config)
    if Path(config["repo"]).resolve() != ROOT:
        parser.error("Run the model-watch script from the configured source checkout")
    Path(config["state_dir"]).mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "scan":
            result = scan(config)
        elif args.command == "status":
            result = status(config)
        elif args.command == "plan":
            result = read(job_path(config, args.job).parent / "plan.json")
        elif args.command == "start":
            result = start(config, args.job)
        elif args.command == "stop":
            result = stop_job(config, args.job)
        elif args.command == "refresh-job":
            result = refresh_job(config, args.job)
        else:
            with lock(Path(config["state_dir"]) / "run.lock", blocking=False):
                result = file_results(config, job_path(config, args.job))
        print(json.dumps(result, indent=2))
        return 0
    except BlockingIOError:
        print("Another pipeline operation owns the lock; no duplicate work started", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
