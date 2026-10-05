"""Reproducible native-map pilot commands. Run with --help for entry points."""

from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path

DEFAULT_MANIFEST = Path(__file__).parent / "maps" / "nyc_pilot.json"


def parser():
    p = argparse.ArgumentParser(
        description="RT-SAFE native map-transfer pilot (separate from paper results)"
    )
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "doctor",
        help="Validate manifest and inspect local dependencies without touching Unreal",
    )
    verify = sub.add_parser(
        "verify-run", help="Check saved event counts, timestamps, and image checksums"
    )
    verify.add_argument("directory", type=Path)
    init = sub.add_parser(
        "init-project",
        help="Create a local derivative with owned config and output assets",
    )
    init.add_argument("--source-project", type=Path, required=True)
    init.add_argument("--destination", type=Path, required=True)
    launch = sub.add_parser(
        "launch", help="Launch only the owned editor; record its PID and logs"
    )
    launch.add_argument("--project", type=Path, required=True)
    launch.add_argument("--config", type=Path, required=True)
    launch.add_argument("--sdk-python", type=Path, required=True)
    launch.add_argument("--output", type=Path, required=True)
    launch.add_argument("--gpu", type=int, required=True)
    launch.add_argument(
        "--source-map",
        action="store_true",
        help="Open the original map for initial collision preparation",
    )
    for name, help_ in [
        ("smoke", "Run scripted physical and timing assertions"),
        ("calibrate-actions", "Measure all 16 actions in native physics"),
        (
            "audit-map",
            "Check authored route ground and capsule clearance in native map geometry",
        ),
        ("rollout", "Run one bounded scripted or visual-policy episode"),
        (
            "prepare-map",
            "Create owned triangle-collision copies and save a derived map",
        ),
    ]:
        c = sub.add_parser(name, help=help_)
        c.add_argument(
            "--config",
            type=Path,
            required=True,
            help="Local SPEAR asynchronous configuration",
        )
        c.add_argument(
            "--editor-pid",
            type=int,
            required=True,
            help="PID of the editor process you own",
        )
        c.add_argument(
            "--output", type=Path, required=True, help="New or empty evidence directory"
        )
        if name == "rollout":
            c.add_argument("--task", required=True)
            c.add_argument("--mode", choices=["static", "realtime"], default="realtime")
            c.add_argument("--seed", type=int, default=0)
            c.add_argument(
                "--policy",
                choices=["scripted-forward", "haiku", "codex-luna"],
                default="scripted-forward",
            )
            c.add_argument("--max-decisions", type=int, default=10)
            c.add_argument("--key-file", type=Path)
            c.add_argument("--budget-usd", type=float, default=0.25)
        if name == "calibrate-actions":
            c.add_argument("--task", default="nyc-park-clear")
        if name == "prepare-map":
            c.add_argument(
                "--extend-owned-map",
                action="store_true",
                help="Prepare additional source meshes in the already owned derived map",
            )
            c.add_argument(
                "--source-prefix",
                action="append",
                required=True,
                help="Explicit static-geometry asset prefix; repeat as needed",
            )
    return p


def prepare_map(client, manifest, args):
    from .project import source_digests

    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Output directory must be empty")
    args.output.mkdir(parents=True, exist_ok=True)
    client.load_runtime(Path(__file__).resolve().parents[2])
    state = client.execute(
        "import benchmark.map_transfer.geometry as geometry\n"
        "import unreal\n"
        "w=unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).get_editor_world()\n"
        'result={"map":w.get_path_name(),"content":unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_content_dir())}'
    )["result"]
    expected = manifest.get(
        "map_path" if args.extend_owned_map else "source_map_path", ""
    )
    if not expected or state["map"].split(".")[0] != expected:
        raise ValueError(f"Load {expected} before preparing collision geometry")
    inventory = client.execute(
        f"import benchmark.map_transfer.geometry as geometry\nresult=geometry.inventory({args.source_prefix!r})"
    )["result"]
    (args.output / "inventory.json").write_text(json.dumps(inventory, indent=2))
    if not inventory:
        raise RuntimeError("No source geometry matched the supplied prefixes")
    reports = []
    sources = [row["source"] for row in inventory]
    # The owned map is intentionally saved; only original packages are immutable.
    packages = sources + [manifest["source_map_path"]]
    before = source_digests(Path(state["content"]), packages)
    (args.output / "source-sha256-before.json").write_text(json.dumps(before, indent=2))
    for index in range(0, len(sources), 5):
        report = client.execute(
            f"import benchmark.map_transfer.geometry as geometry\nresult=geometry.prepare_assets({sources[index:index+5]!r})"
        )["result"]
        reports.append(report)
        (args.output / "collision-copies.json").write_text(
            json.dumps(reports, indent=2)
        )
        print(
            f"Prepared {min(index+5,len(sources))}/{len(sources)} owned collision copies",
            flush=True,
        )
    saved = client.execute(
        f'import benchmark.map_transfer.geometry as geometry\nresult=geometry.save_owned_map({manifest["map_path"]!r})'
    )["result"]
    after = source_digests(Path(state["content"]), packages)
    (args.output / "source-sha256-after.json").write_text(json.dumps(after, indent=2))
    if after != before:
        raise RuntimeError(
            "Source package digest changed during preparation; inspect the evidence before continuing"
        )
    saved["source_packages_unchanged"] = len(before)
    saved["next_step"] = (
        "Restart the owned editor without --source-map to load the derived map."
    )
    saved["extended_owned_map"] = args.extend_owned_map
    (args.output / "saved-map.json").write_text(json.dumps(saved, indent=2))
    return saved


def rollout(client, manifest, args):
    from .runner import NativeEpisode
    from .contract import Action

    if not 1 <= args.max_decisions <= 64:
        raise ValueError("Use 1–64 decisions for a transfer pilot")
    policy = None
    if args.policy == "haiku":
        if not args.key_file:
            raise ValueError("--key-file is required for the Haiku policy")
        from .policy import BudgetedHaikuPolicy

        policy = BudgetedHaikuPolicy(
            args.key_file, budget_usd=args.budget_usd, max_calls=args.max_decisions
        )
    elif args.policy == "codex-luna":
        from .cli_policy import CodexVisualPolicy

        policy = CodexVisualPolicy(max_calls=args.max_decisions)
    with NativeEpisode(
        client,
        manifest,
        args.task,
        args.output,
        mode=args.mode,
        seed=args.seed,
        reload_source=True,
    ) as episode:
        if policy:
            from .policy import SYSTEM_PROMPT, policy_prompt

            (args.output / "system-prompt.txt").write_text(SYSTEM_PROMPT)
        for index in range(args.max_decisions):
            image, state = episode.observe(index)
            if state["phase"] == "terminal":
                break
            metadata = None
            if policy:
                action, metadata = policy.decide(
                    image, policy_prompt(state, episode.task, episode.trace)
                )
                (args.output / f"policy-{index:03d}.json").write_text(
                    json.dumps(metadata, indent=2)
                )
                if client.call("status")["phase"] == "terminal":
                    break
            else:
                action = Action("move_to", "5")
            state = episode.step(action, policy_metadata=metadata)
            print(
                json.dumps(
                    {
                        "decision": index,
                        "action": action.as_dict(),
                        "phase": state["phase"],
                        "events": state["counts"],
                    }
                ),
                flush=True,
            )
            if state["phase"] == "terminal":
                break
        summary = episode.finish()
        if policy:
            summary["model_calls"] = policy.calls
            summary["model_cost_usd"] = policy.spent
            (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def main(argv=None):
    args = parser().parse_args(argv)
    from .manifest import load_manifest

    manifest = load_manifest(args.manifest)
    if args.command == "verify-run":
        from .evidence import verify_run

        result = verify_run(args.directory)
    elif args.command == "init-project":
        from .project import create_project

        result = create_project(args.source_project, args.destination)
    elif args.command == "launch":
        from .project import launch_project

        launch_project(
            args.project,
            args.config,
            args.sdk_python,
            args.output,
            manifest["source_map_path" if args.source_map else "map_path"],
            args.gpu,
        )
        return
    elif args.command == "doctor":
        result = {
            "manifest_valid": True,
            "pilot": True,
            "map": manifest["map_path"],
            "tasks": [t["id"] for t in manifest["tasks"]],
            "dependencies": {
                name: importlib.util.find_spec(name) is not None
                for name in ("spear", "spear_ext", "PIL", "requests")
            },
            "live_runtime_checked": False,
        }
    else:
        from .client import NativeClient

        with NativeClient(args.config, expected_pid=args.editor_pid) as client:
            if args.command == "prepare-map":
                result = prepare_map(client, manifest, args)
            elif args.command == "audit-map":
                import time

                if args.output.exists() and any(args.output.iterdir()):
                    raise ValueError("Audit output must be empty")
                args.output.mkdir(parents=True, exist_ok=True)
                client.load_runtime(Path(__file__).resolve().parents[2])
                client.call("cleanup_scene")
                time.sleep(0.2)
                result = client.execute(
                    "import benchmark.map_transfer.audit_native as audit\n"
                    f"result=audit.audit_map({manifest!r})"
                )["result"]
                (args.output / "route-audit.json").write_text(
                    json.dumps(result, indent=2)
                )
            elif args.command == "rollout":
                result = rollout(client, manifest, args)
            elif args.command == "calibrate-actions":
                from .calibration import run_calibration

                result = run_calibration(client, manifest, args.output, args.task)
            else:
                from .smoke import run_suite

                result = run_suite(client, manifest, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
