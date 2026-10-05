#!/usr/bin/env python3
"""Plot measured park timing/contact evidence from a completed native smoke suite.

Requires matplotlib in the reporting environment. Raw runs remain local;
the exported JSON contains only the measured traces and provenance hashes.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.map_transfer.evidence import verify_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Choose an empty reporting directory")
    args.output.mkdir(parents=True, exist_ok=True)
    traces = []
    for name, label in [
        ("pedestrian-park-static", "Static"),
        ("pedestrian-park-real", "Real-time"),
    ]:
        root = args.suite / name
        verify_run(root)
        trajectory = json.loads((root / "trajectory.json").read_text())["samples"]
        state = json.loads((root / "final-state.json").read_text())
        provenance = json.loads((root / "provenance.json").read_text())
        origin = trajectory[0]["wall_time"]
        traces.append(
            {
                "label": label,
                "mode": provenance["mode"],
                "manifest_sha256": provenance["manifest_sha256"],
                "native_binary_sha256": provenance["runtime_fingerprint"][
                    "native_binary_sha256"
                ],
                "source_trajectory_sha256": hashlib.sha256(
                    (root / "trajectory.json").read_bytes()
                ).hexdigest(),
                "source_final_state_sha256": hashlib.sha256(
                    (root / "final-state.json").read_bytes()
                ).hexdigest(),
                "samples": [
                    {
                        "wall_seconds": point["wall_time"] - origin,
                        "simulation_seconds": point["simulation_time"],
                        "center_distance_m": math.dist(
                            point["agent_position_cm"][:2],
                            point["entities"]["walker"][:2],
                        )
                        / 100,
                    }
                    for point in trajectory
                ],
                "collisions": [
                    {
                        "wall_seconds": event["wall_time"] - origin,
                        "simulation_seconds": event["simulation_time"],
                        "phase": event["phase"],
                        "source": event["evidence"]["source"],
                    }
                    for event in state["events"]
                    if event["type"] == "collision"
                ],
            }
        )
    if traces[0]["manifest_sha256"] != traces[1]["manifest_sha256"]:
        raise ValueError("Paired plots require the same manifest")
    artifact = {
        "description": "Native NYC park encounter; agent remains in inference without acting",
        "pilot": True,
        "traces": traces,
    }
    (args.output / "contact-and-timing.json").write_text(json.dumps(artifact, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = {"Static": "#286B73", "Real-time": "#C16A35"}
    for trace in traces:
        points = trace["samples"]
        x = [p["wall_seconds"] for p in points]
        for ax, key in zip(axes, ["center_distance_m", "simulation_seconds"]):
            ax.plot(
                x,
                [p[key] for p in points],
                label=trace["label"],
                color=colors[trace["label"]],
                linewidth=2.5,
            )
            ax.set_xlabel("Wall time since episode start (s)")
            ax.grid(alpha=0.18)
            ax.set_xlim(left=0)
        for event in trace["collisions"]:
            axes[0].axvline(
                event["wall_seconds"], color="#B33C49", linestyle=":", linewidth=1.5
            )
            axes[0].annotate(
                "Native blocking hit",
                xy=(event["wall_seconds"], 0.7),
                xytext=(event["wall_seconds"] - 3.0, 2.2),
                arrowprops={"arrowstyle": "->", "color": "#B33C49"},
                color="#B33C49",
                fontsize=10,
            )
    axes[0].set(
        title="Pedestrian approaches a stationary agent",
        ylabel="Distance between actor centers (m)",
    )
    axes[1].set(title="World time during inference", ylabel="Simulation time (s)")
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.suptitle("Native NYC timing and contact check", fontsize=15, fontweight="bold")
    fig.savefig(args.output / "contact-and-timing.png", dpi=180)
    svg = args.output / "contact-and-timing.svg"
    fig.savefig(svg)
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    print(
        json.dumps(
            {"output": str(args.output), "matplotlib_version": matplotlib.__version__}
        )
    )


if __name__ == "__main__":
    main()
