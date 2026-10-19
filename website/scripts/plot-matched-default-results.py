"""Export the exact website comparison as a shareable PNG/PDF (matplotlib)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SITE = Path(__file__).resolve().parents[1]
data = json.loads((SITE / "app/matched-default-results.json").read_text())
rows = sorted(data["models"], key=lambda r: r["collisions"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(15, 7.5), sharey=True,
                         gridspec_kw={"width_ratios": [2.3, 1.05, 1.1]})
fig.patch.set_facecolor("#fafaf7")
active, passive = "#1f5663", "#a9cbd1"
for ax in axes:
    ax.set_facecolor("#fafaf7")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#b9c3c6")
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.18)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(rows)-.5, -.7)
for i, row in enumerate(rows):
    if row["name"] == "Gemini 4":
        for ax in axes:
            ax.axhspan(i-.43, i+.43, color="#e4ebff", zorder=0)
    axes[0].barh(i, row["active"], color=active, height=.56)
    axes[0].barh(i, row["passive"], left=row["active"], color=passive, height=.56)
    axes[0].text(row["collisions"]+1.8, i, f"{row['collisions']:.2f}", va="center", fontsize=10)
    axes[1].barh(i, row["success"], color="#6d8293", height=.56)
    axes[1].text(row["success"]+2, i, f"{row['success']:.0f}%", va="center", fontsize=10)
    axes[2].barh(i, row["latency"], color="#9e829d", height=.56)
    axes[2].text(row["latency"]+.7, i, f"{row['latency']:.1f}", va="center", fontsize=10)
labels = [r["fullName"].replace("DeepSeek-V4-Flash-Vision-Exp", "DeepSeek V4 Flash Vision") + f" ({r['effort'].lower()})" for r in rows]
axes[0].set_yticks(range(len(rows)), labels)
for tick, row in zip(axes[0].get_yticklabels(), rows):
    if row["name"] == "Gemini 4":
        tick.set_fontweight("bold")
axes[0].set_xlim(0, max(r["collisions"] for r in rows)*1.21)
axes[1].set_xlim(0, 124)
axes[1].set_xticks([0, 50, 100])
axes[2].set_xlim(0, max(r["latency"] for r in rows)*1.25)
axes[0].set_title("Contacts per episode ↓", loc="left", fontweight="bold", pad=18)
axes[1].set_title("Goal success ↑", loc="left", fontweight="bold", pad=18)
axes[2].set_title("Response time (s)", loc="left", fontweight="bold", pad=18)
axes[0].legend(handles=[Patch(facecolor=active, label="Active · execution"), Patch(facecolor=passive, label="Passive · inference")],
               loc="upper left", bbox_to_anchor=(0, -.08), ncol=2, frameon=False, fontsize=10)
fig.text(.03, .955, "RT-SAFE  /  20 matched tasks, nine models", fontsize=21, fontweight="bold", color="#18333c")
fig.text(.03, .915, "Configured-default reasoning · 5 tasks per map × 4 maps · Hard, real-time · Seed 0", fontsize=12, color="#4b5c64")
safe = "; ".join(f"{r['name']}: {r['safeSuccess']:.0f}%" for r in rows if r["safeSuccess"] > 0)
safe_note = f"Safe success: {safe}; all other models 0%." if safe else "Safe success is 0% for all nine models."
fig.text(.03, .112, safe_note, fontsize=10, fontweight="bold", color="#18333c")
fig.text(.03, .08, "All 20 episodes count, including failures. Contacts may repeat; lower counts can coexist with failed navigation.\nHistorical defaults vs. Gemini 4 explicit medium: different collection dates/transports, original scoring. Latency averages episode means.", fontsize=9, color="#53616a", linespacing=1.5, va="top")
fig.subplots_adjust(left=.285, right=.97, top=.81, bottom=.23, wspace=.28)
destination = SITE / "public/downloads/matched-default-leaderboard"
fig.savefig(destination.with_suffix(".png"), dpi=200, facecolor=fig.get_facecolor())
fig.savefig(destination.with_suffix(".pdf"), facecolor=fig.get_facecolor())
print(destination)
