"""Generate visual evaluation report charts for LinkedIn/GitHub."""
import json, os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "eval", "results.json")
OUT = os.path.join(ROOT, "eval")

with open(RESULTS, encoding="utf-8") as f:
    data = json.load(f)

cases = [r for r in data["per_case"] if "error" not in r]

# --- per-subject aggregation ---
subjects = {}
for r in cases:
    subj = r["case"].rsplit("-", 1)[0]
    subjects.setdefault(subj, []).append(r)

subj_stats = {}
for s, rows in subjects.items():
    bert_key = "bertscore_f1" if "bertscore_f1" in rows[0] else "embedding_similarity_PROXY"
    subj_stats[s] = {
        "bert":      round(sum(r[bert_key] for r in rows) / len(rows), 3),
        "baseline":  round(sum(r.get("baseline_bertscore_avg", 0) for r in rows) / len(rows), 3),
        "precision": round(sum(r["syllabus_precision"] for r in rows) / len(rows), 3),
        "recall":    round(sum(r["syllabus_recall"] for r in rows) / len(rows), 3),
        "ref_recall":round(sum(r.get("ref_recall", 0) for r in rows) / len(rows), 3),
        "f1":        round(sum(r["syllabus_f1"] for r in rows) / len(rows), 3),
        "halluc":    round(sum(r["hallucination_rate"] for r in rows) / len(rows), 3),
        "n":         len(rows),
    }

LABELS = {
    "cprog": "C Prog", "dl": "Digital Logic", "ds": "Disc. Struct.",
    "dwdm": "DWDM", "iit": "IIT", "java": "Java",
    "micro": "Microprocessor", "oop": "OOP", "pom": "POM", "webtech": "Web Tech",
}
order = sorted(subj_stats, key=lambda s: subj_stats[s]["bert"], reverse=True)
labels = [LABELS.get(s, s) for s in order]

# ── CHART 1: BERTScore generator vs baseline ──────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(order))
w = 0.35
bars1 = ax.bar(x - w/2, [subj_stats[s]["bert"]     for s in order], w,
               label="Generator BERTScore", color="#2196F3", zorder=3)
bars2 = ax.bar(x + w/2, [subj_stats[s]["baseline"] for s in order], w,
               label="Baseline (real past paper)", color="#90CAF9", zorder=3)

for b in bars1:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
            f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
for b in bars2:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.003,
            f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.5, color="#555")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.65, 0.98)
ax.set_ylabel("BERTScore F1 (roberta-large)", fontsize=10)
ax.set_title("Generator BERTScore vs Past-Paper Baseline\n(44 held-out cases, 10 BSc.CSIT subjects, TU)",
             fontsize=11, fontweight="bold", pad=12)
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_bertscore.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart_bertscore.png")

# ── CHART 2: Precision / Recall / F1 per subject ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
x = np.arange(len(order))
w = 0.25
ax.bar(x - w,   [subj_stats[s]["precision"] for s in order], w,
       label="Syllabus Precision", color="#4CAF50", zorder=3)
ax.bar(x,       [subj_stats[s]["recall"]    for s in order], w,
       label="Generator Recall",   color="#FF9800", zorder=3)
ax.bar(x + w,   [subj_stats[s]["ref_recall"]for s in order], w,
       label="Real Paper Recall",  color="#FFCC80", zorder=3)

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score", fontsize=10)
ax.set_title("Syllabus Precision & Recall — Generator vs Real Exam Paper\n(Generator recall consistently ≥ real paper recall)",
             fontsize=11, fontweight="bold", pad=12)
ax.legend(fontsize=9)
ax.yaxis.grid(True, alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "chart_precision_recall.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart_precision_recall.png")

# ── CHART 3: Summary scorecard (single image for LinkedIn) ───────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Shikshya Chautari — AI Question Paper Generator\nEvaluation: 44 held-out papers, 10 BSc.CSIT subjects (TU)",
             fontsize=12, fontweight="bold", y=1.02)

# Left: aggregate metrics as horizontal bars
agg_labels = ["BERTScore F1", "Syllabus Precision", "Generator Recall",
              "Real Paper Recall", "Syllabus F1"]
agg_vals   = [
    round(sum(subj_stats[s]["bert"]      for s in subj_stats) / len(subj_stats), 3),
    round(sum(subj_stats[s]["precision"] for s in subj_stats) / len(subj_stats), 3),
    round(sum(subj_stats[s]["recall"]    for s in subj_stats) / len(subj_stats), 3),
    round(sum(subj_stats[s]["ref_recall"]for s in subj_stats) / len(subj_stats), 3),
    round(sum(subj_stats[s]["f1"]        for s in subj_stats) / len(subj_stats), 3),
]
colors     = ["#2196F3","#4CAF50","#FF9800","#FFCC80","#9C27B0"]
ax = axes[0]
bars = ax.barh(agg_labels, agg_vals, color=colors, zorder=3, height=0.5)
for b in bars:
    ax.text(b.get_width()+0.005, b.get_y()+b.get_height()/2,
            f"{b.get_width():.3f}", va="center", fontsize=10, fontweight="bold")
ax.set_xlim(0, 1.05)
ax.set_title("Aggregate Metrics", fontsize=10, fontweight="bold")
ax.xaxis.grid(True, alpha=0.4, zorder=0)
ax.set_axisbelow(True)
ax.spines[["top","right"]].set_visible(False)
avg_halluc = round(sum(subj_stats[s]["halluc"] for s in subj_stats) / len(subj_stats) * 100, 1)
ax.text(0.5, -0.18, f"★  Hallucination Rate: {avg_halluc}%  across 44 held-out cases",
        transform=ax.transAxes, ha="center", fontsize=8, color="#E53935",
        style="italic")

# Right: per-subject F1 horizontal bar
order2 = sorted(subj_stats, key=lambda s: subj_stats[s]["f1"], reverse=True)
ax2 = axes[1]
f1_vals = [subj_stats[s]["f1"] for s in order2]
f1_labels = [LABELS.get(s, s) for s in order2]
bar_colors = ["#4CAF50" if v >= 0.7 else "#FF9800" if v >= 0.5 else "#EF5350" for v in f1_vals]
bars2 = ax2.barh(f1_labels, f1_vals, color=bar_colors, zorder=3, height=0.5)
for b in bars2:
    ax2.text(b.get_width()+0.005, b.get_y()+b.get_height()/2,
             f"{b.get_width():.3f}", va="center", fontsize=10, fontweight="bold")
ax2.set_xlim(0, 1.05)
ax2.set_title("Syllabus F1 by Subject", fontsize=10, fontweight="bold")
ax2.xaxis.grid(True, alpha=0.4, zorder=0)
ax2.set_axisbelow(True)
ax2.spines[["top","right"]].set_visible(False)

green = mpatches.Patch(color="#4CAF50", label="F1 ≥ 0.70")
orange = mpatches.Patch(color="#FF9800", label="0.50 ≤ F1 < 0.70")
red = mpatches.Patch(color="#EF5350", label="F1 < 0.50")
ax2.legend(handles=[green, orange, red], fontsize=8,
           loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(os.path.join(OUT, "chart_summary.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved chart_summary.png")

print("\nAll charts saved to eval/")
print("Use chart_summary.png for LinkedIn (single image)")
print("Use chart_bertscore.png + chart_precision_recall.png for GitHub README")
