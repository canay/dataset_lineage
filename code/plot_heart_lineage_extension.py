#!/usr/bin/env python3
"""Render Heart Disease lineage extension figures and manuscript table from saved outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VAR_LABELS = {
    "cleveland_impute_any": "Cleveland\nimpute\nany",
    "cleveland_drop_any": "Cleveland\ndrop\nany",
    "multi_site_impute_any": "Four-site\nimpute\nany",
    "multi_site_drop_any": "Four-site\ndrop\nany",
    "cleveland_impute_severe": "Cleveland\nimpute\nsevere",
}
MODEL_LABELS = {"logreg": "LR", "rf": "RF", "hgb": "HGB"}
COLORS = {"logreg": "#0072B2", "rf": "#D55E00", "hgb": "#009E73"}


def mean_sd(x: pd.Series) -> str:
    return f"{x.mean():.3f} $\\pm$ {x.std(ddof=1):.3f}"


def render_table(result_dir: Path) -> None:
    unit = pd.read_csv(result_dir / "heart_unit_metrics.csv")
    rows = []
    for variant in VAR_LABELS:
        sub = unit[unit["variant"] == variant]
        rows.append(
            {
                "Variant": VAR_LABELS[variant].replace("\n", " "),
                "Target": sub["label_policy"].iloc[0].replace("_", " "),
                "Train n": f"{sub['n_train'].mean():.1f}",
                "Common acc.": mean_sd(sub["common_acc"]),
                "Common AUC": mean_sd(sub["common_auc"]),
                "ECE": mean_sd(sub["common_ece"]),
                "Pred. positive": mean_sd(sub["common_pred_pos_rate"]),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(result_dir / "heart_extension_table.csv", index=False)
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Variant & Target & Train n & Common acc. & Common AUC & ECE & Pred. positive \\\\",
        "\\midrule",
    ]
    for _, r in table.iterrows():
        lines.append(
            f"{r['Variant']} & {r['Target']} & {r['Train n']} & {r['Common acc.']} & {r['Common AUC']} & {r['ECE']} & {r['Pred. positive']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (result_dir / "heart_extension_table.tex").write_text("\n".join(lines), encoding="utf-8")


def render_figure(result_dir: Path, figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    deltas = pd.read_csv(result_dir / "heart_deltas_vs_baseline.csv")
    unit = pd.read_csv(result_dir / "heart_unit_metrics.csv")
    effects = pd.read_csv(result_dir / "heart_effect_sizes.csv")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.5))

    ax = axes[0]
    compatible = deltas[deltas["same_target_as_baseline"] == True].copy()
    for model, sub in compatible.groupby("model"):
        ax.scatter(
            sub["d_common_acc"].abs(),
            sub["churn_vs_baseline"],
            s=42,
            alpha=0.82,
            color=COLORS[model],
            edgecolor="white",
            linewidth=0.4,
            label=MODEL_LABELS[model],
        )
    lim = max(compatible["d_common_acc"].abs().max(), compatible["churn_vs_baseline"].max()) * 1.15
    ax.plot([0, lim], [0, lim], ls=":", lw=0.8, color="#888888")
    ax.set_xlabel("|Delta common accuracy|")
    ax.set_ylabel("Prediction churn vs. Cleveland baseline")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    order = ["cleveland_impute_any", "cleveland_drop_any", "multi_site_impute_any", "multi_site_drop_any"]
    means = [unit[unit["variant"] == v]["common_ece"].mean() for v in order]
    sds = [unit[unit["variant"] == v]["common_ece"].std(ddof=1) for v in order]
    ax.bar(range(len(order)), means, yerr=sds, color="#999999", edgecolor="#333333", linewidth=0.5, capsize=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([VAR_LABELS[v] for v in order], fontsize=7)
    ax.set_ylabel("Common-evaluation ECE")

    ax = axes[2]
    order2 = ["cleveland_impute_any", "cleveland_impute_severe"]
    label_pos = [unit[unit["variant"] == v]["common_label_pos_rate"].mean() for v in order2]
    pred_pos = [unit[unit["variant"] == v]["common_pred_pos_rate"].mean() for v in order2]
    width = 0.36
    x = np.arange(len(order2))
    ax.bar(x - width / 2, label_pos, width=width, color="#56B4E9", label="Label positive rate")
    ax.bar(x + width / 2, pred_pos, width=width, color="#E69F00", label="Predicted positive rate")
    severe_churn = effects[(effects["variant"] == "cleveland_impute_severe") & (effects["outcome"] == "churn_vs_baseline")]
    if len(severe_churn):
        ax.text(0.5, 0.94, f"decision churn={severe_churn['mean'].iloc[0]:.3f}", ha="center", va="top", transform=ax.transAxes, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Any disease", "Severity >= 2"], fontsize=8)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Rate on common evaluation set")
    ax.legend(frameon=False, fontsize=7, loc="lower left")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_heart_lineage_extension.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_heart_lineage_extension.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

    manifest = {
        "figure": "fig_heart_lineage_extension",
        "inputs": [
            str(result_dir / "heart_unit_metrics.csv"),
            str(result_dir / "heart_deltas_vs_baseline.csv"),
            str(result_dir / "heart_effect_sizes.csv"),
        ],
        "outputs": [
            str(figure_dir / "fig_heart_lineage_extension.pdf"),
            str(figure_dir / "fig_heart_lineage_extension.png"),
        ],
    }
    (result_dir / "heart_figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", default=str(ROOT / "results" / "heart_lineage_extension"))
    p.add_argument("--figure-dir", default=str(ROOT / "figures"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result_dir = Path(args.result_dir)
    render_table(result_dir)
    render_figure(result_dir, Path(args.figure_dir))
