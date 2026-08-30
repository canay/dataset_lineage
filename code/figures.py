#!/usr/bin/env python3
"""Paper 4 figures. Usage: figures.py <name|all>.
Scripted, colorblind-aware figures with no redundant in-figure titles."""
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = str(Path(__file__).resolve().parents[1])
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.dpi": 110})
OI = {"logreg": "#0072B2", "rf": "#D55E00", "hgb": "#009E73", "extra": "#CC79A7",
      "grey": "#666666", "gold": "#E69F00"}
MODELS = ["logreg", "rf", "hgb"]
MODEL_LAB = {"logreg": "Logistic regression", "rf": "Random forest", "hgb": "Hist.\\ gradient boosting"}
MODEL_LAB2 = {"logreg": "Logistic regression", "rf": "Random forest", "hgb": "Hist. gradient boosting"}
ADULT_VARIANTS = ["V0", "V1_workforce", "V3_drop", "V3_impute", "V4_dedup", "V5_uniform", "V5_hist"]
VLAB = {"V0": "V0 as distributed", "V1_workforce": "V1 workforce filter",
        "V3_drop": "V3a drop missing", "V3_impute": "V3b impute mode",
        "V4_dedup": "V4 deduplicate", "V5_uniform": "V5a uniform split", "V5_hist": "V5b historical split"}
ACS_VARIANTS = ["T25", "T50", "T85"]
TLAB = {"T25": "$25{,}000", "T50": "$50{,}000", "T85": "$84{,}700"}


def save(fig, name, png_dpi=300, pad_inches=0.04):
    fig.savefig(f"{ROOT}/figures/{name}.pdf", bbox_inches="tight",
                pad_inches=pad_inches)
    fig.savefig(f"{ROOT}/figures/{name}.png", bbox_inches="tight",
                pad_inches=pad_inches, dpi=png_dpi)
    plt.close(fig)
    print("saved", name)


def fig_lineage_map():
    L = json.load(open(f"{ROOT}/results/lineage_stats.json"))
    A = json.load(open(f"{ROOT}/results/acs_stats.json"))
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    def box(x, y, w, h, text, fc="#F0F0F0", ec="#333333", fs=7.6, weight="normal"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight)

    def arrow(x1, y1, x2, y2, text="", fs=6.8, color="#333333", ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11,
                                     color=color, lw=1.0, linestyle=ls))
        if text:
            ax.text((x1 + x2) / 2 + 1.0, (y1 + y2) / 2, text, fontsize=fs, ha="left", va="center",
                    color="#111111")

    # left lineage: Adult
    box(2, 88, 30, 8, "1994 U.S. Census Bureau\ncurrent survey database", fc="#DCE9F5", weight="bold")
    arrow(17, 87, 17, 77.5, "Becker extraction (1994):\nAAGE>16, AGI>100,\nAFNLWGT>1, HRSWK>0;\nlabel: income > \\$50,000")
    box(2, 66, 30, 10, "UCI Adult (1996)\nadult.data 32,561 + adult.test 16,281\n'?' missing tokens; test labels with\ntrailing periods; MLC++ 2/3 vs 1/3 split", fc="#DCE9F5")
    arrow(17, 65, 17, 56.5, "merge + recode")
    box(2, 46, 30, 9, "OpenML adult v1 (ID 179, 2014)\ndiscretized features\nOpenML adult v2 (ID 1590, 2015)\n48,842 rows; '?' recoded to NA", fc="#DCE9F5")
    arrow(17, 45, 17, 36.5, "cleaning re-issues")
    box(2, 26, 30, 9, "OpenML adult v3 (ID 43898, 2022)\n52 exact duplicates dropped\nOpenML adult v4 (ID 45068, 2023)", fc="#DCE9F5")

    # middle: reproduced decisions
    box(38, 46, 27, 9, "Reproduced lineage decisions\n(this paper, controlled variants)", fc="#FDEBD3", weight="bold", fs=8)
    txt = (f"V1 workforce filter: $-${L['workforce_filter_removed']:,} rows\n"
           f"V3a drop missing: $-${L['rows_with_any_missing']:,} rows\n"
           f"V3b impute mode: 0 rows\n"
           f"V4 deduplicate: $-${L['exact_duplicates_features_and_label']:,} rows\n"
           f"V5 split policy: historical vs.\nuniform vs. stratified")
    box(38, 26, 27, 16, txt, fc="#FDF6EC", fs=7.4)
    arrow(33, 50, 37.6, 50)
    arrow(51, 45.6, 51, 42.8)

    # right lineage: ACSIncome
    box(70, 88, 28, 8, "ACS PUMS 2018 1-Year\n(U.S. Census Bureau)", fc="#DFF2E9", weight="bold")
    arrow(84, 87, 84, 77.5, "folktables ACSIncome filter:\nAGEP>16, PINCP>100,\nWKHP>0, PWGTP$\\geq$1")
    box(70, 66, 28, 10, "ACSIncome (Ding et al. 2021)\n1,664,500 rows (OpenML 43141)\nlabel: PINCP > \\$50,000", fc="#DFF2E9")
    arrow(84, 65, 84, 56.5, f"Oregon subset (this paper):\n{A['n_raw_person_records']:,} person records\n$\\rightarrow$ {A['n_after_feature_na_drop']:,} after filter")
    t2 = (f"V2 label threshold re-derivation\n"
          f"$>\\$25{{,}}000$: pos.\\ rate {A['pos_rate_T25']:.3f}\n"
          f"$>\\$50{{,}}000$: pos.\\ rate {A['pos_rate_T50']:.3f}\n"
          f"$>\\$84{{,}}700$ (CPI 1994$\\rightarrow$2018): {A['pos_rate_T85']:.3f}")
    box(70, 42, 28, 13, t2, fc="#FDF6EC", fs=7.4)

    # dashed re-derivation arrow
    arrow(32.5, 70, 69.5, 70, "re-derivation\n(same task concept)", color="#888888", ls="--")
    save(fig, "fig_lineage_map")


def fig_effect_forest():
    eff = pd.read_csv(f"{ROOT}/results/adult_effect_sizes.csv")
    outs = [("d_common_acc", "$\\Delta$ accuracy"), ("d_common_auc", "$\\Delta$ AUC"),
            ("d_common_ece", "$\\Delta$ ECE"), ("d_sex_tpr_gap", "$\\Delta$ TPR gap (sex)"),
            ("churn_vs_base", "Churn vs. V0")]
    variants = [v for v in ADULT_VARIANTS if v != "V0"]
    fig, axes = plt.subplots(1, len(outs), figsize=(11.5, 3.4), sharey=True)
    for j, (o, lab) in enumerate(outs):
        ax = axes[j]
        for i, v in enumerate(variants):
            for k, m in enumerate(MODELS):
                r = eff[(eff.variant == v) & (eff.outcome == o) & (eff.model == m)]
                if not len(r): continue
                y = len(variants) - 1 - i + (k - 1) * 0.22
                ax.errorbar(r["mean"].iloc[0], y,
                            xerr=[[r["mean"].iloc[0] - r["min"].iloc[0]],
                                  [r["max"].iloc[0] - r["mean"].iloc[0]]],
                            fmt="o", ms=3.5, color=OI[m], elinewidth=1, capsize=2,
                            label=MODEL_LAB2[m] if (i == 0 and j == 0) else None)
        if o != "churn_vs_base":
            ax.axvline(0, color="#999999", lw=0.8, ls=":")
        ax.set_xlabel(lab, fontsize=8.5)
        ax.tick_params(labelsize=8)
        if j == 0:
            ax.set_yticks(range(len(variants)))
            ax.set_yticklabels([VLAB[v] for v in variants[::-1]], fontsize=8)
    axes[0].legend(loc="lower left", fontsize=7, frameon=False, bbox_to_anchor=(0, 1.0), ncol=3)
    fig.tight_layout()
    save(fig, "fig_effect_forest")


def fig_acc_churn():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5))
    for ax, case, variants in ((axes[0], "adult", ADULT_VARIANTS[1:]), (axes[1], "acs", ["T25", "T85"])):
        dd = pd.read_csv(f"{ROOT}/results/{case}_deltas_vs_base.csv")
        markers = dict(zip(variants, ["o", "s", "^", "D", "v", "P", "X"]))
        for m in MODELS:
            for v in variants:
                sub = dd[(dd.model == m) & (dd.variant == v)]
                ax.scatter(sub["abs_d_common_acc"], sub["churn_vs_base"], s=22,
                           marker=markers[v], facecolor=OI[m], edgecolor="white", lw=0.3, alpha=0.85)
        lim = max(dd["churn_vs_base"].max(), dd["abs_d_common_acc"].max()) * 1.15
        ax.plot([0, lim], [0, lim], color="#999999", lw=0.8, ls=":")
        ax.set_xlabel("$|\\Delta$ accuracy$|$ on common evaluation set")
        ax.set_ylabel("Prediction churn vs. baseline")
        ax.tick_params(labelsize=8)
        # marker legend
        from matplotlib.lines import Line2D
        tlab2 = {"T25": "> \\$25,000", "T85": "> \\$84,700"}
        h1 = [Line2D([0], [0], marker=markers[v], color="#555555", ls="", ms=5,
                     label=tlab2[v] if v.startswith("T") else VLAB[v])
              for v in variants]
        h2 = [Line2D([0], [0], marker="o", color=OI[m], ls="", ms=5, label=MODEL_LAB2[m]) for m in MODELS]
        ax.legend(handles=h1 + h2, fontsize=6.4, frameon=False, loc="lower right", ncol=1)
    fig.tight_layout()
    save(fig, "fig_acc_churn")


def fig_subgroup():
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.3))
    um = pd.read_csv(f"{ROOT}/results/unit_metrics.csv")
    panels = [("adult", "sex_dp_diff", "Demographic parity diff. (sex)", ADULT_VARIANTS),
              ("adult", "sex_tpr_gap", "TPR gap (sex)", ADULT_VARIANTS),
              ("acs", "sex_tpr_gap", "TPR gap (sex), ACSIncome", ACS_VARIANTS)]
    for ax, (case, met, lab, variants) in zip(axes, panels):
        sub = um[um.case == case]
        for i, v in enumerate(variants):
            for k, m in enumerate(MODELS):
                x = sub[(sub.variant == v) & (sub.model == m)][met]
                xpos = i + (k - 1) * 0.22
                ax.errorbar(xpos, x.mean(), yerr=[[x.mean() - x.min()], [x.max() - x.mean()]],
                            fmt="o", ms=4, color=OI[m], elinewidth=1, capsize=2)
        ax.set_xticks(range(len(variants)))
        labels = [VLAB.get(v, v).replace(" ", "\n", 1) if case == "adult" else TLAB[v].replace("{,}", ",").replace("$", "\\$") for v in variants]
        if case == "acs":
            labels = ["> $25,000", "> $50,000", "> $84,700"]
        ax.set_xticklabels(labels, fontsize=6.8, rotation=38, ha="right")
        ax.set_ylabel(lab, fontsize=8.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.axhline(0, color="#999999", lw=0.7, ls=":")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color=OI[m], ls="", ms=5, label=MODEL_LAB2[m]) for m in MODELS]
    axes[0].legend(handles=handles, fontsize=6.5, frameon=False, loc="lower left")
    axes[0].set_ylim(bottom=0)
    fig.tight_layout()
    save(fig, "fig_subgroup")


def fig_imp_heatmap():
    ic = pd.read_csv(f"{ROOT}/results/adult_importance_rankcorr.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    arr = ic.loc[ADULT_VARIANTS, ADULT_VARIANTS].to_numpy()
    im = ax.imshow(arr, vmin=arr.min() - 0.02, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(ADULT_VARIANTS)))
    ax.set_yticks(range(len(ADULT_VARIANTS)))
    ax.set_xticklabels([VLAB[v] for v in ADULT_VARIANTS], rotation=38, ha="right", fontsize=7.5)
    ax.set_yticklabels([VLAB[v] for v in ADULT_VARIANTS], fontsize=7.5)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=6.6,
                    color="white" if arr[i, j] < (arr.min() + 1.0) / 2 else "black")
    cb = fig.colorbar(im, ax=ax, shrink=0.85)
    cb.set_label("Spearman rank correlation of permutation importances", fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    save(fig, "fig_imp_heatmap")


def fig_workflow():
    """Render the six-stage audit as a single left-to-right methodology flow."""
    fig, ax = plt.subplots(figsize=(7.15, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    lato_bold = font_dir / "Lato-Bold.ttf"
    liberation_regular = font_dir / "LiberationSans-Regular.ttf"
    missing_fonts = [str(path) for path in (lato_bold, liberation_regular) if not path.is_file()]
    if missing_fonts:
        raise FileNotFoundError(f"Required embedded figure fonts are missing: {missing_fonts}")
    heading_font = FontProperties(fname=str(lato_bold), size=8.4)
    body_font = FontProperties(fname=str(liberation_regular), size=8.0)
    group_font = FontProperties(fname=str(lato_bold), size=8.5)
    edge = "#17324D"
    text = "#1F2933"
    muted = "#4F6B7A"
    fill_documentation = "#EAF0F4"
    fill_behavior = "#DDE6EB"

    def box(x, y, w, h, heading, body, fc):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            fc=fc, ec=edge, lw=1.05
        ))
        ax.text(
            x + w / 2, y + h * 0.70, heading,
            ha="center", va="center",
            fontproperties=heading_font, color=edge
        )
        ax.text(
            x + w / 2, y + h * 0.36, body,
            ha="center", va="center",
            fontproperties=body_font, color=text, linespacing=1.12
        )

    def connector(start, end):
        ax.add_patch(FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=10,
            color=edge, lw=1.05, shrinkA=0, shrinkB=0,
            connectionstyle="arc3,rad=0"
        ))

    stages = [
        ("DATASET\nEVIDENCE", "Files, versions,\nand metadata", fill_documentation),
        ("GENEALOGY", "Reconstructed\nderivation paths", fill_documentation),
        ("INTERVENTIONS", "Controlled lineage\nvariants", fill_behavior),
        ("EVALUATION", "Fixed models and\ncommon records", fill_behavior),
        ("MEASUREMENT", "Accuracy, AUC, ECE,\nchurn, gaps, and\nimportance", fill_behavior),
        ("INTERPRETATION", "Bounded lineage-\nsensitivity claims", fill_behavior),
    ]

    w, h = 0.142, 0.44
    y = 0.25
    positions = [(x, y) for x in (0.018, 0.184, 0.350, 0.516, 0.682, 0.848)]
    for (heading, body, fc), (x, y) in zip(stages, positions):
        box(x, y, w, h, heading, body, fc)

    mid = y + h / 2
    for (x0, _), (x1, _) in zip(positions[:-1], positions[1:]):
        connector((x0 + w + 0.004, mid), (x1 - 0.004, mid))

    ax.text(
        0.171, 0.86, "DOCUMENTATION EVIDENCE",
        ha="center", va="center",
        fontproperties=group_font, color=muted
    )
    ax.text(
        0.670, 0.86, "BEHAVIORAL EVIDENCE ON FROZEN COMMON RECORDS",
        ha="center", va="center",
        fontproperties=group_font, color=muted
    )
    ax.plot([0.018, 0.324], [0.79, 0.79], color="#8FA7B5", lw=1.0)
    ax.plot([0.350, 0.990], [0.79, 0.79], color="#8FA7B5", lw=1.0)
    ax.plot([0.334, 0.334], [0.19, 0.89], color="#8FA7B5", lw=0.9, ls="--")

    fig.subplots_adjust(left=0.004, right=0.996, bottom=0.05, top=0.98)
    save(fig, "fig_workflow", png_dpi=600, pad_inches=0.02)


FIGS = {"lineage": fig_lineage_map, "forest": fig_effect_forest, "churn": fig_acc_churn,
        "subgroup": fig_subgroup, "heatmap": fig_imp_heatmap, "workflow": fig_workflow}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for k, f in FIGS.items():
        if which in ("all", k):
            f()

