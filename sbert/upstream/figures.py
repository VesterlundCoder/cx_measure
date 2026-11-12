"""Figures for the coherence test. Usage: python figures.py OUT_DIR"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(sys.argv[1])
T = pd.read_csv(OUT / "leuven_categories.csv")
R = json.load(open(OUT / "results.json"))
T11 = T[T.domain != "other"]

NAT, ART = "#2a78d6", "#eb6834"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SV = {"bird": "fågel", "fish": "fisk", "insect": "insekt", "mammal": "däggdjur", "reptile": "reptil",
      "clothing": "kläder", "kitchen_utensil": "köksredskap", "musical_instrument": "musikinstrument",
      "tool": "verktyg", "vehicle": "fordon", "weapon": "vapen"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def style(ax):
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def scatter_cats(ax, x, y, label_all=True):
    for dom, col, mk, name in [("animal", NAT, "o", "Djur (naturliga arter)"), ("artifact", ART, "s", "Artefakter")]:
        S = T11[T11.domain == dom]
        ax.scatter(S[x], S[y], s=64, c=col, marker=mk, edgecolors=SURF, linewidths=2, label=name, zorder=3)
        if label_all:
            for r in S.itertuples():
                off = (5, -12) if (r.category == "musical_instrument" and x == "C_own_sal") else (5, 4)
                ax.annotate(SV[r.category], (getattr(r, x), getattr(r, y)), xytext=off,
                            textcoords="offset points", fontsize=8, color=INK2)


# Fig 1: C_X (universal prototype) is a distinctiveness measure
fig, ax = plt.subplots(figsize=(6.4, 4.6))
style(ax)
scatter_cats(ax, "dist_share", "C_univ_sal")
rho = R["H3_category_level"]["C_univ_sal_vs_dist_share"]["rho"]
ax.set_xlabel("Andel av matrisen som bara beror på avståndet till den universella prototypen")
ax.set_ylabel("C_X enligt artikeln")
ax.set_title(f"C_X följer avståndet, inte samvariationen (Spearman ρ = {rho:.2f}, n = 11)", fontsize=10.5, loc="left", color=INK)
ax.legend(frameon=False, loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig1_cx_vs_distinctiveness.png", dpi=200); plt.close(fig)

# Fig 2: two versions vs human similarity
fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2), sharey=True)
for ax, v, ttl in [(axs[0], "C_univ_sal", "Artikelns version (universell prototyp)"),
                   (axs[1], "C_own_sal", "Ren samvariation (eget medelvärde)")]:
    style(ax)
    scatter_cats(ax, v, "human_sim")
    rr = R["H3_category_level"][f"11_domain_categories:{v}"]
    ax.set_title(f"{ttl}\nρ = {rr['rho']:.2f}, p = {rr['p']:.2f}", fontsize=10, loc="left", color=INK)
    ax.set_xlabel("C_X")
axs[0].set_ylabel("Människors bedömda likhet inom kategorin")
axs[1].legend(frameon=False, loc="lower right", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig2_cx_vs_human_similarity.png", dpi=200); plt.close(fig)

# Fig 3: typicality predictors (single series)
H4 = R["H4_typicality"]
items = [("PC1_score_univ", "PC1, universell prototyp (artikeln)"),
         ("contrast_proj_on_m", "Kontrast: riktning mot kategorins mitt"),
         ("family_resemblance_RM", "Familjelikhet (Rosch & Mervis)"),
         ("human_sim_to_others", "Bedömd likhet med övriga exemplar"),
         ("prototype_cosine_unweighted", "Cosinus mot prototypen"),
         ("PC1_alignment_own", "PC1-anpassning, ren samvariation"),
         ("PC1_absscore_own", "|PC1-poäng|, ren samvariation")]
fig, ax = plt.subplots(figsize=(7.4, 3.9))
ax.grid(True, axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
vals = [H4[k]["mean_rho"] for k, _ in items]
y = np.arange(len(items))[::-1]
ax.barh(y, vals, height=0.6, color=NAT, edgecolor=SURF, linewidth=2)
for yi, v, (k, _) in zip(y, vals, items):
    ax.text(v + 0.015, yi, f"{v:.2f}  ({H4[k]['n_positive']}/{H4[k]['n_cats']} kat. positiva)", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(y, [l for _, l in items])
ax.set_xlim(0, 1.0)
ax.set_xlabel("Medelkorrelation med typikalitetsbetyg (Spearman ρ, Fisher-z-medel)")
ax.set_title("Typikalitet förutsägs av kontrast, inte av samvariation", fontsize=10.5, loc="left", color=INK)
fig.tight_layout(); fig.savefig(OUT / "fig3_typicality.png", dpi=200); plt.close(fig)

# Fig 4: size-matched null (pure covariation)
SM = pd.DataFrame(R["H2_superordinate"]["size_matched"]).T
SM["dom"] = ["animal" if c in ("bird", "fish", "insect", "mammal", "reptile") else "artifact" for c in SM.index]
SM = SM.sort_values("real_own")
fig, ax = plt.subplots(figsize=(6.8, 4.3))
style(ax)
yy = np.arange(len(SM))
ax.scatter(SM.null_own_mean, yy, marker="|", s=220, c=INK2, linewidths=2, label="Slumpkategori av samma storlek (medel)", zorder=2)
for i, r in enumerate(SM.itertuples()):
    col, mk = (NAT, "o") if r.dom == "animal" else (ART, "s")
    ax.plot([r.null_own_mean, r.real_own], [i, i], color=GRID, lw=2, zorder=1)
    ax.scatter(r.real_own, i, c=col, marker=mk, s=64, edgecolors=SURF, linewidths=2, zorder=3)
    ax.text(max(r.real_own, r.null_own_mean) + 0.004, i, ("under alla 500 slumpdragningar" if r.pct_own == 0 else f"{int(round(r.pct_own * 100))}:e percentilen"), va="center", fontsize=8, color=INK2)
ax.set_yticks(yy, [SV[c] for c in SM.index])
ax.set_xlabel("C_X utan prototypterm (ren samvariation)")
ax.set_title("Verkliga kategorier jämfört med slumpvis hopblandade", fontsize=10.5, loc="left", color=INK)
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=NAT, label="Djur"),
                   Line2D([], [], marker="s", ls="", color=ART, label="Artefakter"),
                   Line2D([], [], marker="|", ls="", color=INK2, markersize=12, mew=2, label="Slumpkategori, samma n")],
          frameon=False, fontsize=8.5, loc="lower right")
ax.set_xlim(0.12, 0.30)
fig.tight_layout(); fig.savefig(OUT / "fig4_size_matched_null.png", dpi=200); plt.close(fig)
print("ok")
