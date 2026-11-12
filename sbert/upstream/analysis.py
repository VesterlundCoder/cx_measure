"""Test Osta-Velez & Gardenfors' coherence measure C_X on human norm data.

Data
  Leuven Natural Concept Database (De Deyne et al., 2008; cleaned English-label copy
  github.com/mikulatomas/dutch-concepts): exemplar x feature applicability (0-4 raters),
  feature importance ratings (-3..3), typicality ratings, pairwise similarity ratings.
  McRae et al. (2005) feature production norms as a replication.

Usage: python analysis.py LEUVEN_DIR MCRAE_XLSX OUT_DIR
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from coherence import coherence, pc_scores

LEU, MCRAE, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
EF = LEU / "exemplar_feature_judgments/exemplar_features"

DOMAINS = {"animal": ["bird", "fish", "insect", "mammal", "reptile"],
           "artifact": ["clothing", "kitchen_utensil", "musical_instrument", "tool", "vehicle", "weapon"]}
OTHER = ["fruit", "vegetable", "profession", "sport"]
NATURAL = set(DOMAINS["animal"])
ALL15 = DOMAINS["animal"] + DOMAINS["artifact"] + OTHER
R_MAIN = 0.5
R_GRID = [0.3, 0.5, 0.7, 0.9]


def load_cat(c):
    X = pd.read_csv(EF / f"categories/{c}/{c}_exemplar_features_sum.csv", index_col=0) / 4.0
    imp = pd.read_csv(EF / f"feature_importance_ratings/{c}_feature_importance_ratings.csv", index_col=0)["mean"]
    sal = ((imp.reindex(X.columns) + 3) / 6).fillna(0).values      # map -3..3 -> 0..1
    typ = pd.read_csv(LEU / f"exemplar_judgments/typicality_ratings/{c}_typicality_ratings.csv", index_col=0)["mean"]
    sim = pd.read_csv(LEU / f"pairwise_similarities/{c}/{c}_pairwise_similarities_mean.csv", index_col=0)
    return X, sal, typ, sim


def load_domain(dom):
    return pd.read_csv(EF / f"domains/{dom}/{dom}_exemplar_features_sum.csv", index_col=0) / 4.0


def mean_offdiag(M):
    M = np.asarray(M, float)
    return M[~np.eye(len(M), dtype=bool)].mean()


def family_resemblance(X, sal):
    """Baseline: mean pairwise cosine similarity of salience-weighted feature vectors."""
    Z = X * sal
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    return mean_offdiag(Z @ Z.T)


def perm_test_diff(vals, labels, stat=np.mean):
    """Exact permutation test for difference natural - artifact."""
    vals, labels = np.asarray(vals), np.asarray(labels, bool)
    obs = stat(vals[labels]) - stat(vals[~labels])
    k, n = labels.sum(), len(labels)
    diffs = []
    for idx in itertools.combinations(range(n), k):
        m = np.zeros(n, bool); m[list(idx)] = True
        diffs.append(stat(vals[m]) - stat(vals[~m]))
    diffs = np.array(diffs)
    return float(obs), float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12))


res = {}

# ---------------------------------------------------------------- Leuven: per category
rows = []
cats = {c: load_cat(c) for c in ALL15}
doms = {d: load_domain(d) for d in DOMAINS}
for dom, members in DOMAINS.items():
    D = doms[dom]
    for c in members:
        Xc, sal, typ, sim = cats[c]
        feats = Xc.columns
        # domain matrix (one rater group for all exemplars) -> consistent universal prototype
        X = D.loc[Xc.index, feats]
        u = D[feats].values.mean(0)                        # universal prototype (domain mean)
        u_full = D.values.mean(0)
        Xfull = D.loc[X.index].values
        row = dict(category=c, domain=dom, n=len(X), d=len(feats))
        for r in R_GRID:
            row[f"C_univ_sal_r{r}"] = coherence(X.values, sal, u, r)["C"]
            row[f"C_own_sal_r{r}"] = coherence(X.values, sal, "own", r)["C"]
        cu = coherence(X.values, sal, u, R_MAIN)
        row.update(C_univ_sal=cu["C"], dist_share=cu["dist_share"], p1_univ=cu["p1"],
                   C_own_sal=coherence(X.values, sal, "own", R_MAIN)["C"],
                   C_univ_uni=coherence(X.values, None, u, R_MAIN)["C"],
                   C_own_uni=coherence(X.values, None, "own", R_MAIN)["C"],
                   C_univ_fullspace=coherence(Xfull, None, u_full, R_MAIN)["C"],
                   C_own_fullspace=coherence(Xfull, None, "own", R_MAIN)["C"],
                   C_own_sal_catmat=coherence(Xc.values, sal, "own", R_MAIN)["C"],
                   fam_res=family_resemblance(X.values, sal),
                   fam_res_catmat=family_resemblance(Xc.values, sal),
                   human_sim=mean_offdiag(sim.loc[X.index, X.index].values),
                   typ_mean=typ.reindex(X.index).mean(), typ_sd=typ.reindex(X.index).std())
        # size-matched (n=20) versions: mean over 300 random subsamples
        rng2 = np.random.default_rng(2)
        subs = [rng2.choice(len(X), 20, replace=False) for _ in range(300)]
        row["C_univ_sal_n20"] = float(np.mean([coherence(X.values[i], sal, u, R_MAIN)["C"] for i in subs]))
        row["C_own_sal_n20"] = float(np.mean([coherence(X.values[i], sal, "own", R_MAIN)["C"] for i in subs]))
        # bootstrap CI for headline variant
        rng = np.random.default_rng(1)
        bs = [coherence(X.values[rng.integers(0, len(X), len(X))], sal, u, R_MAIN)["C"] for _ in range(1000)]
        row["C_univ_sal_lo"], row["C_univ_sal_hi"] = np.percentile(bs, [2.5, 97.5])
        rows.append(row)
for c in OTHER:                                            # not part of a domain matrix
    X, sal, typ, sim = cats[c]
    row = dict(category=c, domain="other", n=len(X), d=X.shape[1],
               C_own_sal=np.nan, C_own_sal_catmat=coherence(X.values, sal, "own", R_MAIN)["C"],
               fam_res_catmat=family_resemblance(X.values, sal),
               C_own_uni=coherence(X.values, None, "own", R_MAIN)["C"],
               fam_res=family_resemblance(X.values, sal),
               human_sim=mean_offdiag(sim.loc[X.index, X.index].values),
               typ_mean=typ.reindex(X.index).mean(), typ_sd=typ.reindex(X.index).std())
    for r in R_GRID:
        row[f"C_own_sal_r{r}"] = coherence(X.values, sal, "own", r)["C"]
    rows.append(row)
T = pd.DataFrame(rows)
T.to_csv(OUT / "leuven_categories.csv", index=False)

# ---------------------------------------------------------------- H1 natural vs artifact
T11 = T[T.domain != "other"].reset_index(drop=True)
lab = T11.domain.eq("animal").values
H1 = {}
for v in ["C_univ_sal", "C_own_sal", "C_univ_uni", "C_own_uni", "C_univ_fullspace", "C_own_fullspace",
          "C_univ_sal_n20", "C_own_sal_n20", "dist_share", "fam_res", "human_sim"] + [f"C_univ_sal_r{r}" for r in R_GRID]:
    diff, p = perm_test_diff(T11[v].values, lab)
    H1[v] = dict(mean_natural=float(T11.loc[lab, v].mean()), mean_artifact=float(T11.loc[~lab, v].mean()),
                 diff=diff, p_perm=p)
res["H1_natural_vs_artifact"] = H1

# ---------------------------------------------------------------- H2 superordinate vs subordinate
H2 = {}
for dom, members in DOMAINS.items():
    D = doms[dom]
    sup_own = coherence(D.values, None, "own", R_MAIN)["C"]  # superordinate: ref = its own mean = domain prototype
    subs = {c: float(T11.set_index("category").loc[c, "C_univ_fullspace"]) for c in members}
    subs_own = {c: float(T11.set_index("category").loc[c, "C_own_fullspace"]) for c in members}
    H2[dom] = dict(superordinate_C=sup_own, sub_C_univ=subs, sub_C_own=subs_own,
                   n_sub_above_univ=int(sum(v > sup_own for v in subs.values())),
                   n_sub_above_own=int(sum(v > sup_own for v in subs_own.values())), n_sub=len(members))
# control: random 'pseudo-categories' of 25 exemplars drawn across the domain's subcategories
rng3 = np.random.default_rng(3)
for dom in DOMAINS:
    D = doms[dom].values
    sizes = [len(cats[c][0]) for c in DOMAINS[dom]]
    mix = [coherence(D[rng3.choice(len(D), int(np.median(sizes)), replace=False)], None, "own", R_MAIN)["C"]
           for _ in range(500)]
    H2[dom]["random_mix_C_mean"] = float(np.mean(mix))
    H2[dom]["random_mix_C_95"] = [float(x) for x in np.percentile(mix, [2.5, 97.5])]
# size-matched test: each real subcategory vs 500 random same-size draws from its whole domain
H2["size_matched"] = {}
for dom, members in DOMAINS.items():
    D = doms[dom]
    u_full = D.values.mean(0)
    for c in members:
        idx = cats[c][0].index
        Xr = D.loc[idx].values
        real_own = coherence(Xr, None, "own", R_MAIN)["C"]
        real_univ = coherence(Xr, None, u_full, R_MAIN)["C"]
        draws = [D.values[rng3.choice(len(D), len(idx), replace=False)] for _ in range(500)]
        own_null = np.array([coherence(Z, None, "own", R_MAIN)["C"] for Z in draws])
        univ_null = np.array([coherence(Z, None, u_full, R_MAIN)["C"] for Z in draws])
        H2["size_matched"][c] = dict(n=len(idx), real_own=real_own, null_own_mean=float(own_null.mean()),
                                     pct_own=float((own_null < real_own).mean()),
                                     real_univ=real_univ, null_univ_mean=float(univ_null.mean()),
                                     pct_univ=float((univ_null < real_univ).mean()))
res["H2_superordinate"] = H2

# ---------------------------------------------------------------- H3 category level vs human similarity
H3 = {}
for subset, TT in [("11_domain_categories", T11), ("all_15", T)]:
    for v in ["C_univ_sal", "C_own_sal", "C_univ_uni", "C_own_uni", "C_univ_fullspace", "dist_share", "fam_res",
              "C_own_sal_catmat", "fam_res_catmat", "C_univ_sal_n20", "C_own_sal_n20"]:
        if TT[v].isna().any():
            continue
        rho, p = stats.spearmanr(TT[v], TT.human_sim)
        H3[f"{subset}:{v}"] = dict(rho=float(rho), p=float(p), n=len(TT))
# partial: does C_own_sal predict human_sim beyond family resemblance? (rank-based)
def partial_spearman(x, y, z):
    rx, ry, rz = (stats.rankdata(a) for a in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return stats.pearsonr(ex, ey)
for v, fr in [("C_own_sal", "fam_res"), ("C_univ_sal", "fam_res"), ("C_own_sal_catmat", "fam_res_catmat")]:
    TT = T if not T[v].isna().any() else T11
    r_, p_ = partial_spearman(TT[v], TT.human_sim, TT[fr])
    H3[f"partial|fam_res:{v}"] = dict(r=float(r_), p=float(p_), n=len(TT))
for v in ["C_univ_sal", "C_own_sal"]:
    rho, p = stats.spearmanr(T11[v], T11.dist_share)
    H3[f"{v}_vs_dist_share"] = dict(rho=float(rho), p=float(p), n=len(T11))
H3["r_robustness"] = {}
for r in R_GRID:
    a = stats.spearmanr(T11[f"C_univ_sal_r{r}"], T11.human_sim)[0]
    b = stats.spearmanr(T11[f"C_own_sal_r{r}"], T11.human_sim)[0]
    c_ = stats.spearmanr(T11[f"C_univ_sal_r{r}"], T11.dist_share)[0]
    H3["r_robustness"][str(r)] = dict(univ_vs_humansim=float(a), own_vs_humansim=float(b), univ_vs_distshare=float(c_))
res["H3_category_level"] = H3

# ---------------------------------------------------------------- H4 exemplar typicality
H4rows = []
for c in ALL15:
    X, sal, typ, sim = cats[c]
    if c not in OTHER:
        dom = [d for d, m in DOMAINS.items() if c in m][0]
        X = doms[dom].loc[X.index, X.columns]
    t = typ.reindex(X.index).values
    ok = ~np.isnan(t)
    row = dict(category=c)
    preds = {}
    if c not in OTHER:
        dom = [d for d, m in DOMAINS.items() if c in m][0]
        u = doms[dom][X.columns].values.mean(0)
        sc, _ = pc_scores(X.values, sal, u, k=1)
        preds["PC1_score_univ"] = sc[:, 0]
    sc_o, Z_o = pc_scores(X.values, sal, "own", k=1)
    preds["PC1_absscore_own"] = np.abs(sc_o[:, 0])
    preds["PC1_alignment_own"] = sc_o[:, 0] ** 2 / ((Z_o ** 2).sum(1) + 1e-12)
    Zw = X.values * sal
    cen = Zw.mean(0)
    preds["family_resemblance_RM"] = Zw @ X.values.mean(0)          # Rosch & Mervis style
    Xu = X.values; cu_ = Xu.mean(0)
    preds["prototype_cosine_unweighted"] = Xu @ cu_ / (np.linalg.norm(Xu, axis=1) * np.linalg.norm(cu_) + 1e-12)
    if c not in OTHER:
        m = (X.values.mean(0) - u) * sal
        preds["contrast_proj_on_m"] = ((X.values - u) * sal) @ (m / np.linalg.norm(m))
        _, Zu = pc_scores(X.values, sal, u, k=1)
        U_, S_, Vt_ = np.linalg.svd(Zu, full_matrices=False)
        row["cos_PC1_m"] = float(abs(Vt_[0] @ m) / np.linalg.norm(m))
    preds["prototype_cosine"] = Zw @ cen / (np.linalg.norm(Zw, axis=1) * np.linalg.norm(cen) + 1e-12)
    S = sim.loc[X.index, X.index].values.copy(); np.fill_diagonal(S, np.nan)
    preds["human_sim_to_others"] = np.nanmean(S, 1)
    for k, v in preds.items():
        row[k] = float(stats.spearmanr(v[ok], t[ok])[0])
    if "contrast_proj_on_m" in preds:
        row["rho_PC1univ_contrast"] = float(stats.spearmanr(preds["PC1_score_univ"], preds["contrast_proj_on_m"])[0])
    H4rows.append(row)
H4 = pd.DataFrame(H4rows)
H4.to_csv(OUT / "leuven_typicality.csv", index=False)
summ = {}
for k in [c_ for c_ in H4.columns[1:] if c_ not in ("cos_PC1_m", "rho_PC1univ_contrast")]:
    v = H4[k].dropna().values
    z = np.arctanh(np.clip(v, -0.999, 0.999))
    summ[k] = dict(mean_rho=float(np.tanh(z.mean())), n_cats=len(v), n_positive=int((v > 0).sum()),
                   wilcoxon_p=float(stats.wilcoxon(v).pvalue))
# head-to-head: PC1 univ vs prototype within categories
both = H4.dropna(subset=["PC1_score_univ"])
summ["PC1_univ_minus_prototype"] = dict(mean_diff=float((both.PC1_score_univ - both.prototype_cosine).mean()),
                                        n_PC1_better=int((both.PC1_score_univ > both.prototype_cosine).sum()),
                                        n=len(both))
summ["cos_PC1_m_mean"] = float(H4.cos_PC1_m.mean())
summ["rho_PC1univ_contrast_mean"] = float(H4.rho_PC1univ_contrast.mean())
summ["PC1_univ_minus_contrast"] = dict(mean_diff=float((both.PC1_score_univ - both.contrast_proj_on_m).mean()),
                                       n_PC1_better=int((both.PC1_score_univ > both.contrast_proj_on_m).sum()), n=len(both))
res["H4_typicality"] = summ

# ---------------------------------------------------------------- McRae replication
mc = pd.read_excel(MCRAE)
tax = mc[mc.BR_Label == "taxonomic"]
non = mc[mc.BR_Label != "taxonomic"]
M = non.pivot_table(index="Concept", columns="Feature", values="Prod_Freq", aggfunc="sum").fillna(0) / 30.0
M = M.reindex(sorted(mc.Concept.unique())).fillna(0)
u_all = M.values.mean(0)
MC_CATS = {"mammal": "a_mammal", "bird": "a_bird", "fish": "a_fish", "fruit": "a_fruit",
           "vegetable": "a_vegetable", "weapon": "a_weapon", "musical_instrument": "a_musical_instrument",
           "tool": "a_tool", "utensil": "a_utensil", "vehicle": "a_vehicle"}
MC_NAT = {"mammal", "bird", "fish"}
mrows = []
for c, f in MC_CATS.items():
    mem = sorted(tax[tax.Feature == f].Concept.unique())
    X = M.loc[mem].values
    keep = X.sum(0) > 0
    cu = coherence(X[:, keep], None, u_all[keep], R_MAIN)
    mrows.append(dict(category=c, kind="natural" if c in MC_NAT else ("food" if c in {"fruit", "vegetable"} else "artifact"),
                      n=len(mem), d=int(keep.sum()), C_univ=cu["C"], dist_share=cu["dist_share"],
                      C_own=coherence(X[:, keep], None, "own", R_MAIN)["C"],
                      C_univ_fullspace=coherence(X, None, u_all, R_MAIN)["C"],
                      C_own_fullspace=coherence(X, None, "own", R_MAIN)["C"]))
MT = pd.DataFrame(mrows)
MT.to_csv(OUT / "mcrae_categories.csv", index=False)
m8 = MT[MT.kind != "food"].reset_index(drop=True)
lab = m8.kind.eq("natural").values
res["McRae_H1"] = {v: dict(zip(["diff", "p_perm"], perm_test_diff(m8[v].values, lab)),
                           mean_natural=float(m8.loc[lab, v].mean()), mean_artifact=float(m8.loc[~lab, v].mean()))
                   for v in ["C_univ", "C_own", "C_univ_fullspace", "C_own_fullspace", "dist_share"]}
living = sorted(set(tax[tax.Feature.isin([MC_CATS[c] for c in MC_NAT])].Concept))
res["McRae_H2"] = dict(superordinate_animals_C=coherence(M.loc[living].values, None, "own", R_MAIN)["C"],
                       thing_all541_C=coherence(M.values, None, "own", R_MAIN)["C"],
                       sub_C_univ_fullspace={r.category: r.C_univ_fullspace for r in MT.itertuples() if r.category in MC_NAT},
                       sub_C_own_fullspace={r.category: r.C_own_fullspace for r in MT.itertuples() if r.category in MC_NAT})
rho, p = stats.spearmanr(MT.C_univ, MT.dist_share)
res["McRae_C_univ_vs_dist_share"] = dict(rho=float(rho), p=float(p), n=len(MT))

json.dump(res, open(OUT / "results.json", "w"), indent=2, default=float)
pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
print(T[["category", "domain", "n", "d", "C_univ_sal", "C_univ_sal_n20", "C_own_sal_n20", "C_own_sal_catmat", "C_univ_sal_lo", "C_univ_sal_hi", "C_own_sal", "C_univ_uni",
         "C_own_uni", "C_univ_fullspace", "dist_share", "fam_res", "human_sim", "typ_mean"]])
print(H4)
print(MT)
print(json.dumps(res, indent=1, default=float))
