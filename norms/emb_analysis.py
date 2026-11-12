"""Sentence-embedding version of the coherence test.

Embeds every Leuven exemplar name with sentence encoders and asks:
  E1  pair level: does embedding cosine track human pairwise similarity (vs the feature space)?
  E2  category level: does mean within-category cosine track human within-category similarity,
      and does it separate natural kinds from artifacts?
  E3  C_X computed in embedding space (universal prototype = mean of ALL 415 exemplars, i.e. a real
      'thing' prototype across both domains): H1, link to human similarity, origin artefact.
  E4  typicality: PC1 (universal) vs contrast vs cosine to centroid vs cosine to the category label.
  E5  real categories vs size-matched random mixtures, pure-covariation C_X.

Usage: python emb_analysis.py LEUVEN_EN_DIR OUT_DIR MODEL_SPEC...
MODEL_SPEC = name:model.onnx:tokenizer.json:pooling
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "emb"))
from coherence import coherence, pc_scores  # noqa: E402
from embed import Embedder  # noqa: E402

LEU, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SPECS = [s.split(":") for s in sys.argv[3:]]
OUT.mkdir(parents=True, exist_ok=True)
EF = LEU / "exemplar_feature_judgments/exemplar_features"

ANIMAL = ["bird", "fish", "insect", "mammal", "reptile"]
ARTIFACT = ["clothing", "kitchen_utensil", "musical_instrument", "tool", "vehicle", "weapon"]
OTHER = ["fruit", "vegetable", "profession", "sport"]
ALL15 = ANIMAL + ARTIFACT + OTHER
LABEL = {c: c.replace("_", " ") for c in ALL15}
R = 0.5


def mean_offdiag(M):
    return M[~np.eye(len(M), dtype=bool)].mean()


def offdiag_pairs(M):
    i, j = np.triu_indices(len(M), 1)
    return M[i, j]


def perm_diff(vals, lab):
    vals, lab = np.asarray(vals), np.asarray(lab, bool)
    obs = vals[lab].mean() - vals[~lab].mean()
    n, k = len(vals), lab.sum()
    d = []
    for idx in itertools.combinations(range(n), k):
        m = np.zeros(n, bool); m[list(idx)] = True
        d.append(vals[m].mean() - vals[~m].mean())
    return float(obs), float(np.mean(np.abs(d) >= abs(obs) - 1e-12))


# ---- data
cats = {}
for c in ALL15:
    X = pd.read_csv(EF / f"categories/{c}/{c}_exemplar_features_sum.csv", index_col=0) / 4.0
    imp = pd.read_csv(EF / f"feature_importance_ratings/{c}_feature_importance_ratings.csv", index_col=0)["mean"]
    sal = ((imp.reindex(X.columns) + 3) / 6).fillna(0).values
    typ = pd.read_csv(LEU / f"exemplar_judgments/typicality_ratings/{c}_typicality_ratings.csv", index_col=0)["mean"]
    sim = pd.read_csv(LEU / f"pairwise_similarities/{c}/{c}_pairwise_similarities_mean.csv", index_col=0)
    cats[c] = dict(names=list(X.index), X=X.values, sal=sal, typ=typ.reindex(X.index).values,
                   sim=sim.loc[X.index, X.index].values)

res = {}
for name, mpath, tpath, pool in SPECS:
    emb = Embedder(mpath, tpath, pooling=pool)
    R_ = {}
    for template in ["{}", "a {}"]:
        E = {c: emb.encode([template.format(w) for w in cats[c]["names"]]) for c in ALL15}
        L = {c: emb.encode([template.format(LABEL[c])])[0] for c in ALL15}
        allE = np.vstack([E[c] for c in ALL15])
        u_all = allE.mean(0)
        u_dom = {c: np.vstack([E[k] for k in (ANIMAL if c in ANIMAL else ARTIFACT if c in ARTIFACT else [c])]).mean(0)
                 for c in ALL15}
        rows, trows = [], []
        rng = np.random.default_rng(7)
        for c in ALL15:
            d = cats[c]
            S_emb = E[c] @ E[c].T
            Zf = d["X"] * d["sal"]
            Zf = Zf / (np.linalg.norm(Zf, axis=1, keepdims=True) + 1e-12)
            S_feat = Zf @ Zf.T
            hp = offdiag_pairs(d["sim"])
            cu = coherence(E[c], None, u_all, R)
            # size-matched random mixtures from ALL exemplars (pure covariation)
            null = [coherence(allE[rng.choice(len(allE), len(E[c]), replace=False)], None, "own", R)["C"]
                    for _ in range(300)]
            real_own = coherence(E[c], None, "own", R)["C"]
            rows.append(dict(
                category=c, kind="natural" if c in ANIMAL else ("artifact" if c in ARTIFACT else "other"),
                pair_rho_emb=float(stats.spearmanr(offdiag_pairs(S_emb), hp)[0]),
                pair_rho_feat=float(stats.spearmanr(offdiag_pairs(S_feat), hp)[0]),
                emb_sim=float(mean_offdiag(S_emb)), feat_sim=float(mean_offdiag(S_feat)),
                human_sim=float(mean_offdiag(d["sim"])),
                C_univ_all=cu["C"], dist_share_all=cu["dist_share"],
                C_univ_dom=coherence(E[c], None, u_dom[c], R)["C"] if c not in OTHER else np.nan,
                C_own=real_own, null_own_mean=float(np.mean(null)),
                pct_own_vs_random=float(np.mean(np.array(null) < real_own))))
            # typicality
            t = d["typ"]; ok = ~np.isnan(t)
            sc, _ = pc_scores(E[c], None, u_all, k=1)
            m = E[c].mean(0) - u_all
            preds = dict(PC1_univ=sc[:, 0],
                         contrast=(E[c] - u_all) @ (m / np.linalg.norm(m)),
                         cos_centroid=E[c] @ E[c].mean(0) / np.linalg.norm(E[c].mean(0)),
                         cos_label=E[c] @ L[c])
            sc_o, Z_o = pc_scores(E[c], None, "own", k=1)
            preds["PC1_own_abs"] = np.abs(sc_o[:, 0])
            trow = dict(category=c)
            for k, v in preds.items():
                trow[k] = float(stats.spearmanr(v[ok], t[ok])[0])
            trows.append(trow)
        T = pd.DataFrame(rows); TT = pd.DataFrame(trows)
        T.to_csv(OUT / f"emb_{name}_{'bare' if template == '{}' else 'a'}_categories.csv", index=False)
        TT.to_csv(OUT / f"emb_{name}_{'bare' if template == '{}' else 'a'}_typicality.csv", index=False)
        T11 = T[T.kind != "other"]
        lab = T11.kind.eq("natural").values
        z = lambda v: float(np.tanh(np.arctanh(np.clip(v, -.999, .999)).mean()))
        out = dict(
            E1_pair_rho_emb_mean=z(T.pair_rho_emb.values), E1_pair_rho_feat_mean=z(T.pair_rho_feat.values),
            E1_n_emb_better=int((T.pair_rho_emb > T.pair_rho_feat).sum()),
            E2_rho_embsim_humansim=dict(zip(["rho", "p"], map(float, stats.spearmanr(T.emb_sim, T.human_sim)))),
            E2_rho_featsim_humansim=dict(zip(["rho", "p"], map(float, stats.spearmanr(T.feat_sim, T.human_sim)))),
            E2_H1_embsim=dict(zip(["diff", "p"], perm_diff(T11.emb_sim, lab))),
            E3_H1_C_univ_all=dict(zip(["diff", "p"], perm_diff(T11.C_univ_all, lab))),
            E3_H1_C_own=dict(zip(["diff", "p"], perm_diff(T11.C_own, lab))),
            E3_rho_Cuniv_humansim=dict(zip(["rho", "p"], map(float, stats.spearmanr(T.C_univ_all, T.human_sim)))),
            E3_rho_Cown_humansim=dict(zip(["rho", "p"], map(float, stats.spearmanr(T.C_own, T.human_sim)))),
            E3_rho_Cuniv_distshare=dict(zip(["rho", "p"], map(float, stats.spearmanr(T.C_univ_all, T.dist_share_all)))),
            E3_rho_Cuniv_embsim=dict(zip(["rho", "p"], map(float, stats.spearmanr(T.C_univ_all, T.emb_sim)))),
            E4_typicality_mean_rho={k: z(TT[k].values) for k in TT.columns[1:]},
            E4_typicality_n_pos={k: int((TT[k] > 0).sum()) for k in TT.columns[1:]},
            E5_n_real_above_95pct_random=int((T.pct_own_vs_random > 0.95).sum()),
            E5_n_real_below_50pct_random=int((T.pct_own_vs_random < 0.5).sum()),
        )
        R_["bare" if template == "{}" else "a"] = out
    res[name] = R_
json.dump(res, open(OUT / "emb_results.json", "w"), indent=1)
print(json.dumps(res, indent=1))
