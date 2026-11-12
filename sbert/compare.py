"""Reproduce Leuven baselines and compare frozen SBERT cosine with human norms."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from upstream.coherence import coherence, pc_scores

ROOT = Path(__file__).resolve().parent
LEUVEN = ROOT / "data/dutch-concepts/dutch_concepts/data/en"
EF = LEUVEN / "exemplar_feature_judgments/exemplar_features"
MODEL = "sentence-transformers/all-mpnet-base-v2"
REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
DOMAINS = {
    "animal": ["bird", "fish", "insect", "mammal", "reptile"],
    "artifact": ["clothing", "kitchen_utensil", "musical_instrument", "tool", "vehicle", "weapon"],
}
CATEGORIES = sum(DOMAINS.values(), []) + ["fruit", "vegetable", "profession", "sport"]
SEED = 20261112


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    keep = np.isfinite(x) & np.isfinite(y)
    r, p = stats.spearmanr(x[keep], y[keep])
    return {"rho": float(r), "p": float(p), "n": int(keep.sum())}


def cosine_matrix(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Zero-length vector: cosine undefined.")
    z = x / norms
    return z @ z.T


def offdiag(m):
    return m[np.triu_indices(len(m), 1)]


def fisher_mean(x):
    return float(np.tanh(np.arctanh(np.clip(x, -.999999, .999999)).mean()))


def exact_group_test(x, animal):
    x, animal = np.asarray(x), np.asarray(animal)
    delta = x[animal].mean() - x[~animal].mean()
    null = []
    for idx in itertools.combinations(range(len(x)), int(animal.sum())):
        mask = np.zeros(len(x), bool)
        mask[list(idx)] = True
        null.append(x[mask].mean() - x[~mask].mean())
    return {"animal": float(x[animal].mean()), "artifact": float(x[~animal].mean()),
            "difference": float(delta), "p_exact": float(np.mean(np.abs(null) >= abs(delta) - 1e-12))}


def mantel_spearman(model, human, rng, permutations):
    """Permute exemplar identities, preserving dependence among shared pairs."""
    n = len(model)
    ij = np.triu_indices(n, 1)
    a = stats.rankdata(model[ij])
    b = stats.rankdata(human[ij])
    a = (a - a.mean()) / np.linalg.norm(a - a.mean())
    b = (b - b.mean()) / np.linalg.norm(b - b.mean())
    ranks = np.zeros_like(human)
    ranks[ij] = b
    ranks[(ij[1], ij[0])] = b
    observed = float(a @ b)
    exceed = 0
    for _ in range(permutations):
        order = rng.permutation(n)
        permuted = float(a @ ranks[order[ij[0]], order[ij[1]]])
        exceed += abs(permuted) >= abs(observed) - 1e-12
    return observed, (exceed + 1) / (permutations + 1)


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty_like(p)
    result[order] = adjusted
    return result


def bootstrap_correlations(frame, a, b, target, rng, draws=3000):
    """Paired category bootstrap; both predictors use identical sampled categories."""
    ranks = [stats.rankdata(frame[c]) for c in (a, b, target)]
    observed = float(np.corrcoef(ranks[0], ranks[2])[0, 1] -
                     np.corrcoef(ranks[1], ranks[2])[0, 1])
    samples = []
    for _ in range(draws):
        idx = rng.integers(0, len(frame), len(frame))
        # Re-rank each bootstrap sample, including its ties.
        ra, rb, ry = [stats.rankdata(frame[c].to_numpy()[idx]) for c in (a, b, target)]
        if min(np.std(ra), np.std(rb), np.std(ry)) == 0:
            continue
        samples.append(np.corrcoef(ra, ry)[0, 1] - np.corrcoef(rb, ry)[0, 1])
    return {"delta_rho": observed, "ci95": np.quantile(samples, [.025, .975]).tolist(),
            "bootstrap_draws": len(samples)}


def load_data():
    cats = {}
    domains = {
        d: pd.read_csv(EF / f"domains/{d}/{d}_exemplar_features_sum.csv", index_col=0) / 4
        for d in DOMAINS
    }
    for c in CATEGORIES:
        x = pd.read_csv(EF / f"categories/{c}/{c}_exemplar_features_sum.csv", index_col=0) / 4
        imp = pd.read_csv(EF / f"feature_importance_ratings/{c}_feature_importance_ratings.csv",
                          index_col=0)["mean"]
        salience = ((imp.reindex(x.columns) + 3) / 6).fillna(0).to_numpy()
        typicality = pd.read_csv(LEUVEN / f"exemplar_judgments/typicality_ratings/{c}_typicality_ratings.csv",
                                 index_col=0)["mean"].reindex(x.index).to_numpy()
        human = pd.read_csv(LEUVEN / f"pairwise_similarities/{c}/{c}_pairwise_similarities_mean.csv",
                            index_col=0).loc[x.index, x.index].to_numpy()
        assert x.index.is_unique and np.isfinite(human).all()
        assert np.isfinite(typicality).all() and np.allclose(human, human.T)
        cats[c] = (x, salience, typicality, human)
    return cats, domains


def embeddings(labels, template, device):
    texts = [template.format(label=label) for label in labels]
    key = hashlib.sha256(json.dumps([MODEL, REVISION, texts]).encode()).hexdigest()[:16]
    path = ROOT / "data" / f"embeddings_{key}.npz"
    if path.exists():
        with np.load(path) as cached:
            assert cached["labels"].tolist() == labels
            return cached["vectors"].astype(float), texts
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(4)
    model = SentenceTransformer(MODEL, revision=REVISION, device=device)
    vectors = model.encode(texts, batch_size=32, normalize_embeddings=True,
                           show_progress_bar=True, convert_to_numpy=True)
    assert np.isfinite(vectors).all()
    np.savez_compressed(path, labels=np.array(labels), vectors=vectors)
    return vectors.astype(float), texts


def run(args):
    out = ROOT / "results" / args.condition
    out.mkdir(parents=True, exist_ok=True)
    cats, domains = load_data()
    labels = sorted(set().union(*(set(v[0].index) for v in cats.values()),
                                *(set(d.index) for d in domains.values())))
    template = "{label}" if args.condition == "labels" else "This is {label}."
    vectors, texts = embeddings(labels, template, args.device)
    lookup = dict(zip(labels, vectors))
    pd.DataFrame({"label": labels, "model_input": texts}).to_csv(out / "inputs.csv", index=False)
    global_ref = vectors.mean(0)
    category_rows, typicality_rows, exemplar_rows, pair_rows, replication = [], [], [], [], []
    supplied = pd.read_csv(ROOT / "upstream/leuven_categories.csv").set_index("category")
    supplied_typ = pd.read_csv(ROOT / "upstream/leuven_typicality.csv").set_index("category")
    rng = np.random.default_rng(SEED)
    for c, (xc, salience, typicality, human) in cats.items():
        print(f"Analyzing {args.condition}: {c}", flush=True)
        domain = next((d for d in DOMAINS if c in DOMAINS[d]), "other")
        x = xc if domain == "other" else domains[domain].loc[xc.index, xc.columns]
        e = np.array([lookup[label] for label in x.index])
        sim = cosine_matrix(e)
        features = cosine_matrix(x.to_numpy() * salience)
        pair_rho, pair_p = mantel_spearman(sim, human, rng, args.permutations)
        feature_rho, feature_p = mantel_spearman(features, human, rng, args.permutations)
        row = {"category": c, "domain": domain, "n": len(x),
               "human_similarity": float(offdiag(human).mean()),
               "sbert_cosine": float(offdiag(sim).mean()),
               "feature_cosine": float(offdiag(features).mean()),
               "cx_features_own_category": coherence(xc, salience)["C"],
               "cx_features_own_uniform": coherence(x, None)["C"],
               "cx_sbert_own": coherence(e)["C"],
               "cx_sbert_global": coherence(e, ref=global_ref)["C"],
               "pairwise_rho_sbert": pair_rho, "pairwise_p_mantel": pair_p,
               "pairwise_rho_features": feature_rho, "pairwise_p_features_mantel": feature_p}
        # Mean similarity to all OTHER members. No diagonal self-similarities.
        mean_cosine = (sim.sum(1) - np.diag(sim)) / (len(e) - 1)
        # Leave the scored exemplar out of its prototype.
        other_sum = e.sum(0) - e
        prototype = np.sum(e * other_sum, 1) / (np.linalg.norm(e, axis=1) *
                                               np.linalg.norm(other_sum, axis=1))
        pred = {
            "sbert_mean_cosine": mean_cosine,
            "sbert_prototype_loo": prototype,
            "feature_family_resemblance": (x.to_numpy() * salience) @ x.to_numpy().mean(0),
            "human_similarity_to_others": (human.sum(1) - np.diag(human)) / (len(e) - 1),
        }
        weighted = x.to_numpy() * salience
        centroid = weighted.mean(0)
        pred["feature_prototype_cosine"] = weighted @ centroid / (
            np.linalg.norm(weighted, axis=1) * np.linalg.norm(centroid))
        if domain != "other":
            u = domains[domain][x.columns].to_numpy().mean(0)
            cu = coherence(x, salience, u)
            row.update(cx_features_universal=cu["C"], feature_distance_share=cu["dist_share"],
                       cx_features_own=coherence(x, salience)["C"])
            domain_embeddings = np.array([lookup[label] for label in domains[domain].index])
            ce = coherence(e, ref=domain_embeddings.mean(0))
            row.update(cx_sbert_domain=ce["C"], sbert_distance_share=ce["dist_share"])
            pc, z = pc_scores(x, salience, u)
            pred["feature_PC1_signed"] = pc[:, 0]
            pred["feature_PC1_absolute"] = np.abs(pc[:, 0])
            pred["feature_PC1_alignment"] = pc[:, 0] ** 2 / np.sum(z ** 2, axis=1)
            m = (x.to_numpy().mean(0) - u) * salience
            pred["feature_contrast"] = z @ (m / np.linalg.norm(m))
        mapping = {"human_similarity": "human_sim", "feature_cosine": "fam_res",
                   "cx_features_own_category": "C_own_sal_catmat",
                   "cx_features_own_uniform": "C_own_uni",
                   "cx_features_universal": "C_univ_sal", "cx_features_own": "C_own_sal"}
        for key, original in mapping.items():
            if key in row:
                error = abs(row[key] - supplied.loc[c, original])
                replication.append({"category": c, "metric": key, "absolute_error": error})
                assert error < 1e-9, (c, key, error)
        tr = {"category": c, "domain": domain}
        for name, values in pred.items():
            tr[name] = corr(values, typicality)["rho"]
        typ_mapping = {"feature_PC1_signed": "PC1_score_univ",
                       "feature_family_resemblance": "family_resemblance_RM",
                       "feature_prototype_cosine": "prototype_cosine"}
        for key, original in typ_mapping.items():
            if key in tr:
                error = abs(tr[key] - supplied_typ.loc[c, original])
                replication.append({"category": c, "metric": key, "absolute_error": error})
                assert error < 1e-9, (c, key, error)
        for i, label in enumerate(x.index):
            exemplar_rows.append({"category": c, "exemplar": label, "human_typicality": typicality[i],
                                  **{name: float(v[i]) for name, v in pred.items()}})
        for i, j in zip(*np.triu_indices(len(x), 1)):
            pair_rows.append({"category": c, "exemplar_a": x.index[i], "exemplar_b": x.index[j],
                              "human_similarity": human[i, j], "sbert_cosine": sim[i, j],
                              "feature_cosine": features[i, j]})
        category_rows.append(row)
        typicality_rows.append(tr)
    table, typ = pd.DataFrame(category_rows), pd.DataFrame(typicality_rows)
    table["pairwise_p_holm"] = holm(table.pairwise_p_mantel)
    table["pairwise_p_features_holm"] = holm(table.pairwise_p_features_mantel)
    summary = {"model": MODEL, "revision": REVISION, "condition": args.condition,
               "seed": SEED, "permutations": args.permutations,
               "unique_labels": len(labels), "category_memberships": int(table.n.sum()),
               "pairs": len(pair_rows), "category_level": {}, "typicality": {}, "H1": {}, "H2": {},
               "baseline_max_absolute_error": max(r["absolute_error"] for r in replication),
               "leuven_commit": subprocess.check_output(
                   ["git", "-C", str(ROOT / "data/dutch-concepts"), "rev-parse", "HEAD"], text=True).strip(),
               "python": platform.python_version(), "device": args.device}
    metrics = ["sbert_cosine", "feature_cosine", "cx_features_universal", "cx_features_own",
               "cx_features_own_category", "cx_features_own_uniform", "cx_sbert_own",
               "cx_sbert_domain", "cx_sbert_global"]
    for scope, subset in [("11", table[table.domain != "other"]), ("15", table)]:
        summary["category_level"][scope] = {
            m: corr(subset[m], subset.human_similarity)
            for m in metrics if not subset[m].isna().any()
        }
        tt = typ[typ.category.isin(subset.category)]
        summary["typicality"][scope] = {
            m: {"fisher_mean_rho": fisher_mean(tt[m]), "arithmetic_mean_rho": float(tt[m].mean()),
                "positive": int((tt[m] > 0).sum()), "n": len(tt)}
            for m in typ.columns[2:] if not tt[m].isna().any()
        }
    t11 = table[table.domain != "other"]
    for m in ["sbert_cosine", "feature_cosine", "human_similarity", "cx_features_universal",
              "cx_features_own", "cx_sbert_domain", "cx_sbert_own"]:
        summary["H1"][m] = exact_group_test(t11[m], t11.domain.eq("animal"))
    summary["category_delta_sbert_minus_cx_11"] = bootstrap_correlations(
        t11, "sbert_cosine", "cx_features_universal", "human_similarity", rng)
    paired = typ[typ.domain != "other"]
    delta = (paired.sbert_mean_cosine - paired.feature_PC1_signed).to_numpy()
    boot = [delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(3000)]
    summary["typicality_delta_sbert_minus_PC1_11"] = {
        "mean_delta_rho": float(delta.mean()), "ci95": np.quantile(boot, [.025, .975]).tolist(),
        "n": len(delta), "sbert_better": int((delta > 0).sum())}
    summary["pairwise"] = {
        "sbert_fisher_mean_rho": fisher_mean(table.pairwise_rho_sbert),
        "features_fisher_mean_rho": fisher_mean(table.pairwise_rho_features),
        "sbert_positive": int((table.pairwise_rho_sbert > 0).sum()),
        "sbert_holm_significant": int((table.pairwise_p_holm < .05).sum()), "n_categories": len(table)}
    summary["cx_sbert_vs_distance_share"] = corr(t11.cx_sbert_domain, t11.sbert_distance_share)
    summary["sbert_cosine_vs_cx_features_11"] = corr(t11.sbert_cosine, t11.cx_features_universal)
    for domain, members in DOMAINS.items():
        e = np.array([lookup[label] for label in domains[domain].index])
        summary["H2"][domain] = {
            "n": len(e), "superordinate_cosine": float(offdiag(cosine_matrix(e)).mean()),
            "subordinate_cosines": table.set_index("category").loc[members, "sbert_cosine"].to_dict()}
    for name, frame in [("categories", table), ("typicality", typ),
                        ("exemplars", pd.DataFrame(exemplar_rows)), ("pairs", pd.DataFrame(pair_rows)),
                        ("baseline_replication", pd.DataFrame(replication))]:
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    plot(table, typ, out)
    return summary


def plot(table, typ, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, metric, title in zip(axes, ["cx_features_universal", "sbert_cosine"],
                                 ["C_X on feature norms (11 categories)", "SBERT cosine (15 categories)"]):
        t = table.dropna(subset=[metric])
        for _, row in t.iterrows():
            ax.scatter(row[metric], row.human_similarity,
                       color={"animal": "#26734d", "artifact": "#b75d25", "other": "#5567aa"}[row.domain])
            near_right = row[metric] > t[metric].quantile(.8)
            ax.annotate(row.category.replace("_", " "), (row[metric], row.human_similarity),
                        fontsize=7, xytext=(-3 if near_right else 3, 3),
                        ha="right" if near_right else "left", textcoords="offset points")
        ax.set(xlabel=metric, ylabel="Mean human similarity", title=title)
    fig.savefig(out / "category_similarity.png", dpi=180)
    plt.close(fig)
    t = typ.set_index("category")[["sbert_mean_cosine", "feature_PC1_signed", "feature_family_resemblance"]]
    ax = t.plot.bar(figsize=(13, 5))
    ax.set(ylabel="Spearman rho with human typicality", ylim=(-.3, 1))
    ax.legend(["SBERT mean cosine", "Feature PC1 (universal)", "Feature family resemblance"], fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "typicality.png", dpi=180)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=["labels", "sentence"], default="labels")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--permutations", type=int, default=1999)
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations must be positive")
    run(args)
