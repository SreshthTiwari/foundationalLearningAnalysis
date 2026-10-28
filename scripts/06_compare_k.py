"""Step 6: compare candidate numbers of clusters (k = 2..8). Does not overwrite clusters.csv.

Selection criteria use contextual variables only. Outcome rates are shown for information
and must not drive the choice (that would leak the outcomes into the profiles).

Outputs (outputs/06_k_comparison/):
  k_comparison.csv      metrics per (MCA dimensions, k)
  k_top_levels.csv      the most distinctive feature levels per cluster for each k (main dims)
  console_log.txt       everything this script printed
"""
import numpy as np
import pandas as pd
import prince
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, silhouette_score)

import config

OUT = config.ensure(config.OUT_K_COMPARE)
config.log_console(OUT)
SEED, KS, BOOT = config.SEED, config.KS_COMPARE, config.BOOT_COMPARE
DIMS = config.DIMS_COMPARE          # 6 = the dimensionality used in step 3
d = pd.read_csv(config.ANALYTIC_CSV)
feats = [c for c in d.columns if c.startswith(config.FEATURE_PREFIX)]
w = d["weight"].to_numpy()
X = d[feats].copy()
for c in feats:
    X[c] = X[c].fillna(X[c].mode()[0])

mca = prince.MCA(n_components=config.MCA_COMPONENTS, n_iter=config.MCA_N_ITER, one_hot=True,
                 random_state=SEED).fit(X)
coords = mca.row_coordinates(X).to_numpy()


def km(Z, k, weight, seed, n_init=20):
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Z, sample_weight=weight)


rows, top_rows = [], []
for nd in DIMS:
    Z = coords[:, :nd]
    rng = np.random.default_rng(SEED)
    prev = None
    for k in KS:
        m = km(Z, k, w, SEED, 50)
        lab = m.labels_
        wsh = np.bincount(lab, weights=w, minlength=k) / w.sum()
        nn = np.bincount(lab, minlength=k)
        order = np.argsort(-wsh)
        lab = pd.Series(lab).map({o: i + 1 for i, o in enumerate(order)}).to_numpy()

        boot, jac = [], np.zeros((BOOT, k))
        for b in range(BOOT):
            idx = rng.integers(0, len(Z), len(Z))
            pb = km(Z[idx], k, w[idx], b, 10).predict(Z)
            boot.append(adjusted_rand_score(m.labels_, pb))
            for c in range(1, k + 1):  # best-matching bootstrap cluster (Jaccard), clusterboot-style
                A = lab == c
                jac[b, c - 1] = max((A & (pb == j)).sum() / (A | (pb == j)).sum() for j in range(k))
        seeds = [adjusted_rand_score(m.labels_, km(Z, k, w, s, 5).labels_) for s in range(config.STABILITY_SEED_START, config.STABILITY_SEED_START + 10)]

        r = dict(dims=nd, k=k, silhouette=silhouette_score(Z, lab), calinski_harabasz=calinski_harabasz_score(Z, lab),
                 wss=m.inertia_, boot_ari_mean=np.mean(boot), boot_ari_p10=np.percentile(boot, 10),
                 seed_ari_min=min(seeds), min_n=int(nn.min()), min_weighted_pct=wsh.min() * 100,
                 ari_vs_prev_k=adjusted_rand_score(prev, lab) if prev is not None else np.nan,
                 min_cluster_jaccard=jac.mean(axis=0).min(),
                 jaccard_by_cluster="/".join(f"{x:.2f}" for x in jac.mean(axis=0)))
        # outcomes, information only
        for o in config.OUTCOMES_PRIMARY:
            ok = d[o].notna().to_numpy()
            rates = [np.average(d[o][ok & (lab == c)], weights=w[ok & (lab == c)]) * 100
                     if (ok & (lab == c)).sum() else np.nan for c in range(1, k + 1)]
            r[f"{o}_range_pp"] = np.nanmax(rates) - np.nanmin(rates)
            r[f"{o}_by_cluster"] = "/".join(f"{x:.0f}" for x in rates)
        r["n_reading_by_cluster"] = "/".join(str(int((d["reading_strict"].notna().to_numpy() & (lab == c)).sum()))
                                             for c in range(1, k + 1))
        rows.append(r)
        prev = lab

        if nd == 6:  # most distinctive levels per cluster (percentage-point gap vs overall)
            for c in range(1, k + 1):
                gaps = []
                for f in feats:
                    overall = pd.Series(w).groupby(X[f].to_numpy()).sum() / w.sum() * 100
                    inc = pd.Series(w[lab == c]).groupby(X[f].to_numpy()[lab == c]).sum() / w[lab == c].sum() * 100
                    for lvl in overall.index:
                        gaps.append((f.replace("f_", "") + "=" + str(lvl), inc.get(lvl, 0) - overall[lvl], inc.get(lvl, 0)))
                gaps.sort(key=lambda t: -abs(t[1]))
                top_rows.append(dict(k=k, cluster=c, n=int((lab == c).sum()), weighted_pct=round(wsh[order[c - 1]] * 100, 1),
                                     top_levels="; ".join(f"{n} ({g:+.0f}pp)" for n, g, _ in gaps[:4]),
                                     n_gaps_ge_15pp=sum(abs(g) >= 15 for _, g, _ in gaps)))

res = pd.DataFrame(rows)
res.round(3).to_csv(OUT / "k_comparison.csv", index=False)
pd.DataFrame(top_rows).to_csv(OUT / "k_top_levels.csv", index=False)

pd.set_option("display.width", 250)
cols = ["dims", "k", "silhouette", "calinski_harabasz", "boot_ari_mean", "boot_ari_p10", "seed_ari_min",
        "min_n", "min_weighted_pct", "ari_vs_prev_k"]
print("=== Context-only criteria ===")
print(res[cols].round(3).to_string(index=False))
print("\n=== Per-cluster bootstrap Jaccard (clusters ordered by size; >0.75 stable, 0.6-0.75 indicative, <0.5 dissolves) ===")
print(res[["dims", "k", "min_cluster_jaccard", "jaccard_by_cluster"]].round(3).to_string(index=False))
print("\n=== Outcomes by cluster, information only (reading% / numeracy%), dims=6 ===")
print(res[res.dims == 6][["k", "n_reading_by_cluster", "reading_strict_by_cluster", "numeracy_strict_by_cluster"]]
      .to_string(index=False))
print("\n=== Most distinctive levels per cluster, dims=6 ===")
print(pd.DataFrame(top_rows).to_string(index=False))
print("\nSaved k_comparison.csv, k_top_levels.csv and console_log.txt to", OUT)
