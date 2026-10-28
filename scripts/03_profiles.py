"""Step 3: child profiles via MCA + survey-weighted k-means.

Usage: python 03_profiles.py [--k N]
  Without --k the script picks k in 3..6 with the highest bootstrap stability among
  solutions whose smallest cluster holds >= 8% of children. Outcomes are NOT used.

Outputs (outputs/03_cluster_profiles/):
  cluster_diagnostics.csv   WSS, silhouette, bootstrap ARI, min cluster share per k
  clusters.csv              row-aligned cluster labels (chosen k + every k)
  profiles_long.csv         weighted level shares by cluster for every feature
  mca_eigenvalues.csv       MCA eigenvalues and which dimensions were kept
  mode_filled_missing.csv   features with mode-filled missing values
  cluster_sizes.csv         n and weighted share per cluster at the chosen k
  descriptive_by_cluster.csv  weighted % of sex and grade by cluster (not used in clustering)
  clustering_settings.json  chosen k, MCA dimensions, seed, bootstrap count
  console_log.txt           everything this script printed
"""
import argparse
import json

import numpy as np
import pandas as pd
import prince
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

import config

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=None)
ap.add_argument("--boot", type=int, default=config.BOOT_PROFILES)
args = ap.parse_args()

OUT = config.ensure(config.OUT_PROFILES)
config.log_console(OUT)
SEED, KS = config.SEED, config.KS_PROFILES
d = pd.read_csv(config.ANALYTIC_CSV)
feats = [c for c in d.columns if c.startswith(config.FEATURE_PREFIX)]
w = d["weight"].to_numpy()

# Non-structural missingness is tiny (<3% per variable): fill with the modal level.
# Structural cases (online-learning routing) are already explicit levels.
X = d[feats].copy()
filled = []
for c in feats:
    if X[c].isna().any():
        print(f"mode-filling {int(X[c].isna().sum())} missing in {c} -> {X[c].mode()[0]}")
        filled.append(dict(feature=c, n_missing=int(X[c].isna().sum()), filled_with=X[c].mode()[0]))
        X[c] = X[c].fillna(X[c].mode()[0])
pd.DataFrame(filled, columns=["feature", "n_missing", "filled_with"]).to_csv(OUT / "mode_filled_missing.csv", index=False)

# ------------------------------------------------------------------ MCA
Q = len(feats)
mca = prince.MCA(n_components=config.MCA_COMPONENTS, n_iter=config.MCA_N_ITER, one_hot=True,
                 random_state=SEED).fit(X)
eig = np.asarray(mca.eigenvalues_)
n_comp = int(np.clip((eig > 1 / Q).sum(), config.MCA_KEEP_MIN, config.MCA_KEEP_MAX))
pd.DataFrame({"dimension": range(1, len(eig) + 1), "eigenvalue": eig, "kept": np.arange(len(eig)) < n_comp}
             ).round(4).to_csv(OUT / "mca_eigenvalues.csv", index=False)
print(f"\nMCA eigenvalues (first 8): {np.round(eig[:8], 3)}  | keep {n_comp} (eigenvalue > 1/{Q})")
Z = mca.row_coordinates(X).iloc[:, :n_comp].to_numpy()


# ------------------------------------------------------------------ k-means
def fit_km(Z, k, weight, seed=SEED):
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z, sample_weight=weight)


rng = np.random.default_rng(SEED)
rows, labels_all = [], {}
for k in KS:
    km = fit_km(Z, k, w)
    lab = km.labels_
    share = np.bincount(lab, weights=w, minlength=k) / w.sum()
    n_share = np.bincount(lab, minlength=k) / len(lab)
    aris = []
    for b in range(args.boot):
        idx = rng.integers(0, len(Z), len(Z))
        kb = fit_km(Z[idx], k, w[idx], seed=b)
        aris.append(adjusted_rand_score(lab, kb.predict(Z)))
    rows.append(dict(k=k, wss=km.inertia_, silhouette=silhouette_score(Z, lab),
                     boot_ari_mean=np.mean(aris), boot_ari_p10=np.percentile(aris, 10),
                     min_cluster_share=n_share.min(), min_weighted_share=share.min()))
    # relabel by descending weighted size so labels are comparable across runs
    order = np.argsort(-share)
    labels_all[k] = pd.Series(lab).map({old: new + 1 for new, old in enumerate(order)}).to_numpy()

diag = pd.DataFrame(rows)
diag["wss_drop_pct"] = (-diag["wss"].pct_change() * 100).round(1)
diag.round(3).to_csv(OUT / "cluster_diagnostics.csv", index=False)
print("\n=== Cluster diagnostics ===\n", diag.round(3).to_string(index=False))

if args.k:
    k_final = args.k
else:
    ok = diag[(diag.k >= config.AUTO_K_MIN) & (diag.min_cluster_share >= config.AUTO_K_MIN_CLUSTER_SHARE)]
    k_final = int(ok.sort_values("boot_ari_mean", ascending=False).k.iloc[0]) if len(ok) else config.AUTO_K_FALLBACK
print(f"\nChosen k = {k_final}" + ("" if args.k else "  (auto; override with --k)"))

clusters = pd.DataFrame({f"cluster_k{k}": v for k, v in labels_all.items()})
clusters.insert(0, "cluster", labels_all[k_final])
clusters.to_csv(OUT / "clusters.csv", index=False)

# ------------------------------------------------------------------ profiles
d["cluster"] = labels_all[k_final]
sizes = (d.groupby("cluster").agg(n=("weight", "size"), weight=("weight", "sum")))
sizes["weighted_share_pct"] = (sizes["weight"] / sizes["weight"].sum() * 100).round(1)
print("\n=== Cluster sizes ===\n", sizes[["n", "weighted_share_pct"]].to_string())
sizes[["n", "weighted_share_pct"]].to_csv(OUT / "cluster_sizes.csv")
with open(OUT / "clustering_settings.json", "w") as fh:
    json.dump(dict(k_final=k_final, k_chosen_by="--k override" if args.k else "auto", n_mca_dims_kept=n_comp,
                   n_features=Q, seed=SEED, n_bootstrap=args.boot, ks_tried=list(KS)), fh, indent=2)

recs = []
for c in feats:
    cat = X[c]
    overall = pd.Series(w, index=d.index).groupby(cat).sum() / w.sum() * 100
    by = d.assign(_c=cat).groupby(["cluster", "_c"])["weight"].sum().unstack(0).fillna(0)
    by = by / by.sum() * 100
    for lvl in overall.index:
        r = dict(variable=c, level=lvl, overall=round(overall[lvl], 1))
        r.update({f"cluster_{j}": round(by.loc[lvl, j], 1) if lvl in by.index else 0.0 for j in by.columns})
        recs.append(r)
prof = pd.DataFrame(recs)
prof.to_csv(OUT / "profiles_long.csv", index=False)
print("\n=== Weighted level shares (%) by cluster ===")
print(prof.to_string(index=False))

print("\nDescriptive (not used in clustering), weighted % by cluster:")
desc_tables = []
for c in ["d_sex", "d_grade"]:
    t = d.groupby(["cluster", c])["weight"].sum().unstack(0).fillna(0)
    t = (t / t.sum() * 100).round(1)
    print(t.to_string(), "\n")
    desc_tables.append(t.reset_index().rename(columns={c: "level"}).assign(variable=c))
pd.concat(desc_tables)[["variable", "level"] + [c for c in desc_tables[0].columns if c not in ("level", "variable")]
                       ].to_csv(OUT / "descriptive_by_cluster.csv", index=False)
print("Saved tables and console_log.txt in", OUT)
