"""Step 4: explain the profiles and compare foundational learning across them.

Outputs (outputs/):
  proficiency_by_cluster.csv   weighted rates + design-based 95% CIs (strict and lenient outcomes)
  proficiency_by_k.csv         strict-outcome rates by cluster for every k (robustness)
  shap_*.csv / shap_*.png      cluster explanations (XGBoost if libomp is installed, else random forest)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

OUT = Path(__file__).parent / "outputs"
SEED = 42
d = pd.read_csv(OUT / "analytic.csv")
cl = pd.read_csv(OUT / "clusters.csv")
d["cluster"] = cl["cluster"]
feats = [c for c in d.columns if c.startswith("f_")]
OUTCOMES = ["reading_strict", "numeracy_strict", "reading_lenient", "numeracy_lenient",
            "reading_strict_incl"]

n_psu = d[["stratum", "psu"]].drop_duplicates().shape[0]
n_strata = d["stratum"].nunique()
DF = n_psu - n_strata


# ------------------------------------------------------------ design-based estimate
def prop_ci(y, mask):
    """Weighted proportion in a domain with Taylor-linearised, PSU-clustered, stratified 95% CI.
    Strata with a single PSU are centred on the overall PSU mean (lonely-PSU 'adjust')."""
    m = mask & y.notna()
    W = d.loc[m, "weight"].sum()
    p = (d.loc[m, "weight"] * y[m]).sum() / W
    z = np.where(m, d["weight"] * (y.fillna(0) - p) / W, 0.0)
    tot = pd.DataFrame({"s": d["stratum"], "p": d["psu"], "z": z}).groupby(["s", "p"])["z"].sum()
    zbar_all = tot.mean()
    var = 0.0
    for _, zs in tot.groupby(level=0):
        n = len(zs)
        var += ((zs - zs.mean()) ** 2).sum() * n / (n - 1) if n > 1 else ((zs - zbar_all) ** 2).sum()
    se = np.sqrt(var)
    t = stats.t.ppf(0.975, DF)
    return p * 100, max(p - t * se, 0) * 100, min(p + t * se, 1) * 100, int(m.sum())


# ------------------------------------------------------------ proficiency by cluster
rows = []
for c in ["All"] + sorted(d["cluster"].unique()):
    mask = pd.Series(True, index=d.index) if c == "All" else d["cluster"] == c
    r = dict(cluster=c, n=int(mask.sum()),
             weighted_share_pct=round(d.loc[mask, "weight"].sum() / d["weight"].sum() * 100, 1))
    for o in OUTCOMES:
        p, lo, hi, n = prop_ci(d[o], mask)
        r.update({f"{o}_n": n, f"{o}_pct": round(p, 1), f"{o}_ci": f"{lo:.1f}-{hi:.1f}"})
    rows.append(r)
prof = pd.DataFrame(rows)
prof.to_csv(OUT / "proficiency_by_cluster.csv", index=False)
print(f"=== Weighted proficiency by cluster (design df = {DF}; {n_psu} PSUs, {n_strata} strata) ===")
print(prof[["cluster", "n", "weighted_share_pct"] +
           [f"{o}_{s}" for o in OUTCOMES[:2] for s in ("n", "pct", "ci")]].to_string(index=False))
print("\nSensitivity definitions (3-of-5 comprehension; 3-of-4 numeracy tasks; reading with"
      " not-assessed children counted as not proficient):")
print(prof[["cluster"] + [f"{o}_{s}" for o in OUTCOMES[2:] for s in ("pct", "ci")]].to_string(index=False))

# ------------------------------------------------------------ robustness over k
krows = []
for k in range(2, 7):
    if f"cluster_k{k}" not in cl:
        continue
    for c in range(1, k + 1):
        mask = cl[f"cluster_k{k}"] == c
        r = dict(k=k, cluster=c, n=int(mask.sum()))
        for o in OUTCOMES[:2]:
            r[o] = round(prop_ci(d[o], mask)[0], 1)
        krows.append(r)
pd.DataFrame(krows).to_csv(OUT / "proficiency_by_k.csv", index=False)
print("\n=== Strict outcomes by cluster for each k (see proficiency_by_k.csv) ===")
print(pd.DataFrame(krows).to_string(index=False))

# ------------------------------------------------------------ explainable classifier + SHAP
Xd = pd.get_dummies(d[feats].fillna("missing"), dtype=float)
y = d["cluster"].to_numpy() - 1
try:
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8,
                          colsample_bytree=0.8, random_state=SEED, eval_metric="mlogloss")
    mname = "XGBoost"
except Exception as e:  # libomp missing
    print(f"\n[xgboost unavailable ({type(e).__name__}); using RandomForest. `brew install libomp` to enable]")
    model = RandomForestClassifier(n_estimators=300, max_depth=5, random_state=SEED)
    mname = "RandomForest"

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
pred = cross_val_predict(model, Xd, y, cv=cv)
print(f"\n=== {mname} predicting cluster from the original features ===")
print(f"5-fold CV accuracy = {(pred == y).mean():.3f}  (explanatory only: labels derive from these variables)")
model.fit(Xd, y)
sv = shap.TreeExplainer(model).shap_values(Xd)
sv = np.stack(sv, axis=-1) if isinstance(sv, list) else sv  # (n, features, classes)

glob = pd.Series(np.abs(sv).mean(axis=(0, 2)), index=Xd.columns).sort_values(ascending=False)
glob.rename("mean_abs_shap").to_csv(OUT / "shap_global.csv")
print("\nTop global features (mean |SHAP|):\n", glob.head(10).round(3).to_string())

top = []
for j, c in enumerate(sorted(d["cluster"].unique())):
    mean_abs = pd.Series(np.abs(sv[:, :, j]).mean(axis=0), index=Xd.columns)
    # direction: mean feature value among members minus non-members
    direction = Xd[d["cluster"] == c].mean() - Xd[d["cluster"] != c].mean()
    t = mean_abs.sort_values(ascending=False).head(8)
    for f_, v in t.items():
        top.append(dict(cluster=c, feature=f_, mean_abs_shap=round(v, 3),
                        member_minus_other=round(direction[f_], 2)))
    plt.figure()
    shap.summary_plot(sv[:, :, j], Xd, max_display=10, show=False)
    plt.title(f"Cluster {c}")
    plt.tight_layout()
    plt.savefig(OUT / f"shap_cluster_{c}.png", dpi=130)
    plt.close()
topdf = pd.DataFrame(top)
topdf.to_csv(OUT / "shap_by_cluster.csv", index=False)
print("\nTop features per cluster (member_minus_other > 0 = more common in cluster):")
print(topdf.to_string(index=False))
print("\nSaved outputs to", OUT)
