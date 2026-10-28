"""Step 7: explainable ML for the existing child profiles (PISA-paper style).

Classifier (Random Forest, XGBoost) predicts the saved cluster labels from the ORIGINAL
categorical features (not MCA components). SHAP then explains which characteristics define
each cluster. Reading/numeracy outcomes are never read, used or explained here.

Small-sample choices (n=368, smallest cluster ~31): shallow trees, no SMOTE/grid search,
class-balanced weights, repeated stratified 5-fold CV, and SHAP computed OUT-OF-FOLD (each
child is explained by a model that never saw them). SHAP explains the classifier, not causes.

Everything is written to outputs/07_explainable_ml/ (other steps' outputs are untouched):
tables, fig1-fig4, run_summary.json, cluster_descriptions.csv, console_log.txt. report.md is
the hand-written write-up of these results.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy.stats import chi2_contingency, spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             normalized_mutual_info_score, precision_recall_fscore_support)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

import config

OUT = config.ensure(config.OUT_EXPLAINABLE)
config.log_console(OUT)

try:
    from xgboost import XGBClassifier
except Exception as e:  # libomp missing
    XGBClassifier = None
    print("xgboost unavailable:", type(e).__name__)

SEED, REPEATS = config.SEED, config.CV_REPEATS

# ------------------------------------------------------------------ data (no outcomes)
d = pd.read_csv(config.ANALYTIC_CSV)
d["cluster"] = pd.read_csv(config.CLUSTERS_CSV)["cluster"]
feats = [c for c in d.columns if c.startswith(config.FEATURE_PREFIX)]
w = d["weight"].to_numpy()
X = d[feats].copy()
for c in feats:  # same mode-fill as step 3, so the labels and the classifier see the same data
    X[c] = X[c].fillna(X[c].mode()[0])
Xd = pd.get_dummies(X, prefix_sep="=", dtype=float)
Xd.columns = [c.replace("f_", "", 1) for c in Xd.columns]
var_of = np.array([c.split("=")[0] for c in Xd.columns])
y = d["cluster"].to_numpy() - 1
K = int(y.max() + 1)
sizes = np.bincount(y)
print(f"n={len(y)}, clusters={K}, sizes={sizes.tolist()}, majority-class baseline={sizes.max()/len(y):.3f}")

DOMAINS = config.DOMAINS
dom_of = {v: dname for dname, vs in DOMAINS.items() for v in vs}
SES = DOMAINS["Socioeconomic"]


# ------------------------------------------------------------------ models
def make(name):
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt",
                                      class_weight="balanced_subsample", random_state=SEED, n_jobs=-1)
    return XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, random_state=SEED, eval_metric="mlogloss", n_jobs=1)


def fit(name, Xtr, ytr):
    m = make(name)
    if name == "XGBoost":
        m.fit(Xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    else:
        m.fit(Xtr, ytr)
    return m


def cv_predict(name, Xmat, repeats=REPEATS):
    """Out-of-fold predictions for each repeat of stratified 5-fold CV. Returns (repeats, n)."""
    preds = np.zeros((repeats, len(y)), dtype=int)
    rskf = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS, n_repeats=repeats, random_state=SEED)
    for i, (tr, te) in enumerate(rskf.split(Xmat, y)):
        preds[i // config.CV_FOLDS, te] = fit(name, Xmat.iloc[tr], y[tr]).predict(Xmat.iloc[te])
    return preds


def summarize(preds):
    rows = []
    for p in preds:
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
        rows.append(dict(accuracy=accuracy_score(y, p), balanced_accuracy=balanced_accuracy_score(y, p),
                         macro_precision=pr, macro_recall=rc, macro_f1=f1))
    r = pd.DataFrame(rows)
    return r.mean(), r.std()


def wilson(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


names = ["RandomForest"] + (["XGBoost"] if XGBClassifier else [])
allpreds, rows = {}, []
for nm in names:
    allpreds[nm] = cv_predict(nm, Xd)
    mu, sd = summarize(allpreds[nm])
    rows.append({"model": nm, **{f"{k}": f"{mu[k]:.3f} ± {sd[k]:.3f}" for k in mu.index}})
metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / "classifier_metrics.csv", index=False)
print(f"\n=== Classifier performance ({REPEATS}x repeated stratified 5-fold CV, mean ± sd across repeats) ===")
print(metrics.to_string(index=False))

best = max(names, key=lambda nm: summarize(allpreds[nm])[0]["balanced_accuracy"])
print(f"\nPrimary model for SHAP (highest balanced accuracy): {best}")
P = allpreds[best]

# per-class metrics (averaged over repeats) + Wilson CI for recall from repeat 0
pc = []
for c in range(K):
    prs, rcs, f1s = [], [], []
    for p in P:
        pr, rc, f1, _ = precision_recall_fscore_support(y, p, labels=[c], zero_division=0)
        prs.append(pr[0]), rcs.append(rc[0]), f1s.append(f1[0])
    k_hit = int(((P[0] == c) & (y == c)).sum())
    lo, hi = wilson(k_hit, int((y == c).sum()))
    pc.append(dict(cluster=c + 1, n=int((y == c).sum()), precision=np.mean(prs), recall=np.mean(rcs),
                   f1=np.mean(f1s), recall_ci_repeat0=f"{lo:.2f}-{hi:.2f}"))
pc = pd.DataFrame(pc).round(3)
pc.to_csv(OUT / "per_class_metrics.csv", index=False)
print("\nPer-cluster metrics:\n", pc.to_string(index=False))

cm = sum(confusion_matrix(y, p, labels=range(K)) for p in P) / len(P)  # mean counts over repeats
pd.DataFrame(cm.round(1), index=[f"true_{i+1}" for i in range(K)],
             columns=[f"pred_{i+1}" for i in range(K)]).to_csv(OUT / "confusion_matrix.csv")
print("\nMean confusion matrix (rows = true cluster, cols = predicted):\n", cm.round(1))

# ------------------------------------------------------------------ is it just SES?
sets = {"all features": list(Xd.columns),
        "SES only (wealth, mother_edu, area)": [c for c in Xd.columns if var_of[list(Xd.columns).index(c)] in SES],
        "without SES": [c for c in Xd.columns if var_of[list(Xd.columns).index(c)] not in SES]}
abl = []
for label, cols in sets.items():
    mu, sd = summarize(cv_predict(best, Xd[cols]))
    abl.append(dict(features=label, n_dummies=len(cols), accuracy=f"{mu['accuracy']:.3f} ± {sd['accuracy']:.3f}",
                    balanced_accuracy=f"{mu['balanced_accuracy']:.3f} ± {sd['balanced_accuracy']:.3f}"))
abl = pd.DataFrame(abl)
abl.to_csv(OUT / "ablation.csv", index=False)
print(f"\n=== Can the clusters be recovered from SES alone / without SES? ({best}) ===\n", abl.to_string(index=False))

cv_rows = []
for f in feats:
    ct = pd.crosstab(X[f], d["cluster"])
    chi2 = chi2_contingency(ct)[0]
    cv_rows.append(dict(variable=f.replace("f_", "", 1),
                        cramers_v=np.sqrt(chi2 / (len(d) * (min(ct.shape) - 1)))))
cram = pd.DataFrame(cv_rows).sort_values("cramers_v", ascending=False).round(3)
cram.to_csv(OUT / "cramers_v.csv", index=False)
print("\nCramér's V, cluster vs each variable (unweighted):\n", cram.to_string(index=False))
nmi_wealth = round(normalized_mutual_info_score(d["cluster"], X["f_wealth"]), 3)
print("NMI(cluster, wealth) =", nmi_wealth)

# ------------------------------------------------------------------ SHAP (out-of-fold)
def shap_arr(model, Xte):
    sv = shap.TreeExplainer(model).shap_values(Xte)
    return np.stack(sv, axis=-1) if isinstance(sv, list) else sv  # (n, features, classes)


sv_oof = np.zeros((len(y), Xd.shape[1], K))
for tr, te in StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=SEED).split(Xd, y):
    sv_oof[te] = shap_arr(fit(best, Xd.iloc[tr], y[tr]), Xd.iloc[te])
sv_full = shap_arr(fit(best, Xd, y), Xd)  # in-sample, for a stability comparison only


def var_level(sv):
    """Sum dummy SHAP within each variable -> (n, variables, classes)."""
    vs = list(dict.fromkeys(var_of))
    return vs, np.stack([sv[:, var_of == v, :].sum(axis=1) for v in vs], axis=1)


vs, svv = var_level(sv_oof)
_, svv_full = var_level(sv_full)
glob = pd.DataFrame({"variable": vs, "domain": [dom_of[v] for v in vs],
                     "mean_abs_shap": np.abs(svv).mean(axis=(0, 2))})
for c in range(K):
    glob[f"cluster_{c+1}"] = np.abs(svv[:, :, c]).mean(axis=0)
glob["share_pct"] = glob["mean_abs_shap"] / glob["mean_abs_shap"].sum() * 100
glob = glob.sort_values("mean_abs_shap", ascending=False).round(4)
glob.to_csv(OUT / "shap_variable_global.csv", index=False)
rho = spearmanr(np.abs(svv).mean(axis=(0, 2)), np.abs(svv_full).mean(axis=(0, 2)))[0]
print(f"\n=== Global SHAP (variable level, out-of-fold {best}) ===  rank agreement with in-sample SHAP: rho={rho:.2f}")
print(glob.to_string(index=False))

dom = (glob.groupby("domain")[["mean_abs_shap"] + [f"cluster_{c+1}" for c in range(K)]].sum())
dom = (dom / dom.sum() * 100).round(1).sort_values("mean_abs_shap", ascending=False)
dom.to_csv(OUT / "domain_importance.csv")
print("\nShare of SHAP importance by domain (%, columns sum to 100):\n", dom.to_string())

# level-level, per cluster
lvl_rows, prof_rows = [], []
overall = {c: (pd.Series(w).groupby(Xd[c].to_numpy()).sum() / w.sum() * 100).get(1.0, 0.0) for c in Xd.columns}
for c in range(K):
    m = y == c
    for j, col in enumerate(Xd.columns):
        share = w[m][Xd[col].to_numpy()[m] == 1].sum() / w[m].sum() * 100
        lvl_rows.append(dict(cluster=c + 1, level=col, mean_abs_shap=np.abs(sv_oof[:, j, c]).mean(),
                             mean_shap_members=sv_oof[m, j, c].mean(), cluster_pct=share,
                             overall_pct=overall[col], gap_pp=share - overall[col]))
lvl = pd.DataFrame(lvl_rows).round(3)
lvl.to_csv(OUT / "shap_level_by_cluster.csv", index=False)

# ------------------------------------------------------------------ cluster descriptions
print("\n=== Cluster descriptions (weighted %, gap vs overall in percentage points) ===")
wshare = [w[y == c].sum() / w.sum() * 100 for c in range(K)]
desc_rows = []
for c in range(K):
    sub = lvl[lvl.cluster == c + 1]
    top_shap = glob.sort_values(f"cluster_{c+1}", ascending=False).head(4)["variable"].tolist()
    top_gap = sub.reindex(sub.gap_pp.abs().sort_values(ascending=False).index).head(6)
    print(f"\nCluster {c+1}: n={sizes[c]} ({wshare[c]:.1f}% weighted). Top SHAP variables: {', '.join(top_shap)}")
    for _, r in top_gap.iterrows():
        print(f"   {r.level:42s} cluster {r.cluster_pct:5.1f}%  overall {r.overall_pct:5.1f}%  ({r.gap_pp:+.0f}pp)")
        desc_rows.append(dict(cluster=c + 1, n=int(sizes[c]), weighted_share_pct=round(wshare[c], 1),
                              top_shap_variables="; ".join(top_shap), level=r.level, cluster_pct=r.cluster_pct,
                              overall_pct=r.overall_pct, gap_pp=r.gap_pp))
pd.DataFrame(desc_rows).to_csv(OUT / "cluster_descriptions.csv", index=False)
with open(OUT / "run_summary.json", "w") as fh:
    json.dump(dict(n=len(y), n_clusters=K, cluster_sizes=sizes.tolist(),
                   majority_class_baseline=round(sizes.max() / len(y), 3), models=names, primary_model=best,
                   shap_rank_agreement_oof_vs_insample=round(float(rho), 3),
                   nmi_cluster_wealth=float(nmi_wealth), cv_repeats=REPEATS, cv_folds=config.CV_FOLDS,
                   seed=SEED), fh, indent=2)

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 9})
cols_c = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756"]

# 1. global SHAP importance, stacked by cluster
g = glob.head(14).iloc[::-1]
fig, ax = plt.subplots(figsize=(7, 5))
left = np.zeros(len(g))
for c in range(K):
    ax.barh(g["variable"], g[f"cluster_{c+1}"], left=left, color=cols_c[c], label=f"Cluster {c+1}")
    left += g[f"cluster_{c+1}"].to_numpy()
ax.set_xlabel(f"Mean |SHAP| (out-of-fold {best}), stacked by cluster")
ax.set_title("Variables that most distinguish the clusters")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUT / "fig1_shap_global.png", dpi=150)
plt.close()

# 2. beeswarm per cluster (dummy level)
for c in range(K):
    plt.figure(figsize=(7, 4.5))
    shap.summary_plot(sv_oof[:, :, c], Xd, max_display=12, show=False)
    plt.title(f"Cluster {c+1} (n={sizes[c]}): SHAP for membership")
    plt.tight_layout()
    plt.savefig(OUT / f"fig2_shap_beeswarm_cluster{c+1}.png", dpi=150)
    plt.close()

# 3. cluster-characteristics heatmap (gap vs overall, annotated with cluster %)
hm = lvl.pivot(index="level", columns="cluster", values="gap_pp").reindex(Xd.columns)
pct = lvl.pivot(index="level", columns="cluster", values="cluster_pct").reindex(Xd.columns)
fig, ax = plt.subplots(figsize=(6.5, 0.28 * len(hm) + 1.5))
im = ax.imshow(hm.to_numpy(), cmap="RdBu_r", vmin=-50, vmax=50, aspect="auto")
ax.set_xticks(range(K), [f"Cluster {c+1}\n(n={sizes[c]})" for c in range(K)])
ax.set_yticks(range(len(hm)), hm.index)
for i in range(len(hm)):
    for j in range(K):
        ax.text(j, i, f"{pct.iloc[i, j]:.0f}", ha="center", va="center", fontsize=7,
                color="white" if abs(hm.iloc[i, j]) > 30 else "black")
fig.colorbar(im, label="Cluster share minus overall (pp)")
ax.set_title("Cluster characteristics\n(cell = weighted % of cluster)", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "fig3_cluster_heatmap.png", dpi=150)
plt.close()

# 4. confusion matrix (row-normalised)
fig, ax = plt.subplots(figsize=(4.2, 3.8))
cmn = cm / cm.sum(axis=1, keepdims=True)
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(K), [f"{i+1}" for i in range(K)])
ax.set_yticks(range(K), [f"{i+1}" for i in range(K)])
ax.set_xlabel("Predicted cluster")
ax.set_ylabel("True cluster")
for i in range(K):
    for j in range(K):
        ax.text(j, i, f"{cm[i, j]:.0f}\n({cmn[i, j]*100:.0f}%)", ha="center", va="center",
                color="white" if cmn[i, j] > 0.6 else "black", fontsize=8)
ax.set_title(f"{best}: out-of-fold confusion matrix")
plt.tight_layout()
plt.savefig(OUT / "fig4_confusion_matrix.png", dpi=150)
plt.close()
print("\nSaved tables and fig1-fig4 to", OUT)
