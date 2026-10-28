"""Step 5: design-based logistic regression in Python (Python stand-in for R's survey::svyglm).

Models (weighted by fsweight), fitted on the same children:
  M0: outcome ~ cluster + grade + age
  M1: M0 + household wealth + mother's education
Comparing the cluster terms between M0 and M1 tests whether the profiles carry information
beyond wealth and mother's education.

Variance: Taylor linearisation sandwich with strata and PSUs (with-replacement within strata);
strata with one PSU are centred on the overall PSU mean (survey's "adjust" rule);
t-based inference with df = #PSUs - #strata. Rows lacking the outcome stay in the design
as zero contributions (domain estimation), so the PSU structure is the full sample's.

Output: outputs/regression_odds_ratios.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrices
from scipy import stats

OUT = Path(__file__).parent / "outputs"
d = pd.read_csv(OUT / "analytic.csv")
d["cluster"] = pd.read_csv(OUT / "clusters.csv")["cluster"]


def svy_cov(X, y, p, w, strata, psu):
    """Sandwich covariance of weighted logit coefficients. Inputs span the full design;
    rows not in the model carry w=0."""
    u = (w * (y - p))[:, None] * X
    A = (X * (w * p * (1 - p))[:, None]).T @ X
    Ainv = np.linalg.inv(A)
    tot = pd.DataFrame(u).groupby([np.asarray(strata), np.asarray(psu)]).sum()
    grand = tot.mean().to_numpy()
    B = np.zeros((X.shape[1], X.shape[1]))
    for _, t in tot.groupby(level=0):
        t = t.to_numpy()
        n = len(t)
        c = t - (t.mean(axis=0) if n > 1 else grand)
        B += (n / (n - 1) if n > 1 else 1.0) * c.T @ c
    return Ainv @ B @ Ainv


# --- self-check: with unit weights and every row its own PSU in one stratum, the result
# --- must equal statsmodels' HC0 covariance times n/(n-1).
rng = np.random.default_rng(0)
Xc = sm.add_constant(rng.normal(size=(200, 2)))
yc = (rng.random(200) < 1 / (1 + np.exp(-(Xc @ [0.2, 0.8, -0.5])))).astype(float)
fc = sm.GLM(yc, Xc, family=sm.families.Binomial()).fit(cov_type="HC0")
mine = svy_cov(Xc, yc, fc.predict(Xc), np.ones(200), np.zeros(200), np.arange(200)) * (199 / 200)
assert np.allclose(mine, fc.cov_params(), rtol=1e-6), "svy_cov self-check failed"
print("svy_cov self-check vs statsmodels HC0: passed")

n_psu = d[["stratum", "psu"]].drop_duplicates().shape[0]
n_strata = d["stratum"].nunique()
DF = n_psu - n_strata
ref = int(d["cluster"].value_counts().idxmax())
print(f"Design: {n_psu} PSUs, {n_strata} strata, df = {DF}; reference cluster = {ref}")

# Same children in both models: drop the few with missing mother's education (outcome -> NaN,
# so they stay in the design as zero-contribution rows).
OUTCOMES = ["reading_strict", "numeracy_strict"]
dd = d.copy()
dd.loc[dd["f_mother_edu"].isna(), OUTCOMES] = np.nan
print(f"Children dropped for missing mother's education: {int(d['f_mother_edu'].isna().sum())}")

CL = f"C(cluster, Treatment(reference={ref}))"
WE = "C(f_wealth, Treatment(reference='lower_40'))"
ME = "C(f_mother_edu, Treatment(reference='lower_sec_or_less'))"
MODELS = {"M0": f"{CL} + C(d_grade) + d_age", "M1": f"{CL} + C(d_grade) + d_age + {WE} + {ME}"}
RENAME = {CL: "cluster", WE: "wealth", ME: "mother_edu"}


def label(name):
    for k, v in RENAME.items():
        name = name.replace(k, v)
    return name


def fit(outcome, rhs):
    y_df, X_df = dmatrices(f"{outcome} ~ {rhs}", dd, return_type="dataframe")  # drops NaN-outcome rows
    ix = y_df.index
    w = np.zeros(len(dd))
    w[ix] = dd.loc[ix, "weight"] / dd.loc[ix, "weight"].mean()
    y, p, X = np.zeros(len(dd)), np.zeros(len(dd)), np.zeros((len(dd), X_df.shape[1]))
    X[ix], y[ix] = X_df.to_numpy(), y_df.iloc[:, 0].to_numpy()
    res = sm.GLM(y_df.iloc[:, 0], X_df, family=sm.families.Binomial(), freq_weights=w[ix]).fit()
    p[ix] = res.predict(X_df).to_numpy()
    V = svy_cov(X, y, p, w, dd["stratum"], dd["psu"])
    return [label(c) for c in X_df.columns], res.params.to_numpy(), V, len(ix)


def wald(names, beta, V, prefix):
    i = [j for j, n in enumerate(names) if n.startswith(prefix)]
    W = beta[i] @ np.linalg.inv(V[np.ix_(i, i)]) @ beta[i]
    return W / len(i), len(i), stats.f.sf(W / len(i), len(i), DF)


tcrit = stats.t.ppf(0.975, DF)
rows = []
for o in OUTCOMES:
    for m, rhs in MODELS.items():
        names, beta, V, n = fit(o, rhs)
        se = np.sqrt(np.diag(V))
        for i, nm in enumerate(names):
            if nm != "Intercept":
                rows.append(dict(outcome=o, model=m, term=nm, OR=np.exp(beta[i]),
                                 lo=np.exp(beta[i] - tcrit * se[i]), hi=np.exp(beta[i] + tcrit * se[i]),
                                 p=2 * stats.t.sf(abs(beta[i] / se[i]), DF), n=n))
        tests = [("cluster", *wald(names, beta, V, "cluster["))]
        if m == "M1":
            tests += [("wealth", *wald(names, beta, V, "wealth[")),
                      ("mother_edu", *wald(names, beta, V, "mother_edu["))]
        print(f"{o} {m} (n={n}): " + "; ".join(f"{t} F({q},{DF})={F:.2f} p={p:.4f}" for t, F, q, p in tests))

res = pd.DataFrame(rows).round(3)
res.to_csv(OUT / "regression_odds_ratios.csv", index=False)
print("\n=== Cluster odds ratios, M0 (cluster + grade + age) vs M1 (+ wealth + mother's education) ===")
cl = res[res.term.str.startswith("cluster")].pivot(index=["outcome", "term"], columns="model",
                                                    values=["OR", "lo", "hi", "p"])
print(cl.round(3).to_string())
print("\n=== Full table ===\n", res.to_string(index=False))
print("\nAbsolute rates per cluster with design-based CIs are in proficiency_by_cluster.csv (step 4).")
