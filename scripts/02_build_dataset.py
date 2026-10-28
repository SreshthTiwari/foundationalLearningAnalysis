"""Step 2: build the analytic dataset (sample, outcomes, clustering features).

Outcome definitions (UNICEF MICS6 foundational learning module, SDG 4.1.1a):
  reading  = reads >= 90% of the 72-word story correctly AND answers all five
             comprehension questions (3 literal + 2 inferential) correctly.
             Children who fail the practice or the 90% stop rule count as NOT
             proficient. Children never given the reading task are missing.
  numeracy = passes all four tasks (number reading, number discrimination,
             addition, pattern completion); a task is passed only if every
             item is correct. Items skipped by discontinue rules count as wrong.
Looser versions (3-of-5 comprehension, 3-of-4 numeracy tasks) are kept as
sensitivity outcomes.

Outputs (outputs/02_analytic_dataset/):
  analytic.csv                  one row per child (f_* = clustering features, d_* = descriptive only)
  reading_status_counts.csv     how many children fall in each reading status
  outcome_rates.csv             unweighted and weighted outcome rates
  numeracy_task_pass_rates.csv  unweighted pass rate per numeracy task
  feature_levels.csv            missing share and level counts for every clustering feature
  digital_vs_online_learning.csv  routing check for the merged digital-resources feature
  console_log.txt               everything this script printed
"""
import math

import numpy as np
import pandas as pd
import pyreadstat

import config

OUT = config.ensure(config.OUT_ANALYTIC)
config.log_console(OUT)

df, meta = pyreadstat.read_sav(config.FS_SAV)
vl = meta.variable_value_labels

# Sample: Grade 2-3 primary pupils with a completed FL interview
stageA = config.stage_a_mask(df)
d = df[stageA & (df["FL28"] == config.COMPLETED)].copy()
print(f"Analytic sample: {len(d)} children (Stage A = {int(stageA.sum())})")

# ---------------------------------------------------------------- outcomes
N_WORDS = config.N_WORDS
MIN_CORRECT = math.ceil(config.MIN_WORD_SHARE * N_WORDS)  # 65
words = d[[f"FL19W{i}" for i in range(1, N_WORDS + 1)]]
d["words_correct"] = (words == 0).sum(axis=1)
has_story = d["FL19INTRO"].notna()
words_ok = has_story & (d["words_correct"] >= MIN_CORRECT)

comp = d[["FL21BA", "FL21BB", "FL21BC", "FL21BD", "FL21BE"]]
comp_n = (comp == 1).sum(axis=1)
assert (comp.notna().all(axis=1) == words_ok).all(), "comprehension routing != 90% rule"

assessed = d["FL13"].notna() & (d["FL14"] != 9)  # reading task actually administered
d["reading_strict"] = (words_ok & (comp_n == 5)).astype(float).where(assessed)
d["reading_lenient"] = (words_ok & (comp_n >= 3)).astype(float).where(assessed)
# MICS manual divides by ALL children in the grade, so also keep a version where
# children never given the reading task count as not proficient.
d["reading_strict_incl"] = (words_ok & (comp_n == 5)).astype(float)
d["reading_status"] = np.select(
    [~assessed, ~has_story, ~words_ok, comp_n < 5],
    ["not_assessed", "failed_practice", "below_90pct_words", "comprehension_not_all"],
    default="proficient",
)

TASKS = {
    "number_reading": [f"FL23{x}" for x in "ABCDEF"],
    "number_discrimination": [f"FL24{x}" for x in "ABCDE"],
    "addition": [f"FL25{x}" for x in "ABCDE"],
    "patterns": [f"FL27{x}" for x in "ABCDE"],
}
passed = pd.DataFrame({t: (d[c] == 1).all(axis=1) for t, c in TASKS.items()})
d["numeracy_strict"] = passed.all(axis=1).astype(float)
d["numeracy_lenient"] = (passed.sum(axis=1) >= 3).astype(float)
for t in TASKS:
    d[f"task_{t}"] = passed[t].astype(float)

# ---------------------------------------------------------------- features
def yn(s, yes=1, no=2):
    return s.map({yes: "yes", no: "no"})


def anyyes(*cols):
    x = d[list(cols)]
    out = pd.Series(np.where((x == 1).any(axis=1), "yes",
                             np.where((x == 2).all(axis=1), "no", None)), index=d.index)
    return out


f = pd.DataFrame(index=d.index)
f["f_wealth"] = d["windex5"].map({1: "lower_40", 2: "lower_40", 3: "middle", 4: "upper_40", 5: "upper_40"})
f["f_mother_edu"] = d["melevel"].map({1: "lower_sec_or_less", 2: "lower_sec_or_less",
                                      3: "upper_sec", 4: "post_sec"})
f["f_area"] = d["HH6"].map({1: "GKMA", 2: "other_urban", 3: "rural"})
f["f_books"] = d["PR3"].map(lambda x: np.nan if pd.isna(x) or x == 99 else
                            "few_0_4" if x <= 4 else "some_5_9" if x <= 9 else "many_10plus")
hw = yn(d["PR6"])
f["f_hw_help"] = hw.where(d["PR5"] != 2, "no_homework")
f["f_pta"] = yn(d["PR8"])
f["f_school_visit"] = yn(d["PR11B"])
f["f_disability"] = d["fsdisability"].map({1: "yes", 2: "no"})
emo = d[["FCF25", "FCF26"]]
f["f_emo_difficulty"] = np.where(emo.isin([1, 2, 3]).any(axis=1), "monthly_or_more",
                                 np.where(emo.isin([4, 5]).all(axis=1), "rarely", None))
f["f_psych_aggr"] = anyyes("FCD2D", "FCD2H")
f["f_phys_punish"] = anyyes("FCD2C", "FCD2F", "FCD2G", "FCD2I", "FCD2J")
f["f_online_learning"] = d["CB10D"].map({0: "rarely_never", 1: "rarely_never", 2: "weekly", 3: "daily"})
# Internet/device questions are only asked of children who did online learning
# (NaN = no online learning). One merged variable avoids a routing-driven mini-cluster.
inet, dev = anyyes("CB10EA", "CB10EB"), yn(d["CB10F"])
f["f_digital_resources"] = np.where((inet == "yes") & (dev == "yes"),
                                    "home_internet_and_own_device", "limited_or_none")
f["f_econ_work"] = anyyes("CL1A", "CL1B", "CL1C", "CL1X")

desc = pd.DataFrame({
    "d_sex": d["HL4"].map(vl["HL4"]),
    "d_grade": d["CB8B"].astype(int),
    "d_age": d["CB3"],
    "d_parish": d["HH7"].map(vl.get("HH7", {})),
    "d_insurance": yn(d["CB11"]),
    "d_school_report": yn(d["PR10"]),
}, index=d.index)

design = pd.DataFrame({"weight": d["fsweight"], "psu": d["PSU"], "stratum": d["stratum"]})
outcomes = d[["reading_strict", "reading_lenient", "reading_strict_incl", "reading_status", "numeracy_strict",
              "numeracy_lenient", "words_correct"] + [f"task_{t}" for t in TASKS]]
out = pd.concat([d[["HH1", "HH2", "LN"]] if {"HH1", "HH2", "LN"} <= set(d.columns) else
                 pd.Series(d.index, index=d.index, name="row_id"),
                 design, outcomes, f, desc], axis=1)
out.to_csv(OUT / "analytic.csv", index=False)

# ---------------------------------------------------------------- diagnostics
def wrate(y, w):
    m = y.notna()
    return np.average(y[m], weights=w[m]) * 100, int(m.sum())


print("\n=== Outcomes ===")
status = out["reading_status"].value_counts()
print("reading_status:\n", status.to_string())
status.rename_axis("reading_status").rename("n").reset_index().to_csv(OUT / "reading_status_counts.csv", index=False)
rate_rows = []
for o in ["reading_strict", "reading_lenient", "reading_strict_incl", "numeracy_strict", "numeracy_lenient"]:
    r, n = wrate(out[o], out["weight"])
    print(f"{o:17s} n={n:3d}  unweighted={out[o].mean()*100:5.1f}%  weighted={r:5.1f}%")
    rate_rows.append(dict(outcome=o, n=n, unweighted_pct=round(out[o].mean() * 100, 1), weighted_pct=round(r, 1)))
pd.DataFrame(rate_rows).to_csv(OUT / "outcome_rates.csv", index=False)
task_rates = {t: round(passed[t].mean() * 100, 1) for t in TASKS}
print("Numeracy task pass rates (unweighted):", task_rates)
pd.Series(task_rates, name="pass_rate_pct").rename_axis("task").reset_index().to_csv(
    OUT / "numeracy_task_pass_rates.csv", index=False)

print("\n=== Features: missing share and levels ===")
level_rows = []
for c in f.columns:
    vc = f[c].value_counts(dropna=False)
    print(f"{c:18s} missing={f[c].isna().mean()*100:4.1f}%  " +
          ", ".join(f"{k}={v}" for k, v in vc.items()))
    level_rows.append(dict(feature=c, missing_pct=round(f[c].isna().mean() * 100, 1),
                           levels="; ".join(f"{k}={v}" for k, v in vc.items())))
pd.DataFrame(level_rows).to_csv(OUT / "feature_levels.csv", index=False)
print("\nDigital resources vs online learning:")
dig = pd.crosstab(d["CB10D"].map(vl["CB10D"]), f["f_digital_resources"])
print(dig.to_string())
dig.to_csv(OUT / "digital_vs_online_learning.csv")
print("\nSaved", OUT / "analytic.csv", out.shape)
