"""

verify the analytic sample and audit candidate variables in fs.sav

outputs (in outputs/01_data_audit/):
  sample_flow.csv                  grade 2-3 eligibility and assessment-completion counts
  variable_audit.csv               one row per candidate variable
  eligibility_value_labels.csv     spss labels for the eligibility variables
  fl28_result_counts.csv           grade 2-3 children by assessment result
  grade_counts.csv                 grade 2-3 children by grade
  usable_item_counts.csv           non-missing / no-response counts for reading and numeracy items
  reading_routing_checks.csv       who got the story and who got the comprehension questions
  missed_words_by_comprehension.csv  missed words for children with vs without comprehension answers
  assessed_vs_not.csv              assessed vs non-assessed shares on a few descriptors (unweighted)
  survey_design_variables.csv      weight / psu / stratum summaries
  survey_design_summary.csv        weight correlation, psu and strata counts
  console_log.txt                  everything this script printed

"""
import pandas as pd
import pyreadstat

import config

OUT = config.ensure(config.OUT_DATA_AUDIT)
config.log_console(OUT)

df, meta = pyreadstat.read_sav(config.FS_SAV)
labels = meta.column_names_to_labels
vlabels = meta.variable_value_labels

ATTENDING, PRIMARY, COMPLETED = config.ATTENDING, config.PRIMARY, config.COMPLETED

# 1. value labels for eligibility variables
print("=== Eligibility variable labels ===")
elig_labels = []
for v in ["CB7", "CB8A", "CB8B", "FL28"]:
    print(f"{v}: {labels[v]}\n  {vlabels.get(v)}")
    elig_labels.append(dict(variable=v, label=labels[v], value_labels=str(vlabels.get(v))))
pd.DataFrame(elig_labels).to_csv(OUT / "eligibility_value_labels.csv", index=False)

# 2. sample flow
stageA = config.stage_a_mask(df)
elig = df[stageA]
completed = elig["FL28"] == COMPLETED

flow = [
    ("All children 5-17 in fs.sav", len(df), None),
    ("Attending school this year (CB7=1)", int((df["CB7"] == ATTENDING).sum()), None),
    ("... at primary level (CB8A=1)", int(((df["CB7"] == ATTENDING) & (df["CB8A"] == PRIMARY)).sum()), None),
    ("Stage A: Grade 2-3 (CB8B in 2,3)", len(elig), None),
    ("Stage B: FL28 = completed", int(completed.sum()), None),
]
flow = pd.DataFrame(flow, columns=["step", "n", "pct_of_stageA"])
flow["pct_of_stageA"] = (flow["n"] / len(elig) * 100).round(1).where(flow["step"].str.startswith("Stage"))
print("\n=== Sample flow ===")
print(flow.to_string(index=False))
flow.to_csv(OUT / "sample_flow.csv", index=False)

fl28_counts = elig["FL28"].map(vlabels["FL28"]).value_counts(dropna=False)
grade_counts = elig["CB8B"].map(vlabels["CB8B"]).value_counts()
print("\nGrade 2-3 by FL28 result:")
print(fl28_counts.to_string())
print("\nGrade 2-3 by grade:")
print(grade_counts.to_string())
fl28_counts.rename_axis("fl28_result").rename("n").reset_index().to_csv(OUT / "fl28_result_counts.csv", index=False)
grade_counts.rename_axis("grade").rename("n").reset_index().to_csv(OUT / "grade_counts.csv", index=False)

# valid outcome counts among completed (non-missing, non-special codes)
comp = elig[completed]
item_counts = []
print("\nCompleted-assessment children with usable reading items:")
for v in ["FL20A", "FL20B", "FL21BA", "FL21BE"]:
    n_ok, n_nr = int(comp[v].notna().sum()), int(comp[v].isin([9, 99]).sum())
    print(f"  {v}: {n_ok} non-missing, {n_nr} coded no-response")
    item_counts.append(dict(task="reading", variable=v, n_non_missing=n_ok, n_coded_no_response=n_nr))
print("Completed-assessment children with usable numeracy items:")
for v in ["FL23A", "FL24A", "FL25A", "FL27A"]:
    n_ok = int(comp[v].notna().sum())
    print(f"  {v}: {n_ok} non-missing")
    item_counts.append(dict(task="numeracy", variable=v, n_non_missing=n_ok, n_coded_no_response=None))
pd.DataFrame(item_counts).to_csv(OUT / "usable_item_counts.csv", index=False)

# reading routing: who got the story, and who got the comprehension questions
print("\n=== Reading routing (completed assessments) ===")
routing = []


def stack(check, ct):
    long = ct.stack().rename("n").reset_index()
    long.columns = ["group", "column", "n"]
    routing.append(long.assign(check=check))


ct_practice = pd.crosstab(comp["FL14"].map(vlabels["FL14"]).fillna("(blank)"), comp["FL20A"].notna().rename("has FL20A"))
print("FL14 practice read correctly vs has word count (FL20A):")
print(ct_practice.to_string())
stack("FL14 practice read correctly vs has word count (FL20A)", ct_practice)
ct_comp = pd.crosstab(comp["FL20A"].notna().rename("has FL20A"), comp["FL21BA"].notna().rename("has FL21BA"))
print("Has word count vs has comprehension answers (FL21BA):")
print(ct_comp.to_string())
stack("has word count vs has comprehension answers (FL21BA)", ct_comp)
lang = comp["FL9"].map(vlabels["FL9"]).value_counts(dropna=False)
print("Language of instruction (FL9):")
print(lang.to_string())
routing.append(lang.rename_axis("group").rename("n").reset_index().assign(column="", check="language of instruction (FL9)"))
pd.concat(routing)[["check", "group", "column", "n"]].to_csv(OUT / "reading_routing_checks.csv", index=False)

missed = comp.groupby(comp["FL21BA"].notna())["FL20B"].describe().round(1)
print("Missed words (FL20B) among those with comprehension answers vs not:")
print(missed.to_string())
missed.rename_axis("has_comprehension_answers").to_csv(OUT / "missed_words_by_comprehension.csv")

# assessed vs non-assessed comparison on a few descriptors
print("\nAssessed vs not (Grade 2-3), unweighted:")
avn = []
for v in ["HL4", "windex5", "HH6"]:
    t = pd.crosstab(elig[v].map(vlabels.get(v, {})), completed.rename("assessed"), normalize="columns").round(3)
    print(f"-- {labels[v]}\n{t.to_string()}")
    avn.append(t.reset_index().rename(columns={v: "level"}).assign(variable=v, label=labels[v]))
pd.concat(avn)[["variable", "label", "level", False, True]].rename(
    columns={False: "share_not_assessed", True: "share_assessed"}).to_csv(OUT / "assessed_vs_not.csv", index=False)

# 3. survey design variables
print("\n=== Survey design variables (Grade 2-3 sample) ===")
design = []
for v in ["fsweight", "fshweight", "PSU", "stratum"]:
    s = elig[v]
    print(f"{v}: label={labels.get(v)!r}, missing={int(s.isna().sum())}, "
          f"n_unique={s.nunique()}, min={s.min()}, max={s.max()}")
    design.append(dict(variable=v, label=labels.get(v), missing=int(s.isna().sum()),
                       n_unique=s.nunique(), min=s.min(), max=s.max()))
pd.DataFrame(design).to_csv(OUT / "survey_design_variables.csv", index=False)
w_corr = round(elig["fsweight"].corr(elig["fshweight"]), 3)
min_psu = int(elig.groupby("stratum")["PSU"].nunique().min())
print("Correlation fsweight vs fshweight:", w_corr)
print("PSUs:", elig["PSU"].nunique(), "| strata:", elig["stratum"].nunique(),
      "| min PSUs per stratum:", min_psu)
pd.DataFrame([dict(corr_fsweight_fshweight=w_corr, n_psu=elig["PSU"].nunique(),
                   n_strata=elig["stratum"].nunique(), min_psu_per_stratum=min_psu)]
             ).to_csv(OUT / "survey_design_summary.csv", index=False)

# 4. candidate variable audit (restricted to the stage a sample)
rows = []
for domain, vars_ in config.CANDIDATES.items():
    for v in vars_:
        if v not in df.columns:
            rows.append(dict(domain=domain, variable=v, label="NOT IN fs.sav"))
            continue
        s = elig[v]
        vl = vlabels.get(v, {})
        special_codes = [k for k, lab in vl.items() if any(t in str(lab).upper() for t in config.SPECIAL)]
        is_special = s.isin(special_codes)
        valid = s.notna() & ~is_special
        freq = (s[valid].map(lambda x: vl.get(x, x)).value_counts(normalize=True)
                .round(3).head(6))
        rows.append(dict(
            domain=domain, variable=v, label=labels.get(v),
            numeric_unlabelled=len(vl) == 0, n_stageA=len(s),
            n_valid=int(valid.sum()),
            pct_sysmissing=round(s.isna().mean() * 100, 1),
            pct_special_code=round(is_special.mean() * 100, 1),
            pct_unusable=round((~valid).mean() * 100, 1),
            n_categories=int(s[valid].nunique()),
            top_categories="; ".join(f"{k}={p}" for k, p in freq.items()),
            value_labels=str(vl) if vl else "",
        ))

audit = pd.DataFrame(rows)
audit.to_csv(OUT / "variable_audit.csv", index=False)
print("\n=== Candidate variable audit (Stage A sample) ===")
cols = ["domain", "variable", "n_valid", "pct_sysmissing", "pct_special_code", "n_categories"]
print(audit[[c for c in cols if c in audit]].to_string(index=False))
print(f"\nSaved tables and console_log.txt to {OUT}")
