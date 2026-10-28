"""

verify the analytic sample and audit candidate variables in fs.sav

outputs (in outputs/):
  sample_flow.csv      grade 2-3 eligibility and assessment-completion counts
  variable_audit.csv   one row per candidate variable

"""
from pathlib import Path

import pandas as pd
import pyreadstat

ROOT = Path(__file__).parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df, meta = pyreadstat.read_sav(ROOT / "fs.sav")
labels = meta.column_names_to_labels
vlabels = meta.variable_value_labels

# codes confirmed against the spss value labels
ATTENDING, PRIMARY, GRADES, COMPLETED = 1, 1, [2, 3], 1

# 1. value labels for eligibility variables
print("=== Eligibility variable labels ===")
for v in ["CB7", "CB8A", "CB8B", "FL28"]:
    print(f"{v}: {labels[v]}\n  {vlabels.get(v)}")

# 2. sample flow
stageA = (df["CB7"] == ATTENDING) & (df["CB8A"] == PRIMARY) & df["CB8B"].isin(GRADES)
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

print("\nGrade 2-3 by FL28 result:")
print(elig["FL28"].map(vlabels["FL28"]).value_counts(dropna=False).to_string())
print("\nGrade 2-3 by grade:")
print(elig["CB8B"].map(vlabels["CB8B"]).value_counts().to_string())

# valid outcome counts among completed (non-missing, non-special codes)
comp = elig[completed]
print("\nCompleted-assessment children with usable reading items:")
for v in ["FL20A", "FL20B", "FL21BA", "FL21BE"]:
    print(f"  {v}: {int(comp[v].notna().sum())} non-missing, "
          f"{int(comp[v].isin([9, 99]).sum())} coded no-response")
print("Completed-assessment children with usable numeracy items:")
for v in ["FL23A", "FL24A", "FL25A", "FL27A"]:
    print(f"  {v}: {int(comp[v].notna().sum())} non-missing")

# reading routing: who got the story, and who got the comprehension questions
print("\n=== Reading routing (completed assessments) ===")
print("FL14 practice read correctly vs has word count (FL20A):")
print(pd.crosstab(comp["FL14"].map(vlabels["FL14"]).fillna("(blank)"), comp["FL20A"].notna().rename("has FL20A")).to_string())
print("Has word count vs has comprehension answers (FL21BA):")
print(pd.crosstab(comp["FL20A"].notna().rename("has FL20A"), comp["FL21BA"].notna().rename("has FL21BA")).to_string())
print("Language of instruction (FL9):")
print(comp["FL9"].map(vlabels["FL9"]).value_counts(dropna=False).to_string())
print("Missed words (FL20B) among those with comprehension answers vs not:")
print(comp.groupby(comp["FL21BA"].notna())["FL20B"].describe().round(1).to_string())

# assessed vs non-assessed comparison on a few descriptors
print("\nAssessed vs not (Grade 2-3), unweighted:")
for v in ["HL4", "windex5", "HH6"]:
    t = pd.crosstab(elig[v].map(vlabels.get(v, {})), completed.rename("assessed"), normalize="columns").round(3)
    print(f"-- {labels[v]}\n{t.to_string()}")

flow.to_csv(OUT / "sample_flow.csv", index=False)

# 3. survey design variables
print("\n=== Survey design variables (Grade 2-3 sample) ===")
for v in ["fsweight", "fshweight", "PSU", "stratum"]:
    s = elig[v]
    print(f"{v}: label={labels.get(v)!r}, missing={int(s.isna().sum())}, "
          f"n_unique={s.nunique()}, min={s.min()}, max={s.max()}")
print("Correlation fsweight vs fshweight:", round(elig["fsweight"].corr(elig["fshweight"]), 3))
print("PSUs:", elig["PSU"].nunique(), "| strata:", elig["stratum"].nunique(),
      "| min PSUs per stratum:", int(elig.groupby("stratum")["PSU"].nunique().min()))

# 4. candidate variable audit (restricted to the stage a sample)
CANDIDATES = {
    "socioeconomic/demographic": ["windex5", "wscore", "melevel", "HH6", "HH7", "HL4", "FSAGE"],
    "home learning/parental involvement": ["PR3", "PR5", "PR6", "PR8", "PR10", "PR11B"],
    "functioning/wellbeing": ["fsdisability", "FCF19", "FCF20", "FCF21", "FCF24", "FCF25", "FCF26"],
    "discipline": ["FCD2C", "FCD2D", "FCD2F", "FCD2G", "FCD2H", "FCD2I", "FCD2J"],
    "digital access": ["CB10D", "CB10EA", "CB10EB", "CB10EC", "CB10ED", "CB10EE", "CB10F"],
    "child labour": ["CL1A", "CL1B", "CL1C", "CL1X", "CL3", "CL4", "CL5", "CL7", "CL9",
                     "CL11A", "CL11B", "CL11C", "CL11D", "CL11E", "CL11F", "CL13"],
    "other": ["CB11", "fsinsurance", "PR12", "PR13"],
}
SPECIAL = ("NO RESPONSE", "DK", "DON'T KNOW", "MISSING", "REFUSED", "NOT STATED")

rows = []
for domain, vars_ in CANDIDATES.items():
    for v in vars_:
        if v not in df.columns:
            rows.append(dict(domain=domain, variable=v, label="NOT IN fs.sav"))
            continue
        s = elig[v]
        vl = vlabels.get(v, {})
        special_codes = [k for k, lab in vl.items() if any(t in str(lab).upper() for t in SPECIAL)]
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
print(f"\nSaved {OUT/'sample_flow.csv'} and {OUT/'variable_audit.csv'}")
