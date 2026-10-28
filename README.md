# Child profiles and foundational learning in Jamaica (MICS6 2022)

## Goal
Identify different profiles of Grade 2 - 3 children from their individual, household, school support and protection circumstances, then compare foundational reading and numeracy proficiency across profiles (SDG 4.1.1a). Method adapted from Alvarez-Garcia et al. (2024), Uncovering student profiles (MCA, weighted k-means, classifier + SHAP). Profiles are built without the outcomes.

## Important files
- Data: `fs.sav` (MICS6 children 5–17; the only dataset used). `Jamaica MICS6 Datasets/` holds the others (hh, hl, wm, ch) and the Readme, unused.
- Docs: handover PDF (`Jamaica_Child_Development_Conference_Current_Summary.pdf`), the PISA reference paper, `MICS6-manual-for-stats-data-analysis-English_2020.pdf` (used to verify scoring and weights).
- `prelim_sorting.ipynb`: the old exploratory notebook; superseded, not used.
- Code: `01`–`07_*.py` (below), `requirements.txt`. Outputs go to `outputs/` and `outputs/explainable/`.

## Run (from this folder; Python 3.13 venv in `.venv`, already built)
```
.venv/bin/pip install -r requirements.txt     # only if .venv is missing
.venv/bin/python 01_data_audit.py
.venv/bin/python 02_build_dataset.py
.venv/bin/python 03_profiles.py [--k N]       # default: auto-picks k in 3..6 (gives k=3)
.venv/bin/python 04_explain_and_outcomes.py
.venv/bin/python 05_survey_inference.py
.venv/bin/python 06_compare_k.py              # does not change clusters.csv
.venv/bin/python 07_explainable_ml.py         # writes only to outputs/explainable/
```
Steps are deterministic (seed 42) and run in order; 03 writes `clusters.csv`, which 04–07 read.

## What each step does (all done)
| Step | Result |
|---|---|
| 01 audit | Sample flow, reading-routing checks, candidate-variable audit |
| 02 build | `analytic.csv`: outcomes, 14 categorical features (`f_*`), descriptive vars (`d_*`), weight/PSU/stratum |
| 03 profiles | MCA (6 dims), survey-weighted k-means k=2–6, bootstrap stability; saves labels for every k and the chosen k |
| 04 explain | Weighted proficiency by cluster with design-based CIs (strict, lenient and not-assessed-as-fail versions), rates for every k, XGBoost + SHAP at cluster level |
| 05 survey | Design-based logistic regression in Python (replaces R's `survey`): M0 = cluster + grade + age; M1 = M0 + wealth + mother's education |
| 06 compare k | k=2–8 × MCA dims 3/4/6: bootstrap ARI, per-cluster Jaccard, sizes, distinctive levels |
| 07 explainable ML | RF + XGBoost predict clusters from original features (3× repeated stratified 5-fold CV), out-of-fold SHAP, SES-only/without-SES ablation, 4 figure types |

## Sample and outcomes
2,725 children → 2,531 attending → 1,197 primary → **394 in Grades 2–3** (193 Gr 2, 201 Gr 3) → **368 completed the interview (analytic sample)**. Design: 234 PSUs in the sample, 27 strata (one stratum has a single PSU; handled with a lonely-PSU adjustment).

Verified against the MICS6 manual:
- **Reading** = reads ≥90% of the 72-word story (≥65 words) **and** answers all 5 comprehension questions (3 literal + 2 inferential). The handover's "3 of 5" rule is not in the manual.
- **Numeracy** = every item correct in all four tasks (number reading, discrimination, addition, patterns).
- **Weight** = `fsweight`; strata = `stratum`; PSU = `PSU`. (`fshweight` is the household weight; not used.)
- **Coding quirk:** `FL19W*` is 0 = correct, 1 = incorrect, 2 = not reached in Jamaica's file (manual says 1 = correct). Confirmed: `FL20B` equals the count of 1s, and comprehension questions were asked exactly when ≥65 words were correct.
- Failing the practice or the 90% stop rule counts as **not proficient**; numeracy items skipped by discontinue rules count as wrong.
- 36 children were never given the reading task. Main reading outcome treats them as missing (n=332); `reading_strict_incl` counts them as not proficient (manual divides by all children).
- Sensitivity outcomes: `reading_lenient` (3 of 5 comprehension), `numeracy_lenient` (3 of 4 tasks).

## Clustering features (14, all categorical)
wealth (3 levels), mother's education, area, books, homework help, PTA, school visit, disability, emotional difficulty, psychological aggression, physical punishment, online learning, digital resources, child economic work. Descriptive-only variables (not clustered): sex, grade, age, parish, insurance, school report. Four variables had 1–8 missing values mode-filled (non-structural); routing-driven missingness was never imputed.

**Routing artifact (fixed):** internet and device questions (`CB10EA`–`CB10F`) are only asked of children who did online learning. As separate features, the 18 children with no online learning formed their own cluster and made the smallest cluster 4.9% at every k. Fix: online learning stays one variable; internet + device are merged into `f_digital_resources` (`limited_or_none` includes the not-asked children).

## Key results (k = 3; details in `outputs/results_summary.md`)
| Cluster | n (weighted share) | Profile | Strict reading | Strict numeracy |
|---|---|---|---|---|
| 1 | 193 (49%) | Wealthier, more books, home internet and device, daily online learning | 60.6% (52.4–68.8) | 35.9% (28.1–43.6) |
| 2 | 144 (41%) | Poorer, fewer books, limited digital resources, more physical punishment | 27.0% (18.2–35.8) | 20.8% (13.1–28.4) |
| 3 | 31 (10%) | Functional and emotional difficulty, little online learning; wealth average | 29.7% (10.1–49.3) | 27.6% (7.9–47.2) |
| All | 368 | | 44.1% (37.9–50.4) | 28.9% (23.3–34.5) |

- **Regression (05):** the reading gap survives adjusting for wealth and mother's education (cluster 2 vs 1 OR 0.22 → 0.35, p=0.009; joint p<0.001 → 0.014). The numeracy gap disappears (joint p=0.027 → 0.996): it is a wealth gap.
- **k choice (06):** k=3 is the most stable choice with ≥25 children per cluster. k=4 adds a school-contact profile (exploratory). k≥5 produces clusters of ~20 or fewer, driven by single variables: rejected.
- **Explainable ML (07):** XGBoost recovers clusters with 0.92 accuracy (0.90 balanced; baseline 0.52); confusion is between clusters 1 and 2. SES alone recovers them poorly (0.46 balanced accuracy); without SES 0.82. Cluster 3 is a functioning-based profile, not an SES group; clusters 1 vs 2 are mostly a resource gradient; parental-involvement variables contribute little.

## Caveats
- Small sample (368; reading 332). Cluster 3 has 31 children (26 with reading), in MICS's "interpret with caution" band (25–49 unweighted cases).
- Cluster structure is weak (silhouette 0.17–0.31) and only moderately stable (k=3 bootstrap ARI ≈ 0.67; smallest cluster Jaccard 0.60). Treat profiles as a summary of a continuum.
- Cluster p-values are optimistic: clusters were found in the same data, and several k values and outcome definitions were tried.
- Classifier accuracy is expected to be high (labels derive from the same variables); it exists to support SHAP. SHAP explains the classifier, not causes.
- The wealth control in 05 is the coarse 3-level `f_wealth`; finer control could shrink the reading effect further.
- Tooling note: the handover PDF and manual can't be read with the built-in PDF reader (no poppler); text was extracted with `pypdf` in a throwaway venv.

## Left to do
1. Finer wealth control: add `d_wscore`/`d_windex5` (and optionally language of instruction `FL9`) to `02`, rerun 02–07, and refit 05.
2. Run the 05 two-model comparison for k = 4 (does the school-contact profile matter beyond wealth?).
3. Consider rerunning 03 with 3–4 MCA dimensions instead of 6 (06 found 3–4 more stable; k=3 clusters at 4 dims agree with the current ones, ARI 0.88).
4. Robustness: alternative feature sets (individual functioning items, insurance), FAMD or latent class analysis.

