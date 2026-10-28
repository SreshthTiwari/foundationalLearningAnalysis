# Results summary (k = 3, n = 368 Grade 2–3 children, MICS6 Jamaica 2022)

All estimates are survey-weighted (`fsweight`) with design-based 95% CIs (strata, PSUs). Clusters were built without reading or numeracy. Everything is association, not cause. Sources: `proficiency_by_cluster.csv` (04), `regression_odds_ratios.csv` (05), `k_comparison.csv` (06), `explainable/` (07).

## 1. Proficiency by cluster
Strict MICS definitions (reading: ≥90% of words + all 5 comprehension; numeracy: all items in all 4 tasks).

| Cluster | Children | Weighted share | Reading (CI), n with outcome | Numeracy (CI) |
|---|---|---|---|---|
| 1 | 193 | 49.3% | 60.6% (52.4–68.8), 178 | 35.9% (28.1–43.6) |
| 2 | 144 | 40.9% | 27.0% (18.2–35.8), 128 | 20.8% (13.1–28.4) |
| 3 | 31 | 9.8% | 29.7% (10.1–49.3), 26 | 27.6% (7.9–47.2) |
| All | 368 | 100% | 44.1% (37.9–50.4), 332 | 28.9% (23.3–34.5) |

Sensitivity definitions, same clusters:

| Cluster | Reading, 3-of-5 comprehension | Reading, not-assessed counted as not proficient | Numeracy, 3-of-4 tasks |
|---|---|---|---|
| 1 | 69.8% (62.1–77.5) | 56.3% (48.7–63.8) | 64.2% (56.0–72.5) |
| 2 | 36.2% (26.2–46.1) | 24.4% (16.4–32.5) | 40.7% (30.8–50.6) |
| 3 | 38.9% (15.6–62.3) | 27.9% (10.4–45.4) | 33.2% (12.7–53.7) |
| All | 53.3% (47.0–59.6) | 40.5% (34.8–46.1) | 51.6% (44.8–58.4) |

The ordering (cluster 1 clearly highest) holds under every definition. Cluster 3's intervals are very wide (31 children; 26 with reading).

## 2. Survey-weighted logistic regression (05)
Odds ratios vs cluster 1 (largest cluster). M0 = cluster + grade + age; M1 = M0 + wealth (3 levels) + mother's education. Same children in both models (reading n=330, numeracy n=366). PSU/strata sandwich SEs, df = 207.

| Outcome | Term | M0 OR (CI), p | M1 OR (CI), p |
|---|---|---|---|
| Reading | Cluster 2 | 0.22 (0.12–0.39), <0.001 | 0.35 (0.16–0.77), 0.009 |
| Reading | Cluster 3 | 0.26 (0.10–0.67), 0.005 | 0.30 (0.11–0.87), 0.026 |
| Numeracy | Cluster 2 | 0.45 (0.25–0.80), 0.007 | 0.98 (0.46–2.09), 0.96 |
| Numeracy | Cluster 3 | 0.66 (0.24–1.86), 0.43 | 0.95 (0.33–2.76), 0.93 |

Joint test of cluster terms: reading p<0.001 → 0.014 after adjustment; numeracy p=0.027 → 0.996. Upper-40% wealth vs lower-40% in M1: reading OR 2.69 (p=0.014), numeracy OR 3.08 (p=0.003). Mother's education is not significant in either. **Reading differences between profiles survive controlling for wealth and mother's education; numeracy differences do not.** Caveats: coarse wealth control, clusters estimated from the same data (optimistic p-values), small sample.

## 3. Choice of k (06)
At 6 MCA dimensions (the step-3 setting):

| k | Bootstrap ARI | Smallest-cluster Jaccard | Smallest cluster (n) | Verdict |
|---|---|---|---|---|
| 2 | 0.61 | 0.75 | 152 | Wealth/digital split only |
| **3** | **0.67** | **0.60** | **31** | **Chosen** |
| 4 | 0.52 | 0.50 | 36 | Exploratory (adds a school-contact profile) |
| 5 | 0.46 | 0.48 | 20 | Rejected |
| 6–8 | 0.52–0.54 | 0.41–0.51 | 15–20 | Rejected |

At 3 MCA dimensions k=3 is more stable (ARI 0.74; cluster Jaccards 0.82/0.86/0.73). k≥5 clusters are single-variable groups with <25 children.

## 4. What defines the clusters (07; details in `explainable/report.md`)
- XGBoost predicts cluster from the original features: accuracy 0.921, balanced accuracy 0.900 (Random Forest 0.895/0.868; baseline 0.524). Errors are mostly between clusters 1 and 2. Cluster 3: precision 0.81, recall 0.85 (CI 0.71–0.95).
- Cluster 1: wealthier (60% upper 40% vs 36% overall), mother post-secondary (48 vs 32), 10+ books (64 vs 43), home internet and own device (96 vs 70), daily online learning (94 vs 75), PTA (91 vs 80), no functional difficulty.
- Cluster 2: lower 40% wealth (81 vs 45), 0–4 books (68 vs 41), limited digital resources (58 vs 30), more physical punishment (74 vs 61), more child work (18 vs 10).
- Cluster 3: functional difficulty (78 vs 9), anxiety/sadness monthly or more (64 vs 11), rarely any online learning (41 vs 8), less homework help (29 vs 7 none/no homework), wealth near average.
- SHAP share by domain: child functioning 28%, socioeconomic 24%, digital 21%, home learning and school contact 19%, discipline 7%, child work 2%. Parental-involvement variables contribute little.
- **Is it just SES?** SES alone (wealth, mother's education, area) recovers the clusters poorly (balanced accuracy 0.46); without SES 0.82. Cluster 3 is a functioning-based profile, not an SES group; clusters 1 vs 2 are mostly a resource gradient (books and digital resources are partly SES proxies).

## 5. Open items
See "Left to do" in `README.md`.
