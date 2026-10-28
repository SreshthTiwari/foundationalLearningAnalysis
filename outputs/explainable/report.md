# Explaining the child profiles (step 7)

Script: `07_explainable_ml.py`. Reads `analytic.csv` and `clusters.csv` (k = 3, n = 368); writes only to this folder. Reading and numeracy are never read, used or explained.

## Methods
- **Target:** the existing cluster labels (step 3). **Predictors:** the 14 original categorical features (34 one-hot columns), not MCA components. Missing values mode-filled exactly as in step 3 so the labels and classifier see the same data.
- **Models:** Random Forest and XGBoost, deliberately small for n = 368 (shallow trees, class-balanced weights, no SMOTE or grid search; the smallest cluster has 31 children).
- **Evaluation:** 3 × repeated stratified 5-fold CV; accuracy, balanced accuracy, macro and per-cluster precision/recall, confusion matrix.
- **SHAP:** TreeSHAP on the better model (XGBoost), computed **out-of-fold** so each child is explained by a model that never saw them; dummy SHAP values are summed to variable level. Ranking agrees with a full-data fit (Spearman ρ = 0.99).
- **"Is it just SES?" checks:** retrain on SES-only features (wealth, mother's education, area) and on everything except them; Cramér's V per variable.
- SHAP explains the classifier's reading of the clusters, not causes. The classifier is not a validation of the clusters: the labels come from the same variables.

## Classifier performance (mean ± sd over repeats; majority-class baseline accuracy 0.524)
| Model | Accuracy | Balanced acc. | Macro precision | Macro recall |
|---|---|---|---|---|
| Random Forest | 0.895 ± 0.003 | 0.868 | 0.866 | 0.868 |
| **XGBoost** | **0.921 ± 0.003** | **0.900** | 0.888 | 0.900 |

| Cluster | n | Precision | Recall (95% CI, one repeat) |
|---|---|---|---|
| 1 | 193 | 0.95 | 0.95 (0.90–0.97) |
| 2 | 144 | 0.91 | 0.90 (0.85–0.95) |
| 3 | 31 | 0.81 | 0.85 (0.71–0.95) |

Nearly all errors are between clusters 1 and 2 (about 9 children each way); cluster 3 is rarely confused. The cluster 3 estimates rest on 31 children and are noisy. Figure: `fig4_confusion_matrix.png`.

## What distinguishes the clusters (figures: `fig1_shap_global.png`, `fig2_shap_beeswarm_cluster{1,2,3}.png`, `fig3_cluster_heatmap.png`)
Share of total SHAP importance by domain: child functioning 28%, socioeconomic 24%, digital access 21%, home learning and school contact 19%, discipline 7%, child work 2%. Top variables: disability (18.5%), wealth (12.5%), online learning (12.1%), emotional difficulty (9.4%), digital resources (8.6%), books (8.5%), mother's education (6.9%).

| Cluster | Weighted share | Profile (cluster % vs overall %) |
|---|---|---|
| **1: Well-resourced, connected** | 49% (n = 193) | Wealth upper 40%: 60 vs 36; lower 40%: 14 vs 45. Mother post-secondary: 48 vs 32. 10+ books: 64 vs 43. Home internet and own device: 96 vs 70. Online learning daily: 94 vs 75. Attends PTA: 91 vs 80. Almost no functional difficulty (0% vs 9%). Less physical punishment (49% vs 61%). |
| **2: Lower-resourced** | 41% (n = 144) | Wealth lower 40%: 81 vs 45. Mother lower-secondary or less: 28 vs 19. 0–4 books: 68 vs 41. Limited or no digital resources: 58 vs 30. Rural: 60 vs 51. Did not attend PTA: 28 vs 20. Physical punishment 74% vs 61%; economic work 18% vs 10%. |
| **3: Children with functional and emotional difficulty** | 10% (n = 31) | Functional difficulty 78% vs 9%. Anxiety or sadness at least monthly: 64 vs 11. Rarely or never does online learning: 41 vs 8. No homework help (or no homework): 29 vs 7. No PTA attendance: 41 vs 20. Wealth and mother's education close to average (48% lowest 40% vs 45%). |

## Do the profiles reflect more than socioeconomic status?
| Features used | Accuracy | Balanced accuracy |
|---|---|---|
| All | 0.921 | 0.900 |
| SES only (wealth, mother's education, area) | 0.605 | 0.463 |
| Everything except SES | 0.822 | 0.824 |

- **Not just SES.** SES alone barely beats guessing (balanced accuracy 0.46 vs 0.33 chance; accuracy 0.61 vs 0.52 baseline). Cluster 3 is not an SES group at all: it is defined by functioning, emotional difficulty and online learning, and its wealth profile is near average. Disability alone has the strongest association with cluster (Cramér's V = 0.80; wealth 0.46).
- **Clusters 1 and 2 are mostly a resource gradient.** They differ together on wealth, books, digital resources and mother's education, and they are the pair the classifier confuses. Digital resources and books are partly SES proxies, so the "without SES" model overstates independence from SES.
- **Parental involvement is a minor distinguishing factor.** PTA attendance (4%), homework help (4%) and school visits (2%) matter little, except that cluster 3 children less often get homework help or have PTA attendance.

## Limits
- n = 368 and a 31-child cluster; precision and recall for cluster 3 are uncertain, and MICS treats 25–49 cases as "interpret with caution".
- Cluster 3's prominence partly reflects a rare binary trait (disability, 9%) that k-means and MCA pick up easily.
- Cluster stability from step 6 was only moderate (k = 3 bootstrap ARI about 0.67, smallest cluster 0.60), so treat the profiles as a useful summary of a continuum, not sharply separated types.
- All findings are associations. Do not name or interpret clusters as deficits; they describe circumstances.
