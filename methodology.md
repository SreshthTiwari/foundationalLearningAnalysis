# Methodology

## What is this project trying to do?
We want to know whether children in Grades 2–3 in Jamaica fall into a few natural "types" based on their home life, school support and personal circumstances. Then we check whether children of different types read and do maths differently.

The data comes from the 2022 MICS6 household survey in Jamaica (a large UN-backed survey). We use the file with children aged 5–17 and keep only those in Grades 2–3.

**The key rule:** the groups are made *without* looking at reading or maths scores. Scores are only checked afterwards. This way the groups are not built to produce a result.

The approach is adapted from a published study (Alvarez-Garcia et al., 2024) that grouped students using the same kind of steps.

---

## Step 1: Pick the children
We started with 2,725 children and narrowed down:

| Who | Number |
|---|---|
| All children in the file | 2,725 |
| Attending school | 2,531 |
| In primary school | 1,197 |
| In Grades 2–3 | 394 |
| Finished the interview (our sample) | **368** |

The survey did not pick children completely evenly. Some kinds of children were more likely to be chosen than others. So every result uses the survey's **weights**, which correct for this so the numbers represent Jamaica's children more fairly. We also account for the fact that children were sampled in clusters (neighbourhoods), which affects how confident we can be.

## Step 2: Define "can read" and "can do maths"
We followed the official MICS6 rules:

- **Reading:** the child reads at least 90% of a short 72-word story correctly **and** answers all 5 questions about it.
- **Maths:** the child gets every question right in all four tasks (reading numbers, telling which number is bigger, adding, and completing number patterns).

These are strict rules. So we also check two looser versions (3 of 5 reading questions; 3 of 4 maths tasks) to see whether conclusions change.

Two practical details:
- In Jamaica's data file, the answer codes for the reading task were flipped compared with the manual (0 meant correct). We checked this carefully and fixed it.
- 36 children were never given the reading task. We leave them out of the main reading result (leaving 332 children). We also show a version that counts them as "not able to read" to test whether that matters.

## Step 3: Choose the things used to describe each child
We used 14 pieces of information, all in categories (like "high / medium / low"):

- **Household:** wealth, mother's education, urban or rural.
- **Home learning:** number of books at home, help with homework, online learning, internet and device access.
- **School contact:** whether parents are in the PTA, whether they visit the school.
- **Child's wellbeing:** disability, emotional difficulties.
- **Discipline:** psychological aggression, physical punishment.
- **Work:** whether the child does paid or economic work.

We did **not** use reading or maths scores, nor sex, grade, age or parish (these are only used to describe the groups).

A small number of answers were missing (1–8 per question). We filled these with the most common answer. Where an answer was missing only because the question was never asked to that child, we did not fill it in.

**One problem we fixed:** the internet and device questions were only asked of children who did online learning. Left as they were, the children with no online learning formed their own artificial group. We combined these questions into one "digital resources" measure to avoid this.

## Step 4: Find the groups
This is done in two parts:

1. **Simplify the data (MCA).** With 14 questions, it is hard to see patterns. A technique called Multiple Correspondence Analysis squeezes the information into 6 summary dimensions, a bit like turning many survey answers into a few overall scores. It is the category-based version of a common tool called PCA.
2. **Group similar children (weighted k-means).** The method then sorts children into groups so that children in the same group are similar to each other and different from other groups. The "weighted" part means the survey weights are used. We tried 2 to 6 groups.

## Step 5: Decide how many groups
There is no single right answer, so we tested how reliable each choice was:

- **Stability:** we re-ran the grouping many times on slightly reshuffled samples and checked whether the same groups kept appearing.
- **Size:** very small groups are unreliable.

We also tested 2 to 8 groups with 3, 4 or 6 summary dimensions.

**Result:** 3 groups was the best balance. Four groups added one more group (about school contact), which we treat as exploratory only. Five or more gave tiny groups of about 20 children or fewer, each defined by a single answer, so we rejected them.

## Step 6: Describe the three groups
| Group | Children | Share of Jamaica's children | What they look like |
|---|---|---|---|
| 1 | 193 | 49% | Wealthier, more books, home internet and a device, daily online learning |
| 2 | 144 | 41% | Poorer, fewer books, limited digital access, more physical punishment |
| 3 | 31 | 10% | Difficulty with daily functioning and emotions, little online learning, average wealth |

## Step 7: Compare reading and maths across groups
We calculated the share of children who are proficient in each group. Each figure comes with a **confidence interval**, a range that shows how uncertain the estimate is. With small groups the range is wide. We always show how many children each estimate is based on.

## Step 8: Is it really just about money?
A fair question is whether the group differences are simply wealth differences. We tested this with **logistic regression**, a method that estimates how much more or less likely a child is to be proficient while holding other factors constant. We ran two versions:

- **Model 1:** group, grade and age only.
- **Model 2:** the same, plus household wealth and mother's education.

If a group difference disappears in Model 2, it was really about wealth. If it stays, the group captures something beyond wealth.

**Finding:** the reading gap between groups stays after adjusting for wealth and mother's education. The maths gap disappears, so it looks like a wealth gap.

## Step 9: What makes the groups different? (the "explainable machine learning" step)

**The problem.** The grouping method gives each child a label (1, 2 or 3) but doesn't say *why*. This step asks: which answers define each group?

**The idea.** We train a prediction model to guess each child's group from their 14 answers. Then we open the model up and see which answers it relied on. Those answers are what define the groups. The model is only a means to an end. Its real job is to give the opening-up tool (SHAP) something to explain.

### 9a. The prediction models
- Two standard models were used: **Random Forest** and **XGBoost**. Both learn rules like "if the child has few books and no home internet, they are probably in group 2".
- They use the original answers (not the 6 summary dimensions from Step 4), so results can be read in everyday terms.
- With only 368 children, we kept things deliberately simple: shallow models, no fine-tuning, and extra weight on the small group so the model doesn't just ignore it.

### 9b. Fair testing
We repeatedly split the children: train on most of them, then test on the rest. This was done 5 ways, repeated 3 times. Every child is predicted by a model that never saw them.

### 9c. SHAP: opening the model up
**SHAP** gives each answer a score for each child, showing how strongly that answer pushed the model towards a group. For example, "having a disability pushed this child towards group 3".

- SHAP scores were also computed on children the model hadn't seen, so they aren't flattered by the model memorising the data.
- Scores were added up for each of the 14 questions, then for 6 themes (domains).

### 9d. "Is it just wealth?" test
We re-ran the model three ways to see whether wealth alone can explain the groups:
1. all 14 answers;
2. only wealth-type answers (household wealth, mother's education, urban/rural);
3. everything except those.

### 9e. Simple extra checks
- **Cramér's V:** a plain 0-to-1 score for how strongly each answer is linked to group membership.
- **Heatmap:** shows how much each group differs from the overall average on each answer.

### Findings
- **Accuracy:** XGBoost guessed the right group about 92% of the time (Random Forest about 90%). Always guessing the biggest group would be right about 52%. Mistakes were mostly mix-ups between groups 1 and 2. This is high partly by design, since the groups were built from the same answers, so it shows the groups are recoverable, not that the model is impressive.
- **Which themes matter most (share of SHAP importance):**

| Theme | Share |
|---|---|
| Child functioning (disability, emotional difficulty) | 28% |
| Wealth and household background | 24% |
| Digital access | 21% |
| Home learning and school contact | 19% |
| Discipline | 7% |
| Child work | 2% |

- **Wealth test:** wealth-type answers alone recovered the groups poorly (balanced accuracy 0.46). Without them, the model still did well (0.82). So the groups are not just wealth bands.
- **Group 3** is mainly defined by children's functioning and emotional difficulties, not by wealth.
- **Groups 1 and 2** mostly differ in household resources such as books and digital access. Those are partly stand-ins for wealth.
- **Parental involvement** (PTA, school visits) mattered little.

### Caution
SHAP explains the *model*, not the real world. "Books mattered" means the model used books to tell groups apart. It does not mean books cause better outcomes.

## Step 10: Repeatability
The whole analysis runs as seven scripts in order, with a fixed random seed (42). Anyone running them should get identical results. A record of software versions and settings is saved with the outputs.

---

## Design choices and why

Every analysis involves choices: who to include, what to measure, which method to use. This section lists the main ones, what the alternatives were, and the reason given. Each reason is tagged:

- **[Documented]**: the project's README or code comments state the reason.
- **[Inferred]**: no reason is written down; this is the most likely one, based on the code and outputs. It may not be what the author intended.
- **[Not recorded]**: no reason is stated and none is obvious.

### A. Who is in the sample

| Choice | Alternatives | Why |
|---|---|---|
| Only children in **Grades 2–3** (394 children) | All school-age children in the file (2,725) | **[Documented]** The project is about the foundational learning indicator (SDG 4.1.1a). **[Inferred]** That indicator is defined for Grades 2–3, which is why only those grades got the reading and maths test. |
| Must be attending **primary school** this year | Include out-of-school children | **[Inferred]** The grade question is only meaningful for pupils in school. The effect is that out-of-school children are not represented at all. |
| Must have **completed the interview** (368 of 394) | Keep the 26 who didn't | **[Inferred]** Without the interview there is no reading or maths result. |
| Use only the **child file (`fs.sav`)** | Also use the household, women's and other MICS files | **[Documented]** The README lists the other files as unused. **[Inferred]** Every variable the project needed (wealth, mother's education, area, outcomes) already sits in this file. |

**What it didn't check:** the 26 children who didn't complete the interview look somewhat different from the 368 who did (more in Kingston, more boys, fewer in the second-poorest wealth group). The group is tiny, the differences weren't tested, and the code doesn't adjust for it, so some drop-out bias can't be ruled out.

### B. How reading and maths were measured

| Choice | Alternatives | Why |
|---|---|---|
| **Strict official rules** as the main outcome (reading: 90% of words and all 5 questions right; maths: every item right) | Looser rules, such as 3 of 5 questions | **[Documented]** Checked against the MICS6 manual. The handover document's "3 of 5" rule isn't in the manual. The looser versions are kept as a check, and the ordering of groups is the same under all of them. |
| Children who **failed the practice round count as "not proficient"** | Treat them as missing | **[Documented]** The manual does this. Note this is large: 100 of the 368 children (27%) are in this category, so it strongly affects the overall reading rate. |
| The **36 children never given the reading task** are left out of the main reading result (leaving 332) | Count them as "not proficient" | **[Documented]** The manual divides by all children, so that version is also reported. The two give similar results (44% vs 41%). |
| Maths items skipped because the child got too many wrong count as wrong | Treat skipped items as missing | **[Documented]** The MICS rule. |
| The flipped answer codes in the reading data were reversed | Trust the manual's codes | **[Documented]** Checked against two internal consistency tests. A wrong guess here would have broken every reading result. |

### C. Which information describes each child (the 14 features)

Step 01 audited about 50 candidate questions. Only 14 variables made it into the grouping. The README says which were chosen but not why the rest were dropped, so most of the reasons below are inferred.

| Dropped | Reason |
|---|---|
| Hours-worked and dangerous-work questions (`CL3`, `CL4`, `CL5`, `CL13`) | **[Inferred]** Mostly not asked: 90% blank for the first three and 49% for hours. They only apply to children who work. |
| Chores (fetching water, cooking, shopping, ...) | **[Not recorded]** Only paid, family-business or income work is used for "child works". |
| Several functioning and emotional questions (`FCF19`–`FCF24`) | **[Inferred]** The overall disability measure (`fsdisability`) and the two anxiety/sadness questions are used instead. |
| Health insurance, school report, parish, sex, grade, age | **[Documented]** Kept as descriptive only, not used to form groups. **[Inferred]** Parish has 14 categories, too many for 368 children. |
| A finer wealth score (`wscore`) | **[Not recorded]** Only 3 wealth levels were used. The README lists the finer score as a to-do. |

**How the answers were simplified into categories** (all **[Not recorded]**, but sensible for a small sample):
- Wealth: 5 groups reduced to 3 (poorest 40%, middle 20%, richest 40%).
- Mother's education: 4 levels reduced to 3.
- Books at home: 0–4, 5–9 and 10 or more.
- Discipline and work: several questions combined into "any yes".
- With only 368 children, fewer, bigger categories mean fewer tiny cells, which tends to give steadier groups.

**Missing answers:** 12 values in total were filled with the most common answer (up to 8 per question, in 4 questions). **[Documented]** The code notes that gaps are under 3% per question. Gaps that exist only because a question wasn't asked were never filled.

**The online-learning fix:** **[Documented]** the internet and device questions were only asked of children who did online learning. Left separate, the 18 children who did none formed their own artificial group at every number of groups. Merging them into one "digital resources" measure removed it.

### D. Which method found the groups

| Choice | Alternatives | Why |
|---|---|---|
| **MCA, then k-means** | Latent class analysis, or FAMD (another method for mixed data) | **[Documented]** Adapted from the PISA reference paper (Alvarez-Garcia et al., 2024). **[Inferred]** k-means needs numbers, and the answers are categories, so MCA converts them first. The README lists latent class analysis and FAMD as untried checks. |
| **Survey weights** used in k-means | Unweighted | **[Inferred]** So the groups reflect Jamaica's children, not just who happened to be sampled. The MCA step itself is not weighted. |
| **6 summary dimensions** | 3 or 4 | **[Documented]** A rule keeps dimensions above a size threshold, with a cap at 6. **[Not recorded]** In practice the cap decided it: 8 dimensions pass the threshold. The results show **3–4 dimensions are more stable**, and re-running with fewer is on the to-do list. |
| **20 random restarts** in k-means | Fewer or more | **[Not recorded]** Step 06 used 50. |

### E. How many groups (k)

The main rule in Step 03 picks the number of groups (from 3 to 6) with the **highest stability (bootstrap ARI)**, among options where **the smallest group is at least 8% of children**. Two stability measures were used, both from re-running the grouping on reshuffled data:
- **ARI:** how much the grouping agrees with the original (1 = identical).
- **Jaccard:** how well each individual group is reproduced. Above 0.75 is stable, 0.6–0.75 indicative, below 0.5 dissolves. **[Documented]** The code states these cut-offs.

Results with the 6 dimensions actually used (from Step 06):

| Groups | Stability (ARI) | Weakest group's Jaccard | Smallest group | Verdict |
|---|---|---|---|---|
| 2 | 0.61 | 0.75 | 152 | Only a wealth/digital split. Excluded because the rule starts at 3. |
| **3** | **0.67** | **0.60** | **31** | **Chosen: highest stability among options meeting the size rule.** |
| 4 | 0.52 | 0.50 | 36 | Less stable. Fails the 8% rule in Step 03 (29 children). Kept as exploratory. |
| 5 | 0.46 | 0.48 | 20 | Too small and unstable. |
| 6–8 | 0.52–0.54 | 0.41–0.51 | 15–20 | Too small. Each group is driven by a single answer. |

**Why not 2?** **[Documented]** Two groups is only a wealth-and-digital split. **[Inferred]** The rule starts at 3 to get more than a rich/poor division. This is a judgement call: k=2 is actually the most reproducible on several measures.

**Why not 5 or more?** **[Documented]** Clusters of about 20 or fewer, each defined by one answer. This holds at 6 dimensions. At 4 dimensions, k=5 has a smallest group of 27 and a good stability score (0.71, slightly above k=3's 0.68), so the rejection depends on the 6-dimension setting.

**The 8% size rule** is **[Not recorded]** as to why 8%. It sits just below the smallest chosen group (8.4%), so it fits k=3. The README says "at least 25 children", which is a slightly different rule.

### F. How outcomes were compared across groups

| Choice | Why |
|---|---|
| **Weighted percentages with design-based confidence intervals** that account for strata and clusters (PSUs) | **[Documented]** MICS is a complex survey, so ordinary formulas give intervals that are too narrow. One stratum has a single PSU, handled with a standard adjustment. |
| **Logistic regression** (outcome is yes or no), run twice: M0 (group, grade, age) and M1 (also wealth and mother's education) | **[Documented]** Comparing the two tests whether the groups carry information beyond wealth and mother's education. |
| **Written in Python instead of R's `survey` package** | **[Not recorded]** Probably to keep the whole project in one language. A built-in self-check compares the code with a standard library on test data, to confirm the variance formula. |

### G. How the groups were explained

| Choice | Why |
|---|---|
| **Random Forest and XGBoost** classifiers | **[Documented]** Follows the PISA reference paper. **[Inferred]** Tree-based models work well with category answers and are what SHAP handles best. |
| **Shallow trees, no tuning, class weights** | **[Documented]** Chosen for the small sample (368, smallest group 31). Tuning on so few children would mostly fit noise. |
| **SHAP on children the model hadn't seen** | **[Documented]** So each child is explained by a model that never saw them. |
| **Wealth-only and without-wealth tests** | **[Documented]** To check whether the groups are just wealth bands. |

### Choices that deserve a second look

These come from reading the code and outputs. None is an error, but they affect how far the results can be trusted:

1. **The 3-group solution depends on the random start.** At 6 dimensions, re-running with different random seeds gave quite different 3-group solutions in some runs (agreement as low as 0.42). At 3 dimensions it is steady (0.92). The group of 31 is most likely the fragile one.
2. **The number of dimensions was set by a cap, not by the data.** The rule would have kept 8, and the cap forced 6. Using 3–4 looks more stable.
3. **Wealth and mother's education are both inputs to the groups and controls in the regression.** The "reading gap survives wealth" result therefore describes what's left after removing the part of the groups that wealth explains.
4. **Several choices were made after seeing results,** including which k and which outcome definition. The project's own caveats acknowledge this, and it means p-values are somewhat optimistic.
5. **27% of children failed the reading practice round,** and they count as not proficient. That is the official rule, but it has a large effect on the reading rate and may reflect the test setup (such as language of instruction) rather than skill.

## Limits to keep in mind
- **Small sample.** Only 368 children (332 for reading). Group 3 has just 31, so its results are very uncertain.
- **Weak groups.** The children form more of a continuum than clearly separate types. The groups are a useful summary, not sharply defined categories. They were only moderately stable when the data was reshuffled.
- **P-values are slightly optimistic.** The groups were found in the same data they are tested on, and several choices were tried.
- **Association, not cause.** We show that things go together, not that one causes the other. SHAP explains the prediction model, not real-world causes.
- **Wealth control is coarse.** It has only 3 levels, so a finer measure might shrink the reading difference further.

## Possible next steps
1. Use a finer wealth measure.
2. Repeat the wealth comparison for 4 groups.
3. Try 3–4 summary dimensions instead of 6, which looked more stable.
4. Try other grouping methods as a robustness check.
