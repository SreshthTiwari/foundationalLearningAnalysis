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
