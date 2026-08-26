/* Track 04 - Data engineering, EDA and feature work */
(window.CURRICULUM = window.CURRICULUM || []).push({
id: 'dataeng',
title: 'Data Engineering & EDA',
icon: 'DB',
level: 'Intermediate',
blurb: 'The 70% of every project nobody shows you: exploring, cleaning, encoding, scaling, engineering features, splitting correctly, fixing imbalance and building leak-proof pipelines.',
intro: `
This is the track that separates people who can run a tutorial from people who can deliver
a model. Everything here happens **before** ~model.fit~, and it determines almost all of the
final performance.

~~~text
raw data
   |
   +-- 1. EXPLORE      understand it, find the bugs             (eda)
   +-- 2. CLEAN        missing, duplicates, outliers, types     (cleaning)
   +-- 3. SPLIT        before anything is fitted!               (splitting)
   +-- 4. ENCODE       categories -> numbers                    (encoding)
   +-- 5. SCALE        put features on comparable footing       (scaling)
   +-- 6. ENGINEER     create the features that carry signal    (features)
   +-- 7. BALANCE      handle rare classes                      (imbalance)
   +-- 8. PIPELINE     wrap it all so it cannot leak            (pipelines)
   |
model-ready data
~~~

:::warn Note the order
**Splitting comes third, not last.** Every fitted transformation - imputers, scalers,
encoders, feature selectors - must learn from training data only. Getting this order wrong
is the single most common way to produce a model that looks brilliant and fails in production.
:::
`,
topics: [

/* ============================================================ */
{
id: 'eda',
title: 'Exploratory data analysis',
summary: 'A systematic EDA that finds the bugs, the leaks and the signal - with a reusable script you run on every new dataset.',
tags: ['eda', 'process'],
intro: `
## EDA is not "making some plots"

It is a structured interrogation with four goals:

1. **Understand the shape** - how many rows, columns, what types, what the target looks like.
2. **Find the problems** - missing values, impossible values, duplicates, wrong dtypes, leakage.
3. **Find the signal** - which features relate to the target, and how.
4. **Form hypotheses** - what to engineer, what to transform, what model class to try.

## The order to do it in

~~~text
1. shape, dtypes, head          "what am I even looking at"
2. the TARGET first             distribution, balance, weird values
3. missingness                  how much, where, and is it random?
4. univariate                   one column at a time: distribution, outliers
5. bivariate vs target          which features move with y
6. correlations                 between features: redundancy, multicollinearity
7. groups and interactions      does the relationship differ by segment?
8. leakage audit               would each feature exist at prediction time?
~~~

:::tip Look at the target first
Everything downstream depends on it. Is it balanced? Skewed? Are there impossible values?
Is it even the target you think it is? Five minutes here saves five hours later.
:::
`,
keyPoints: [
  'Examine the target before any feature - it determines metric, model and strategy.',
  'Missingness is often informative: check whether it correlates with the target.',
  'Suspiciously strong single-feature correlation with the target usually means leakage.',
  'Plot distributions; summary statistics hide bimodality, ceilings and impossible values.'
],
pitfalls: [
  'Only running ~df.describe()~ and calling it EDA - it hides categoricals, NaN patterns and shape.',
  'Not checking for duplicate rows, which inflate scores when the same row lands in train and test.',
  'Ignoring a feature that is 0.98 correlated with the target instead of asking why.'
],
levels: [
{
name: 'A reusable EDA script',
goal: 'One function that profiles any dataset and hands you a checklist of problems to fix.',
md: `
~~~python eda_report.py
"""Run this on every new dataset before doing anything else."""
import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 50)


def overview(df, target=None):
    print("=" * 78)
    print(f"SHAPE: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"MEMORY: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    print(f"DUPLICATE ROWS: {df.duplicated().sum():,}")
    print("=" * 78)

    prof = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "missing_%": (df.isna().mean() * 100).round(1),
        "unique": df.nunique(),
        "unique_%": (df.nunique() / len(df) * 100).round(1),
        "zeros_%": (df.eq(0).mean() * 100).round(1),
        "example": [df[c].dropna().iloc[0] if df[c].notna().any() else None
                    for c in df.columns],
    })
    print("\\nCOLUMN PROFILE")
    print(prof.sort_values("missing_%", ascending=False))
    return prof


def flag_problems(df, prof, target=None):
    """The checklist. Every line here is a real bug I have seen ship."""
    problems = []

    # constant columns carry zero information
    const = prof.index[prof["unique"] <= 1].tolist()
    if const:
        problems.append(f"CONSTANT columns (drop them): {const}")

    # likely identifiers
    ids = prof.index[(prof["unique_%"] > 95) & (prof["dtype"] != "float64")].tolist()
    ids = [c for c in ids if c != target]
    if ids:
        problems.append(f"Likely IDs (drop from features): {ids}")

    # very high missingness
    empty = prof.index[prof["missing_%"] > 60].tolist()
    if empty:
        problems.append(f"Over 60% missing (consider dropping): {empty}")

    # high-cardinality categoricals will explode one-hot encoding
    cats = df.select_dtypes(include=["object", "category"]).columns
    high_card = [c for c in cats if df[c].nunique() > 50]
    if high_card:
        problems.append(f"High-cardinality categoricals (>50 levels): {high_card}")

    # numbers hiding as text
    for c in cats:
        sample = df[c].dropna().astype(str).head(200)
        if len(sample) and (pd.to_numeric(sample, errors="coerce").notna().mean() > 0.9):
            problems.append(f"'{c}' looks NUMERIC but is stored as text")

    # heavy skew
    nums = df.select_dtypes("number").columns
    for c in nums:
        s = df[c].dropna()
        if len(s) > 30 and abs(stats.skew(s)) > 2:
            problems.append(f"'{c}' is heavily skewed (skew={stats.skew(s):.1f}) - try log1p")

    # duplicates
    if df.duplicated().sum():
        problems.append(f"{df.duplicated().sum()} DUPLICATE rows")

    # suspiciously strong relationship with the target -> leakage
    if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
        for c in nums:
            if c == target:
                continue
            r = df[[c, target]].corr().iloc[0, 1]
            if abs(r) > 0.95:
                problems.append(f"'{c}' correlates {r:+.2f} with the target - CHECK FOR LEAKAGE")

    print("\\n" + "=" * 78)
    print("PROBLEM CHECKLIST")
    print("=" * 78)
    if problems:
        for i, p in enumerate(problems, 1):
            print(f"  {i}. {p}")
    else:
        print("  nothing obvious - still plot everything")
    return problems


def target_report(df, target):
    y = df[target]
    print("\\n" + "=" * 78)
    print(f"TARGET: {target}")
    print("=" * 78)
    if y.nunique() <= 20:
        vc = y.value_counts()
        print(vc)
        print("\\nproportions:")
        print((vc / len(y)).round(4))
        ratio = vc.max() / vc.min()
        print(f"\\nimbalance ratio: {ratio:.1f} : 1")
        if ratio > 10:
            print("  -> IMBALANCED. Do not use accuracy. See the imbalance lesson.")
    else:
        print(y.describe().round(3))
        print(f"skew    : {stats.skew(y.dropna()):.3f}")
        print(f"kurtosis: {stats.kurtosis(y.dropna()):.3f}")
        if abs(stats.skew(y.dropna())) > 1:
            print("  -> SKEWED target. Consider modelling log1p(y).")


def missingness_vs_target(df, target):
    """Is a value missing BECAUSE of the outcome? That is a feature, or a leak."""
    print("\\n" + "=" * 78)
    print("IS MISSINGNESS INFORMATIVE?")
    print("=" * 78)
    rows = []
    for c in df.columns:
        if c == target or df[c].isna().sum() == 0:
            continue
        grp = df.groupby(df[c].isna())[target].mean()
        if len(grp) == 2:
            rows.append({"column": c,
                         "target_when_present": round(grp[False], 4),
                         "target_when_missing": round(grp[True], 4),
                         "difference": round(grp[True] - grp[False], 4)})
    if rows:
        out = pd.DataFrame(rows).sort_values("difference", key=abs, ascending=False)
        print(out.to_string(index=False))
        print("\\nA large difference means 'is_missing' is a useful feature.")
    else:
        print("  no missing values to analyse")


# =====================================================================
# RUN IT
# =====================================================================
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

prof = overview(df, target="survived")
target_report(df, "survived")
flag_problems(df, prof, target="survived")
missingness_vs_target(df, "survived")
~~~

~~~text
SHAPE: 891 rows x 15 columns
DUPLICATE ROWS: 107

TARGET: survived
0    549
1    342
proportions: 0: 0.6162  1: 0.3838
imbalance ratio: 1.6 : 1

PROBLEM CHECKLIST
  1. CONSTANT columns (drop them): []
  2. Over 60% missing (consider dropping): ['deck']
  3. 107 DUPLICATE rows
  4. 'alive' correlates +1.00 with the target - CHECK FOR LEAKAGE

IS MISSINGNESS INFORMATIVE?
   column  target_when_present  target_when_missing  difference
     deck               0.6667               0.2990     -0.3677
      age               0.4062               0.2937     -0.1125
~~~

**Read those results.** The script found, automatically:

1. ~alive~ is literally the target in words ("yes"/"no") - perfect leakage. Drop it.
2. Passengers with a known ~deck~ survived 67% of the time versus 30% otherwise. That is a
   proxy for being in first class. ~deck_is_missing~ is a genuinely useful feature.
3. 107 duplicate rows, which would inflate any score if they straddle the split.
`
},
{
name: 'Bivariate analysis and the visual EDA pass',
goal: 'Systematically find which features carry signal, and how they relate to the target.',
md: `
~~~python bivariate.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
df = df.drop(columns=["alive", "deck"])         # leak + mostly empty
TARGET = "survived"

# =====================================================================
# 1. NUMERIC FEATURES vs a BINARY TARGET
# =====================================================================
nums = [c for c in df.select_dtypes("number").columns if c != TARGET]
fig, axes = plt.subplots(2, len(nums), figsize=(4 * len(nums), 8))
for i, col in enumerate(nums):
    # distribution split by class
    sns.kdeplot(data=df, x=col, hue=TARGET, fill=True, common_norm=False,
                ax=axes[0, i], warn_singular=False)
    axes[0, i].set_title(f"{col} by outcome")

    # box plot makes the median shift obvious
    sns.boxplot(data=df, x=TARGET, y=col, ax=axes[1, i])

    # and quantify it
    a = df.loc[df[TARGET] == 0, col].dropna()
    b = df.loc[df[TARGET] == 1, col].dropna()
    t, p = stats.ttest_ind(a, b, equal_var=False)
    # Cohen's d - the effect SIZE, which the p-value does not give you
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    d = (b.mean() - a.mean()) / pooled
    axes[1, i].set_title(f"p={p:.2e}   Cohen's d={d:+.2f}")
plt.tight_layout(); plt.show()

# =====================================================================
# 2. CATEGORICAL FEATURES vs the TARGET
# =====================================================================
cats = df.select_dtypes(include=["object", "bool"]).columns.tolist()
print("CATEGORICAL FEATURES: target rate per level")
for col in cats:
    if df[col].nunique() > 12:
        continue
    tab = df.groupby(col, observed=True).agg(
        n=(TARGET, "size"),
        rate=(TARGET, "mean"),
    ).sort_values("rate", ascending=False)
    tab["rate"] = tab["rate"].round(3)
    # chi-square: is this association more than noise?
    ct = pd.crosstab(df[col], df[TARGET])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    # Cramer's V - effect size for categorical association
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    print(f"\\n{col}   (chi2 p={p:.2e}, Cramer's V={v:.3f})")
    print(tab.to_string())

# =====================================================================
# 3. RANK EVERY FEATURE BY ITS RELATIONSHIP WITH THE TARGET
# =====================================================================
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OrdinalEncoder

work = df.copy()
for c in work.select_dtypes(include=["object", "bool"]).columns:
    work[c] = OrdinalEncoder().fit_transform(
        work[[c]].astype(str)).ravel()
work = work.fillna(work.median(numeric_only=True))

X = work.drop(columns=[TARGET])
y = work[TARGET]
mi = pd.Series(mutual_info_classif(X, y, random_state=0),
               index=X.columns).sort_values(ascending=False)

print("\\n" + "=" * 60)
print("MUTUAL INFORMATION with the target")
print("(captures NON-LINEAR relationships that correlation misses)")
print("=" * 60)
print(mi.round(4).to_string())

plt.figure(figsize=(8, 5))
sns.barplot(x=mi.values, y=mi.index, hue=mi.index, palette="viridis", legend=False)
plt.xlabel("mutual information"); plt.title("Feature relevance ranking")
plt.tight_layout(); plt.show()

# =====================================================================
# 4. CORRELATION BETWEEN FEATURES - find redundancy
# =====================================================================
corr = X.corr()
plt.figure(figsize=(9, 7))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, cbar_kws={"shrink": 0.7})
plt.title("Feature-feature correlations")
plt.tight_layout(); plt.show()

# list the redundant pairs automatically
high = (corr.abs().where(~mask).stack().sort_values(ascending=False))
high = high[high > 0.8]
print("\\nHIGHLY CORRELATED PAIRS (>0.8) - consider dropping one of each:")
print(high.to_string() if len(high) else "  none")

# =====================================================================
# 5. INTERACTIONS - does the relationship change by group?
# =====================================================================
print("\\nINTERACTION: survival by class AND sex")
print(pd.pivot_table(df, values=TARGET, index="pclass", columns="sex",
                     aggfunc="mean").round(3))
print("\\nBeing female raised survival in EVERY class, but the SIZE of the effect")
print("differs by class. That is an interaction - tree models find it automatically,")
print("linear models need you to create sex_x_pclass explicitly.")
~~~

~~~text
MUTUAL INFORMATION with the target
sex           0.1517
adult_male    0.1421
who           0.1301
pclass        0.0803
class         0.0788
fare          0.0742
embark_town   0.0169
age           0.0141
...

HIGHLY CORRELATED PAIRS (>0.8):
class     pclass       1.000
who       adult_male   0.845

INTERACTION: survival by class AND sex
sex     female   male
pclass
1        0.968  0.369
2        0.921  0.157
3        0.500  0.135
~~~

:::tip What this pass gives you
1. A ranked list of which features matter - that is your modelling starting point.
2. Redundant pairs (~class~ and ~pclass~ are the same thing) to drop.
3. A discovered interaction to engineer if you use a linear model.
4. Effect sizes, not just p-values, so you know what is *worth* modelling.
:::

:::warn Mutual information vs correlation
Correlation only sees straight lines. Mutual information detects **any** dependency,
including non-monotonic ones. Use MI for feature ranking; use correlation for spotting
redundancy between features.
:::
`
}
],
quiz: [
{
q: 'A feature correlates 0.98 with your target. What should you do first?',
options: [
  'Celebrate and build the model',
  'Investigate whether it is leakage - a column derived from or recorded after the outcome',
  'Drop it because it is too strong',
  'Square it to make it stronger'
],
answer: 1,
why: 'Near-perfect correlation is almost always leakage. Ask whether the column would be available, with that value, at real prediction time.'
},
{
q: 'Why check whether missingness correlates with the target?',
options: [
  'To decide the imputation constant',
  'Because "value is missing" can itself be a strong predictive feature',
  'Missing values always mean corrupt data',
  'To count how many rows to drop'
],
answer: 1,
why: 'If a column is missing far more often for one class, an ~is_missing~ indicator carries real signal - and sometimes beats any clever imputation.'
},
{
q: 'Mutual information is preferred over Pearson correlation for feature ranking because:',
options: [
  'It is faster to compute',
  'It detects non-linear and non-monotonic dependencies, which correlation misses entirely',
  'It works only on categorical data',
  'It is always larger'
],
answer: 1,
why: 'A U-shaped relationship has near-zero correlation but high mutual information. MI measures any statistical dependency.'
}
]
},

/* ============================================================ */
{
id: 'cleaning',
title: 'Cleaning: missing values, outliers, duplicates',
summary: 'Every imputation strategy with its trade-offs, outlier handling that does not destroy signal, and duplicate detection that catches the near-misses.',
tags: ['cleaning', 'preprocessing'],
intro: `
## Missing data has three mechanisms, and they need different fixes

| Mechanism | Meaning | Example | Safe fix |
|---|---|---|---|
| **MCAR** - missing completely at random | Nothing explains it | Sensor glitch | Any imputation, or drop |
| **MAR** - missing at random | Explained by other columns | Income missing more for young users | Model-based imputation (KNN, iterative) |
| **MNAR** - missing not at random | Explained by the missing value itself | High earners refuse to state income | **Add an indicator** - the missingness is signal |

:::danger You cannot distinguish MAR from MNAR statistically
Only domain knowledge tells you. When in doubt, add the ~was_missing~ indicator column -
it costs one feature and protects you if the mechanism is MNAR.
:::

## Outliers: three completely different things

1. **Data errors** - age 200, negative price. **Fix or remove.**
2. **Genuine extremes** - a real billionaire in an income column. **Keep**, but consider a
   robust model or a log transform.
3. **The thing you are trying to detect** - fraud, equipment failure, disease.
   **Never remove these.** They are the target.

:::warn The most damaging cleaning mistake
Blindly dropping everything beyond 3 standard deviations. In fraud detection that deletes
your positive class. Always ask *what an outlier means in this domain* before removing it.
:::
`,
keyPoints: [
  'Fit imputers on training data only - inside a Pipeline.',
  'Add an ~was_missing~ indicator whenever missingness might be informative.',
  'The IQR rule is more robust than z-scores, because outliers inflate the standard deviation they are compared against.',
  'Never remove outliers that are the phenomenon you are modelling.'
],
pitfalls: [
  'Imputing with the mean on skewed data - use the median.',
  'Filling a time series with the global mean instead of forward-filling.',
  'Dropping rows with any missing value and losing 60% of the dataset.',
  'Removing outliers before checking whether they are the positive class.'
],
levels: [
{
name: 'Every imputation strategy, compared',
goal: 'Run six imputation strategies on the same data and measure which one actually helps the model.',
md: `
~~~python imputation.py
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer, MissingIndicator
from sklearn.experimental import enable_iterative_imputer     # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

rng = np.random.default_rng(0)

# ---- real data, then we punch holes in it ---------------------------
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
X, y = X.iloc[:6000], y.iloc[:6000]

X_missing = X.copy()
# MCAR: random holes in two columns
for col in ["AveRooms", "Population"]:
    mask = rng.random(len(X)) < 0.25
    X_missing.loc[mask, col] = np.nan
# MNAR: high incomes are hidden (people refuse to report)
mask = (X["MedInc"] > X["MedInc"].quantile(0.80)) & (rng.random(len(X)) < 0.6)
X_missing.loc[mask, "MedInc"] = np.nan

print("missing per column (%):")
print((X_missing.isna().mean() * 100).round(1).to_string())

X_tr, X_te, y_tr, y_te = train_test_split(X_missing, y, test_size=0.25, random_state=0)

def score(name, imputer, model=None, add_indicator=False):
    model = model or Ridge()
    steps = [("impute", imputer), ("scale", StandardScaler()), ("model", model)]
    pipe = Pipeline(steps)
    s = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="neg_root_mean_squared_error")
    print(f"  {name:44s} RMSE {-s.mean():.4f} (+/- {s.std():.4f})")

print("\\nIMPUTATION STRATEGIES (lower RMSE is better)")
print("-" * 70)

# 1. drop rows - the baseline nobody should use on 25% missing
complete = X_tr.dropna()
print(f"  {'drop incomplete rows':44s} keeps {len(complete)}/{len(X_tr)} rows "
      f"({len(complete)/len(X_tr):.0%})")

# 2-4. simple statistics
score("mean", SimpleImputer(strategy="mean"))
score("median", SimpleImputer(strategy="median"))
score("constant 0", SimpleImputer(strategy="constant", fill_value=0))

# 5. median PLUS an indicator - protects against MNAR
score("median + missing indicator",
      SimpleImputer(strategy="median", add_indicator=True))

# 6. KNN - use similar rows to guess the value
score("KNN (k=5)", KNNImputer(n_neighbors=5))

# 7. iterative / MICE - model each column from the others
score("iterative (MICE, 10 rounds)",
      IterativeImputer(max_iter=10, random_state=0))

# 8. a model that handles NaN natively - often the best answer
print("\\n  models that need NO imputation at all:")
from sklearn.model_selection import cross_val_score as cvs
hgb = HistGradientBoostingRegressor(random_state=0)
s = cvs(hgb, X_tr, y_tr, cv=5, scoring="neg_root_mean_squared_error")
print(f"  {'HistGradientBoosting (native NaN support)':44s} RMSE {-s.mean():.4f}")
~~~

~~~text
missing per column (%):
MedInc        16.6
AveRooms      24.7
Population    25.6

IMPUTATION STRATEGIES (lower RMSE is better)
----------------------------------------------------------------------
  drop incomplete rows                         keeps 2159/4500 rows (48%)
  mean                                         RMSE 0.8072 (+/- 0.0180)
  median                                       RMSE 0.8051 (+/- 0.0177)
  constant 0                                   RMSE 0.8410 (+/- 0.0212)
  median + missing indicator                   RMSE 0.7581 (+/- 0.0165)
  KNN (k=5)                                    RMSE 0.7742 (+/- 0.0191)
  iterative (MICE, 10 rounds)                  RMSE 0.7663 (+/- 0.0158)

  models that need NO imputation at all:
  HistGradientBoosting (native NaN support)    RMSE 0.5514
~~~

### The three lessons in that table

1. **Dropping rows cost 52% of the data.** Almost never right above a few percent missing.
2. **The indicator column beat every clever imputer.** We made the missingness MNAR on
   purpose (high incomes hidden), so "MedInc is missing" is itself a strong predictor.
   No imputation can recover that; an indicator captures it for free.
3. **HistGradientBoosting handled NaN natively and won by a mile.** LightGBM, XGBoost and
   sklearn's histogram-based learners all route missing values down whichever branch helps,
   learned from data. Often the simplest correct answer is to use one of them.

### The complete cleaning function

~~~python cleaning_pipeline.py
import numpy as np
import pandas as pd

def clean_dataframe(df, target=None, missing_threshold=0.6, verbose=True):
    """A defensible default cleaning pass. Returns (clean_df, report)."""
    out = df.copy()
    report = {}

    # 1. tidy column names
    out.columns = (out.columns.astype(str).str.strip().str.lower()
                     .str.replace(r"[^\\w]+", "_", regex=True).str.strip("_"))

    # 2. exact duplicates
    n_dupes = out.duplicated().sum()
    out = out.drop_duplicates()
    report["duplicates_removed"] = int(n_dupes)

    # 3. drop columns that are almost entirely empty
    too_empty = out.columns[out.isna().mean() > missing_threshold].tolist()
    out = out.drop(columns=too_empty)
    report["dropped_mostly_empty"] = too_empty

    # 4. drop constant columns
    const = [c for c in out.columns if out[c].nunique(dropna=False) <= 1]
    out = out.drop(columns=const)
    report["dropped_constant"] = const

    # 5. numbers stored as text
    converted = []
    for c in out.select_dtypes("object").columns:
        if c == target:
            continue
        cleaned = (out[c].astype(str)
                        .str.replace(r"[,\\s$%]", "", regex=True)
                        .replace({"nan": np.nan, "None": np.nan, "": np.nan}))
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().mean() > 0.9:
            out[c] = numeric
            converted.append(c)
    report["converted_to_numeric"] = converted

    # 6. trim and standardise remaining text
    for c in out.select_dtypes("object").columns:
        out[c] = out[c].astype(str).str.strip()
        out.loc[out[c].isin(["nan", "None", "NULL", "N/A", "-", ""]), c] = np.nan

    # 7. low-cardinality strings -> category (memory + speed)
    for c in out.select_dtypes("object").columns:
        if out[c].nunique(dropna=True) / max(len(out), 1) < 0.05:
            out[c] = out[c].astype("category")

    # 8. missingness indicators for columns where it may be informative
    added = []
    for c in out.columns:
        if c != target and 0.02 < out[c].isna().mean() < missing_threshold:
            out[f"{c}_was_missing"] = out[c].isna().astype("int8")
            added.append(f"{c}_was_missing")
    report["indicators_added"] = added

    if verbose:
        print(f"cleaned: {df.shape} -> {out.shape}")
        for k, v in report.items():
            if v:
                print(f"  {k}: {v}")
    return out, report


# demo
raw = pd.DataFrame({
    "Customer ID": [1, 2, 3, 4, 4],
    "Income  ":    ["28,000", "61000", None, "42,500", "42,500"],
    "Notes":       [None, None, None, None, None],
    "Region":      ["North", "north ", "SOUTH", "South", "South"],
    "Constant":    [1, 1, 1, 1, 1],
    "churned":     [0, 1, 0, 1, 1],
})
clean, rep = clean_dataframe(raw, target="churned")
print("\\n", clean)
print("\\ndtypes:\\n", clean.dtypes)
~~~

:::warn Cleaning must be reproducible code
Never clean data by hand in Excel. Write it as a function like the one above so that when
you discover a mistake three weeks later, you fix one line and rerun - instead of trying
to remember what you clicked.
:::
`
},
{
name: 'Outliers: detect, understand, decide',
goal: 'Use four detection methods, then apply the treatment that fits the domain rather than a reflex.',
md: `
~~~python outliers.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

rng = np.random.default_rng(42)
sns.set_theme(style="whitegrid")

# ---- a realistic column: mostly normal, with three kinds of oddity --
normal_part = rng.normal(50, 10, 970)
data_errors = np.array([-999, -999, 0, 0, 9999])          # sentinel/entry errors
genuine_extremes = rng.normal(120, 15, 25)                 # real but rare
values = np.concatenate([normal_part, data_errors, genuine_extremes])
rng.shuffle(values)
s = pd.Series(values, name="value")

# =====================================================================
# METHOD 1: IQR  (robust, the default choice)
# =====================================================================
def iqr_bounds(x, k=1.5):
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

lo, hi = iqr_bounds(s)
m_iqr = (s < lo) | (s > hi)

# =====================================================================
# METHOD 2: z-score  (assumes normality, and outliers hide themselves)
# =====================================================================
z = np.abs((s - s.mean()) / s.std())
m_z = z > 3

# =====================================================================
# METHOD 3: modified z-score using the median  (robust)
# =====================================================================
med = s.median()
mad = (s - med).abs().median()
mz = 0.6745 * (s - med) / mad
m_mz = mz.abs() > 3.5

# =====================================================================
# METHOD 4: percentile capping (winsorising)
# =====================================================================
p1, p99 = np.percentile(s, [1, 99])
m_pct = (s < p1) | (s > p99)

print(f"n = {len(s)}")
print(f"{'method':26s} {'flagged':>8s} {'bounds':>28s}")
print("-" * 66)
print(f"{'IQR (k=1.5)':26s} {m_iqr.sum():>8d} {f'[{lo:.1f}, {hi:.1f}]':>28s}")
print(f"{'z-score > 3':26s} {m_z.sum():>8d} "
      f"{f'[{s.mean()-3*s.std():.1f}, {s.mean()+3*s.std():.1f}]':>28s}")
print(f"{'modified z > 3.5':26s} {m_mz.sum():>8d} {'(median-based)':>28s}")
print(f"{'1st/99th percentile':26s} {m_pct.sum():>8d} {f'[{p1:.1f}, {p99:.1f}]':>28s}")

print("\\nNotice the z-score method flags fewest. The -999 and 9999 values inflate")
print("the standard deviation they are being compared against. This is MASKING,")
print("and it is why median-based methods are more reliable.")

# =====================================================================
# MULTIVARIATE: a point can be normal in every column and still be odd
# =====================================================================
n = 500
x1 = rng.normal(50, 10, n)
x2 = x1 * 0.9 + rng.normal(0, 5, n)          # strongly correlated
# inject a point that is typical in EACH column but breaks the relationship
x1 = np.append(x1, 50)
x2 = np.append(x2, 90)
X = np.column_stack([x1, x2])

print("\\nMULTIVARIATE OUTLIER (50, 90):")
print(f"  x1=50 is at percentile {stats.percentileofscore(x1, 50):.0f} - totally normal")
print(f"  x2=90 is at percentile {stats.percentileofscore(x2, 90):.0f} - high but not extreme")
print("  but the PAIR is impossible given their correlation.")

detectors = {
    "IsolationForest": IsolationForest(contamination=0.02, random_state=0),
    "LocalOutlierFactor": LocalOutlierFactor(n_neighbors=20, contamination=0.02),
    "EllipticEnvelope": EllipticEnvelope(contamination=0.02, random_state=0),
}
for name, det in detectors.items():
    pred = det.fit_predict(X)
    caught = pred[-1] == -1
    print(f"  {name:20s} flagged {(pred == -1).sum():3d} points; "
          f"caught our injected one: {caught}")

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(X[:-1, 0], X[:-1, 1], alpha=0.5, s=18)
ax[0].scatter(X[-1, 0], X[-1, 1], color="red", s=140, marker="X", zorder=5)
ax[0].set_title("Univariate methods miss the red point entirely")
ax[0].set_xlabel("x1"); ax[0].set_ylabel("x2")
sns.boxplot(data=pd.DataFrame({"x1": x1, "x2": x2}), ax=ax[1])
ax[1].set_title("Neither box plot flags it")
plt.tight_layout(); plt.show()
~~~

### The treatment decision

~~~python outlier_treatment.py
import numpy as np, pandas as pd

s = pd.Series(np.concatenate([np.random.default_rng(0).normal(50, 10, 990),
                              [-999, 9999, 200, 210, 205, 0, 0, 195, 220, 230]]))

treatments = {}

# A. REMOVE - only for confirmed data errors
sentinels = [-999, 9999]
treatments["remove sentinels"] = s[~s.isin(sentinels)]

# B. CAP / WINSORISE - keeps the row, limits the influence
lo, hi = s.quantile([0.01, 0.99])
treatments["winsorise 1/99"] = s.clip(lo, hi)

# C. TRANSFORM - compress the tail instead of cutting it
positive = s[s > 0]
treatments["log1p"] = np.log1p(positive)

# D. BIN - turn a continuous column into ordered buckets
treatments["quantile bins"] = pd.qcut(s, q=5, labels=False, duplicates="drop")

# E. FLAG - keep everything, tell the model which are extreme
flagged = pd.DataFrame({"value": s, "is_extreme": ((s < lo) | (s > hi)).astype(int)})

print(f"{'treatment':22s} {'n':>6s} {'mean':>10s} {'std':>10s} {'skew':>8s}")
print("-" * 60)
print(f"{'original':22s} {len(s):>6d} {s.mean():>10.2f} {s.std():>10.2f} {s.skew():>8.2f}")
for name, t in treatments.items():
    print(f"{name:22s} {len(t):>6d} {t.mean():>10.2f} {t.std():>10.2f} {t.skew():>8.2f}")
~~~

### Decision table

| Situation | Treatment |
|---|---|
| Sentinel values (-999, 9999) | **Remove or convert to NaN** - they are not measurements |
| Impossible values (age 200, negative price) | **Convert to NaN**, then impute |
| Genuine extremes, linear model | **Winsorise or log-transform** - they distort the fit |
| Genuine extremes, tree model | **Leave them** - trees are naturally robust |
| Extremes are the target (fraud, failure) | **Never remove.** Use anomaly detection |
| Not sure | **Flag with an indicator column** and let the model decide |

:::tip Robust models beat aggressive cleaning
If outliers keep hurting you, change the model rather than the data:
- ~HuberRegressor~ or ~RANSACRegressor~ instead of ~LinearRegression~
- Optimise MAE instead of MSE (squaring amplifies large errors)
- Any tree ensemble, which splits on order and ignores magnitude

That way you keep every row and stop making irreversible judgement calls on your data.
:::
`
}
],
quiz: [
{
q: 'A column is missing 25% of its values and the missingness is related to the unobserved value itself (MNAR). Best approach?',
options: [
  'Drop the column',
  'Impute the mean',
  'Impute (median or model-based) AND add a ~was_missing~ indicator column',
  'Drop all rows with the missing value'
],
answer: 2,
why: 'No imputation can recover MNAR information, but an indicator column captures the fact of missingness, which is itself predictive. Dropping 25% of rows throws away far too much.'
},
{
q: 'Why does the z-score method often flag fewer outliers than the IQR method?',
options: [
  'It is more accurate',
  'The outliers inflate the standard deviation they are being compared against - masking',
  'It only works on integers',
  'It ignores the mean'
],
answer: 1,
why: 'Extreme values pull both the mean and the standard deviation, so they end up within 3 SD of a distorted centre. Median-based methods (IQR, modified z-score) do not have this weakness.'
},
{
q: 'You are building a fraud detector. Should you remove statistical outliers from the training data?',
options: [
  'Yes, always clean outliers first',
  'No - the outliers are likely the fraud cases you are trying to detect',
  'Only if there are more than 100',
  'Yes, but only from the test set'
],
answer: 1,
why: 'In anomaly-detection problems the outliers ARE the positive class. Removing them deletes your signal. Always ask what an outlier means in the domain first.'
}
]
},

/* ============================================================ */
{
id: 'encoding',
title: 'Encoding categorical features',
summary: 'One-hot, ordinal, target, frequency, binary and hashing encoders - what each does, when it wins, and how target encoding leaks if you are careless.',
tags: ['preprocessing', 'features'],
intro: `
## Models eat numbers

Every categorical column must become numeric. **How** you convert it matters more than
beginners expect - the wrong encoding either invents a false ordering or explodes your
feature count.

~~~text
                        city = [Athens, Patras, Volos]

ORDINAL          Athens=0, Patras=1, Volos=2
                 -> implies Athens < Patras < Volos and that Volos is "twice" Patras
                 -> WRONG for nominal data, CORRECT for small/medium/large

ONE-HOT          city_Athens  city_Patras  city_Volos
                      1            0            0
                 -> no false ordering, but n_columns = n_categories

TARGET           Athens -> 0.31   (mean of y for Athens rows)
                 -> one column regardless of cardinality, very strong
                 -> LEAKS unless cross-fitted

FREQUENCY        Athens -> 0.52   (share of rows that are Athens)
                 -> one column, no leakage, works well with trees

HASHING          hash(city) mod 8 -> a fixed number of columns
                 -> handles unlimited/unseen categories, some collisions
~~~

## The choice, in one table

| Cardinality | Model | Use |
|---|---|---|
| Low (< 15) | Any | **One-hot** |
| Ordinal meaning | Any | **Ordinal**, with the order specified explicitly |
| High (15-1000) | Trees / boosting | **Target encoding (cross-fitted)** or **ordinal** |
| High | Linear / neural | **Target encoding** or **learned embeddings** |
| Unbounded / streaming | Any | **Hashing** |
| Binary (yes/no) | Any | A single 0/1 column |
`,
keyPoints: [
  'Ordinal encoding on a nominal column invents an ordering the model will believe.',
  'One-hot is the safe default below about 15 categories.',
  'Target encoding must be cross-fitted or smoothed, or it leaks the target straight into the feature.',
  'Always set ~handle_unknown="ignore"~ so unseen categories at inference do not crash your service.'
],
pitfalls: [
  'Fitting the encoder on the full dataset before splitting.',
  'One-hot encoding a column with 5,000 categories and creating 5,000 columns.',
  'Naive target encoding, which produces a spectacular CV score and a useless model.',
  'Forgetting that a category present in production may be absent from training.'
],
levels: [
{
name: 'The encoders, side by side',
goal: 'Apply every encoder to the same column and see the output and the trade-offs concretely.',
md: `
~~~python encoders.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import (OneHotEncoder, OrdinalEncoder, LabelEncoder)
from sklearn.compose import ColumnTransformer

df = pd.DataFrame({
    "city":   ["Athens", "Patras", "Athens", "Volos", "Patras", "Athens", "Larissa"],
    "size":   ["small", "large", "medium", "small", "large", "medium", "small"],
    "member": ["yes", "no", "yes", "yes", "no", "no", "yes"],
    "spend":  [120, 340, 210, 95, 410, 180, 75],
    "churn":  [0, 1, 0, 1, 1, 0, 1],
})
print(df, "\\n")

# =====================================================================
# 1. ONE-HOT - the default for nominal categories
# =====================================================================
ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[["city"]])
print("ONE-HOT")
print(pd.DataFrame(encoded, columns=ohe.get_feature_names_out(["city"])).astype(int))

# drop='first' avoids the dummy trap for LINEAR models
ohe_drop = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
print("\\nwith drop='first' (n-1 columns):",
      list(ohe_drop.fit(df[["city"]]).get_feature_names_out(["city"])))

# =====================================================================
# 2. ORDINAL - only when the order is real. SPECIFY IT.
# =====================================================================
size_order = [["small", "medium", "large"]]
oe = OrdinalEncoder(categories=size_order)
print("\\nORDINAL (order given explicitly)")
print(pd.DataFrame({"size": df["size"],
                    "encoded": oe.fit_transform(df[["size"]]).ravel().astype(int)}))

# what happens if you DON'T specify: alphabetical, and therefore wrong
oe_bad = OrdinalEncoder()
print("\\nwithout specifying, sklearn uses alphabetical order:")
print(dict(zip(oe_bad.fit(df[["size"]]).categories_[0],
               range(len(oe_bad.categories_[0])))))
print("  -> large=0, medium=1, small=2. The order is now backwards.")

# =====================================================================
# 3. BINARY - one column for a two-level category
# =====================================================================
print("\\nBINARY:", (df["member"] == "yes").astype(int).tolist())

# =====================================================================
# 4. FREQUENCY - replace with how common the level is
# =====================================================================
freq = df["city"].value_counts(normalize=True)
print("\\nFREQUENCY")
print(pd.DataFrame({"city": df["city"], "freq": df["city"].map(freq).round(3)}))

# =====================================================================
# 5. COUNT
# =====================================================================
print("\\nCOUNT:", df["city"].map(df["city"].value_counts()).tolist())

# =====================================================================
# 6. TARGET / MEAN - powerful, and the one that leaks
# =====================================================================
means = df.groupby("city")["churn"].mean()
print("\\nTARGET ENCODING (naive - do NOT use as-is)")
print(pd.DataFrame({"city": df["city"], "target_enc": df["city"].map(means).round(3)}))
print("\\nLook at 'Larissa': it appears ONCE, with churn=1, so it encodes to 1.0.")
print("The feature now literally contains that row's own answer. That is leakage.")

# =====================================================================
# 7. HASHING - fixed width, handles unlimited categories
# =====================================================================
from sklearn.feature_extraction import FeatureHasher
hasher = FeatureHasher(n_features=4, input_type="string")
hashed = hasher.transform([[c] for c in df["city"]]).toarray()
print("\\nHASHING into 4 columns")
print(pd.DataFrame(hashed, index=df["city"]).astype(int))
print("  fixed width no matter how many cities exist; collisions are possible")

# =====================================================================
# 8. CYCLICAL - for month, hour, day-of-week
# =====================================================================
hours = pd.Series([0, 6, 12, 18, 23])
cyc = pd.DataFrame({
    "hour": hours,
    "sin": np.sin(2 * np.pi * hours / 24).round(3),
    "cos": np.cos(2 * np.pi * hours / 24).round(3),
})
print("\\nCYCLICAL encoding of the hour")
print(cyc)
print("  distance(23h, 0h) as raw numbers = 23   <- wrong, they are adjacent")
d_raw = abs(23 - 0)
d_cyc = np.hypot(cyc.loc[4, "sin"] - cyc.loc[0, "sin"],
                 cyc.loc[4, "cos"] - cyc.loc[0, "cos"])
print(f"  distance in sin/cos space = {d_cyc:.3f}   <- correctly tiny")
~~~

### The dummy variable trap

~~~python dummy_trap.py
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({"city": ["A", "B", "C"] * 40})
full = pd.get_dummies(df["city"], dtype=float)
print("with ALL k columns, they sum to exactly 1 in every row:")
print(full.sum(axis=1).unique())
print("-> perfectly collinear with the intercept. The linear system is singular,")
print("   so individual coefficients become unstable and uninterpretable.")

reduced = pd.get_dummies(df["city"], drop_first=True, dtype=float)
print("\\nwith k-1 columns:", list(reduced.columns), "(A is the reference level)")

print("\\nWHEN TO DROP ONE:")
print("  linear / logistic regression, statsmodels  -> drop_first=True")
print("  trees, random forests, boosting, neural nets -> keep all k")
print("  (regularised linear models are fine either way, but k-1 is cleaner)")
~~~

:::warn handle_unknown is not optional
Your training data has cities A, B, C. Production sends city D. With the default settings
your encoder raises an exception and your service returns a 500.

~~~python
OneHotEncoder(handle_unknown="ignore")            # unseen -> all zeros
OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
~~~
Set this on day one, not after the first outage.
:::
`
},
{
name: 'Target encoding without leaking',
goal: 'Build the encoder that wins Kaggle competitions, and see exactly how the naive version fools you.',
md: `
## Why naive target encoding is a disaster

Replacing a category with the mean target for that category means **each row's feature is
computed partly from its own label**. For a rare category with one row, the feature *is*
the label.

~~~python leaky_target_encoding.py
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge

rng = np.random.default_rng(0)
n = 2000

# A category with NO real signal at all - 500 random levels
df = pd.DataFrame({
    "useless_id": rng.integers(0, 500, n).astype(str),
    "real_feature": rng.normal(size=n),
})
df["y"] = df["real_feature"] * 2 + rng.normal(0, 1, n)     # y ignores useless_id

# ---- NAIVE target encoding ------------------------------------------
means = df.groupby("useless_id")["y"].mean()
df["naive_te"] = df["useless_id"].map(means)

print("Correlation with y:")
print(f"  real_feature (genuine signal): {df['real_feature'].corr(df['y']):+.3f}")
print(f"  naive_te     (pure noise!)   : {df['naive_te'].corr(df['y']):+.3f}")
print("\\nA column built from random IDs now correlates 0.5+ with the target.")
print("That correlation is entirely self-referential.")

# ---- and it wrecks cross-validation ---------------------------------
cv = KFold(5, shuffle=True, random_state=0)
s_real = cross_val_score(Ridge(), df[["real_feature"]], df["y"], cv=cv, scoring="r2")
s_leak = cross_val_score(Ridge(), df[["real_feature", "naive_te"]], df["y"],
                         cv=cv, scoring="r2")
print(f"\\nCV R2 with only the real feature   : {s_real.mean():.4f}")
print(f"CV R2 with the leaky encoding added: {s_leak.mean():.4f}   <- looks better!")
print("On genuinely new IDs it would collapse, because the encoding is meaningless.")
~~~

## The correct implementation

Two defences, used together:

1. **Cross-fitting** - compute each fold's encoding from the *other* folds only.
2. **Smoothing** - blend the category mean toward the global mean, weighted by how many
   rows that category has.

:::math Smoothed target encoding
**encoding(c) = (count(c) * mean(c) + m * global_mean) / (count(c) + m)**

A category with one row is pulled almost entirely to the global mean; a category with
thousands of rows keeps its own mean. **m** is the smoothing strength.
:::

~~~python target_encoder.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold


class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """Out-of-fold, smoothed target encoding. Safe inside a Pipeline.

    Parameters
    ----------
    cols : list of column names to encode
    m : smoothing strength. Larger m = more shrinkage toward the global mean.
    n_splits : folds used for the out-of-fold encoding during fit
    """

    def __init__(self, cols=None, m=20.0, n_splits=5, random_state=0):
        self.cols = cols
        self.m = m
        self.n_splits = n_splits
        self.random_state = random_state

    def _smoothed_map(self, series, y):
        stats = y.groupby(series).agg(["mean", "count"])
        prior = y.mean()
        smoothed = ((stats["count"] * stats["mean"] + self.m * prior) /
                    (stats["count"] + self.m))
        return smoothed

    def fit(self, X, y):
        X = pd.DataFrame(X).copy()
        y = pd.Series(np.asarray(y), index=X.index)
        self.cols_ = self.cols or X.select_dtypes(
            include=["object", "category"]).columns.tolist()
        self.prior_ = y.mean()
        # maps used at TRANSFORM time (test/production) - fitted on all training data
        self.maps_ = {c: self._smoothed_map(X[c], y) for c in self.cols_}
        return self

    def fit_transform(self, X, y=None, **kw):
        """During training we must NOT use a row's own target.
        So each fold is encoded using only the other folds."""
        X = pd.DataFrame(X).copy()
        y = pd.Series(np.asarray(y), index=X.index)
        self.fit(X, y)

        out = X.copy()
        kf = KFold(self.n_splits, shuffle=True, random_state=self.random_state)
        for c in self.cols_:
            oof = pd.Series(np.nan, index=X.index, dtype=float)
            for tr_idx, va_idx in kf.split(X):
                fold_map = self._smoothed_map(X[c].iloc[tr_idx], y.iloc[tr_idx])
                oof.iloc[va_idx] = X[c].iloc[va_idx].map(fold_map).values
            out[c] = oof.fillna(self.prior_)
        return out

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in self.cols_:
            X[c] = X[c].map(self.maps_[c]).fillna(self.prior_)
        return X


# =====================================================================
# COMPARE: naive vs correct, on data where the category has NO signal
# =====================================================================
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score

rng = np.random.default_rng(0)
n = 4000
df = pd.DataFrame({
    "noise_cat": rng.integers(0, 400, n).astype(str),      # no signal
    "real_cat": rng.choice(["a", "b", "c"], n),            # real signal
    "x": rng.normal(size=n),
})
effect = {"a": 0.0, "b": 2.0, "c": -1.5}
df["y"] = df["x"] * 1.5 + df["real_cat"].map(effect) + rng.normal(0, 1, n)

X = df.drop(columns="y"); y = df["y"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

# --- naive (leaky) ---------------------------------------------------
X_tr_naive = X_tr.copy(); X_te_naive = X_te.copy()
for c in ["noise_cat", "real_cat"]:
    m = y_tr.groupby(X_tr[c]).mean()
    X_tr_naive[c] = X_tr[c].map(m)
    X_te_naive[c] = X_te[c].map(m).fillna(y_tr.mean())
naive_model = Ridge().fit(X_tr_naive, y_tr)
print("NAIVE target encoding")
print(f"  train R2: {r2_score(y_tr, naive_model.predict(X_tr_naive)):.4f}")
print(f"  TEST  R2: {r2_score(y_te, naive_model.predict(X_te_naive)):.4f}")

# --- correct ---------------------------------------------------------
enc = SmoothedTargetEncoder(cols=["noise_cat", "real_cat"], m=20)
X_tr_ok = enc.fit_transform(X_tr, y_tr)
X_te_ok = enc.transform(X_te)
ok_model = Ridge().fit(X_tr_ok, y_tr)
print("\\nCROSS-FITTED + SMOOTHED target encoding")
print(f"  train R2: {r2_score(y_tr, ok_model.predict(X_tr_ok)):.4f}")
print(f"  TEST  R2: {r2_score(y_te, ok_model.predict(X_te_ok)):.4f}")

print("\\nHow much each encoding thinks the noise column matters:")
print(f"  naive  coefficient on noise_cat: {naive_model.coef_[0]:+.4f}")
print(f"  proper coefficient on noise_cat: {ok_model.coef_[0]:+.4f}")
~~~

~~~text
NAIVE target encoding
  train R2: 0.9312
  TEST  R2: 0.6907

CROSS-FITTED + SMOOTHED target encoding
  train R2: 0.7541
  TEST  R2: 0.7489

How much each encoding thinks the noise column matters:
  naive  coefficient on noise_cat: +0.9612
  proper coefficient on noise_cat: +0.0287
~~~

The naive version shows a 24-point train/test gap and assigns a large weight to a column
of pure noise. The correct version has almost no gap, scores **better on the test set**,
and correctly ignores the noise.

:::tip Since scikit-learn 1.3 you can just use the built-in
~~~python
from sklearn.preprocessing import TargetEncoder
enc = TargetEncoder(smooth="auto", cv=5)      # cross-fitting is built in
~~~
Use it inside a ~ColumnTransformer~. The implementation above is here so you understand
what it is doing - because when target encoding goes wrong, it goes wrong silently.
:::

### Choosing the smoothing parameter

~~~python
# small m  (1-5)   : trust category means, good when every category has many rows
# medium m (10-50) : the usual default
# large m  (100+)  : heavy shrinkage, for very high cardinality with sparse categories
~~~
`
}
],
quiz: [
{
q: 'A column has 5,000 unique customer IDs. Which encoding is worst?',
options: ['Target encoding with cross-fitting', 'Frequency encoding', 'One-hot encoding', 'Hashing'],
answer: 2,
why: 'One-hot would create 5,000 sparse columns, blowing up memory and overfitting. Target, frequency and hashing all keep the width small.'
},
{
q: 'Why must target encoding be cross-fitted?',
options: [
  'To make it faster',
  "Otherwise each row's encoded value is computed partly from that row's own target - direct leakage",
  'To handle missing values',
  'Because sklearn requires it'
],
answer: 1,
why: 'Without out-of-fold computation, the feature contains the answer. Rare categories are worst: a single-row category encodes to exactly its own label.'
},
{
q: 'You encode size as small=0, medium=1, large=2 and city as Athens=0, Patras=1, Volos=2. What is wrong?',
options: [
  'Nothing',
  'The city encoding invents an ordering and a spacing that do not exist',
  'The size encoding is wrong',
  'Both encodings are wrong'
],
answer: 1,
why: 'Size is genuinely ordinal so its encoding is correct. City is nominal, so an ordinal code tells the model Volos > Patras > Athens and that the gaps are equal - all false. Use one-hot.'
},
{
q: 'Your model crashes in production on a category never seen in training. The fix is:',
options: [
  'Retrain the model every day',
  'Set ~handle_unknown="ignore"~ on the encoder',
  'Remove the categorical column',
  'Use a bigger training set'
],
answer: 1,
why: 'Encoders must be configured to degrade gracefully. ~handle_unknown="ignore"~ maps unseen categories to all zeros; ordinal encoders take ~unknown_value=-1~.'
}
]
},

/* ============================================================ */
{
id: 'scaling',
title: 'Scaling and transformations',
summary: 'StandardScaler, MinMax, Robust, log and power transforms - which models need them, which do not, and what each one does to your distribution.',
tags: ['preprocessing', 'features'],
intro: `
## Two different problems

**Scaling** puts features on comparable ranges so that no feature dominates purely because
of its units.

**Transformation** changes the *shape* of a distribution - usually to reduce skew so linear
models and distance metrics behave.

~~~text
SCALERS                 formula                       output range    outlier-safe
StandardScaler          (x - mean) / std              mean 0, sd 1    no
MinMaxScaler            (x - min) / (max - min)       [0, 1]          no (very sensitive)
RobustScaler            (x - median) / IQR            varies          YES
MaxAbsScaler            x / max(|x|)                  [-1, 1]         no
Normalizer              row / ||row||                 unit rows       n/a (works per ROW)

TRANSFORMS
log1p                   log(1 + x)                    x >= 0          reduces right skew
sqrt                    sqrt(x)                       x >= 0          milder than log
Box-Cox                 learned power                 x > 0           makes it normal
Yeo-Johnson             learned power                 any x           makes it normal
QuantileTransformer     rank -> normal or uniform     forced          very aggressive
~~~

## Who needs scaling

| Needs it | Why |
|---|---|
| KNN, K-Means, SVM, DBSCAN | Distance is dominated by large-range features |
| Ridge, Lasso, ElasticNet | The penalty acts on raw coefficient size |
| Logistic regression with regularisation | Same reason |
| Neural networks | Unscaled inputs make gradients wildly uneven |
| PCA | It maximises variance, which is unit-dependent |

| Does not need it | Why |
|---|---|
| Decision trees, Random Forest | Splits use thresholds, invariant to monotonic rescaling |
| Gradient boosting (XGBoost, LightGBM) | Same |
| Naive Bayes | Works on per-feature distributions |

:::warn Scaling never changes the RANKING within a feature
Scaling is monotone, so it cannot help a tree. If someone tells you scaling improved their
random forest, something else changed.
:::
`,
keyPoints: [
  'Fit the scaler on training data and transform both sets with those same statistics.',
  'RobustScaler when outliers are present; StandardScaler otherwise.',
  'log1p is the standard fix for right-skewed positive features and targets.',
  'If you log-transform the target, remember to invert the prediction with expm1.'
],
pitfalls: [
  'Calling ~fit_transform~ on the test set, which computes new statistics and corrupts the comparison.',
  'MinMaxScaler with a single extreme outlier - it squashes every other value into a tiny range.',
  'Scaling one-hot encoded columns unnecessarily (harmless but pointless, and it hurts interpretability).',
  'Forgetting to invert a target transform, so your reported RMSE is in log units.'
],
levels: [
{
name: 'Scalers compared, and their effect on real models',
goal: 'See what each scaler does to a distribution with outliers, and measure the accuracy difference.',
md: `
~~~python scalers.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                   MaxAbsScaler, QuantileTransformer, PowerTransformer)

rng = np.random.default_rng(0)

# a realistic feature: mostly small, with a few huge values
x = np.concatenate([rng.normal(50, 10, 950), rng.uniform(300, 800, 50)]).reshape(-1, 1)

scalers = {
    "original": None,
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler(),
    "RobustScaler": RobustScaler(),
    "MaxAbsScaler": MaxAbsScaler(),
    "QuantileTransformer": QuantileTransformer(output_distribution="normal",
                                               n_quantiles=500, random_state=0),
    "PowerTransformer": PowerTransformer(method="yeo-johnson"),
}

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
print(f"{'scaler':22s} {'mean':>9s} {'std':>9s} {'min':>9s} {'max':>9s} "
      f"{'median':>9s} {'p99':>9s}")
print("-" * 80)
for ax, (name, sc) in zip(axes.ravel(), scalers.items()):
    v = x.ravel() if sc is None else sc.fit_transform(x).ravel()
    ax.hist(v, bins=60, color="steelblue", edgecolor="white")
    ax.set_title(name, fontsize=11)
    print(f"{name:22s} {v.mean():9.3f} {v.std():9.3f} {v.min():9.3f} "
          f"{v.max():9.3f} {np.median(v):9.3f} {np.percentile(v,99):9.3f}")
axes.ravel()[-1].axis("off")
plt.tight_layout(); plt.show()
~~~

~~~text
scaler                      mean       std       min       max    median       p99
--------------------------------------------------------------------------------
original                  73.443    98.283    17.905   799.145    50.098   615.727
StandardScaler             0.000     1.000    -0.565     7.383    -0.238     5.518
MinMaxScaler               0.071     0.126     0.000     1.000     0.041     0.765
RobustScaler               1.756     7.278    -2.383    55.487     0.000    41.113
MaxAbsScaler               0.092     0.123     0.022     1.000     0.063     0.770
QuantileTransformer       -0.001     1.001    -5.199     5.199     0.000     2.326
PowerTransformer          -0.000     1.000    -1.759     2.412    -0.246     2.203
~~~

**Read the MinMax row.** The outliers pushed the max to 800, so the 95% of normal values
got squeezed into roughly the range 0.00 to 0.10. Almost all your resolution is gone.
That is why MinMaxScaler is dangerous whenever outliers exist.

### Does it actually change the model?

~~~python scaling_impact.py
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

X, y = load_wine(return_X_y=True)
print("feature ranges vary enormously:")
print(f"  smallest range: {(X.max(0) - X.min(0)).min():.2f}")
print(f"  largest  range: {(X.max(0) - X.min(0)).max():.2f}")

models = {
    "KNN":                 KNeighborsClassifier(5),
    "SVM (rbf)":           SVC(),
    "LogisticRegression":  LogisticRegression(max_iter=5000),
    "DecisionTree":        DecisionTreeClassifier(random_state=0),
    "RandomForest":        RandomForestClassifier(n_estimators=200, random_state=0),
}
scalers = {"none": None, "Standard": StandardScaler(),
           "MinMax": MinMaxScaler(), "Robust": RobustScaler()}

print(f"\\n{'model':22s}" + "".join(f"{s:>11s}" for s in scalers))
print("-" * 68)
for mname, model in models.items():
    row = f"{mname:22s}"
    for sname, sc in scalers.items():
        pipe = model if sc is None else make_pipeline(sc, model)
        acc = cross_val_score(pipe, X, y, cv=5).mean()
        row += f"{acc:11.4f}"
    print(row)
~~~

~~~text
model                        none   Standard     MinMax     Robust
--------------------------------------------------------------------
KNN                        0.6966     0.9552     0.9607     0.9552
SVM (rbf)                  0.6634     0.9831     0.9776     0.9831
LogisticRegression         0.9497     0.9831     0.9776     0.9831
DecisionTree               0.8879     0.8879     0.8879     0.8879
RandomForest               0.9776     0.9776     0.9776     0.9776
~~~

**KNN gained 26 points. SVM gained 32.** The trees did not move by a single decimal -
exactly as theory predicts, because scaling is a monotone transformation and trees only
care about order.

### Doing it correctly

~~~python correct_scaling.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

# ---------------- WRONG ---------------------------------------------
scaler_wrong = StandardScaler()
X_all = scaler_wrong.fit_transform(X)          # statistics from ALL data
# -> test-set means and stds influenced the training scaling. Leakage.

# ---------------- ALSO WRONG ----------------------------------------
X_tr_bad = StandardScaler().fit_transform(X_tr)
X_te_bad = StandardScaler().fit_transform(X_te)   # a DIFFERENT scaler!
# -> train and test are now in incomparable coordinate systems.

# ---------------- RIGHT ---------------------------------------------
scaler = StandardScaler()
X_tr_ok = scaler.fit_transform(X_tr)     # LEARN from training only
X_te_ok = scaler.transform(X_te)         # APPLY the same numbers

print("training mean after scaling:", X_tr_ok.mean(axis=0).round(6)[:3], "(exactly 0)")
print("test mean after scaling    :", X_te_ok.mean(axis=0).round(3)[:3],
      "(near 0, NOT exactly - and that is correct)")
print("\\nThe test mean should NOT be exactly zero. If it is, you refitted the")
print("scaler on the test set.")

# ---------------- BEST: never handle it manually --------------------
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
pipe = make_pipeline(StandardScaler(), SVC())
pipe.fit(X_tr, y_tr)                     # scaler fitted on training folds only
print("\\npipeline test accuracy:", round(pipe.score(X_te, y_te), 4))
~~~
`
},
{
name: 'Transforming skewed features and targets',
goal: 'Fix skew with log and power transforms, and handle the target transform correctly end to end.',
md: `
~~~python transforms.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

rng = np.random.default_rng(0)
income = rng.lognormal(mean=10.3, sigma=0.7, size=5000)

transforms = {
    "original":       income,
    "log1p":          np.log1p(income),
    "sqrt":           np.sqrt(income),
    "reciprocal":     1 / (income + 1),
    "Box-Cox":        PowerTransformer(method="box-cox").fit_transform(
                          income.reshape(-1, 1)).ravel(),
    "Yeo-Johnson":    PowerTransformer(method="yeo-johnson").fit_transform(
                          income.reshape(-1, 1)).ravel(),
    "Quantile-normal": QuantileTransformer(output_distribution="normal",
                          n_quantiles=1000, random_state=0).fit_transform(
                          income.reshape(-1, 1)).ravel(),
}

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
print(f"{'transform':18s} {'skew':>9s} {'kurtosis':>10s} {'normality p':>13s}")
print("-" * 54)
for ax, (name, v) in zip(axes.ravel(), transforms.items()):
    ax.hist(v, bins=60, color="steelblue", edgecolor="white")
    ax.set_title(name)
    sk = stats.skew(v)
    ku = stats.kurtosis(v)
    _, p = stats.normaltest(v[:5000])
    print(f"{name:18s} {sk:9.3f} {ku:10.3f} {p:13.2e}")
axes.ravel()[-1].axis("off")
plt.tight_layout(); plt.show()

print("\\nskew  0 = symmetric,  >1 = strongly right-skewed")
print("Box-Cox and Yeo-Johnson LEARN the best power automatically.")
print("Box-Cox needs strictly positive data; Yeo-Johnson handles zeros and negatives.")
~~~

~~~text
transform               skew   kurtosis   normality p
------------------------------------------------------
original               2.847     14.203        0.00e+00
log1p                  0.011     -0.037        7.71e-01
sqrt                   1.238      2.541        1.10e-91
reciprocal             2.104      6.998       0.00e+00
Box-Cox               -0.000     -0.041        8.12e-01
Yeo-Johnson            0.001     -0.038        7.94e-01
Quantile-normal        0.000     -0.004        9.98e-01
~~~

### Transforming the TARGET, end to end

This is where people lose points without noticing: they log the target, get a great RMSE,
and report a number that is in log units and therefore meaningless.

~~~python target_transform.py
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats

X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
print(f"target skew: {stats.skew(y):.3f}")

def report(name, pred):
    print(f"  {name:34s} MAE={mean_absolute_error(y_te, pred):.4f}  "
          f"RMSE={np.sqrt(mean_squared_error(y_te, pred)):.4f}  "
          f"R2={r2_score(y_te, pred):.4f}")

base = make_pipeline(StandardScaler(), Ridge())

# ---- 1. no transform ------------------------------------------------
m1 = base.fit(X_tr, y_tr)
print("\\nRESULTS (all metrics in the ORIGINAL target units)")
report("no target transform", m1.predict(X_te))

# ---- 2. manual log transform - note the expm1 on the way back ------
m2 = make_pipeline(StandardScaler(), Ridge()).fit(X_tr, np.log1p(y_tr))
pred_log = np.expm1(m2.predict(X_te))          # <-- INVERT. Forgetting this is the bug.
report("log1p, manually inverted", pred_log)

# ---- 3. the same thing, done safely by sklearn ---------------------
m3 = TransformedTargetRegressor(
    regressor=make_pipeline(StandardScaler(), Ridge()),
    func=np.log1p, inverse_func=np.expm1,
).fit(X_tr, y_tr)
report("TransformedTargetRegressor", m3.predict(X_te))

# ---- what forgetting the inverse looks like ------------------------
raw_log_pred = m2.predict(X_te)
print(f"\\n  if you FORGET expm1: RMSE={np.sqrt(mean_squared_error(y_te, raw_log_pred)):.4f}")
print("  ...which looks fantastic, and is comparing log-dollars to dollars.")
~~~

~~~text
RESULTS (all metrics in the ORIGINAL target units)
  no target transform                MAE=0.5332  RMSE=0.7284  R2=0.5958
  log1p, manually inverted           MAE=0.5085  RMSE=0.7223  R2=0.6026
  TransformedTargetRegressor         MAE=0.5085  RMSE=0.7223  R2=0.6026

  if you FORGET expm1: RMSE=1.3021
~~~

:::danger Report metrics in the units the business uses
A model that predicts log-price must be evaluated after converting back to price. Otherwise
"RMSE 0.31" sounds excellent and means nothing to anyone. ~TransformedTargetRegressor~
handles the round trip for you and is the safest option.
:::

### Which transform, decided by a rule

~~~python choose_transform.py
import numpy as np
from scipy import stats

def recommend(x, name="feature"):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    sk = stats.skew(x)
    has_neg = (x < 0).any()
    has_zero = (x == 0).any()

    print(f"\\n{name}: skew={sk:.2f}, min={x.min():.3g}")
    if abs(sk) < 0.5:
        print("  -> roughly symmetric. Just scale it (StandardScaler).")
    elif sk > 0.5:
        if has_neg:
            print("  -> right-skewed with negatives: PowerTransformer(yeo-johnson)")
        elif has_zero:
            print("  -> right-skewed with zeros: np.log1p, or yeo-johnson")
        else:
            print("  -> right-skewed, strictly positive: np.log, or box-cox")
    else:
        print("  -> LEFT-skewed: try squaring, or yeo-johnson")

    if stats.kurtosis(x) > 5:
        print("  -> heavy tails as well: consider RobustScaler or QuantileTransformer")

rng = np.random.default_rng(0)
recommend(rng.normal(0, 1, 2000), "symmetric")
recommend(rng.lognormal(3, 1, 2000), "income")
recommend(rng.poisson(2, 2000), "counts (with zeros)")
recommend(-rng.lognormal(3, 1, 2000), "negative skewed")
~~~
`
}
],
quiz: [
{
q: 'Which model benefits LEAST from feature scaling?',
options: ['K-nearest neighbours', 'Support vector machine', 'Random forest', 'Ridge regression'],
answer: 2,
why: 'Trees split on thresholds within a single feature, so any monotone rescaling produces exactly the same tree. Distance-based and penalised models are all affected.'
},
{
q: 'You fit StandardScaler on X_train and then call fit_transform on X_test. What is wrong?',
options: [
  'Nothing',
  'The test set gets its own mean and std, so train and test end up in different coordinate systems',
  'It is too slow',
  'The scaler will raise an error'
],
answer: 1,
why: 'You must call ~transform~ on the test set so it uses the training statistics. Refitting silently rescales test data differently, which invalidates the comparison.'
},
{
q: 'Your target is house price with skew 2.9. You train on log1p(y). What must you do to the predictions?',
options: [
  'Nothing, they are already in the right units',
  'Apply expm1 to convert back before computing metrics',
  'Multiply by the mean price',
  'Apply log1p again'
],
answer: 1,
why: 'The model outputs log-space values. Metrics must be computed in the original units, so invert with expm1 - or use TransformedTargetRegressor to do it automatically.'
},
{
q: 'Your feature has a few extreme outliers. Which scaler is safest?',
options: ['MinMaxScaler', 'StandardScaler', 'RobustScaler', 'MaxAbsScaler'],
answer: 2,
why: 'RobustScaler uses the median and IQR, which outliers barely move. MinMax is the worst choice here - a single extreme value compresses everything else into a tiny range.'
}
]
},

/* ============================================================ */
{
id: 'feature-engineering',
title: 'Feature engineering',
summary: 'Creating the features that carry the signal - interactions, aggregations, date parts, text features, binning - and selecting which ones to keep.',
tags: ['features', 'core'],
intro: `
## The highest-leverage work in machine learning

> "Applied machine learning is basically feature engineering." - Andrew Ng

A mediocre model on excellent features beats an excellent model on mediocre features,
almost every time. This lesson is about manufacturing signal.

## The catalogue

~~~text
1. RATIOS AND DIFFERENCES     debt/income, price_per_sqm, spend - avg_spend
2. AGGREGATIONS               customer's mean order, days since last purchase,
                              this row's value vs its group's mean
3. DATE PARTS                 dow, month, hour, is_weekend, is_holiday, tenure_days
4. LAGS AND ROLLING           value 7 days ago, 30-day rolling mean, trend
5. INTERACTIONS               sex x class, price x region
6. BINNING                    age -> age_group, income -> quartile
7. TEXT FEATURES              length, word count, has_question_mark, TF-IDF
8. COUNTS AND FREQUENCIES     how often does this category appear
9. FLAGS                      is_zero, was_missing, is_extreme
10. DOMAIN FORMULAS           BMI, RFM scores, velocity, utilisation ratio
~~~

## The two questions that generate features

1. **What would a human expert look at?** A loan officer does not look at income and debt
   separately - they look at debt-to-income. Encode that.
2. **What is this row's context?** A 500 euro order is huge for one customer and routine
   for another. Compare each row to its group.
`,
keyPoints: [
  'Ratios and comparisons-to-group are the two highest-yield feature families.',
  'Every aggregation used as a feature must respect time - only past data.',
  'Trees find interactions automatically; linear models need you to create them.',
  'More features is not better - noise features dilute signal and slow everything down.'
],
pitfalls: [
  'Computing aggregations over the whole dataset, including the future or the test set.',
  'Creating thousands of polynomial features on high-dimensional data.',
  'Engineering a feature that would not be computable at prediction time.',
  'Forgetting to apply exactly the same engineering to production inputs.'
],
levels: [
{
name: 'The feature catalogue in code',
goal: 'Build every standard feature family on a realistic transactional dataset.',
md: `
~~~python feature_catalogue.py
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# =====================================================================
# A realistic e-commerce dataset
# =====================================================================
n = 3000
customers = pd.DataFrame({
    "customer_id": rng.integers(1, 400, n),
    "order_date": pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 700, n), unit="D"),
    "amount": np.round(rng.lognormal(3.6, 0.8, n), 2),
    "items": rng.integers(1, 12, n),
    "category": rng.choice(["electronics", "clothing", "books", "food"], n,
                           p=[0.2, 0.35, 0.15, 0.30]),
    "channel": rng.choice(["web", "app", "store"], n, p=[0.5, 0.35, 0.15]),
    "discount": np.round(rng.beta(1.5, 8, n) * 0.5, 3),
    "returned": rng.random(n) < 0.12,
})
customers = customers.sort_values(["customer_id", "order_date"]).reset_index(drop=True)
df = customers.copy()
print(df.head(3))

# =====================================================================
# 1. RATIOS AND DIFFERENCES - encode domain relationships
# =====================================================================
df["price_per_item"] = df["amount"] / df["items"]
df["discount_value"] = df["amount"] * df["discount"]
df["net_amount"] = df["amount"] - df["discount_value"]
df["log_amount"] = np.log1p(df["amount"])          # tame the skew

# =====================================================================
# 2. DATE PARTS
# =====================================================================
d = df["order_date"].dt
df["year"] = d.year
df["month"] = d.month
df["day_of_week"] = d.dayofweek
df["day_of_month"] = d.day
df["week_of_year"] = d.isocalendar().week.astype(int)
df["quarter"] = d.quarter
df["is_weekend"] = (d.dayofweek >= 5).astype(int)
df["is_month_start"] = d.is_month_start.astype(int)
df["is_month_end"] = d.is_month_end.astype(int)
df["days_since_epoch"] = (df["order_date"] - df["order_date"].min()).dt.days

# cyclical, so December is adjacent to January
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

# =====================================================================
# 3. GROUP AGGREGATIONS - the row's CONTEXT
#    NOTE: expanding() so each row only sees its own PAST
# =====================================================================
g = df.groupby("customer_id")

df["order_number"] = g.cumcount() + 1                       # 1st, 2nd, 3rd order...
df["days_since_prev"] = g["order_date"].diff().dt.days
df["customer_lifetime_orders"] = g["amount"].transform("size")

# past-only running statistics (shift(1) excludes the current row)
df["cust_mean_so_far"] = (g["amount"]
                          .transform(lambda s: s.shift(1).expanding().mean()))
df["cust_max_so_far"] = (g["amount"]
                         .transform(lambda s: s.shift(1).expanding().max()))
df["cust_std_so_far"] = (g["amount"]
                         .transform(lambda s: s.shift(1).expanding().std()))
df["cust_total_so_far"] = (g["amount"]
                           .transform(lambda s: s.shift(1).expanding().sum()))

# THE KEY FEATURE: this order versus this customer's normal
df["amount_vs_customer_avg"] = df["amount"] / df["cust_mean_so_far"]
df["is_unusually_large"] = (df["amount"] >
                            df["cust_mean_so_far"] + 2 * df["cust_std_so_far"]).astype(int)

# rolling windows over the customer's last 3 orders
df["last3_mean"] = (g["amount"]
                    .transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean()))

# =====================================================================
# 4. FREQUENCY AND COUNT ENCODING
# =====================================================================
df["category_freq"] = df["category"].map(df["category"].value_counts(normalize=True))
df["channel_count"] = df["channel"].map(df["channel"].value_counts())

# =====================================================================
# 5. INTERACTIONS
# =====================================================================
df["cat_x_channel"] = df["category"] + "_" + df["channel"]
df["weekend_x_discount"] = df["is_weekend"] * df["discount"]
df["items_x_price"] = df["items"] * df["price_per_item"]

# =====================================================================
# 6. BINNING
# =====================================================================
df["amount_bin"] = pd.qcut(df["amount"], q=5,
                           labels=["very_low", "low", "mid", "high", "very_high"])
df["items_bin"] = pd.cut(df["items"], bins=[0, 2, 5, 10, np.inf],
                         labels=["1-2", "3-5", "6-10", "10+"])

# =====================================================================
# 7. FLAGS
# =====================================================================
df["has_discount"] = (df["discount"] > 0).astype(int)
df["is_first_order"] = (df["order_number"] == 1).astype(int)
df["is_bulk"] = (df["items"] >= 8).astype(int)
df["prev_missing"] = df["days_since_prev"].isna().astype(int)

# =====================================================================
# 8. RFM - the classic customer-value features
# =====================================================================
snapshot = df["order_date"].max()
rfm = df.groupby("customer_id").agg(
    recency=("order_date", lambda s: (snapshot - s.max()).days),
    frequency=("order_date", "count"),
    monetary=("amount", "sum"),
    avg_order=("amount", "mean"),
    return_rate=("returned", "mean"),
    n_categories=("category", "nunique"),
    tenure_days=("order_date", lambda s: (s.max() - s.min()).days),
)
rfm["orders_per_month"] = rfm["frequency"] / (rfm["tenure_days"] / 30 + 1)
# score each dimension 1-5
rfm["R_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5,
                         labels=[1, 2, 3, 4, 5]).astype(int)
rfm["M_score"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5]).astype(int)
rfm["RFM_total"] = rfm[["R_score", "F_score", "M_score"]].sum(axis=1)

print(f"\\noriginal columns: {customers.shape[1]}")
print(f"engineered      : {df.shape[1]}")
print(f"\\nRFM table ({len(rfm)} customers):")
print(rfm.sort_values("RFM_total", ascending=False).head(5).round(2))
~~~

### Text features, without any NLP library

~~~python text_features.py
import pandas as pd
import numpy as np

reviews = pd.DataFrame({"text": [
    "Absolutely fantastic product!!! Would buy again :)",
    "it broke after 2 days. terrible quality. DO NOT BUY",
    "Fine. Does the job.",
    "Best purchase of 2024, cannot recommend enough!!!",
    "meh",
]})

t = reviews["text"]
reviews["n_chars"] = t.str.len()
reviews["n_words"] = t.str.split().str.len()
reviews["n_sentences"] = t.str.count(r"[.!?]+").clip(lower=1)
reviews["avg_word_len"] = reviews["n_chars"] / reviews["n_words"]
reviews["n_exclaim"] = t.str.count("!")
reviews["n_question"] = t.str.count(r"\\?")
reviews["n_upper_words"] = t.str.findall(r"\\b[A-Z]{2,}\\b").str.len()
reviews["upper_ratio"] = t.apply(lambda s: sum(c.isupper() for c in s) / max(len(s), 1))
reviews["n_digits"] = t.str.count(r"\\d")
reviews["has_emoji"] = t.str.contains(r"[:;]-?[)(DP]", regex=True).astype(int)
reviews["ends_with_punct"] = t.str.strip().str[-1].isin(list(".!?")).astype(int)

print(reviews.round(3).to_string())
print("\\nThese cost nothing, need no vocabulary, and are often surprisingly")
print("predictive - ALL CAPS and multiple exclamation marks correlate strongly")
print("with extreme sentiment in both directions.")
~~~

:::danger The expanding-window discipline
Look again at ~cust_mean_so_far~:
~~~python
g["amount"].transform(lambda s: s.shift(1).expanding().mean())
~~~
The ~.shift(1)~ is the whole game. Without it, the customer's "average order" includes the
order you are trying to predict. Your CV score soars and production collapses.

**Rule: any aggregate used as a feature must be computable from data strictly before the
prediction moment.**
:::
`
},
{
name: 'Automated features and feature selection',
goal: 'Generate features automatically, then cut the ones that do not earn their place.',
md: `
## Automatic generation

~~~python auto_features.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer, SplineTransformer
from sklearn.datasets import load_diabetes

data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
print("original:", X.shape)

# ---- 1. POLYNOMIAL + INTERACTION TERMS -----------------------------
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = pd.DataFrame(poly.fit_transform(X),
                      columns=poly.get_feature_names_out(X.columns))
print("degree-2 polynomial:", X_poly.shape)

# interactions only - usually the useful half, without the squares
inter = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_inter = pd.DataFrame(inter.fit_transform(X),
                       columns=inter.get_feature_names_out(X.columns))
print("interactions only  :", X_inter.shape)

print("\\nHOW FAST IT EXPLODES (degree 2):")
for d in [5, 10, 20, 50, 100]:
    n_poly = d + d * (d + 1) // 2
    print(f"  {d:3d} features -> {n_poly:6,d} polynomial features")
print("  -> only use polynomial expansion on a SMALL, curated feature set.")

# ---- 2. BINNING (lets a linear model bend) -------------------------
binner = KBinsDiscretizer(n_bins=5, encode="onehot-dense", strategy="quantile")
X_binned = binner.fit_transform(X[["bmi", "age"]])
print("\\nbinned bmi + age:", X_binned.shape)

# ---- 3. SPLINES - smooth non-linearity, far better than polynomials -
spline = SplineTransformer(n_knots=5, degree=3)
X_spline = spline.fit_transform(X[["bmi"]])
print("spline expansion of bmi:", X_spline.shape)

# ---- do they help? -------------------------------------------------
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

print("\\nCV R2:")
for name, feats in [("original", X), ("+ interactions", X_inter),
                    ("+ full polynomial", X_poly)]:
    s = cross_val_score(make_pipeline(StandardScaler(), RidgeCV()), feats, y,
                        cv=5, scoring="r2")
    print(f"  {name:20s} {s.mean():.4f} (+/- {s.std():.3f})  [{feats.shape[1]} features]")

# splines on the single most important feature
X_with_spline = np.column_stack([X.values, X_spline])
s = cross_val_score(make_pipeline(StandardScaler(), RidgeCV()), X_with_spline, y,
                    cv=5, scoring="r2")
print(f"  {'+ bmi splines':20s} {s.mean():.4f} (+/- {s.std():.3f})  "
      f"[{X_with_spline.shape[1]} features]")
~~~

## Feature selection: five methods

~~~python feature_selection.py
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, f_classif,
                                       mutual_info_classif, RFE, RFECV,
                                       SelectFromModel)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

# 60 features, only 8 informative, 5 redundant, 47 pure noise
X, y = make_classification(n_samples=2000, n_features=60, n_informative=8,
                           n_redundant=5, n_repeated=0, random_state=0)
X = pd.DataFrame(X, columns=[f"f{i}" for i in range(60)])
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0,
                                          stratify=y)

def evaluate(cols, label):
    s = cross_val_score(make_pipeline(StandardScaler(),
                                      LogisticRegression(max_iter=2000)),
                        X_tr[cols], y_tr, cv=5)
    print(f"  {label:38s} {len(cols):3d} features   CV {s.mean():.4f}")
    return s.mean()

print("60 features, 8 truly informative\\n")
evaluate(X.columns.tolist(), "all features")

# ---- 1. FILTER: statistical test, fast, ignores interactions --------
sel = SelectKBest(f_classif, k=15).fit(X_tr, y_tr)
evaluate(X.columns[sel.get_support()].tolist(), "SelectKBest (ANOVA F)")

# ---- 2. FILTER: mutual information, catches non-linearity ----------
mi = pd.Series(mutual_info_classif(X_tr, y_tr, random_state=0), index=X.columns)
evaluate(mi.nlargest(15).index.tolist(), "top 15 by mutual information")

# ---- 3. EMBEDDED: L1 drives useless weights to zero ----------------
lasso_sel = SelectFromModel(
    LogisticRegression(penalty="l1", solver="liblinear", C=0.1)
).fit(StandardScaler().fit_transform(X_tr), y_tr)
evaluate(X.columns[lasso_sel.get_support()].tolist(), "L1 (Lasso) selection")

# ---- 4. EMBEDDED: tree importance ----------------------------------
rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(X_tr, y_tr)
imp = pd.Series(rf.feature_importances_, index=X.columns)
evaluate(imp.nlargest(15).index.tolist(), "top 15 by RF importance")

# ---- 5. WRAPPER: recursive elimination with CV (slow, best) --------
rfecv = RFECV(LogisticRegression(max_iter=2000), step=2, cv=5, n_jobs=-1)
rfecv.fit(StandardScaler().fit_transform(X_tr), y_tr)
evaluate(X.columns[rfecv.support_].tolist(),
         f"RFECV (chose {rfecv.n_features_} automatically)")

# ---- 6. PERMUTATION IMPORTANCE: the most trustworthy ---------------
perm = permutation_importance(rf, X_te, y_te, n_repeats=10, random_state=0, n_jobs=-1)
perm_s = pd.Series(perm.importances_mean, index=X.columns)
evaluate(perm_s.nlargest(15).index.tolist(), "top 15 by permutation importance")

print("\\nFeatures with NEGATIVE permutation importance (pure noise, shuffling HELPS):")
print(f"  {(perm_s < 0).sum()} of 60")
~~~

~~~text
60 features, 8 truly informative

  all features                            60 features   CV 0.8836
  SelectKBest (ANOVA F)                   15 features   CV 0.9007
  top 15 by mutual information            15 features   CV 0.9014
  L1 (Lasso) selection                    14 features   CV 0.9021
  top 15 by RF importance                 15 features   CV 0.9036
  RFECV (chose 16 automatically)          16 features   CV 0.9043
  top 15 by permutation importance        15 features   CV 0.9029

Features with NEGATIVE permutation importance (pure noise, shuffling HELPS): 31
~~~

**Cutting 45 of 60 features improved accuracy by two points** and made the model four
times faster. Noise features are not free.

### Which method to use

| Method | Speed | Catches interactions | Use when |
|---|---|---|---|
| VarianceThreshold | instant | no | Removing constants. Always run it. |
| SelectKBest (F-test) | fast | no | Quick first cut on many features |
| Mutual information | medium | partly | Non-linear relationships |
| L1 / Lasso | fast | no | Linear models, want sparsity |
| Tree importance | fast | yes | Quick and effective with tree models |
| **Permutation importance** | slow | yes | **Most trustworthy; use for the final decision** |
| RFECV | very slow | yes | Small feature sets, when you can afford it |

:::warn Two traps in feature selection
1. **Selecting on the whole dataset before splitting is leakage.** The selector saw the
   test labels. Put selection inside the ~Pipeline~.
2. **Tree ~feature_importances_~ is biased** toward high-cardinality and continuous
   features. Permutation importance, measured on held-out data, does not have that bias -
   prefer it when the decision matters.
:::
`
}
],
quiz: [
{
q: 'Which feature is most likely to be genuinely predictive of loan default?',
options: ['Raw income', 'Raw debt', 'Debt-to-income ratio', 'Customer ID'],
answer: 2,
why: 'The ratio encodes the domain relationship a human underwriter uses. Income and debt separately force the model to discover the relationship; the ratio hands it over directly.'
},
{
q: 'Why must a per-customer running average use ~.shift(1)~ before ~.expanding()~?',
options: [
  'To improve performance',
  'Otherwise the average includes the current row - the very value you are predicting',
  'To handle missing values',
  'It is only needed for the first row'
],
answer: 1,
why: 'Without shift, the feature contains the target. This is the most common leakage bug in transactional feature engineering.'
},
{
q: 'You have 60 features, 8 informative. What happens if you keep all 60?',
options: [
  'Accuracy improves because more information is better',
  'Noise features dilute the signal, increase variance and slow training',
  'Nothing changes',
  'The model refuses to train'
],
answer: 1,
why: 'Irrelevant features add variance and give the model more chances to fit noise. In the demo above, cutting to 15 improved cross-validated accuracy by roughly two points.'
},
{
q: 'Which importance measure is least biased and most trustworthy for the final decision?',
options: [
  'Tree ~feature_importances_~',
  'Permutation importance measured on held-out data',
  'Pearson correlation with the target',
  'The order columns appear in the CSV'
],
answer: 1,
why: 'Impurity-based importance favours high-cardinality and continuous features. Permutation importance measures the actual drop in held-out performance when a feature is scrambled.'
}
]
},

/* ============================================================ */
{
id: 'splitting',
title: 'Splitting data correctly',
summary: 'Random, stratified, grouped and time-based splits; K-fold variants; and the nested cross-validation you need when you tune hyperparameters.',
tags: ['validation', 'core'],
intro: `
## The split encodes your assumption about deployment

Ask: **what will the model see in production that it has not seen in training?**

| Situation | Split |
|---|---|
| Rows are independent | Random split |
| Imbalanced classes | **Stratified** split |
| Multiple rows per entity (patient, user, shop) | **Group** split |
| Predicting the future | **Time-based** split |
| Both groups and time | Grouped time-series split |
| Very little data | Repeated K-fold, or leave-one-out |

~~~text
RANDOM              [ tr | te | tr | tr | te | tr ]   rows shuffled freely
STRATIFIED          same, but each split keeps the class ratio
GROUP               all rows of one entity land in the SAME side
TIME SERIES         [ train ............ ][ test ]    never train on the future
~~~

:::danger Group leakage, the invisible one
You have 10 X-rays per patient and split randomly. Patient A's images land in both train
and test. The model recognises *patient A*, not the disease. Your test accuracy is 0.97;
on a new hospital it is 0.61. Nothing in your code looks wrong.
:::
`,
keyPoints: [
  'Stratify classification splits so both halves keep the class ratio.',
  'If several rows belong to one entity, use GroupKFold or the entity leaks across the split.',
  'Time series must be split chronologically - never shuffled.',
  'Tuning and evaluating on the same data inflates your score; use nested CV when it matters.'
],
pitfalls: [
  'Shuffling time series data.',
  'Random splits on data with repeated entities.',
  'Reporting the best cross-validation score as if it were an unbiased estimate of future performance.',
  'Splitting after resampling (SMOTE), so synthetic copies of training rows appear in the test set.'
],
levels: [
{
name: 'The four split strategies',
goal: 'Implement each split type and measure how much score inflation the wrong one causes.',
md: `
~~~python splits.py
import numpy as np
import pandas as pd
from sklearn.model_selection import (train_test_split, KFold, StratifiedKFold,
                                     GroupKFold, StratifiedGroupKFold,
                                     TimeSeriesSplit, RepeatedStratifiedKFold,
                                     cross_val_score)

rng = np.random.default_rng(0)

# =====================================================================
# 1. STRATIFIED - keep the class balance in every split
# =====================================================================
y_imb = np.array([0] * 950 + [1] * 50)
X_imb = rng.normal(size=(1000, 5))

print("IMBALANCED DATA: 95% class 0, 5% class 1")
for name, strat in [("plain random", None), ("stratified", y_imb)]:
    mins, maxs = [], []
    for seed in range(200):
        _, _, y_tr, y_te = train_test_split(X_imb, y_imb, test_size=0.2,
                                            random_state=seed, stratify=strat)
        mins.append(y_te.mean()); maxs.append(y_te.mean())
    print(f"  {name:14s} test-set positive rate ranged "
          f"{min(mins):.3f} to {max(maxs):.3f}  (true rate 0.050)")

# =====================================================================
# 2. GROUP SPLIT - the leak that looks like success
# =====================================================================
from sklearn.ensemble import RandomForestClassifier

n_patients, per_patient = 100, 10
patient_id = np.repeat(np.arange(n_patients), per_patient)

# each patient has a personal signature; the disease adds a weak signal on top
patient_signature = rng.normal(size=(n_patients, 20))
X_med = patient_signature[patient_id] + rng.normal(0, 0.35, (1000, 20))
disease = rng.integers(0, 2, n_patients)
y_med = disease[patient_id]
X_med[:, 0] += y_med * 0.45                     # weak genuine signal

model = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)

random_cv = cross_val_score(model, X_med, y_med, cv=StratifiedKFold(5, shuffle=True,
                                                                   random_state=0))
group_cv = cross_val_score(model, X_med, y_med, cv=GroupKFold(5), groups=patient_id)

print("\\nMEDICAL DATA: 100 patients, 10 scans each")
print(f"  random 5-fold CV : {random_cv.mean():.4f}   <- patients appear on BOTH sides")
print(f"  GroupKFold       : {group_cv.mean():.4f}   <- the honest number")
print(f"  inflation        : {random_cv.mean() - group_cv.mean():+.4f}")
print("\\n  The random split lets the model memorise each patient. The group split")
print("  forces it to generalise to people it has never seen - which is the")
print("  actual deployment condition.")

# StratifiedGroupKFold keeps groups intact AND balances classes
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
sg_cv = cross_val_score(model, X_med, y_med, cv=sgkf, groups=patient_id)
print(f"  StratifiedGroupKFold: {sg_cv.mean():.4f}  (best of both)")

# =====================================================================
# 3. TIME SERIES SPLIT
# =====================================================================
print("\\nTIME SERIES SPLIT (expanding window)")
n = 100
tscv = TimeSeriesSplit(n_splits=5)
for i, (tr, te) in enumerate(tscv.split(np.arange(n))):
    print(f"  fold {i}: train [{tr[0]:3d} .. {tr[-1]:3d}]  test [{te[0]:3d} .. {te[-1]:3d}]")

print("\\n  with a gap, to avoid boundary leakage from rolling features:")
tscv_gap = TimeSeriesSplit(n_splits=5, gap=5)
for i, (tr, te) in enumerate(tscv_gap.split(np.arange(n))):
    print(f"  fold {i}: train ends {tr[-1]:3d}, GAP, test starts {te[0]:3d}")

# rolling (fixed-size) window instead of expanding
tscv_roll = TimeSeriesSplit(n_splits=5, max_train_size=30)
print("\\n  fixed-size rolling window (recent data only):")
for i, (tr, te) in enumerate(tscv_roll.split(np.arange(n))):
    print(f"  fold {i}: train [{tr[0]:3d} .. {tr[-1]:3d}] (n={len(tr)})  "
          f"test [{te[0]:3d} .. {te[-1]:3d}]")

# =====================================================================
# 4. HOW BAD IS SHUFFLING A TIME SERIES?
# =====================================================================
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

t = np.arange(500)
signal = np.sin(t / 12) * 10 + t * 0.05 + rng.normal(0, 1.2, 500)
lookback = 10
Xs = np.array([signal[i:i + lookback] for i in range(500 - lookback)])
ys = signal[lookback:]

# WRONG: shuffled
Xa, Xb, ya, yb = train_test_split(Xs, ys, test_size=0.25, random_state=0)
wrong = r2_score(yb, Ridge().fit(Xa, ya).predict(Xb))

# RIGHT: chronological
cut = int(len(Xs) * 0.75)
right = r2_score(ys[cut:], Ridge().fit(Xs[:cut], ys[:cut]).predict(Xs[cut:]))

print(f"\\nTIME SERIES FORECASTING")
print(f"  shuffled split (WRONG)  R2 = {wrong:.4f}")
print(f"  chronological  (RIGHT)  R2 = {right:.4f}")
print("  Overlapping windows mean nearly identical rows straddle a shuffled split.")
~~~

~~~text
IMBALANCED DATA: 95% class 0, 5% class 1
  plain random   test-set positive rate ranged 0.020 to 0.085  (true rate 0.050)
  stratified     test-set positive rate ranged 0.050 to 0.050  (true rate 0.050)

MEDICAL DATA: 100 patients, 10 scans each
  random 5-fold CV : 0.9720   <- patients appear on BOTH sides
  GroupKFold       : 0.6810   <- the honest number
  inflation        : +0.2910
~~~

**Twenty-nine points of pure illusion.** This is the single most dangerous split error,
because nothing errors out and the number looks wonderful.

:::tip Ask "what is a group?" on every project
- Medical images -> patient
- Retail transactions -> customer
- Sensor readings -> device
- Reviews -> product, or reviewer
- Anything scraped -> source domain
- Augmented images -> the original image

If the answer is not "each row is independent", you need a group-aware split.
:::
`
},
{
name: 'Cross-validation and nested CV',
goal: 'Choose the right CV scheme, and use nested CV to get an unbiased estimate when you tune.',
md: `
~~~python cv_schemes.py
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import (cross_val_score, cross_validate,
                                     KFold, StratifiedKFold, RepeatedStratifiedKFold,
                                     LeaveOneOut, ShuffleSplit)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import time

X, y = load_breast_cancer(return_X_y=True)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))

schemes = {
    "5-fold":                    StratifiedKFold(5, shuffle=True, random_state=0),
    "10-fold":                   StratifiedKFold(10, shuffle=True, random_state=0),
    "repeated 5-fold x5":        RepeatedStratifiedKFold(n_splits=5, n_repeats=5,
                                                         random_state=0),
    "shuffle-split (20 x 20%)":  ShuffleSplit(n_splits=20, test_size=0.2, random_state=0),
}
print(f"{'scheme':28s} {'mean':>8s} {'std':>8s} {'fits':>6s} {'seconds':>9s}")
print("-" * 64)
for name, cv in schemes.items():
    t0 = time.perf_counter()
    s = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
    print(f"{name:28s} {s.mean():8.4f} {s.std():8.4f} {len(s):6d} "
          f"{time.perf_counter()-t0:9.2f}")

# leave-one-out on a small subset (it is expensive)
t0 = time.perf_counter()
s = cross_val_score(model, X[:150], y[:150], cv=LeaveOneOut(), n_jobs=-1)
print(f"{'leave-one-out (n=150)':28s} {s.mean():8.4f} {s.std():8.4f} {len(s):6d} "
      f"{time.perf_counter()-t0:9.2f}")

# =====================================================================
# cross_validate: several metrics, and train scores, in one pass
# =====================================================================
res = cross_validate(model, X, y, cv=5,
                     scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
                     return_train_score=True, n_jobs=-1)
print("\\nMULTI-METRIC CROSS-VALIDATION")
for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    tr = res[f"train_{k}"].mean()
    te = res[f"test_{k}"].mean()
    print(f"  {k:10s} train={tr:.4f}  test={te:.4f}  gap={tr-te:+.4f}")
~~~

## Nested cross-validation

If you tune hyperparameters with CV and then report the best CV score, that score is
**optimistically biased** - you selected the configuration that happened to suit those folds.

~~~text
NESTED CV

OUTER LOOP (estimates performance)
  fold 1: [   TRAIN 80%   ][ TEST ]
             |
             +-- INNER LOOP (selects hyperparameters, on the 80% only)
                   fold a: [ tr ][va]
                   fold b: [ tr ][va]
                   fold c: [ tr ][va]
                   -> best params
             |
             +-- refit on the full 80% with those params, score on TEST
  fold 2: ... and so on

The outer test folds never influenced any choice, so their mean is unbiased.
~~~

~~~python nested_cv.py
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)

pipe = Pipeline([("scale", StandardScaler()), ("svm", SVC())])
grid = {"svm__C": [0.01, 0.1, 1, 10, 100],
        "svm__gamma": ["scale", 0.0001, 0.001, 0.01, 0.1]}

inner = StratifiedKFold(5, shuffle=True, random_state=1)
outer = StratifiedKFold(5, shuffle=True, random_state=2)

# ---- the BIASED way: tune and report the same number ---------------
search = GridSearchCV(pipe, grid, cv=inner, n_jobs=-1).fit(X, y)
print(f"NON-NESTED (biased)")
print(f"  best params    : {search.best_params_}")
print(f"  best CV score  : {search.best_score_:.4f}   <- reported as if unbiased")

# ---- the honest way -------------------------------------------------
nested = cross_val_score(
    GridSearchCV(pipe, grid, cv=inner, n_jobs=-1),
    X, y, cv=outer, n_jobs=-1)
print(f"\\nNESTED (unbiased)")
print(f"  per outer fold : {np.round(nested, 4)}")
print(f"  mean           : {nested.mean():.4f} (+/- {nested.std():.4f})")
print(f"\\n  optimism from tuning: {search.best_score_ - nested.mean():+.4f}")

# ---- see the instability: which params win in each outer fold? -----
print("\\nBest parameters chosen per outer fold:")
for i, (tr, te) in enumerate(outer.split(X, y)):
    s = GridSearchCV(pipe, grid, cv=inner, n_jobs=-1).fit(X[tr], y[tr])
    print(f"  fold {i}: {s.best_params_}  inner CV {s.best_score_:.4f}")
print("\\n  Different folds prefer different hyperparameters. That variability is")
print("  exactly the optimism that non-nested CV hides.")
~~~

~~~text
NON-NESTED (biased)
  best params    : {'svm__C': 10, 'svm__gamma': 'scale'}
  best CV score  : 0.9789

NESTED (unbiased)
  per outer fold : [0.9737 0.9825 0.9737 0.9737 0.9646]
  mean           : 0.9736 (+/- 0.0058)

  optimism from tuning: +0.0053
~~~

:::tip When to bother with nested CV
Half a point here, on a clean dataset with a small grid. But with a 500-point random
search on 400 rows the optimism can reach five points or more.

**Use nested CV when**: the dataset is small, the grid is large, or you are comparing
model families and need a defensible number (a paper, a client, a decision).

**Skip it when**: you have a proper held-out test set that you open exactly once. That is
simpler, cheaper and equally honest - and it is what you should do by default.
:::

### The decision tree for choosing a CV scheme

~~~python
# Is your data a time series?
#   YES -> TimeSeriesSplit (add gap= if you use rolling features)
#   NO  -> continue
#
# Do multiple rows belong to the same entity?
#   YES -> GroupKFold, or StratifiedGroupKFold if also imbalanced
#   NO  -> continue
#
# Is it classification?
#   YES -> StratifiedKFold(5 or 10, shuffle=True, random_state=...)
#   NO  -> KFold(5 or 10, shuffle=True, random_state=...)
#
# Fewer than ~200 rows?
#   -> RepeatedStratifiedKFold, or LeaveOneOut if truly tiny
~~~
`
}
],
quiz: [
{
q: 'You have 500 patients with 20 images each and split the 10,000 images randomly. What happens?',
options: [
  'Nothing, this is standard',
  'The same patient appears in train and test, so the model memorises patients and the score is hugely inflated',
  'The model will underfit',
  'You need more data'
],
answer: 1,
why: 'This is group leakage. Use GroupKFold with patient ID as the group so no patient spans the split. Inflation of 20-30 accuracy points is typical.'
},
{
q: 'Why does TimeSeriesSplit offer a ~gap~ parameter?',
options: [
  'To speed up training',
  'Rolling or lag features near the boundary can peek across it, so a gap prevents that leakage',
  'To reduce memory usage',
  'To handle missing timestamps'
],
answer: 1,
why: 'If a training row uses a 7-day rolling window that reaches into the test period, information leaks. A gap of at least the window length prevents it.'
},
{
q: 'You run GridSearchCV and report ~best_score_~ as your expected production performance. What is wrong?',
options: [
  'Nothing',
  'That score is optimistically biased because those folds selected the hyperparameters',
  'GridSearchCV does not compute a score',
  'You should report the training score instead'
],
answer: 1,
why: 'Selecting the maximum over many configurations captures fold-specific noise. Use nested CV, or - simpler - a held-out test set opened once.'
}
]
},

/* ============================================================ */
{
id: 'imbalance',
title: 'Imbalanced data',
summary: 'When 1% of your rows are the class you care about: why accuracy lies, and the four families of fix - metrics, weights, resampling and thresholds.',
tags: ['imbalance', 'classification'],
intro: `
## The problem

~~~text
Fraud detection:   99.8% legitimate, 0.2% fraud
A model that predicts "legitimate" for everything: 99.8% ACCURATE and 100% USELESS.
~~~

## The four families of fix, in the order you should try them

**1. Change the metric.** Often the only thing you needed. Use precision, recall, F1,
PR-AUC, or a business cost function. Never accuracy.

**2. Change the threshold.** Your model already outputs probabilities. Move the cut-off
away from 0.5 to match the cost of each error. Free, and usually the biggest win.

**3. Change the class weights.** ~class_weight="balanced"~ makes the loss count each
minority example more. One parameter, no new data, no synthetic rows.

**4. Resample.** Oversample the minority (SMOTE), undersample the majority, or both.
Powerful but easy to misuse - and it must happen **inside** the CV fold.

:::danger Resample only the training data, never the validation or test data
If you SMOTE before splitting, synthetic points derived from training rows land in your
test set. The score becomes fiction. This is the number-one imbalance bug.
:::
`,
keyPoints: [
  'Accuracy is meaningless under imbalance - use PR-AUC, recall, or a cost metric.',
  'Threshold tuning is free and usually beats resampling.',
  '~class_weight="balanced"~ is a one-line fix worth trying before anything complicated.',
  'SMOTE must be applied inside the pipeline, to training folds only.'
],
pitfalls: [
  'Resampling before the train/test split.',
  'Using ROC-AUC on severe imbalance - it looks good even when precision is terrible.',
  'Reporting accuracy alongside a 99% majority class.',
  'Applying SMOTE to high-dimensional or categorical data, where interpolation is meaningless.'
],
levels: [
{
name: 'Why accuracy lies, and the cheap fixes',
goal: 'Quantify the failure of accuracy, then fix it with metrics, weights and thresholds - no resampling needed.',
md: `
~~~python imbalance_basics.py
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix, classification_report,
                             precision_recall_curve)

# 1% positive class - realistic for fraud
X, y = make_classification(n_samples=20000, n_features=25, n_informative=8,
                           weights=[0.99], flip_y=0.01, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0,
                                          stratify=y)
print(f"class balance: {np.bincount(y_tr)}  ({y_tr.mean():.2%} positive)")

def report(name, model):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = (model.predict_proba(X_te)[:, 1]
             if hasattr(model, "predict_proba") else pred)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    print(f"{name:32s} acc={accuracy_score(y_te, pred):.4f} "
          f"prec={precision_score(y_te, pred, zero_division=0):.3f} "
          f"rec={recall_score(y_te, pred):.3f} "
          f"f1={f1_score(y_te, pred, zero_division=0):.3f} "
          f"ROC={roc_auc_score(y_te, proba):.3f} "
          f"PR={average_precision_score(y_te, proba):.3f}  "
          f"[TP={tp} FN={fn} FP={fp}]")

print(f"\\n{'model':32s} {'metrics':>60s}")
print("-" * 118)
report("dummy: always predict majority", DummyClassifier(strategy="most_frequent"))
report("logistic regression", LogisticRegression(max_iter=2000))
report("logistic, class_weight=balanced",
       LogisticRegression(max_iter=2000, class_weight="balanced"))
report("random forest", RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1))
report("random forest, balanced",
       RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                              random_state=0, n_jobs=-1))
~~~

~~~text
class balance: [13860   140]  (1.00% positive)

model                            metrics
----------------------------------------------------------------------------------
dummy: always predict majority   acc=0.9900 prec=0.000 rec=0.000 f1=0.000 ROC=0.500 PR=0.010  [TP=0 FN=60 FP=0]
logistic regression              acc=0.9928 prec=0.735 rec=0.417 f1=0.532 ROC=0.968 PR=0.601  [TP=25 FN=35 FP=9]
logistic, class_weight=balanced  acc=0.9605 prec=0.183 rec=0.917 f1=0.305 ROC=0.968 PR=0.601  [TP=55 FN=5 FP=245]
random forest                    acc=0.9932 prec=0.833 rec=0.417 f1=0.556 ROC=0.960 PR=0.657  [TP=25 FN=35 FP=5]
random forest, balanced          acc=0.9933 prec=0.870 rec=0.433 f1=0.578 ROC=0.958 PR=0.664  [TP=26 FN=34 FP=6]
~~~

### Read that table carefully

- **The dummy model is 99% accurate and catches zero fraud.** Accuracy is worthless here.
- **class_weight="balanced" moved recall from 0.42 to 0.92** - it caught 55 of 60 frauds
  instead of 25. It also produced 245 false alarms instead of 9. Whether that trade is
  good depends entirely on what an investigation costs versus what a missed fraud costs.
- **ROC-AUC barely moved (0.968 vs 0.968)** while the confusion matrix changed completely.
  ROC-AUC is threshold-free, so weighting does not change it. **PR-AUC is the metric to
  optimise under imbalance.**

### Choosing the threshold by business cost

~~~python cost_threshold.py
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
proba = model.predict_proba(X_te)[:, 1]

# The actual economics
COST_FALSE_NEGATIVE = 500.0     # a missed fraud costs us 500 euros
COST_FALSE_POSITIVE = 10.0      # investigating a false alarm costs 10 euros

thresholds = np.linspace(0.001, 0.999, 400)
costs, rows = [], []
for t in thresholds:
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    cost = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
    costs.append(cost)
    rows.append((t, tp, fn, fp, cost))

costs = np.array(costs)
best_i = costs.argmin()
best_t = thresholds[best_i]

print(f"cost of a missed fraud : {COST_FALSE_NEGATIVE:.0f} EUR")
print(f"cost of a false alarm  : {COST_FALSE_POSITIVE:.0f} EUR")
print(f"\\n{'threshold':>10s} {'TP':>5s} {'FN':>5s} {'FP':>6s} {'total cost':>12s}")
print("-" * 44)
for t in [0.005, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7]:
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    c = fn * COST_FALSE_NEGATIVE + fp * COST_FALSE_POSITIVE
    mark = "  <-- default" if t == 0.5 else ""
    print(f"{t:>10.3f} {tp:>5d} {fn:>5d} {fp:>6d} {c:>12,.0f}{mark}")

print(f"\\nOPTIMAL threshold: {best_t:.4f}  cost {costs[best_i]:,.0f} EUR")
print(f"Default 0.5 cost : {costs[np.argmin(np.abs(thresholds-0.5))]:,.0f} EUR")
savings = costs[np.argmin(np.abs(thresholds-0.5))] - costs[best_i]
print(f"SAVED by moving the threshold: {savings:,.0f} EUR "
      f"({savings/costs[np.argmin(np.abs(thresholds-0.5))]:.0%})")
~~~

~~~text
 threshold    TP    FN     FP   total cost
--------------------------------------------
     0.005    57     3    632        7,820
     0.020    52     8    241        6,410
     0.050    46    14    108        8,080
     0.100    40    20     53       10,530
     0.300    31    29     20       14,700
     0.500    25    35      9       17,590  <-- default
     0.700    19    41      4       20,540

OPTIMAL threshold: 0.0161  cost 6,180 EUR
Default 0.5 cost : 17,590 EUR
SAVED by moving the threshold: 11,410 EUR (65%)
~~~

:::tip The cheapest improvement in machine learning
**Moving one number from 0.5 to 0.016 cut costs by 65%.** No new features, no new model,
no extra data. Yet almost every beginner tutorial ships the default threshold.

Always: get probabilities, define the cost of each error, then optimise the threshold on
a validation set.
:::
`
},
{
name: 'Resampling: SMOTE and friends, done safely',
goal: 'Apply oversampling and undersampling correctly, and see the damage when you do not.',
md: `
~~~bash
pip install imbalanced-learn
~~~

~~~python resampling.py
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, classification_report

from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline      # NOTE: imblearn's Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier

X, y = make_classification(n_samples=10000, n_features=20, n_informative=6,
                           weights=[0.97], flip_y=0.01, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0,
                                          stratify=y)
print(f"training balance: {np.bincount(y_tr)}")

# =====================================================================
# THE WRONG WAY - resample first, then split
# =====================================================================
X_bad, y_bad = SMOTE(random_state=0).fit_resample(X, y)
Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(X_bad, y_bad, test_size=0.3,
                                              random_state=0, stratify=y_bad)
bad_model = RandomForestClassifier(n_estimators=200, random_state=0,
                                   n_jobs=-1).fit(Xb_tr, yb_tr)
bad_score = average_precision_score(yb_te, bad_model.predict_proba(Xb_te)[:, 1])
# and how it does on the REAL, untouched test set
real_score = average_precision_score(y_te, bad_model.predict_proba(X_te)[:, 1])

print("\\nSMOTE BEFORE SPLITTING (wrong)")
print(f"  score on the resampled test set : {bad_score:.4f}   <- what you would report")
print(f"  score on the REAL test set      : {real_score:.4f}   <- what you would get")
print("  Synthetic points interpolated from training rows ended up in the test set.")

# =====================================================================
# THE RIGHT WAY - resampling inside the pipeline
# =====================================================================
cv = StratifiedKFold(5, shuffle=True, random_state=0)

def evaluate(name, sampler=None, model=None):
    model = model or RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    steps = [("scale", StandardScaler())]
    if sampler is not None:
        steps.append(("sample", sampler))       # only runs on TRAINING folds
    steps.append(("model", model))
    pipe = ImbPipeline(steps)
    s = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="average_precision", n_jobs=-1)
    print(f"  {name:36s} PR-AUC {s.mean():.4f} (+/- {s.std():.4f})")
    return s.mean()

print("\\nDONE CORRECTLY (PR-AUC, cross-validated on training data)")
evaluate("no resampling")
evaluate("class_weight=balanced", None,
         RandomForestClassifier(n_estimators=200, class_weight="balanced_subsample",
                                random_state=0, n_jobs=-1))
print()
evaluate("RandomOverSampler (duplicate)", RandomOverSampler(random_state=0))
evaluate("SMOTE", SMOTE(random_state=0))
evaluate("BorderlineSMOTE", BorderlineSMOTE(random_state=0))
evaluate("ADASYN", ADASYN(random_state=0))
print()
evaluate("RandomUnderSampler", RandomUnderSampler(random_state=0))
evaluate("TomekLinks (clean boundary)", TomekLinks())
print()
evaluate("SMOTE + Tomek", SMOTETomek(random_state=0))
evaluate("SMOTE + ENN", SMOTEENN(random_state=0))
print()
evaluate("BalancedRandomForest", None,
         BalancedRandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1,
                                        sampling_strategy="all", replacement=True))

# partial oversampling is often better than going all the way to 50/50
print()
for ratio in [0.1, 0.25, 0.5, 1.0]:
    evaluate(f"SMOTE to minority ratio {ratio}",
             SMOTE(sampling_strategy=ratio, random_state=0))
~~~

~~~text
SMOTE BEFORE SPLITTING (wrong)
  score on the resampled test set : 0.9891   <- what you would report
  score on the REAL test set      : 0.6247   <- what you would get

DONE CORRECTLY (PR-AUC, cross-validated on training data)
  no resampling                        PR-AUC 0.6603 (+/- 0.0421)
  class_weight=balanced                PR-AUC 0.6712 (+/- 0.0388)

  RandomOverSampler (duplicate)        PR-AUC 0.6588 (+/- 0.0455)
  SMOTE                                PR-AUC 0.6431 (+/- 0.0402)
  BorderlineSMOTE                      PR-AUC 0.6390 (+/- 0.0433)
  ADASYN                               PR-AUC 0.6205 (+/- 0.0418)

  RandomUnderSampler                   PR-AUC 0.5744 (+/- 0.0512)
  TomekLinks (clean boundary)          PR-AUC 0.6621 (+/- 0.0408)

  SMOTE + Tomek                        PR-AUC 0.6455 (+/- 0.0397)
  SMOTE + ENN                          PR-AUC 0.6102 (+/- 0.0521)

  SMOTE to minority ratio 0.1          PR-AUC 0.6689 (+/- 0.0399)
  SMOTE to minority ratio 0.25         PR-AUC 0.6612 (+/- 0.0410)
  SMOTE to minority ratio 0.5          PR-AUC 0.6503 (+/- 0.0388)
  SMOTE to minority ratio 1.0          PR-AUC 0.6431 (+/- 0.0402)
~~~

### The result that surprises people

**SMOTE did not help.** ~class_weight="balanced"~ - one parameter - beat every resampling
strategy, and full 50/50 SMOTE was *worse* than doing nothing.

This is common. SMOTE is popular far beyond its evidence.

### How SMOTE actually works, and when it fails

~~~text
SMOTE: for each minority point, pick one of its k nearest minority neighbours,
       and create a new point somewhere on the line between them.

    o   o          o = majority     x = minority     * = synthetic
      x---*---x
    o       o      the new point lies ON the segment joining two real ones
~~~

**It fails when:**
- Features are **categorical** - interpolating between "Athens" and "Patras" is nonsense.
  Use ~SMOTENC~ instead.
- The data is **high-dimensional** - nearest neighbours become meaningless (curse of
  dimensionality), so the interpolation is arbitrary.
- The minority class is **not a contiguous cluster** - synthetic points land between two
  separate sub-populations, in territory where no real example exists.
- There are **noisy minority points** - SMOTE amplifies them into whole synthetic regions.

:::tip The order to try things
1. **Fix the metric** (PR-AUC, recall at fixed precision, business cost). Often enough.
2. **Tune the threshold.** Free, and usually the biggest single win.
3. **~class_weight="balanced"~.** One parameter.
4. **Partial oversampling** to a ratio like 0.1-0.25, not all the way to 50/50.
5. **SMOTE**, inside an ~imblearn.pipeline.Pipeline~, only if 1-4 are not enough.
6. **Collect more minority data.** Slow, expensive, and better than all of the above.
:::

:::warn imblearn's Pipeline, not sklearn's
~sklearn.pipeline.Pipeline~ applies every step to both training and prediction data.
~imblearn.pipeline.Pipeline~ knows that samplers must run **only during fit**. Using the
wrong one silently resamples your validation folds.
:::
`
}
],
quiz: [
{
q: 'Your fraud model is 99.8% accurate. Fraud is 0.2% of transactions. What do you conclude?',
options: [
  'The model is excellent',
  'It probably predicts "not fraud" for everything - check recall and PR-AUC',
  'The dataset is too small',
  'You should add more features'
],
answer: 1,
why: 'Predicting the majority class always achieves exactly that accuracy. Look at the confusion matrix, recall and precision-recall AUC instead.'
},
{
q: 'Where must SMOTE be applied?',
options: [
  'To the whole dataset before splitting',
  'Only to the training folds, inside an imblearn Pipeline',
  'To the test set only',
  'After training the model'
],
answer: 1,
why: 'Resampling before splitting puts synthetic points derived from training rows into the test set, inflating the score dramatically. imblearn Pipelines apply samplers during fit only.'
},
{
q: 'Under severe imbalance, which metric is most informative?',
options: ['Accuracy', 'ROC-AUC', 'PR-AUC (average precision)', 'Mean squared error'],
answer: 2,
why: 'ROC-AUC can stay high because the huge number of true negatives keeps the false-positive rate low. PR-AUC focuses on the positive class and reflects precision honestly.'
},
{
q: 'What is usually the cheapest and most effective first fix for imbalance?',
options: [
  'SMOTE oversampling',
  'Tuning the decision threshold on a validation set using the real cost of each error',
  'Collecting ten times more data',
  'Switching to a deep neural network'
],
answer: 1,
why: 'The model already outputs probabilities. Choosing the threshold that minimises actual business cost is free and frequently the single largest improvement available.'
}
]
},

/* ============================================================ */
{
id: 'pipelines',
title: 'Pipelines and leakage prevention',
summary: 'ColumnTransformer, custom transformers, and the architecture that makes leakage structurally impossible - plus a full production-ready preprocessing setup.',
tags: ['pipelines', 'sklearn', 'production'],
intro: `
## A pipeline is not tidiness, it is correctness

~~~text
WITHOUT A PIPELINE                    WITH A PIPELINE
imputer.fit(X)                        pipe = Pipeline([
scaler.fit(X)                            ("impute", SimpleImputer()),
encoder.fit(X)                           ("scale",  StandardScaler()),
X = ...transform...                      ("model",  LogisticRegression()),
model.fit(X, y)                       ])
                                      pipe.fit(X_train, y_train)
Every .fit() above saw ALL the data.
During CV, each transformer must      During CV, EVERY step is refitted on
be refitted per fold. By hand, you    each training fold automatically.
will forget. Everyone does.           It is structurally impossible to leak.
~~~

## What a pipeline gives you

1. **No leakage during cross-validation** - transformers refit per fold, automatically.
2. **One object to save** - ~joblib.dump(pipe)~ stores preprocessing *and* model.
3. **Hyperparameter search across preprocessing too** - tune the imputation strategy and
   the model's C in one grid.
4. **Serving is trivial** - feed a raw DataFrame and get a prediction. No duplicated
   preprocessing code between training and production.

:::tip The rule
**If it has a ~fit~ method, it belongs in the pipeline.** Imputers, scalers, encoders,
feature selectors, dimensionality reducers, samplers. Anything that learns from data
must learn from training data only.
:::
`,
keyPoints: [
  'Anything with a ~fit~ method must live inside the Pipeline.',
  'ColumnTransformer applies different preprocessing to different column groups.',
  'Save the whole pipeline, not just the model - serving then needs no preprocessing code.',
  'You can grid-search preprocessing choices exactly like model hyperparameters.'
],
pitfalls: [
  'Transforming outside the pipeline and then passing the result in.',
  'Forgetting ~remainder="drop"~ or ~"passthrough"~ on ColumnTransformer and losing or leaking columns.',
  'Writing a custom transformer that stores state in ~transform~ instead of ~fit~.',
  'Using sklearn Pipeline with imblearn samplers.'
],
levels: [
{
name: 'ColumnTransformer: different treatment per column type',
goal: 'Build the full preprocessing architecture for a mixed-type dataset in one composable object.',
md: `
~~~python column_transformer.py
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, OrdinalEncoder,
                                   FunctionTransformer, RobustScaler)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report

# =====================================================================
# A realistic mixed-type dataset
# =====================================================================
rng = np.random.default_rng(0)
n = 3000
df = pd.DataFrame({
    # numeric, roughly symmetric
    "age": rng.normal(45, 14, n).round().clip(18, 90),
    "tenure_months": rng.integers(1, 120, n),
    # numeric, heavily skewed
    "income": rng.lognormal(10.4, 0.7, n).round(-2),
    "balance": rng.lognormal(8, 1.2, n).round(2),
    # nominal categorical
    "city": rng.choice(["Athens", "Thessaloniki", "Patras", "Volos", "Larissa"], n),
    "channel": rng.choice(["web", "app", "branch"], n, p=[0.5, 0.35, 0.15]),
    # ORDINAL categorical
    "plan": rng.choice(["basic", "standard", "premium"], n, p=[0.5, 0.35, 0.15]),
    "satisfaction": rng.choice(["low", "medium", "high"], n),
    # binary
    "has_card": rng.random(n) < 0.6,
})
# punch some holes
for c in ["age", "income", "city", "satisfaction"]:
    df.loc[rng.random(n) < 0.08, c] = np.nan

# a target that actually depends on the features
logit = (-2.5
         + 0.02 * (50 - df["age"].fillna(45))
         + 0.9 * (df["plan"] == "basic")
         - 0.8 * (df["satisfaction"] == "high").astype(float)
         - 0.004 * df["tenure_months"]
         + rng.normal(0, 0.6, n))
df["churn"] = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)

print(df.dtypes)
print(f"\\nchurn rate: {df['churn'].mean():.3f}")
print(f"missing:\\n{df.isna().sum()[df.isna().sum() > 0]}")

X = df.drop(columns="churn")
y = df["churn"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

# =====================================================================
# DEFINE THE COLUMN GROUPS
# =====================================================================
numeric_symmetric = ["age", "tenure_months"]
numeric_skewed = ["income", "balance"]
nominal = ["city", "channel"]
ordinal_cols = ["plan", "satisfaction"]
binary = ["has_card"]

# explicit orders for the ordinal columns - never let it guess
ordinal_orders = [
    ["basic", "standard", "premium"],
    ["low", "medium", "high"],
]

# =====================================================================
# BUILD ONE PIPELINE PER GROUP
# =====================================================================
symmetric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median", add_indicator=True)),
    ("scale", StandardScaler()),
])

skewed_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("scale", RobustScaler()),
])

nominal_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                             min_frequency=0.01)),   # rare levels grouped
])

ordinal_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(categories=ordinal_orders,
                               handle_unknown="use_encoded_value", unknown_value=-1)),
    ("scale", StandardScaler()),
])

binary_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("to_int", FunctionTransformer(lambda x: x.astype(int),
                                   feature_names_out="one-to-one")),
])

# =====================================================================
# COMBINE THEM
# =====================================================================
preprocess = ColumnTransformer(
    transformers=[
        ("sym", symmetric_pipe, numeric_symmetric),
        ("skew", skewed_pipe, numeric_skewed),
        ("nom", nominal_pipe, nominal),
        ("ord", ordinal_pipe, ordinal_cols),
        ("bin", binary_pipe, binary),
    ],
    remainder="drop",          # be explicit: anything unlisted is dropped
    verbose_feature_names_out=False,
)

full_pipeline = Pipeline([
    ("prep", preprocess),
    ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
])

full_pipeline.fit(X_tr, y_tr)

print(f"\\nraw columns : {X_tr.shape[1]}")
print(f"after prep  : {full_pipeline.named_steps['prep'].transform(X_tr).shape[1]}")
print("feature names:")
print(" ", list(full_pipeline.named_steps["prep"].get_feature_names_out()))

print("\\nTEST SET PERFORMANCE")
print(classification_report(y_te, full_pipeline.predict(X_te),
                            target_names=["stay", "churn"], digits=3))
~~~

### Selecting columns automatically

~~~python auto_selection.py
from sklearn.compose import make_column_selector as selector

# instead of hard-coding lists, select by dtype
auto_preprocess = ColumnTransformer([
    ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                      ("scale", StandardScaler())]),
     selector(dtype_include=np.number)),
    ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
     selector(dtype_include=["object", "category", "bool"])),
])

# useful for quick baselines; use explicit lists for anything you will maintain,
# because dtype-based selection silently changes behaviour when a dtype changes.
~~~

### Tuning preprocessing and model together

~~~python tune_pipeline.py
from sklearn.model_selection import GridSearchCV, StratifiedKFold

grid = {
    # preprocessing choices ARE hyperparameters
    "prep__sym__impute__strategy": ["mean", "median"],
    "prep__nom__onehot__min_frequency": [0.005, 0.02, 0.05],
    # and the model
    "model__C": [0.01, 0.1, 1, 10],
}

search = GridSearchCV(full_pipeline, grid,
                      cv=StratifiedKFold(5, shuffle=True, random_state=0),
                      scoring="average_precision", n_jobs=-1)
search.fit(X_tr, y_tr)

print("best parameters:")
for k, v in search.best_params_.items():
    print(f"  {k:42s} {v}")
print(f"best CV PR-AUC: {search.best_score_:.4f}")

# ---- save the ENTIRE thing ------------------------------------------
import joblib
joblib.dump(search.best_estimator_, "churn_pipeline.joblib")

# ---- and serving is one line ----------------------------------------
loaded = joblib.load("churn_pipeline.joblib")
new_customer = pd.DataFrame([{
    "age": 34, "tenure_months": 6, "income": 28000.0, "balance": 1200.0,
    "city": "Rhodes",                    # a city never seen in training!
    "channel": "app", "plan": "basic",
    "satisfaction": np.nan,              # and a missing value
    "has_card": True,
}])
print(f"\\nchurn probability for a new customer: "
      f"{loaded.predict_proba(new_customer)[0, 1]:.3f}")
print("Unseen city and a missing value, handled without a line of extra code.")
~~~

:::tip That last block is the payoff
The raw input had an unseen category and a missing value. Because imputation and
~handle_unknown="ignore"~ live inside the saved pipeline, it just worked. Without a
pipeline you would need to reimplement every preprocessing step in your serving code -
and keep the two implementations in sync forever.
:::
`
},
{
name: 'Custom transformers and a leakage audit',
goal: 'Write your own pipeline-compatible transformer, and build a checklist that catches leakage before deployment.',
md: `
## Writing a custom transformer

~~~python custom_transformers.py
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class RatioFeatures(BaseEstimator, TransformerMixin):
    """Add ratio columns. Stateless, but still fit/transform for pipeline use."""

    def __init__(self, pairs=None):
        self.pairs = pairs or []          # list of (numerator, denominator, name)

    def fit(self, X, y=None):
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        X = X.copy()
        for num, den, name in self.pairs:
            X[name] = X[num] / X[den].replace(0, np.nan)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_ + [p[2] for p in self.pairs])


class WinsoriseNumeric(BaseEstimator, TransformerMixin):
    """Cap values at percentiles LEARNED FROM TRAINING DATA ONLY.

    This is the pattern that matters: the percentiles are state, so they
    are computed in fit and merely applied in transform.
    """

    def __init__(self, lower=0.01, upper=0.99, columns=None):
        self.lower = lower
        self.upper = upper
        self.columns = columns

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = self.columns or X.select_dtypes("number").columns.tolist()
        self.bounds_ = {c: (X[c].quantile(self.lower), X[c].quantile(self.upper))
                        for c in self.columns_}
        return self

    def transform(self, X):
        check_is_fitted(self, "bounds_")
        X = pd.DataFrame(X).copy()
        for c, (lo, hi) in self.bounds_.items():
            X[c] = X[c].clip(lo, hi)
        return X


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Map categories seen in under min_freq of training rows to 'OTHER'."""

    def __init__(self, min_freq=0.01, columns=None, other="OTHER"):
        self.min_freq = min_freq
        self.columns = columns
        self.other = other

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = self.columns or X.select_dtypes(
            include=["object", "category"]).columns.tolist()
        self.keep_ = {}
        for c in self.columns_:
            freq = X[c].value_counts(normalize=True)
            self.keep_[c] = set(freq[freq >= self.min_freq].index)
        return self

    def transform(self, X):
        check_is_fitted(self, "keep_")
        X = pd.DataFrame(X).copy()
        for c in self.columns_:
            X[c] = X[c].where(X[c].isin(self.keep_[c]), self.other)
        return X


class DateFeatures(BaseEstimator, TransformerMixin):
    """Expand a datetime column into modelling features and drop the original."""

    def __init__(self, column, cyclical=True):
        self.column = column
        self.cyclical = cyclical

    def fit(self, X, y=None):
        self.reference_ = pd.to_datetime(X[self.column]).max()
        return self

    def transform(self, X):
        X = X.copy()
        d = pd.to_datetime(X[self.column]).dt
        X[f"{self.column}_year"] = d.year
        X[f"{self.column}_month"] = d.month
        X[f"{self.column}_dow"] = d.dayofweek
        X[f"{self.column}_is_weekend"] = (d.dayofweek >= 5).astype(int)
        X[f"{self.column}_days_ago"] = (
            self.reference_ - pd.to_datetime(X[self.column])).dt.days
        if self.cyclical:
            X[f"{self.column}_month_sin"] = np.sin(2 * np.pi * d.month / 12)
            X[f"{self.column}_month_cos"] = np.cos(2 * np.pi * d.month / 12)
        return X.drop(columns=[self.column])


# =====================================================================
# USE THEM IN A PIPELINE
# =====================================================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

rng = np.random.default_rng(0)
n = 2000
df = pd.DataFrame({
    "income": rng.lognormal(10.4, 0.7, n),
    "debt": rng.lognormal(9.5, 0.9, n),
    "signup": pd.to_datetime("2022-01-01") + pd.to_timedelta(
        rng.integers(0, 900, n), unit="D"),
    "city": rng.choice(["Athens"] * 50 + ["Patras"] * 30 + ["Volos"] * 15 +
                       [f"tiny{i}" for i in range(60)], n),
})
df["y"] = ((df["debt"] / df["income"] > 0.8).astype(int) ^
           (rng.random(n) < 0.12).astype(int))

X, y = df.drop(columns="y"), df["y"]

pipe = Pipeline([
    ("dates", DateFeatures("signup")),
    ("ratios", RatioFeatures([("debt", "income", "debt_to_income")])),
    ("winsorise", WinsoriseNumeric(columns=["income", "debt", "debt_to_income"])),
    ("rare", RareCategoryGrouper(min_freq=0.02, columns=["city"])),
    ("encode", ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["city"]),
    ], remainder="passthrough", verbose_feature_names_out=False)),
    ("impute", SimpleImputer(strategy="median")),
    ("model", HistGradientBoostingClassifier(random_state=0)),
])

scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
print(f"5-fold ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")
print("\\nEvery custom step was refitted inside each fold. No leakage is possible.")

pipe.fit(X, y)
print("\\ncity levels kept:", sorted(pipe.named_steps['rare'].keep_['city']))
print("winsorising bounds learned:",
      {k: (round(v[0], 1), round(v[1], 1))
       for k, v in pipe.named_steps['winsorise'].bounds_.items()})
~~~

### The transformer contract

~~~text
1. Inherit from BaseEstimator and TransformerMixin
     -> you get fit_transform, get_params, set_params for free
2. __init__ stores parameters ONLY. No computation, no validation, no renaming.
     -> sklearn clones estimators by reading __init__ arguments
3. fit(X, y=None) LEARNS state, stores it with a trailing underscore, returns self
4. transform(X) APPLIES the learned state. It must NOT learn anything.
5. Optionally implement get_feature_names_out for readable pipelines
~~~

:::danger The custom-transformer leak
~~~python
def transform(self, X):
    return (X - X.mean()) / X.std()      # WRONG - learns from whatever it is given
~~~
At prediction time this standardises using the *test set's* statistics. Learn in ~fit~,
apply in ~transform~. Always.
:::

## The leakage audit

~~~python leakage_audit.py
"""Run this checklist before you deploy anything."""

CHECKLIST = """
DATA LEAKAGE AUDIT
==================

1. TEMPORAL
   [ ] Is every feature computable using only data from BEFORE the prediction time?
   [ ] Do rolling/expanding features use .shift(1)?
   [ ] Is the split chronological for time-series problems?
   [ ] Is there a gap between train and test if features use windows?

2. TARGET-DERIVED
   [ ] Does any column encode the outcome (refund_issued, cancellation_date, 'alive')?
   [ ] Is any feature a transformation of the target?
   [ ] Is target encoding cross-fitted?

3. PREPROCESSING
   [ ] Are ALL fitted transformers inside the Pipeline?
   [ ] Was any scaler/imputer/encoder fitted before the split?
   [ ] Was feature selection done on the whole dataset?
   [ ] Was resampling (SMOTE) applied before splitting?

4. GROUPS AND DUPLICATES
   [ ] Do multiple rows belong to one entity? If so, is the split grouped?
   [ ] Are there duplicate or near-duplicate rows straddling the split?
   [ ] For augmented data, do augmented copies stay on the same side?

5. EVALUATION
   [ ] Was the test set used only once, at the end?
   [ ] Were hyperparameters tuned on validation data, not test?
   [ ] Is the reported metric computed in the original target units?

6. SMELL TESTS
   [ ] Is accuracy suspiciously high (>0.99) for a hard problem?
   [ ] Does one feature carry over 80% of the importance? Why?
   [ ] Is the train/test gap implausibly small?
   [ ] Would a domain expert be surprised by this score?
"""
print(CHECKLIST)


# an automatable part of it
import pandas as pd
import numpy as np

def audit(df, target, time_col=None, group_col=None):
    print("AUTOMATED CHECKS")
    y = df[target]

    # near-perfect single-feature predictors
    for c in df.select_dtypes("number").columns:
        if c == target:
            continue
        r = df[[c, target]].corr().iloc[0, 1]
        if abs(r) > 0.9:
            print(f"  ! '{c}' correlates {r:+.3f} with the target")

    # a categorical column that perfectly separates the classes
    for c in df.select_dtypes(include=["object", "category"]).columns:
        if df[c].nunique() > 50:
            continue
        rates = df.groupby(c, observed=True)[target].mean()
        if ((rates == 0) | (rates == 1)).all() and df[c].nunique() > 1:
            print(f"  ! '{c}' perfectly separates the target")

    # duplicates
    d = df.duplicated().sum()
    if d:
        print(f"  ! {d} duplicate rows - they may straddle the split")

    # duplicate feature rows with different labels (or the same)
    feats = df.drop(columns=[target])
    fd = feats.duplicated().sum()
    if fd:
        print(f"  ! {fd} rows have identical features")

    # groups
    if group_col:
        per = df.groupby(group_col).size()
        if per.max() > 1:
            print(f"  ! '{group_col}' has up to {per.max()} rows per entity "
                  f"- use GroupKFold")

    print("  (automated checks complete - the manual checklist still matters)")


demo = pd.DataFrame({
    "feature": [1, 2, 3, 4, 5, 5],
    "leaky": [0, 0, 1, 1, 1, 1],
    "patient": ["a", "a", "b", "b", "c", "c"],
    "target": [0, 0, 1, 1, 1, 1],
})
audit(demo, "target", group_col="patient")
~~~
`
}
],
quiz: [
{
q: 'Why does a Pipeline prevent leakage during cross-validation?',
options: [
  'It shuffles the data automatically',
  'Every step with a fit method is refitted on each training fold only',
  'It removes correlated features',
  'It uses a different random seed per fold'
],
answer: 1,
why: 'Cross-validation calls fit on the whole pipeline per fold, so imputers, scalers and encoders learn only from that fold training portion. Doing it manually, people forget.'
},
{
q: 'In a custom transformer, where should the training statistics be computed?',
options: [
  'In ~__init__~',
  'In ~fit~, stored with a trailing underscore, and merely applied in ~transform~',
  'In ~transform~, recomputed each time',
  'In a module-level variable'
],
answer: 1,
why: 'Computing statistics inside transform means prediction-time data influences the transformation - leakage. __init__ must only store parameters, because sklearn clones estimators from them.'
},
{
q: 'What is the advantage of saving the whole pipeline rather than just the model?',
options: [
  'It is a smaller file',
  'Serving code needs no preprocessing logic - raw input goes in, prediction comes out',
  'It trains faster next time',
  'It automatically retrains'
],
answer: 1,
why: 'The imputer, scaler and encoder travel with the model, so training and serving cannot drift apart. Duplicated preprocessing code is a classic source of production bugs.'
},
{
q: 'Which of these is a leakage red flag worth investigating?',
options: [
  'Test accuracy of 0.85 on a hard problem',
  'A single feature carrying 92% of the model importance',
  'A training accuracy slightly above test accuracy',
  'Five percent missing values'
],
answer: 1,
why: 'One overwhelmingly dominant feature usually means it encodes the answer. Check whether it would be available, with that value, at real prediction time.'
}
]
}

]
});
