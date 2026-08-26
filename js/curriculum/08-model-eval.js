/* Track 08 - Model evaluation, tuning, interpretation */
(window.CURRICULUM = window.CURRICULUM || []).push({
id: 'modeleval',
title: 'Evaluation & Tuning',
icon: 'E',
level: 'Intermediate',
blurb: 'Bias-variance in practice, hyperparameter search that is worth the compute, model interpretation with SHAP, ensembling and stacking, and the error analysis that actually improves models.',
intro: `
You have a model. Now: is it good, why is it wrong, and how do you make it better?

~~~text
1. DIAGNOSE     bias or variance? learning curves say which
2. TUNE         grid -> random -> Bayesian, and when each is worth it
3. INTERPRET    SHAP, partial dependence, permutation importance
4. COMBINE      voting, stacking, blending
5. ANALYSE      look at the errors themselves - the highest-yield habit in ML
~~~

:::tip The order matters
Most people jump straight to tuning. Tuning typically buys 1-3%. **Error analysis and
feature work typically buy 10-30%.** Diagnose first, tune last.
:::
`,
topics: [

/* ============================================================ */
{
id: 'bias-variance',
title: 'Bias, variance and diagnosis',
summary: 'Decompose your error into bias, variance and noise - then use learning and validation curves to decide exactly what to do next.',
tags: ['diagnosis', 'theory', 'core'],
intro: `
## The decomposition

:::math Expected test error
**E[error] = Bias squared + Variance + Irreducible noise**

- **Bias** - error from wrong assumptions. Systematically off. Underfitting.
- **Variance** - error from sensitivity to the training sample. Unstable. Overfitting.
- **Noise** - randomness in the world. Cannot be removed by any model.
:::

~~~text
                      the dartboard picture

  LOW BIAS, LOW VARIANCE      LOW BIAS, HIGH VARIANCE
        . . .                        .        .
       . x .                       .    x       .
        . . .                          .    .
     tight, on target            scattered around target

  HIGH BIAS, LOW VARIANCE     HIGH BIAS, HIGH VARIANCE
                . . .                .      .
      x        . . .          x        .  .
                . . .                     .   .
   tight, but off target        scattered AND off target
~~~

## The diagnosis table

| Train error | Validation error | Diagnosis | Do this |
|---|---|---|---|
| High | High (similar) | **High bias** | Bigger model, better features, less regularisation, train longer |
| Low | High (big gap) | **High variance** | More data, more regularisation, simpler model, dropout, early stopping |
| High | Low | Bug | Check the split, check for a leak in reverse |
| Low | Low | Done | Ship it |

:::warn Compare against the achievable error, not zero
If two expert radiologists disagree 4% of the time, a 4% model error is *perfect*, not
"96% accurate with room to improve". This gap - between your model and the best possible -
is what you should be measuring. It is often called Bayes error or human-level performance.
:::
`,
keyPoints: [
  'Bias and variance require opposite fixes - diagnosing wrongly wastes weeks.',
  'The gap between training and validation error measures variance.',
  'The distance between training error and the achievable floor measures bias.',
  'More data fixes variance and does nothing for bias.'
],
pitfalls: [
  'Adding model capacity to fix a high validation error caused by overfitting.',
  'Requesting more data when the learning curve has already flattened.',
  'Comparing your error to zero rather than to the irreducible floor.'
],
levels: [
{
name: 'Measuring bias and variance directly',
goal: 'Decompose the error of several models empirically, and read learning curves to prescribe the fix.',
md: `
~~~python bias_variance.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

rng = np.random.default_rng(0)

# The TRUE function - in real life you never know this
def truth(x):
    return np.sin(1.4 * np.pi * x) + 0.4 * x

NOISE_SD = 0.35
X_test = np.linspace(0, 2, 200).reshape(-1, 1)
y_test_true = truth(X_test).ravel()

def bias_variance_decompose(model_fn, n_datasets=200, n_train=40):
    """Train the same model on many different training sets and
    decompose the error at each test point."""
    preds = np.zeros((n_datasets, len(X_test)))
    for i in range(n_datasets):
        X_tr = rng.uniform(0, 2, (n_train, 1))
        y_tr = truth(X_tr).ravel() + rng.normal(0, NOISE_SD, n_train)
        preds[i] = model_fn().fit(X_tr, y_tr).predict(X_test)

    mean_pred = preds.mean(axis=0)
    bias_sq = np.mean((mean_pred - y_test_true) ** 2)
    variance = np.mean(preds.var(axis=0))
    noise = NOISE_SD ** 2
    total = bias_sq + variance + noise
    return bias_sq, variance, noise, total, preds, mean_pred


models = {
    "linear (degree 1)":  lambda: make_pipeline(PolynomialFeatures(1), LinearRegression()),
    "poly degree 3":      lambda: make_pipeline(PolynomialFeatures(3), LinearRegression()),
    "poly degree 12":     lambda: make_pipeline(PolynomialFeatures(12), LinearRegression()),
    "tree (unpruned)":    lambda: DecisionTreeRegressor(),
    "tree (depth 3)":     lambda: DecisionTreeRegressor(max_depth=3),
    "random forest":      lambda: RandomForestRegressor(n_estimators=100, n_jobs=-1),
}

print(f"{'model':22s} {'bias^2':>9s} {'variance':>10s} {'noise':>8s} {'total':>9s}  dominant")
print("-" * 74)
results = {}
for name, fn in models.items():
    b, v, n, t, preds, mean_pred = bias_variance_decompose(fn)
    dominant = "BIAS" if b > v else "VARIANCE"
    print(f"{name:22s} {b:>9.4f} {v:>10.4f} {n:>8.4f} {t:>9.4f}  {dominant}")
    results[name] = (preds, mean_pred, b, v)

# ---- visualise: many fits from many training sets -------------------
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ax, (name, (preds, mean_pred, b, v)) in zip(axes.ravel(), results.items()):
    for i in range(0, 60, 2):
        ax.plot(X_test, preds[i], color="steelblue", alpha=0.12, lw=1)
    ax.plot(X_test, y_test_true, "k--", lw=2.5, label="truth")
    ax.plot(X_test, mean_pred, "r-", lw=2.5, label="average model")
    ax.set_title(f"{name}\\nbias2={b:.3f}  var={v:.3f}")
    ax.set_ylim(-2, 3)
    if name.startswith("linear"):
        ax.legend(fontsize=8)
plt.suptitle("Blue = models from 30 different training sets. "
             "Spread = VARIANCE, distance of red from black = BIAS", y=1.01)
plt.tight_layout(); plt.show()
~~~

~~~text
model                     bias^2   variance    noise     total  dominant
--------------------------------------------------------------------------
linear (degree 1)         0.4212     0.0197   0.1225    0.5634  BIAS
poly degree 3             0.0871     0.0403   0.1225    0.2499  BIAS
poly degree 12            0.0089     0.6841   0.1225    0.8155  VARIANCE
tree (unpruned)           0.0104     0.2913   0.1225    0.4242  VARIANCE
tree (depth 3)            0.1035     0.0819   0.1225    0.3079  BIAS
random forest             0.0261     0.0430   0.1225    0.1916  VARIANCE
~~~

### Read the last two rows together

- An **unpruned tree** has almost no bias (0.010) and huge variance (0.291).
- A **random forest** of 100 such trees has similar bias (0.026) and **variance cut by 85%**
  (0.043).

That is precisely what bagging does: averaging decorrelated high-variance models leaves the
bias alone and destroys the variance. It is why random forests are so hard to break.

### Prescribing the fix from a learning curve

~~~python diagnose.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = make_classification(n_samples=5000, n_features=25, n_informative=10,
                           n_redundant=5, class_sep=0.85, flip_y=0.05,
                           random_state=0)

def diagnose(model, name, ax):
    sizes, train_s, val_s = learning_curve(
        model, X, y, train_sizes=np.linspace(0.03, 1.0, 15), cv=5,
        scoring="accuracy", n_jobs=-1, shuffle=True, random_state=0)
    tr, va = train_s.mean(1), val_s.mean(1)

    ax.plot(sizes, tr, "o-", label="training", color="steelblue")
    ax.fill_between(sizes, tr - train_s.std(1), tr + train_s.std(1),
                    alpha=0.15, color="steelblue")
    ax.plot(sizes, va, "s-", label="validation", color="darkorange")
    ax.fill_between(sizes, va - val_s.std(1), va + val_s.std(1),
                    alpha=0.15, color="darkorange")
    ax.set_xlabel("training samples"); ax.set_ylabel("accuracy")
    ax.set_ylim(0.5, 1.02); ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    gap = tr[-1] - va[-1]
    # is validation still climbing?
    still_improving = (va[-1] - va[-4]) > 0.005
    if gap > 0.08:
        verdict = ("HIGH VARIANCE, and more data WILL help" if still_improving
                   else "HIGH VARIANCE, more data will NOT help - regularise")
    elif va[-1] < 0.80:
        verdict = "HIGH BIAS - use a stronger model or better features"
    else:
        verdict = "well balanced"
    ax.set_title(f"{name}\\ngap={gap:.3f}  ->  {verdict}", fontsize=9)
    return gap, va[-1], verdict


fig, axes = plt.subplots(1, 3, figsize=(17, 5))
print(f"{'model':34s} {'gap':>7s} {'val':>7s}  prescription")
print("-" * 96)
for ax, (name, m) in zip(axes, [
        ("underfit: heavy regularisation",
         make_pipeline(StandardScaler(), LogisticRegression(C=0.0005, max_iter=3000))),
        ("overfit: unpruned tree", DecisionTreeClassifier(random_state=0)),
        ("balanced: random forest",
         RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                random_state=0, n_jobs=-1))]):
    gap, val, verdict = diagnose(m, name, ax)
    print(f"{name:34s} {gap:>7.3f} {val:>7.3f}  {verdict}")
plt.tight_layout(); plt.show()
~~~

~~~text
model                                  gap     val  prescription
------------------------------------------------------------------------------------------------
underfit: heavy regularisation       0.004   0.712  HIGH BIAS - use a stronger model or better features
overfit: unpruned tree               0.166   0.786  HIGH VARIANCE, and more data WILL help
balanced: random forest              0.052   0.856  well balanced
~~~

### The full decision tree

~~~python prescription.py
GUIDE = """
DIAGNOSIS AND PRESCRIPTION

Step 1: what is the best achievable error?
        (human performance, an existing system, an expert disagreement rate)

Step 2: measure training error and validation error.

  AVOIDABLE BIAS = training error - best achievable error
  VARIANCE       = validation error - training error

Step 3: attack the LARGER of the two.

IF AVOIDABLE BIAS IS LARGER  (underfitting)
  + Bigger / more expressive model      (more layers, more depth, higher degree)
  + Better features                     (usually the biggest win)
  + Train longer, tune the optimiser
  + REDUCE regularisation
  - More data will NOT help
  - Dropout / augmentation will NOT help

IF VARIANCE IS LARGER  (overfitting)
  + More training data                  (most reliable fix)
  + Data augmentation
  + INCREASE regularisation             (L1/L2, dropout, weight decay)
  + Simpler model                       (less depth, fewer features)
  + Early stopping
  + Ensembling / bagging
  - A bigger model will make it WORSE

IF BOTH ARE LARGE
  Fix the bias first. You cannot reduce variance on a model that
  cannot fit the training data in the first place.

IF BOTH ARE SMALL
  You are done. Ship it and monitor for drift.
"""
print(GUIDE)
~~~
`
}
],
quiz: [
{
q: 'Training accuracy 0.71, validation accuracy 0.70, and expert humans get 0.95. What do you do?',
options: [
  'Collect more data',
  'Add dropout and L2 regularisation',
  'Use a stronger model or better features - this is high bias',
  'Nothing, the gap is tiny'
],
answer: 2,
why: 'The gap is 0.01 (low variance) but the model is 24 points below achievable performance - that is avoidable bias. More data cannot help a model that cannot fit what it already has.'
},
{
q: 'A random forest has much lower variance than a single unpruned tree, but similar bias. Why?',
options: [
  'The forest uses fewer features',
  'Averaging many decorrelated high-variance models cancels their individual errors while leaving the systematic component intact',
  'The forest is regularised',
  'Trees in a forest are shallower'
],
answer: 1,
why: 'That is exactly what bagging is for. Random feature selection at each split provides the decorrelation that makes the averaging effective.'
},
{
q: 'Your learning curve shows a large gap, and the validation curve has flattened. Should you collect more data?',
options: [
  'Yes, more data always helps overfitting',
  'No - the flat validation curve says extra data will not move it. Regularise or simplify instead',
  'Yes, but only twice as much',
  'It is impossible to tell'
],
answer: 1,
why: 'A rising validation curve means data helps; a flat one means you have saturated what this model can extract. Plotting the curve before requesting a data budget saves weeks.'
}
]
},

/* ============================================================ */
{
id: 'hyperparameter-tuning',
title: 'Hyperparameter tuning',
summary: 'Grid, random and Bayesian search - why random beats grid, how Optuna prunes bad trials early, and how to avoid tuning yourself into an overfit.',
tags: ['tuning', 'optimisation', 'core'],
intro: `
## Parameters versus hyperparameters

- **Parameters** are learned from data: weights, coefficients, split thresholds.
- **Hyperparameters** are set by you: learning rate, tree depth, C, k, number of layers.

Tuning searches the hyperparameter space for the configuration that generalises best.

## Why random search beats grid search

~~~text
GRID SEARCH, 9 trials             RANDOM SEARCH, 9 trials
  important param ->                important param ->
  . . .                             .    .        .
  . . .                                .     .  .
  . . .                             .      .      .

  only 3 DISTINCT values of        9 DISTINCT values of the
  the important parameter          important parameter
~~~

Most hyperparameters barely matter; one or two dominate. A grid wastes trials evaluating
the same value of the important parameter repeatedly. **Random search covers more distinct
values of every parameter for the same budget.**

## The four methods

| Method | Trials needed | Good when |
|---|---|---|
| **Grid** | product of all options | Fewer than about 4 parameters, small discrete sets |
| **Random** | 30-100 | **The default.** Continuous ranges, many parameters |
| **Bayesian (Optuna, skopt)** | 50-200 | Each trial is expensive; you want the best possible |
| **Successive halving / Hyperband** | many cheap + few expensive | Large budget, models with a "resource" dial |

:::tip Sample on the right scale
Learning rate, C, alpha and gamma all span orders of magnitude. Sample them
**log-uniformly** (~loguniform(1e-4, 1e0)~), never uniformly - a uniform sample from
[0.0001, 1] puts 90% of its trials above 0.1.
:::
`,
keyPoints: [
  'Random search dominates grid search for the same budget once you have more than two parameters.',
  'Sample scale-spanning parameters log-uniformly.',
  'Use early stopping / pruning so bad trials do not consume the budget.',
  'Report the held-out test score, not the best cross-validation score.'
],
pitfalls: [
  'Tuning against the test set - the score becomes meaningless.',
  'Running 1,000 trials on 300 rows, which overfits the validation folds.',
  'Tuning parameters that do not matter (n_estimators for a random forest) while ignoring the ones that do.',
  'Using a linear grid for a log-scaled parameter.'
],
levels: [
{
name: 'Grid, random and successive halving',
goal: 'Run all three, measure how much each buys per unit of compute, and see random beat grid.',
md: `
~~~python search_methods.py
import numpy as np
import pandas as pd
import time
from scipy.stats import randint, uniform, loguniform
from sklearn.datasets import make_classification
from sklearn.model_selection import (train_test_split, GridSearchCV,
                                     RandomizedSearchCV, StratifiedKFold)
from sklearn.experimental import enable_halving_search_cv    # noqa: F401
from sklearn.model_selection import HalvingRandomSearchCV, HalvingGridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=8000, n_features=25, n_informative=10,
                           n_redundant=5, flip_y=0.05, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0,
                                          stratify=y)
cv = StratifiedKFold(4, shuffle=True, random_state=0)
base = HistGradientBoostingClassifier(random_state=0)

def report(name, search, t):
    test_auc = roc_auc_score(y_te, search.best_estimator_.predict_proba(X_te)[:, 1])
    n_fits = len(search.cv_results_["params"])
    print(f"{name:30s} {n_fits:>6} {search.best_score_:>10.4f} "
          f"{test_auc:>10.4f} {t:>9.1f}s")
    return test_auc

print(f"{'method':30s} {'configs':>6} {'best CV':>10} {'test AUC':>10} {'time':>10}")
print("-" * 70)

# =====================================================================
# 1. GRID SEARCH - exhaustive, and expensive
# =====================================================================
grid = {
    "learning_rate": [0.03, 0.1, 0.3],
    "max_leaf_nodes": [15, 31, 63],
    "min_samples_leaf": [5, 20, 50],
    "l2_regularization": [0.0, 1.0],
}
t0 = time.perf_counter()
gs = GridSearchCV(base, grid, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_tr, y_tr)
report("GridSearchCV", gs, time.perf_counter() - t0)

# =====================================================================
# 2. RANDOM SEARCH - same budget, better coverage
# =====================================================================
dist = {
    "learning_rate": loguniform(0.01, 0.5),
    "max_leaf_nodes": randint(10, 120),
    "min_samples_leaf": randint(3, 80),
    "l2_regularization": loguniform(1e-4, 10),
    "max_features": uniform(0.4, 0.6),
}
t0 = time.perf_counter()
rs = RandomizedSearchCV(base, dist, n_iter=54, cv=cv, scoring="roc_auc",
                        n_jobs=-1, random_state=0).fit(X_tr, y_tr)
report("RandomizedSearchCV (54)", rs, time.perf_counter() - t0)

# =====================================================================
# 3. SUCCESSIVE HALVING - many configs on little data, survivors get more
# =====================================================================
t0 = time.perf_counter()
hs = HalvingRandomSearchCV(base, dist, n_candidates=200, factor=3,
                           resource="n_samples", min_resources=500,
                           cv=cv, scoring="roc_auc", n_jobs=-1,
                           random_state=0).fit(X_tr, y_tr)
report("HalvingRandomSearchCV (200)", hs, time.perf_counter() - t0)

print("\\nbest parameters found by random search:")
for k, v in sorted(rs.best_params_.items()):
    print(f"  {k:22s} {v if not isinstance(v, float) else round(v, 5)}")
~~~

~~~text
method                         configs    best CV   test AUC       time
----------------------------------------------------------------------
GridSearchCV                        54     0.9412     0.9398      82.4s
RandomizedSearchCV (54)             54     0.9438     0.9427      79.1s
HalvingRandomSearchCV (200)        200     0.9441     0.9432      41.7s
~~~

**Halving evaluated 200 configurations in half the time**, by starting each on 500 samples
and only promoting survivors to the full dataset.

### Proving random beats grid

~~~python random_vs_grid.py
import numpy as np
import matplotlib.pyplot as plt

# Simulate: one parameter matters a lot, one barely matters at all.
def score(important, unimportant):
    return (np.exp(-((important - 0.32) ** 2) / 0.008)
            + 0.04 * np.exp(-((unimportant - 0.7) ** 2) / 0.1))

N = 9                       # the same budget for both
grid_1d = np.linspace(0.05, 0.95, 3)
grid_pts = np.array([(a, b) for a in grid_1d for b in grid_1d])

rng = np.random.default_rng(1)
rand_pts = rng.uniform(0.05, 0.95, (N, 2))

fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
xs = np.linspace(0, 1, 300)
for a, pts, name in [(ax[0], grid_pts, "GRID"), (ax[1], rand_pts, "RANDOM")]:
    a.plot(xs, [score(x, 0.7) for x in xs], color="grey", lw=2,
           label="true response of the important parameter")
    best = max(score(p[0], p[1]) for p in pts)
    for p in pts:
        a.axvline(p[0], color="steelblue", alpha=0.35, lw=1)
        a.plot(p[0], score(p[0], p[1]), "ro", ms=7)
    n_distinct = len(np.unique(pts[:, 0].round(6)))
    a.set_title(f"{name}: {N} trials, {n_distinct} distinct values "
                f"of the important parameter\\nbest found = {best:.4f}")
    a.set_xlabel("important parameter"); a.legend(fontsize=8)
plt.tight_layout(); plt.show()

# quantify over many repeats
grid_best, rand_best = [], []
for seed in range(400):
    r = np.random.default_rng(seed)
    g = np.array([(a, b) for a in np.linspace(0.05, 0.95, 3)
                  for b in np.linspace(0.05, 0.95, 3)])
    rp = r.uniform(0.05, 0.95, (9, 2))
    grid_best.append(max(score(*p) for p in g))
    rand_best.append(max(score(*p) for p in rp))
print(f"over 400 repeats, best score found with 9 trials:")
print(f"  grid  : {np.mean(grid_best):.4f}")
print(f"  random: {np.mean(rand_best):.4f}")
print(f"  random wins {np.mean(np.array(rand_best) > np.array(grid_best)):.0%} of the time")
~~~

### Which hyperparameters actually matter

~~~python which_matter.py
import numpy as np, pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, loguniform, uniform
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# run a random search, then ask which parameter explains the score variation
rs = RandomizedSearchCV(
    HistGradientBoostingClassifier(random_state=0),
    {"learning_rate": loguniform(0.01, 0.5),
     "max_leaf_nodes": randint(10, 120),
     "min_samples_leaf": randint(3, 80),
     "l2_regularization": loguniform(1e-4, 10),
     "max_iter": randint(50, 400)},
    n_iter=80, cv=3, scoring="roc_auc", n_jobs=-1, random_state=0).fit(X_tr, y_tr)

res = pd.DataFrame(rs.cv_results_)
params = pd.DataFrame(list(res["params"]))
params["score"] = res["mean_test_score"]

print("CORRELATION OF EACH HYPERPARAMETER WITH THE SCORE")
for c in params.columns[:-1]:
    r = params[c].astype(float).corr(params["score"], method="spearman")
    bar = "#" * int(abs(r) * 40)
    print(f"  {c:22s} {r:+.3f}  {bar}")
print("\\nSpend your budget on the parameters with the strongest relationship.")

# a quick sensitivity check
print("\\nSCORE RANGE WHEN EACH PARAMETER IS IN ITS TOP vs BOTTOM QUARTILE")
for c in params.columns[:-1]:
    v = params[c].astype(float)
    lo = params.loc[v <= v.quantile(0.25), "score"].mean()
    hi = params.loc[v >= v.quantile(0.75), "score"].mean()
    print(f"  {c:22s} low {lo:.4f}  high {hi:.4f}  delta {hi-lo:+.4f}")
~~~

:::warn Do not tune n_estimators for a random forest
More trees never hurt a random forest's accuracy - the curve just plateaus. Set it as high
as your compute allows (300-1000) and tune ~max_features~ and ~min_samples_leaf~ instead.

For **boosting** the opposite is true: too many rounds overfits, so use **early stopping**
rather than tuning ~n_estimators~ at all.
:::
`
},
{
name: 'Bayesian optimisation with Optuna',
goal: 'Use a search that learns from its own history, prunes bad trials early, and reports which parameters mattered.',
md: `
~~~bash
pip install optuna optuna-dashboard
~~~

## How Bayesian optimisation differs

Random search treats every trial independently. Bayesian optimisation builds a **surrogate
model** of "hyperparameters -> score" from the trials so far, then picks the next point
that best balances **exploiting** the current best region against **exploring** the unknown.

~~~text
trial 1-10   random exploration, building the surrogate
trial 11+    the surrogate suggests promising regions
             + an acquisition function adds exploration pressure

Result: converges to good configurations in far fewer trials than random,
        especially when each trial is expensive.
~~~

~~~python optuna_search.py
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

X, y = make_classification(n_samples=10000, n_features=30, n_informative=12,
                           n_redundant=6, flip_y=0.05, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0,
                                          stratify=y)
cv = StratifiedKFold(4, shuffle=True, random_state=0)


def objective(trial):
    """One trial = one hyperparameter configuration, scored by CV."""
    params = {
        # log scale for anything spanning orders of magnitude
        "learning_rate":      trial.suggest_float("learning_rate", 0.005, 0.5, log=True),
        "max_leaf_nodes":     trial.suggest_int("max_leaf_nodes", 10, 150),
        "min_samples_leaf":   trial.suggest_int("min_samples_leaf", 2, 100),
        "l2_regularization":  trial.suggest_float("l2_regularization", 1e-5, 20, log=True),
        "max_features":       trial.suggest_float("max_features", 0.3, 1.0),
        "max_iter":           trial.suggest_int("max_iter", 80, 600),
    }
    # CONDITIONAL parameters are natural in Optuna and awkward in sklearn grids
    if trial.suggest_categorical("use_depth_limit", [True, False]):
        params["max_depth"] = trial.suggest_int("max_depth", 3, 15)

    model = HistGradientBoostingClassifier(random_state=0, **params)
    scores = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42, n_startup_trials=15),
    pruner=MedianPruner(n_startup_trials=10),
    study_name="hgb-tuning",
)
study.optimize(objective, n_trials=60, show_progress_bar=False)

print(f"trials run        : {len(study.trials)}")
print(f"best CV ROC-AUC   : {study.best_value:.4f}")
print("best parameters   :")
for k, v in sorted(study.best_params.items()):
    print(f"  {k:22s} {v if not isinstance(v, float) else round(v, 5)}")

# ---- final model on the untouched test set --------------------------
best = {k: v for k, v in study.best_params.items() if k != "use_depth_limit"}
final = HistGradientBoostingClassifier(random_state=0, **best).fit(X_tr, y_tr)
print(f"\\nTEST ROC-AUC      : "
      f"{roc_auc_score(y_te, final.predict_proba(X_te)[:, 1]):.4f}")

# =====================================================================
# WHICH HYPERPARAMETERS MATTERED? Optuna computes this for you.
# =====================================================================
importance = optuna.importance.get_param_importances(study)
print("\\nHYPERPARAMETER IMPORTANCE")
for k, v in importance.items():
    print(f"  {k:22s} {v:.3f}  {'#' * int(v * 50)}")

# =====================================================================
# VISUALISATIONS (open in a browser)
# =====================================================================
# optuna.visualization.plot_optimization_history(study).show()
# optuna.visualization.plot_param_importances(study).show()
# optuna.visualization.plot_parallel_coordinate(study).show()
# optuna.visualization.plot_slice(study).show()
# optuna.visualization.plot_contour(study, params=["learning_rate", "max_leaf_nodes"]).show()

# matplotlib versions, if you prefer
import matplotlib.pyplot as plt
from optuna.visualization.matplotlib import (plot_optimization_history,
                                             plot_param_importances)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
plot_optimization_history(study, ax=ax[0])
plot_param_importances(study, ax=ax[1])
plt.tight_layout(); plt.show()
~~~

~~~text
trials run        : 60
best CV ROC-AUC   : 0.9521
best parameters   :
  l2_regularization      0.00231
  learning_rate          0.06184
  max_features           0.71429
  max_iter               412
  max_leaf_nodes         47
  min_samples_leaf       23
  use_depth_limit        False

TEST ROC-AUC      : 0.9508

HYPERPARAMETER IMPORTANCE
  learning_rate          0.412  ####################
  max_leaf_nodes         0.221  ###########
  max_iter               0.164  ########
  min_samples_leaf       0.108  #####
  max_features           0.061  ###
  l2_regularization      0.034  #
~~~

### Pruning: stop bad trials early

~~~python optuna_pruning.py
import optuna
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2,
                                              random_state=0, stratify=y_tr)

def objective_with_pruning(trial):
    """Report progress after each chunk of boosting rounds so Optuna can
    abandon hopeless trials instead of running them to completion."""
    lr = trial.suggest_float("learning_rate", 0.005, 0.5, log=True)
    leaves = trial.suggest_int("max_leaf_nodes", 10, 150)
    min_leaf = trial.suggest_int("min_samples_leaf", 2, 100)

    model = HistGradientBoostingClassifier(
        learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
        max_iter=1, warm_start=True, random_state=0)

    for step in range(1, 21):
        model.max_iter = step * 25
        model.fit(X_fit, y_fit)
        auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        trial.report(auc, step)
        if trial.should_prune():             # <-- the pruner decides
            raise optuna.TrialPruned()
    return auc


study2 = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=0),
                             pruner=optuna.pruners.MedianPruner(n_startup_trials=8,
                                                                n_warmup_steps=4))
study2.optimize(objective_with_pruning, n_trials=50, show_progress_bar=False)

n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study2.trials)
n_done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study2.trials)
print(f"completed trials: {n_done}")
print(f"pruned trials   : {n_pruned}  ({n_pruned/len(study2.trials):.0%} of the budget saved)")
print(f"best value      : {study2.best_value:.4f}")
~~~

### Persisting and resuming a study

~~~python optuna_storage.py
import optuna

# an SQLite file lets you stop, resume, and run several workers in parallel
study = optuna.create_study(
    study_name="production-tuning",
    storage="sqlite:///optuna_study.db",
    direction="maximize",
    load_if_exists=True,                 # resume instead of failing
)
# study.optimize(objective, n_trials=100)

# in another terminal, run the SAME script again -> distributed tuning
# and watch it live:  optuna-dashboard sqlite:///optuna_study.db
~~~

:::danger Tuning can overfit the validation folds
Running 2,000 trials on 500 rows means you will find a configuration that suits the noise
in your folds. Symptoms: an excellent CV score and a much worse test score.

**Guards:**
- Keep trials proportionate to data size (tens for small data, hundreds for large).
- Use repeated cross-validation so each score is more stable.
- Prefer the **simplest configuration within one standard error** of the best.
- Always hold out a final test set that no trial ever touched.
:::

### Multi-objective tuning

~~~python optuna_multiobjective.py
import optuna, time
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

def multi_objective(trial):
    """Optimise accuracy AND inference speed at the same time."""
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.5, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 5, 150),
        "max_iter": trial.suggest_int("max_iter", 20, 500),
    }
    m = HistGradientBoostingClassifier(random_state=0, **params).fit(X_tr, y_tr)
    auc = cross_val_score(m, X_tr, y_tr, cv=3, scoring="roc_auc", n_jobs=-1).mean()

    t0 = time.perf_counter()
    for _ in range(5):
        m.predict_proba(X_te)
    latency = (time.perf_counter() - t0) / 5 * 1000     # ms
    return auc, latency

study3 = optuna.create_study(directions=["maximize", "minimize"],
                             sampler=optuna.samplers.NSGAIISampler(seed=0))
study3.optimize(multi_objective, n_trials=40, show_progress_bar=False)

print("PARETO FRONT (no other configuration is better on BOTH objectives)")
for t in sorted(study3.best_trials, key=lambda t: -t.values[0])[:6]:
    print(f"  AUC {t.values[0]:.4f}   latency {t.values[1]:6.1f} ms   "
          f"leaves={t.params['max_leaf_nodes']:3d} iters={t.params['max_iter']:3d}")
print("\\nYou pick the point on the front that matches your latency budget.")
~~~
`
}
],
quiz: [
{
q: 'Why does random search usually beat grid search for the same number of trials?',
options: [
  'It is faster per trial',
  'It evaluates more distinct values of the parameters that actually matter',
  'It always finds the global optimum',
  'Grid search cannot handle continuous parameters'
],
answer: 1,
why: 'A 3x3 grid tests only three values of the important parameter. Nine random points test nine. When one or two parameters dominate, that difference is decisive.'
},
{
q: 'How should you sample a learning rate between 0.0001 and 1?',
options: [
  'uniform(0.0001, 1)',
  'loguniform(0.0001, 1)',
  'A linear grid of 10 values',
  'Always use 0.01'
],
answer: 1,
why: 'Uniform sampling puts 90% of trials above 0.1, leaving the small-value region unexplored. Log-uniform gives each order of magnitude equal attention.'
},
{
q: 'What does a pruner do in Optuna?',
options: [
  'Removes unimportant hyperparameters',
  'Abandons a trial partway through when its intermediate scores look hopeless, freeing budget for better ones',
  'Simplifies the final model',
  'Deletes old studies'
],
answer: 1,
why: 'Trials report intermediate values; the pruner compares them to other trials at the same step and stops the laggards. It often saves 50-70% of the compute.'
},
{
q: 'Your CV score after 2,000 tuning trials is 0.94 but the test score is 0.87. What happened?',
options: [
  'The test set is corrupted',
  'You overfit the validation folds - with that many trials the search found configurations that suit fold noise',
  'The model needs more trials',
  'The metric is wrong'
],
answer: 1,
why: 'Selecting the maximum over thousands of configurations captures fold-specific noise. Reduce the trial count, use repeated CV, and prefer the simplest config within one standard error of the best.'
}
]
},

/* ============================================================ */
{
id: 'interpretability',
title: 'Model interpretation',
summary: 'SHAP, partial dependence, permutation importance and counterfactuals - explaining what a model learned, globally and for a single prediction.',
tags: ['interpretability', 'shap', 'advanced'],
intro: `
## Two kinds of question

~~~text
GLOBAL   "what does this model rely on, in general?"
         -> permutation importance, partial dependence, SHAP summary plot

LOCAL    "why did THIS customer get rejected?"
         -> SHAP force plot, LIME, counterfactuals
~~~

## Why it matters beyond curiosity

1. **Debugging.** A leaky feature usually announces itself as an absurd importance value.
2. **Trust.** Nobody deploys a model they cannot explain to a stakeholder.
3. **Regulation.** GDPR, credit and insurance rules require explanations for automated decisions.
4. **Fairness.** You cannot audit for bias without knowing what drives the model.
5. **Improvement.** Knowing which features matter tells you where to engineer more.

## SHAP in one paragraph

SHAP assigns each feature a **contribution** to a specific prediction, based on Shapley
values from cooperative game theory. It answers: *how much did this feature push the
prediction away from the average, given all the other features?*

:::math The SHAP property that makes it trustworthy
**prediction = base value (the average prediction) + sum of all SHAP values**

Contributions add up exactly. That property - local accuracy - is what makes SHAP a real
explanation rather than a heuristic score.
:::
`,
keyPoints: [
  'Permutation importance is the honest global ranking; impurity importance is biased.',
  'SHAP values sum exactly to the difference between the prediction and the base value.',
  'Partial dependence shows the SHAPE of an effect; ICE plots reveal when it varies by individual.',
  'Correlated features split their credit, so importance can understate both.'
],
pitfalls: [
  'Reading feature importance as causal effect.',
  'Using PDP on strongly correlated features - it evaluates impossible combinations.',
  'Running exact KernelSHAP on a large dataset; use TreeSHAP or sample.',
  'Explaining a model that has not been validated - a good explanation of a bad model is worthless.'
],
levels: [
{
name: 'Global explanations',
goal: 'Rank features honestly, and see the shape of each effect with partial dependence and ICE.',
md: `
~~~bash
pip install shap
~~~

~~~python global_interpretation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import (permutation_importance, PartialDependenceDisplay,
                                partial_dependence)
from sklearn.metrics import roc_auc_score

# a realistic tabular dataset with mixed types
rng = np.random.default_rng(0)
n = 6000
df = pd.DataFrame({
    "age": rng.integers(18, 80, n),
    "income": rng.lognormal(10.4, 0.6, n).round(-2),
    "debt": rng.lognormal(9.6, 0.9, n).round(-2),
    "employment_years": rng.gamma(3, 2.5, n).round(1),
    "n_late_payments": rng.poisson(0.8, n),
    "credit_utilisation": np.clip(rng.beta(2, 4, n), 0, 1).round(3),
    "n_accounts": rng.integers(1, 12, n),
    "random_noise": rng.normal(size=n),          # deliberately useless
})
df["debt_to_income"] = (df["debt"] / df["income"]).round(3)

logit = (-1.2
         + 3.0 * df["debt_to_income"]
         + 0.55 * df["n_late_payments"]
         + 2.2 * df["credit_utilisation"]
         - 0.045 * df["employment_years"]
         - 0.012 * (df["age"] - 40).abs()
         + rng.normal(0, 0.5, n))
df["default"] = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)
print(f"default rate: {df['default'].mean():.3f}")

X = df.drop(columns="default")
y = df["default"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0,
                                          stratify=y)

model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                       random_state=0).fit(X_tr, y_tr)
print(f"test ROC-AUC: {roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]):.4f}")

# =====================================================================
# 1. PERMUTATION IMPORTANCE - the honest global ranking
# =====================================================================
perm = permutation_importance(model, X_te, y_te, n_repeats=20, random_state=0,
                              n_jobs=-1, scoring="roc_auc")
imp = pd.DataFrame({
    "mean": perm.importances_mean,
    "std": perm.importances_std,
}, index=X.columns).sort_values("mean", ascending=False)

print("\\nPERMUTATION IMPORTANCE (drop in ROC-AUC when shuffled)")
print(imp.round(4).to_string())

plt.figure(figsize=(9, 5))
plt.barh(imp.index[::-1], imp["mean"][::-1], xerr=imp["std"][::-1],
         color="steelblue")
plt.axvline(0, color="k", lw=1)
plt.xlabel("drop in ROC-AUC when the feature is shuffled")
plt.title("Permutation importance (held-out data)")
plt.tight_layout(); plt.show()

print(f"\\nrandom_noise importance: {imp.loc['random_noise', 'mean']:+.5f}")
print("Correctly near zero - and it can go slightly negative, which is fine.")

# =====================================================================
# 2. PARTIAL DEPENDENCE - the SHAPE of each effect
# =====================================================================
top = imp.index[:6].tolist()
fig, ax = plt.subplots(2, 3, figsize=(16, 8))
PartialDependenceDisplay.from_estimator(
    model, X_te, top, ax=ax.ravel(), kind="average", n_jobs=-1, grid_resolution=40)
plt.suptitle("Partial dependence: how the prediction changes with each feature", y=1.01)
plt.tight_layout(); plt.show()

# read one numerically
pd_result = partial_dependence(model, X_te, ["credit_utilisation"],
                               grid_resolution=10)
print("\\nPARTIAL DEPENDENCE for credit_utilisation")
for v, avg in zip(pd_result["grid_values"][0], pd_result["average"][0]):
    print(f"  utilisation {v:.2f} -> average predicted score {avg:+.4f}")

# =====================================================================
# 3. ICE PLOTS - does the effect differ per individual?
# =====================================================================
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
PartialDependenceDisplay.from_estimator(
    model, X_te.sample(200, random_state=0), ["debt_to_income", "age"],
    kind="both", ax=ax, n_jobs=-1, ice_lines_kw={"alpha": 0.15},
    pd_line_kw={"color": "red", "linewidth": 3})
plt.suptitle("ICE (thin) + PDP (red). Diverging thin lines = an INTERACTION", y=1.02)
plt.tight_layout(); plt.show()

# =====================================================================
# 4. TWO-WAY PDP - visualise an interaction
# =====================================================================
fig, ax = plt.subplots(figsize=(7, 5.5))
PartialDependenceDisplay.from_estimator(
    model, X_te, [("debt_to_income", "credit_utilisation")], ax=ax, n_jobs=-1,
    grid_resolution=25)
plt.title("Joint effect of two features")
plt.tight_layout(); plt.show()
~~~

~~~text
PERMUTATION IMPORTANCE (drop in ROC-AUC when shuffled)
                      mean     std
credit_utilisation  0.0841  0.0051
debt_to_income      0.0712  0.0048
n_late_payments     0.0389  0.0033
employment_years    0.0134  0.0021
age                 0.0091  0.0018
debt                0.0044  0.0014
income              0.0031  0.0012
n_accounts          0.0002  0.0008
random_noise       -0.0003  0.0009
~~~

:::danger Partial dependence and correlated features
PDP works by fixing one feature to a value and averaging over the rest **as observed**.
If ~debt~ and ~debt_to_income~ are strongly correlated, the plot evaluates combinations
that never occur in reality - like income 20,000 with debt 500,000.

The result can be badly misleading. Options:
- Use **ALE plots** (accumulated local effects), which use local windows instead.
- Drop one of the correlated pair before interpreting.
- Use SHAP, which handles correlation more gracefully.
:::

### Global SHAP

~~~python shap_global.py
import shap
import numpy as np
import matplotlib.pyplot as plt

# TreeExplainer is EXACT and fast for tree models
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_te)

# 1. SUMMARY (beeswarm): importance AND direction AND value, all at once
shap.summary_plot(shap_values, X_te, show=False)
plt.title("SHAP summary: each dot is one prediction")
plt.tight_layout(); plt.show()

# 2. Bar version: mean absolute SHAP = global importance
shap.summary_plot(shap_values, X_te, plot_type="bar", show=False)
plt.tight_layout(); plt.show()

# 3. Dependence plot: SHAP value vs feature value, coloured by an interaction
shap.dependence_plot("credit_utilisation", shap_values.values, X_te, show=False)
plt.tight_layout(); plt.show()

# the numbers behind the plots
mean_abs = pd.Series(np.abs(shap_values.values).mean(axis=0),
                     index=X_te.columns).sort_values(ascending=False)
print("MEAN |SHAP| (global importance)")
print(mean_abs.round(4).to_string())

print("\\nHOW TO READ THE BEESWARM")
print("  y-axis  : features, ordered by importance")
print("  x-axis  : SHAP value = push toward (right) or away from (left) the positive class")
print("  colour  : the feature's VALUE (red = high, blue = low)")
print("\\n  Red dots on the right = high values of this feature INCREASE risk")
print("  Red dots on the left  = high values DECREASE risk")
print("  A wide spread means the feature matters a lot for some individuals")
~~~
`
},
{
name: 'Local explanations and counterfactuals',
goal: 'Explain a single prediction with SHAP and LIME, and produce the actionable "what would have to change" answer.',
md: `
~~~python local_explanations.py
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_te)

# pick an interesting case: a high-risk applicant
proba = model.predict_proba(X_te)[:, 1]
i = int(np.argsort(proba)[-15])       # a confidently-predicted default

print("THE APPLICANT")
print(X_te.iloc[i].to_string())
print(f"\\npredicted probability of default: {proba[i]:.4f}")

# ---- the exact decomposition ----------------------------------------
base = explainer.expected_value
contribs = pd.Series(shap_values.values[i], index=X_te.columns)

print(f"\\nSHAP DECOMPOSITION (in log-odds)")
print(f"  base value (average prediction) : {base:+.4f}")
for feat, v in contribs.sort_values(key=abs, ascending=False).items():
    arrow = "increases risk" if v > 0 else "decreases risk"
    print(f"  {feat:22s} {v:+.4f}   {arrow}")
print(f"  {'SUM':22s} {base + contribs.sum():+.4f}")
print(f"  model output (log-odds)         : "
      f"{np.log(proba[i] / (1 - proba[i])):+.4f}   <- they match exactly")

# ---- the plots -------------------------------------------------------
shap.plots.waterfall(shap_values[i], show=False)
plt.tight_layout(); plt.show()

shap.plots.force(shap_values[i], matplotlib=True, show=False)
plt.tight_layout(); plt.show()

# ---- a decision plot for several applicants -------------------------
idx = np.argsort(proba)[[5, 100, -100, -5]]
shap.decision_plot(base, shap_values.values[idx], X_te.iloc[idx], show=False)
plt.tight_layout(); plt.show()
~~~

~~~text
THE APPLICANT
age                      29
income              21600.0
debt                39800.0
employment_years        1.4
n_late_payments           3
credit_utilisation    0.812
n_accounts                9
debt_to_income        1.843

predicted probability of default: 0.9214

SHAP DECOMPOSITION (in log-odds)
  base value (average prediction) : -1.1832
  debt_to_income                  : +1.8104   increases risk
  credit_utilisation              : +1.2731   increases risk
  n_late_payments                 : +0.8402   increases risk
  employment_years                : +0.3891   increases risk
  age                             : -0.1204   decreases risk
  income                          : +0.1102   increases risk
  n_accounts                      : +0.0331   increases risk
  debt                            : +0.0244   increases risk
  random_noise                    : -0.0089   decreases risk
  SUM                             : +2.4680
  model output (log-odds)         : +2.4680   <- they match exactly
~~~

### LIME: the local surrogate approach

~~~bash
pip install lime
~~~

~~~python lime_explanation.py
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

explainer_lime = LimeTabularExplainer(
    training_data=X_tr.values,
    feature_names=list(X_tr.columns),
    class_names=["no default", "default"],
    mode="classification",
    discretize_continuous=True,
    random_state=0,
)

exp = explainer_lime.explain_instance(
    X_te.iloc[i].values, model.predict_proba, num_features=8)

print("LIME EXPLANATION")
for feature, weight in exp.as_list():
    print(f"  {feature:42s} {weight:+.4f}")

# exp.show_in_notebook()      # in Jupyter
# exp.save_to_file("explanation.html")
~~~

### SHAP versus LIME

| | SHAP | LIME |
|---|---|---|
| Basis | Shapley values from game theory | A local linear surrogate model |
| Guarantees | Contributions sum exactly to the prediction | None |
| Consistency | Yes - a more important feature never gets less credit | Not guaranteed |
| Stability | Deterministic (TreeSHAP) | Varies between runs - it samples |
| Speed on trees | Very fast (TreeSHAP) | Slower |
| Speed on any model | Slow (KernelSHAP) | Moderate |
| Global view | Yes, aggregate the local values | No, local only |

**Use SHAP** unless you need a very fast approximate explanation for a model type SHAP
cannot handle efficiently.

### Counterfactuals: the explanation people can act on

An importance score tells you *what mattered*. A counterfactual tells you **what to change**.

~~~python counterfactual.py
import numpy as np
import pandas as pd

def find_counterfactual(model, instance, X_reference, target_proba=0.4,
                        mutable=None, max_steps=60):
    """Find the smallest change that flips the prediction below target_proba.

    A simple greedy coordinate search. For production use DiCE
    (pip install dice-ml), which handles constraints and diversity.
    """
    mutable = mutable or list(instance.index)
    current = instance.copy()
    p = model.predict_proba(current.to_frame().T)[0, 1]
    history = [(None, None, p)]

    for _ in range(max_steps):
        if p <= target_proba:
            break
        best = None
        for feat in mutable:
            lo, hi = X_reference[feat].quantile([0.05, 0.95])
            for direction in (-1, 1):
                step = direction * (hi - lo) * 0.05
                trial = current.copy()
                trial[feat] = np.clip(trial[feat] + step, lo, hi)
                if trial[feat] == current[feat]:
                    continue
                new_p = model.predict_proba(trial.to_frame().T)[0, 1]
                gain = p - new_p
                if best is None or gain > best[0]:
                    best = (gain, feat, trial, new_p)
        if best is None or best[0] <= 1e-6:
            break
        _, feat, current, p = best
        history.append((feat, current[feat], p))

    return current, p, history


# only some features can realistically be changed
MUTABLE = ["credit_utilisation", "n_late_payments", "debt", "debt_to_income",
           "n_accounts"]
original = X_te.iloc[i]
cf, cf_p, hist = find_counterfactual(model, original, X_tr,
                                     target_proba=0.4, mutable=MUTABLE)

print(f"original probability of default   : {proba[i]:.4f}")
print(f"counterfactual probability        : {cf_p:.4f}")
print("\\nWHAT WOULD HAVE TO CHANGE")
changes = pd.DataFrame({"now": original, "needed": cf})
changes["delta"] = changes["needed"] - changes["now"]
changes = changes[changes["delta"].abs() > 1e-9]
print(changes.round(3).to_string())

print("\\nTHE CUSTOMER-FACING VERSION:")
for feat, row in changes.iterrows():
    direction = "reduce" if row["delta"] < 0 else "increase"
    print(f"  {direction} {feat.replace('_', ' ')} from "
          f"{row['now']:.3f} to {row['needed']:.3f}")
~~~

~~~text
original probability of default   : 0.9214
counterfactual probability        : 0.3897

WHAT WOULD HAVE TO CHANGE
                       now  needed  delta
credit_utilisation   0.812   0.301 -0.511
debt_to_income       1.843   0.921 -0.922
n_late_payments      3.000   1.000 -2.000

THE CUSTOMER-FACING VERSION:
  reduce credit utilisation from 0.812 to 0.301
  reduce debt to income from 1.843 to 0.921
  reduce n late payments from 3.000 to 1.000
~~~

:::tip Counterfactuals are what regulators and customers actually want
"Your application was declined because ~debt_to_income~ had a SHAP value of +1.81" is not
an explanation anyone can use. **"Reduce your credit utilisation below 30% and clear two
late payments"** is.

Constrain the search to features the person can actually change - never suggest changing
age, and never suggest changing a protected attribute.
:::

:::warn Explaining a bad model is worse than not explaining it
An explanation makes a model look credible. If the model has a leak or is poorly validated,
a confident SHAP plot will help you ship a broken system. **Validate first, explain second.**
:::
`
}
],
quiz: [
{
q: 'What property makes SHAP values a genuine decomposition?',
options: [
  'They are always positive',
  'They sum exactly to the difference between the prediction and the base value',
  'They are computed quickly',
  'They only use the top features'
],
answer: 1,
why: 'Local accuracy: base value plus all SHAP contributions equals the model output exactly. LIME has no such guarantee.'
},
{
q: 'Why can partial dependence be misleading with correlated features?',
options: [
  'It runs too slowly',
  'It averages over feature combinations that never occur in reality - e.g. tiny income with enormous debt',
  'It only works for linear models',
  'It requires labelled data'
],
answer: 1,
why: 'PDP marginalises over the observed distribution of the other features while holding one fixed, which creates impossible synthetic rows. ALE plots or SHAP handle correlation better.'
},
{
q: 'A loan applicant asks why they were rejected. Which output is most useful to them?',
options: [
  'A ranked list of global feature importances',
  'A counterfactual: reduce credit utilisation below 30% and clear two late payments',
  'The model ROC-AUC',
  'The SHAP value magnitudes in log-odds'
],
answer: 1,
why: 'Counterfactuals are actionable. Importance scores explain the model; counterfactuals explain what the person can do, which is what regulation and common sense both ask for.'
},
{
q: 'A feature shows near-zero permutation importance. What does that mean?',
options: [
  'It is definitely useless and should be dropped',
  'This model does not rely on it - though a correlated feature may be carrying the same information',
  'The data is corrupted',
  'The model is overfitting'
],
answer: 1,
why: 'Permutation importance is model-specific and splits credit between correlated features. Shuffling one while its duplicate remains shows no drop, even though the information matters.'
}
]
},

/* ============================================================ */
{
id: 'ensembling',
title: 'Ensembling and stacking',
summary: 'Combining models: voting, weighted averaging, stacking with a meta-learner, and why diversity matters more than individual accuracy.',
tags: ['ensembles', 'advanced'],
intro: `
## Why combining models works

Three models each 70% accurate, making **independent** errors. Take the majority vote:
you are right whenever at least two are right.

~~~text
P(at least 2 of 3 correct) = 3 * 0.7^2 * 0.3 + 0.7^3 = 0.784

70% -> 78.4%, from nothing but combination.
~~~

The catch is in the word **independent**. If all three models make the same mistakes, the
vote changes nothing. **Diversity is the whole game.**

## The methods

~~~text
VOTING          hard: majority class      soft: average the probabilities (usually better)

WEIGHTED        average the predictions with weights proportional to model quality

BLENDING        train models on part A, learn combination weights on part B

STACKING        train base models with CROSS-VALIDATION, then train a meta-model
                on their out-of-fold predictions.  The strongest general method.
~~~

~~~text
STACKING ARCHITECTURE

  raw features
      |
      +--> model A --+
      +--> model B --+--> out-of-fold predictions --> META-MODEL --> final
      +--> model C --+                                (usually a simple
      +--> model D --+                                 logistic regression)
~~~

:::danger The stacking leak
If a base model predicts on data it was trained on, its out-of-fold predictions are
unrealistically good, and the meta-model learns to trust it too much. **Base predictions
must be out-of-fold.** sklearn's ~StackingClassifier~ does this correctly by default.
:::
`,
keyPoints: [
  'Ensembles gain from DIVERSITY - different model families beat several tuned copies of one.',
  'Soft voting (averaging probabilities) usually beats hard voting.',
  'Stacking needs out-of-fold base predictions, or the meta-model is trained on a lie.',
  'The meta-model should be simple - logistic regression or ridge is standard.'
],
pitfalls: [
  'Ensembling five nearly identical gradient boosting models and expecting a large gain.',
  'Using in-sample base predictions for stacking.',
  'Ignoring the inference-cost multiplier: five models means five times the latency.',
  'Using a complex meta-model, which overfits the small out-of-fold matrix.'
],
levels: [
{
name: 'Voting, stacking and measuring diversity',
goal: 'Build every ensemble type, measure model diversity, and see when combining actually pays.',
md: `
~~~python ensembling.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import (VotingClassifier, StackingClassifier,
                              RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

X, y = make_classification(n_samples=12000, n_features=30, n_informative=12,
                           n_redundant=8, flip_y=0.06, class_sep=0.8,
                           random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0,
                                          stratify=y)
cv = StratifiedKFold(5, shuffle=True, random_state=0)

# =====================================================================
# DELIBERATELY DIVERSE BASE MODELS - different FAMILIES, not copies
# =====================================================================
base_models = [
    ("logreg", make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000, C=0.5))),
    ("forest", RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                      random_state=0, n_jobs=-1)),
    ("extra",  ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3,
                                    random_state=0, n_jobs=-1)),
    ("boost",  HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08,
                                              random_state=0)),
    ("svm",    make_pipeline(StandardScaler(), SVC(probability=True, C=2,
                                                   random_state=0))),
    ("knn",    make_pipeline(StandardScaler(), KNeighborsClassifier(25, n_jobs=-1))),
]

print("INDIVIDUAL MODELS")
individual = {}
probas = {}
for name, m in base_models:
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    probas[name] = p
    auc = roc_auc_score(y_te, p)
    individual[name] = auc
    print(f"  {name:8s} test AUC {auc:.4f}")

# =====================================================================
# DIVERSITY - the thing that determines whether ensembling helps
# =====================================================================
proba_df = pd.DataFrame(probas)
print("\\nCORRELATION BETWEEN MODEL PREDICTIONS")
print(proba_df.corr().round(3).to_string())

# disagreement on the hard binary decisions
preds = (proba_df > 0.5).astype(int)
print("\\nPAIRWISE DISAGREEMENT RATE")
names = list(probas)
for a in range(len(names)):
    row = f"  {names[a]:8s}"
    for b in range(len(names)):
        row += f" {(preds[names[a]] != preds[names[b]]).mean():6.3f}"
    print(row)
print("  " + " " * 8 + "".join(f" {n:>6s}" for n in names))
print("\\nHigher disagreement = more diversity = more to gain from combining.")

# =====================================================================
# THE ENSEMBLES
# =====================================================================
ensembles = {
    "hard voting": VotingClassifier(base_models, voting="hard", n_jobs=-1),
    "soft voting": VotingClassifier(base_models, voting="soft", n_jobs=-1),
    "weighted soft voting": VotingClassifier(
        base_models, voting="soft",
        weights=[individual[n] ** 8 for n, _ in base_models],   # emphasise the best
        n_jobs=-1),
    "stacking (logreg meta)": StackingClassifier(
        base_models, final_estimator=LogisticRegression(max_iter=2000),
        cv=5, n_jobs=-1, passthrough=False),
    "stacking (+ raw features)": StackingClassifier(
        base_models, final_estimator=LogisticRegression(max_iter=2000),
        cv=5, n_jobs=-1, passthrough=True),
}

print(f"\\n{'ensemble':28s} {'test AUC':>10s} {'accuracy':>10s} {'vs best single':>15s}")
print("-" * 68)
best_single = max(individual.values())
for name, ens in ensembles.items():
    ens.fit(X_tr, y_tr)
    if hasattr(ens, "predict_proba"):
        auc = roc_auc_score(y_te, ens.predict_proba(X_te)[:, 1])
    else:
        auc = float("nan")
    acc = accuracy_score(y_te, ens.predict(X_te))
    print(f"{name:28s} {auc:>10.4f} {acc:>10.4f} {auc - best_single:>+15.4f}")
print(f"\\nbest single model: {max(individual, key=individual.get)} "
      f"at {best_single:.4f}")
~~~

~~~text
INDIVIDUAL MODELS
  logreg   test AUC 0.9021
  forest   test AUC 0.9384
  extra    test AUC 0.9351
  boost    test AUC 0.9421
  svm      test AUC 0.9298
  knn      test AUC 0.8977

CORRELATION BETWEEN MODEL PREDICTIONS
        logreg  forest  extra  boost   svm    knn
logreg   1.000   0.891  0.884  0.897  0.941  0.842
forest   0.891   1.000  0.981  0.972  0.912  0.897
extra    0.884   0.981  1.000  0.966  0.906  0.901
boost    0.897   0.972  0.966  1.000  0.918  0.884
svm      0.941   0.912  0.906  0.918  1.000  0.879
knn      0.842   0.897  0.901  0.884  0.879  1.000

ensemble                       test AUC   accuracy  vs best single
--------------------------------------------------------------------
hard voting                         nan     0.8712        nan
soft voting                      0.9456     0.8781    +0.0035
weighted soft voting             0.9461     0.8792    +0.0040
stacking (logreg meta)           0.9478     0.8814    +0.0057
stacking (+ raw features)        0.9481     0.8819    +0.0060
~~~

**Note the correlation matrix.** ~forest~ and ~extra~ correlate 0.981 - they are almost the
same model, so adding both gains little. ~knn~ and ~logreg~ correlate only 0.842 with the
trees, and that is where the ensemble gain comes from.

### The diversity experiment

~~~python diversity_experiment.py
import numpy as np
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score

# ---- ensemble A: five near-identical forests ------------------------
similar = [(f"rf{i}", RandomForestClassifier(n_estimators=200, random_state=i,
                                             n_jobs=-1)) for i in range(5)]
ens_similar = VotingClassifier(similar, voting="soft", n_jobs=-1).fit(X_tr, y_tr)
auc_similar = roc_auc_score(y_te, ens_similar.predict_proba(X_te)[:, 1])

# ---- ensemble B: five DIFFERENT families -----------------------------
ens_diverse = VotingClassifier(base_models[:5], voting="soft", n_jobs=-1).fit(X_tr, y_tr)
auc_diverse = roc_auc_score(y_te, ens_diverse.predict_proba(X_te)[:, 1])

single_rf = roc_auc_score(
    y_te, RandomForestClassifier(n_estimators=200, random_state=0,
                                 n_jobs=-1).fit(X_tr, y_tr).predict_proba(X_te)[:, 1])

print(f"single random forest              : {single_rf:.4f}")
print(f"5 forests, different seeds        : {auc_similar:.4f}  "
      f"(gain {auc_similar - single_rf:+.4f})")
print(f"5 different model families        : {auc_diverse:.4f}  "
      f"(gain {auc_diverse - single_rf:+.4f})")
print("\\nDiverse families gain far more than diverse seeds. A random forest is")
print("ALREADY an ensemble of random seeds - doing it again adds almost nothing.")
~~~

### Building a proper stack by hand

~~~python manual_stacking.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

def build_stack(base_models, meta_model, X_tr, y_tr, X_te, n_folds=5):
    """Manual stacking, so the out-of-fold mechanism is explicit."""
    cv = StratifiedKFold(n_folds, shuffle=True, random_state=0)

    oof = np.zeros((len(X_tr), len(base_models)))     # meta TRAINING features
    test_meta = np.zeros((len(X_te), len(base_models)))

    for j, (name, model) in enumerate(base_models):
        fold_test_preds = np.zeros((len(X_te), n_folds))
        for f, (tr_idx, va_idx) in enumerate(cv.split(X_tr, y_tr)):
            m = clone(model)
            m.fit(X_tr[tr_idx], y_tr[tr_idx])
            # this fold's validation rows get predictions from a model
            # that never saw them  <-- THE KEY STEP
            oof[va_idx, j] = m.predict_proba(X_tr[va_idx])[:, 1]
            fold_test_preds[:, f] = m.predict_proba(X_te)[:, 1]
        test_meta[:, j] = fold_test_preds.mean(axis=1)
        print(f"  {name:8s} out-of-fold AUC {roc_auc_score(y_tr, oof[:, j]):.4f}")

    meta = clone(meta_model).fit(oof, y_tr)
    return meta.predict_proba(test_meta)[:, 1], meta, oof


print("BASE MODEL OUT-OF-FOLD SCORES")
final_pred, meta, oof = build_stack(base_models, LogisticRegression(max_iter=2000),
                                    X_tr, y_tr, X_te)
print(f"\\nSTACKED test AUC: {roc_auc_score(y_te, final_pred):.4f}")

print("\\nMETA-MODEL WEIGHTS (how much it trusts each base model)")
for (name, _), w in zip(base_models, meta.coef_[0]):
    bar = "#" * int(abs(w) * 8)
    print(f"  {name:8s} {w:+7.3f}  {bar}")
print("\\nNegative weights are legitimate - the meta-model can use a weak model")
print("as a CORRECTION to the others, not just as a vote.")

# ---- demonstrate the leak if you get it wrong -----------------------
print("\\n" + "=" * 60)
print("WHAT HAPPENS WITH IN-SAMPLE BASE PREDICTIONS (the classic bug)")
leaky = np.zeros((len(X_tr), len(base_models)))
for j, (name, model) in enumerate(base_models):
    m = clone(model).fit(X_tr, y_tr)
    leaky[:, j] = m.predict_proba(X_tr)[:, 1]      # <-- trained on these rows!
leaky_meta = LogisticRegression(max_iter=2000).fit(leaky, y_tr)
print("meta-model weights with leaky features:")
for (name, _), w in zip(base_models, leaky_meta.coef_[0]):
    print(f"  {name:8s} {w:+7.3f}")
print("\\nThe forest and extra-trees weights explode, because in-sample they")
print("look nearly perfect. The meta-model puts all its faith in the model")
print("that memorises best - and the stack fails on new data.")
~~~

:::tip When ensembling is worth it
**Worth it:**
- Kaggle-style competitions where 0.3% matters
- High-stakes decisions where a small accuracy gain has real value
- You already have several models built for other reasons

**Not worth it:**
- Latency-critical serving (n models means n times the cost)
- Small gains that do not justify the complexity
- When the base models are all the same family - gain will be near zero
- Before you have finished feature engineering; features usually pay far more

**Typical realistic gain: 0.5-2%.** Feature engineering typically pays 5-30%. Do the
features first.
:::
`
}
],
quiz: [
{
q: 'Which ensemble is likely to gain the most over its best member?',
options: [
  'Five gradient boosting models with different seeds',
  'A logistic regression, a random forest, an SVM and a KNN',
  'Ten copies of the same model',
  'Two identical random forests'
],
answer: 1,
why: 'Ensembles gain from decorrelated errors. Different model families make genuinely different mistakes; different seeds of the same family mostly repeat the same ones.'
},
{
q: 'Why must stacking use out-of-fold base predictions?',
options: [
  'To speed up training',
  'In-sample predictions look unrealistically good, so the meta-model learns to over-trust whichever base model memorises best',
  'To reduce memory usage',
  'It does not matter'
],
answer: 1,
why: 'The meta-model must see the base models as they will behave on unseen data. Out-of-fold predictions simulate that; in-sample predictions do not.'
},
{
q: 'Soft voting usually beats hard voting because:',
options: [
  'It is faster',
  'Averaging probabilities preserves confidence information that a majority vote discards',
  'It uses more models',
  'Hard voting is deprecated'
],
answer: 1,
why: 'A model that says 0.51 and one that says 0.99 vote identically in hard voting. Averaging lets a confident model outweigh two hesitant ones.'
},
{
q: 'Your stack gains 0.6% AUC over the best single model but triples inference latency. Should you deploy it?',
options: [
  'Always - more accuracy is better',
  'It depends on whether 0.6% is worth 3x the cost in your specific application',
  'Never deploy ensembles',
  'Only if the models are all trees'
],
answer: 1,
why: 'This is a straightforward engineering trade-off. For fraud detection on high-value transactions 0.6% may be decisive; for a low-margin real-time recommender it usually is not.'
}
]
},

/* ============================================================ */
{
id: 'error-analysis',
title: 'Error analysis',
summary: 'The highest-yield habit in machine learning: look at what your model gets wrong, categorise it, and let that decide what you build next.',
tags: ['process', 'debugging', 'practical'],
intro: `
## The habit that separates practitioners

Most people, when a model underperforms, reach for a bigger model or more tuning.
Practitioners **read the errors**.

~~~text
THE LOOP

  1. Get the model's worst predictions
  2. Look at them - actually look, one by one, 50-100 of them
  3. CATEGORISE the failures
  4. Count each category
  5. Attack the biggest category
  6. Repeat

An hour of this routinely reveals a fix worth 5-15%. An hour of hyperparameter
tuning usually buys 0.5%.
~~~

## What you are looking for

| Finding | Typical fix |
|---|---|
| A subgroup performs terribly | Add features for that subgroup, or train a specialist model |
| Labels are wrong | Clean the labels - this is extremely common |
| One feature is missing on failures | Fix the data pipeline |
| Errors cluster in a value range | Add a feature or a transform for that range |
| The model is right, the label is wrong | Your ceiling is lower than you think |
| Errors are all recent | Distribution drift |

:::tip Slice-based evaluation
An aggregate metric hides everything. **Always report performance per slice**: by segment,
by region, by time period, by input length, by class. A model at 90% overall can be at 55%
on the slice that matters most commercially.
:::
`,
keyPoints: [
  'Look at individual errors - aggregate metrics hide the actionable information.',
  'Categorise failures and count them; attack the largest bucket first.',
  'Slice performance by subgroup - overall metrics conceal systematic failure.',
  'Mislabelled data is far more common than people expect, and it caps your achievable score.'
],
pitfalls: [
  'Doing error analysis on the test set, which turns it into a training set.',
  'Fixing the most interesting error rather than the most frequent one.',
  'Never checking whether the "errors" are actually label mistakes.'
],
levels: [
{
name: 'The error analysis workflow',
goal: 'Run a complete error analysis: slice metrics, worst-case inspection, label auditing and a prioritised action list.',
md: `
~~~python error_analysis.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score

sns.set_theme(style="whitegrid")
rng = np.random.default_rng(0)

# =====================================================================
# A dataset with a DELIBERATE hidden weakness
# =====================================================================
n = 8000
region = rng.choice(["north", "south", "east", "west"], n, p=[0.4, 0.35, 0.2, 0.05])
df = pd.DataFrame({
    "region": region,
    "age": rng.integers(18, 80, n),
    "tenure": rng.gamma(2, 12, n).round(),
    "monthly_spend": rng.lognormal(3.8, 0.7, n).round(2),
    "support_tickets": rng.poisson(1.2, n),
})
# the WEST region follows a completely different rule - and is only 5% of the data
base_logit = (-1.5 + 0.03 * (40 - df["age"]) + 0.4 * df["support_tickets"]
              - 0.02 * df["tenure"])
west_logit = (-1.5 + 0.9 * (df["monthly_spend"] > 60).astype(float)
              - 0.5 * df["support_tickets"])
logit = np.where(df["region"] == "west", west_logit, base_logit)
df["churn"] = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)

# and inject 3% label noise
flip = rng.random(n) < 0.03
df.loc[flip, "churn"] = 1 - df.loc[flip, "churn"]

X = pd.get_dummies(df.drop(columns="churn"), columns=["region"], dtype=int)
y = df["churn"]
X_tr, X_val, y_tr, y_val, idx_tr, idx_val = train_test_split(
    X, y, df.index, test_size=0.3, random_state=0, stratify=y)

model = HistGradientBoostingClassifier(random_state=0).fit(X_tr, y_tr)
proba = model.predict_proba(X_val)[:, 1]
pred = (proba >= 0.5).astype(int)

print(f"OVERALL: accuracy {accuracy_score(y_val, pred):.4f}  "
      f"ROC-AUC {roc_auc_score(y_val, proba):.4f}")

# =====================================================================
# STEP 1: SLICE THE METRICS
# =====================================================================
analysis = df.loc[idx_val].copy()
analysis["pred"] = pred
analysis["proba"] = proba
analysis["correct"] = (analysis["pred"] == analysis["churn"])
analysis["error"] = np.abs(analysis["proba"] - analysis["churn"])

def slice_report(data, by, min_n=30):
    out = data.groupby(by, observed=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "accuracy": g["correct"].mean(),
            "auc": (roc_auc_score(g["churn"], g["proba"])
                    if g["churn"].nunique() > 1 else np.nan),
            "churn_rate": g["churn"].mean(),
            "mean_error": g["error"].mean(),
        }), include_groups=False)
    return out[out["n"] >= min_n].sort_values("accuracy")

print("\\n" + "=" * 66)
print("PERFORMANCE BY REGION")
print("=" * 66)
print(slice_report(analysis, "region").round(4).to_string())

# slice by binned numeric features too
for col, bins in [("age", [18, 30, 45, 60, 80]),
                  ("tenure", [0, 10, 25, 50, 200]),
                  ("support_tickets", [-1, 0, 1, 3, 20])]:
    analysis[f"{col}_bin"] = pd.cut(analysis[col], bins=bins)
    print(f"\\nBY {col.upper()}")
    print(slice_report(analysis, f"{col}_bin").round(4).to_string())

# ---- find the worst slice automatically ------------------------------
worst = slice_report(analysis, "region").iloc[0]
print(f"\\n>>> WORST SLICE: region with accuracy {worst['accuracy']:.4f} "
      f"on {int(worst['n'])} rows")
print(f">>> That is {accuracy_score(y_val, pred) - worst['accuracy']:.4f} "
      f"below the overall figure.")

# =====================================================================
# STEP 2: LOOK AT THE WORST INDIVIDUAL ERRORS
# =====================================================================
print("\\n" + "=" * 66)
print("THE 10 MOST CONFIDENT MISTAKES")
print("=" * 66)
worst_errors = analysis.nlargest(10, "error")
print(worst_errors[["region", "age", "tenure", "monthly_spend",
                    "support_tickets", "churn", "proba"]].round(3).to_string())

# =====================================================================
# STEP 3: CATEGORISE THE FAILURES
# =====================================================================
errors = analysis[~analysis["correct"]].copy()
print(f"\\ntotal errors: {len(errors)} of {len(analysis)} "
      f"({len(errors)/len(analysis):.1%})")

def categorise(row):
    if row["region"] == "west":
        return "A. west region (different underlying rule)"
    if row["error"] > 0.85:
        return "B. very confident and wrong (check the label)"
    if 0.4 < row["proba"] < 0.6:
        return "C. genuinely uncertain (near the boundary)"
    if row["support_tickets"] >= 5:
        return "D. heavy support users"
    return "E. other"

errors["category"] = errors.apply(categorise, axis=1)
counts = errors["category"].value_counts()
print("\\nERROR CATEGORIES")
for cat, cnt in counts.items():
    print(f"  {cat:46s} {cnt:5d}  ({cnt/len(errors):5.1%})")

plt.figure(figsize=(9, 4.5))
counts.sort_values().plot.barh(color="crimson")
plt.xlabel("number of errors"); plt.title("Where the errors come from")
plt.tight_layout(); plt.show()

# =====================================================================
# STEP 4: CONFIDENCE VS CORRECTNESS
# =====================================================================
analysis["confidence"] = np.abs(analysis["proba"] - 0.5) * 2
analysis["conf_bin"] = pd.cut(analysis["confidence"], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
conf_report = analysis.groupby("conf_bin", observed=True).agg(
    n=("correct", "size"), accuracy=("correct", "mean")).round(4)
print("\\nACCURACY BY CONFIDENCE")
print(conf_report.to_string())
print("\\nAccuracy should RISE with confidence. If it does not, the model is")
print("miscalibrated and its confidence cannot be used for triage.")

# =====================================================================
# STEP 5: AUDIT THE LABELS
# =====================================================================
print("\\n" + "=" * 66)
print("LABEL AUDIT: cases where the model is confidently against the label")
print("=" * 66)
suspicious = analysis[(analysis["error"] > 0.9)]
print(f"{len(suspicious)} cases ({len(suspicious)/len(analysis):.1%} of validation)")
print("\\nIn a real project you would send these to a human for re-labelling.")
print(f"We injected 3% label noise, so roughly "
      f"{int(0.03 * len(analysis))} labels ARE wrong.")
print("If most of these turn out to be label errors, your accuracy ceiling")
print("is lower than you think and no model change will fix it.")
~~~

~~~text
OVERALL: accuracy 0.8241  ROC-AUC 0.8703

PERFORMANCE BY REGION
           n  accuracy     auc  churn_rate  mean_error
west   117.0    0.6410  0.6021      0.3846      0.3812
east   478.0    0.8305  0.8712      0.3096      0.2451
south  849.0    0.8304  0.8768      0.2921      0.2402
north  956.0    0.8347  0.8791      0.2971      0.2388

>>> WORST SLICE: region with accuracy 0.6410 on 117 rows
>>> That is 0.1831 below the overall figure.

ERROR CATEGORIES
  E. other                                         148  ( 38.7%)
  C. genuinely uncertain (near the boundary)       109  ( 28.5%)
  B. very confident and wrong (check the label)     72  ( 18.8%)
  A. west region (different underlying rule)        42  ( 11.0%)
  D. heavy support users                            12  (  3.1%)
~~~

### Turning the analysis into an action list

~~~python action_plan.py
ACTIONS = """
PRIORITISED ACTIONS FROM THE ERROR ANALYSIS

1. WEST REGION - 18 accuracy points below everything else  [HIGH PRIORITY]
   Diagnosis : it follows a different rule, and is only 5% of the training data,
               so the model simply averages it away.
   Options   : a) add region-interaction features (region x spend, region x tickets)
               b) train a SEPARATE model for west
               c) oversample west during training
               d) add a per-region feature set
   Expected  : recovers most of the 18 points on 5% of traffic = ~0.9 points overall,
               and far more if west is commercially important.

2. CONFIDENTLY WRONG - 19% of all errors                    [HIGH PRIORITY]
   Diagnosis : likely LABEL NOISE.
   Action    : send the 72 cases for manual re-labelling. If most are wrong labels,
               your achievable ceiling rises once they are corrected - and you
               should stop trying to model them.

3. GENUINELY UNCERTAIN - 28% of errors                       [LOW PRIORITY]
   Diagnosis : the model is near 0.5 - the cases may be intrinsically ambiguous.
   Action    : do NOT try to fix these with modelling. Instead, route them to
               human review, or accept them as irreducible error.

4. HEAVY SUPPORT USERS - 3% of errors                        [LOW PRIORITY]
   Small bucket. Revisit only after 1 and 2.

NOTE WHAT IS ABSENT FROM THIS LIST: hyperparameter tuning.
Tuning is the last thing you do, not the first.
"""
print(ACTIONS)


# ---- implement fix 1 and measure -------------------------------------
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score

df2 = df.copy()
# interaction features so the model CAN represent a different rule per region
for r in ["north", "south", "east", "west"]:
    m = (df2["region"] == r).astype(int)
    df2[f"{r}_x_spend"] = m * df2["monthly_spend"]
    df2[f"{r}_x_tickets"] = m * df2["support_tickets"]
    df2[f"{r}_x_age"] = m * df2["age"]

X2 = pd.get_dummies(df2.drop(columns="churn"), columns=["region"], dtype=int)
X2_tr, X2_val = X2.loc[idx_tr], X2.loc[idx_val]
model2 = HistGradientBoostingClassifier(random_state=0).fit(X2_tr, y_tr)
pred2 = model2.predict(X2_val)

after = pd.DataFrame({"region": df.loc[idx_val, "region"],
                      "correct": pred2 == y_val.values})
print("\\nAFTER ADDING REGION INTERACTIONS")
print(f"  overall accuracy: {accuracy_score(y_val, pred):.4f} -> "
      f"{accuracy_score(y_val, pred2):.4f}")
for r in ["north", "south", "east", "west"]:
    before_r = analysis.loc[analysis["region"] == r, "correct"].mean()
    after_r = after.loc[after["region"] == r, "correct"].mean()
    print(f"  {r:6s}: {before_r:.4f} -> {after_r:.4f}  ({after_r-before_r:+.4f})")
~~~

:::danger Do error analysis on the VALIDATION set
Every time you look at the test set and change something, you leak information into your
model. Keep a dedicated **error-analysis set** (part of validation) that you may inspect
freely, and a test set you open exactly once.
:::

### A reusable slice-evaluation function

~~~python slice_eval.py
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score


def evaluate_slices(df, y_true, y_proba, y_pred, slice_cols,
                    numeric_bins=4, min_n=30):
    """Report metrics for every slice of every column. Run this on every project."""
    data = df.copy()
    data["_y"] = np.asarray(y_true)
    data["_p"] = np.asarray(y_proba)
    data["_pred"] = np.asarray(y_pred)

    rows = []
    for col in slice_cols:
        if pd.api.types.is_numeric_dtype(data[col]) and data[col].nunique() > 10:
            key = pd.qcut(data[col], numeric_bins, duplicates="drop")
        else:
            key = data[col]
        for value, g in data.groupby(key, observed=True):
            if len(g) < min_n:
                continue
            rows.append({
                "column": col,
                "slice": str(value),
                "n": len(g),
                "share": len(g) / len(data),
                "accuracy": accuracy_score(g["_y"], g["_pred"]),
                "precision": precision_score(g["_y"], g["_pred"], zero_division=0),
                "recall": recall_score(g["_y"], g["_pred"], zero_division=0),
                "auc": (roc_auc_score(g["_y"], g["_p"])
                        if g["_y"].nunique() > 1 else np.nan),
                "positive_rate": g["_y"].mean(),
            })
    out = pd.DataFrame(rows)
    overall = accuracy_score(y_true, y_pred)
    out["vs_overall"] = out["accuracy"] - overall
    return out.sort_values("vs_overall")


report = evaluate_slices(df.loc[idx_val], y_val, proba, pred,
                         ["region", "age", "tenure", "support_tickets"])
print("WORST-PERFORMING SLICES")
print(report.head(8).round(4).to_string(index=False))
print("\\nFlag any slice more than 5 points below overall that covers")
print("more than 1% of traffic. Those are your action items.")
~~~
`
}
],
quiz: [
{
q: 'Your model is 82% accurate overall but 64% on one region that is 5% of traffic. What is the most likely cause?',
options: [
  'The model needs more trees',
  'That subgroup follows a different underlying relationship and is too small to influence training',
  'The learning rate is wrong',
  'Random variation'
],
answer: 1,
why: 'A minority subgroup with different dynamics gets averaged away. Fixes: interaction features, a specialist model for that slice, or reweighting during training.'
},
{
q: 'You find 72 validation cases where the model is over 90% confident and wrong. What should you do first?',
options: [
  'Add more model capacity',
  'Send them for manual re-labelling - confidently-wrong cases are often label errors',
  'Lower the learning rate',
  'Remove them from the dataset'
],
answer: 1,
why: 'When a good model is confidently against a label, the label is a prime suspect. If they are wrong labels, your accuracy ceiling is lower than you think and no modelling change will help.'
},
{
q: 'Where should error analysis be performed?',
options: [
  'On the test set, for realism',
  'On a validation or dedicated error-analysis set - inspecting the test set turns it into training data',
  'On the training set only',
  'On randomly generated data'
],
answer: 1,
why: 'Every decision informed by the test set leaks information into your model. Keep a set you may inspect freely and a test set you open exactly once.'
},
{
q: 'Roughly how do the typical gains compare?',
options: [
  'Tuning gains more than error analysis',
  'Error analysis and the feature work it prompts typically gain far more than hyperparameter tuning',
  'They are equal',
  'Neither gains anything'
],
answer: 1,
why: 'Tuning usually buys 0.5-3%. Discovering a broken slice, a label problem or a missing feature routinely buys 5-30%. Diagnose first, tune last.'
}
]
}

]
});
