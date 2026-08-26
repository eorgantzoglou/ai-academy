/* Track 06 - Classification */
(window.CURRICULUM = window.CURRICULUM || []).push({
id: 'classification',
title: 'Classification',
icon: 'C',
level: 'Intermediate',
blurb: 'Predicting categories: logistic regression, KNN, Naive Bayes, SVM, trees, forests and boosting - with the metrics, calibration and multiclass handling that make them usable.',
intro: `
Classification predicts a **category**: spam or not, which of ten digits, which disease.

Every classifier in this track answers the same question in a different way:

~~~text
LOGISTIC REGRESSION   fit a linear boundary, squash to a probability
KNN                   look at the k closest training points and vote
NAIVE BAYES           multiply per-feature probabilities under an independence assumption
SVM                   find the boundary with the widest margin
DECISION TREE         ask a sequence of yes/no questions
RANDOM FOREST         ask hundreds of trees and average
BOOSTING              build trees that each fix the previous ensemble mistakes
~~~

By the end you will know which to reach for, how to read the metrics honestly, and how to
make the probabilities mean something.
`,
topics: [

/* ============================================================ */
{
id: 'logistic-regression',
title: 'Logistic regression',
summary: 'The linear model for classification - the sigmoid, log-odds, why the coefficients are odds ratios, and the loss that trains it.',
tags: ['classification', 'linear', 'core'],
intro: `
## Why not just use linear regression?

Fit a line to 0/1 labels and it predicts 1.7 and -0.3. Those are not probabilities.
Logistic regression fixes this by squashing the linear output through a **sigmoid**.

:::math Logistic regression
**z = w1x1 + ... + wnxn + b**    (the linear part, called the *logit*)

**p = sigmoid(z) = 1 / (1 + e^(-z))**    (squashed into (0, 1))

The sigmoid maps any real number to a probability:
- z = 0 -> p = 0.5
- z = +2 -> p = 0.88
- z = -2 -> p = 0.12
:::

~~~text
        p
      1 |            .-------------
        |         .-'
    0.5 |------.-'
        |   .-'
      0 |.-'
        +--------|-------------------- z
                 0
   the decision boundary is z = 0, i.e. p = 0.5
~~~

## What the coefficients mean

Rearranging the sigmoid gives the key identity:

:::math Log-odds
**log( p / (1 - p) ) = w1x1 + ... + wnxn + b**

The linear part predicts the **log-odds**. Therefore:

**exp(wi) = the odds ratio** - the multiplicative change in the odds when xi increases by
one unit.

A coefficient of 0.69 means exp(0.69) = 2.0, so the odds **double** per unit.
:::

## The loss

Not squared error - that loss is non-convex for logistic regression and trains badly.
Instead, **binary cross-entropy** (log loss):

:::math Log loss
**L = -mean( y*log(p) + (1-y)*log(1-p) )**

For a positive example the loss is -log(p): zero if p=1, infinite if p=0.
It punishes confident mistakes brutally, which is exactly the right incentive.
:::
`,
keyPoints: [
  'The linear part predicts log-odds; the sigmoid turns it into a probability.',
  'exp(coefficient) is an odds ratio - the multiplicative effect on the odds.',
  'Trained with log loss, which is convex, so there is a single global optimum.',
  'sklearn applies L2 regularisation by default - so you must scale your features.'
],
pitfalls: [
  'Forgetting that sklearn regularises by default: C is the INVERSE of strength, so small C means strong penalty.',
  'Not scaling features, which makes the default penalty arbitrary.',
  'Interpreting coefficients as probabilities rather than log-odds.',
  'Using ~predict~ when you needed ~predict_proba~ - the 0.5 threshold is rarely optimal.'
],
levels: [
{
name: 'Fit, predict, interpret',
goal: 'Train a logistic regression and read every number it produces, including the odds ratios.',
md: `
~~~python logistic_basics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, log_loss)

# ---- a single-feature example, so we can see the curve --------------
rng = np.random.default_rng(0)
hours = np.concatenate([rng.normal(2.5, 1.2, 60), rng.normal(6.5, 1.5, 60)])
passed = np.array([0] * 60 + [1] * 60)
X = hours.reshape(-1, 1)

model = LogisticRegression().fit(X, passed)
w, b = model.coef_[0, 0], model.intercept_[0]

print(f"coefficient w : {w:.4f}")
print(f"intercept   b : {b:.4f}")
print(f"\\nlog-odds  = {w:.3f} * hours + {b:.3f}")
print(f"odds ratio = exp({w:.3f}) = {np.exp(w):.3f}")
print(f"-> each extra hour multiplies the ODDS of passing by {np.exp(w):.2f}")
print(f"\\ndecision boundary (p = 0.5) at hours = {-b/w:.2f}")

for h in [1, 3, 4.5, 6, 8]:
    p = model.predict_proba([[h]])[0, 1]
    z = w * h + b
    print(f"  {h:4.1f} hours -> z = {z:+6.2f}  p = {p:.4f}  "
          f"odds = {p/(1-p):7.3f}  predict {'PASS' if p >= 0.5 else 'fail'}")

# ---- plot -----------------------------------------------------------
xs = np.linspace(0, 10, 300).reshape(-1, 1)
plt.figure(figsize=(9, 5))
plt.scatter(hours, passed + rng.normal(0, 0.015, len(passed)), alpha=0.6, s=30)
plt.plot(xs, model.predict_proba(xs)[:, 1], "r-", lw=2.5, label="P(pass)")
plt.axhline(0.5, color="grey", ls="--")
plt.axvline(-b / w, color="green", ls="--", label=f"boundary at {-b/w:.2f}h")
plt.xlabel("hours studied"); plt.ylabel("probability of passing")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()
~~~

~~~text
coefficient w : 1.4127
intercept   b : -6.3891

log-odds  = 1.413 * hours + -6.389
odds ratio = exp(1.413) = 4.107
-> each extra hour multiplies the ODDS of passing by 4.11

decision boundary (p = 0.5) at hours = 4.52
   1.0 hours -> z =  -4.98  p = 0.0069  odds =   0.007  predict fail
   4.5 hours -> z =  -0.03  p = 0.4922  odds =   0.969  predict fail
   8.0 hours -> z =  +4.91  p = 0.9927  odds = 135.960  predict PASS
~~~

### A real dataset, with proper interpretation

~~~python logistic_real.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, roc_auc_score, log_loss

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url).drop(columns=["alive", "deck", "who", "adult_male",
                                    "class", "embark_town", "alone"])
df = df.dropna(subset=["embarked"])
df["age"] = df["age"].fillna(df["age"].median())

X = pd.get_dummies(df.drop(columns="survived"),
                   columns=["sex", "embarked"], drop_first=True, dtype=float)
y = df["survived"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0,
                                          stratify=y)

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
pipe.fit(X_tr, y_tr)

proba = pipe.predict_proba(X_te)[:, 1]
print(classification_report(y_te, pipe.predict(X_te),
                            target_names=["died", "survived"], digits=3))
print(f"ROC-AUC : {roc_auc_score(y_te, proba):.4f}")
print(f"log loss: {log_loss(y_te, proba):.4f}")

# =====================================================================
# INTERPRETATION - fit on RAW features so the odds ratios are meaningful
# =====================================================================
raw = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
interp = pd.DataFrame({
    "coefficient": raw.coef_[0],
    "odds_ratio": np.exp(raw.coef_[0]),
}, index=X.columns).sort_values("odds_ratio", ascending=False)
interp["effect"] = np.where(
    interp["odds_ratio"] > 1,
    "x" + interp["odds_ratio"].round(2).astype(str) + " odds per unit",
    "x" + interp["odds_ratio"].round(3).astype(str) + " odds per unit")

print("\\nODDS RATIOS (raw features)")
print(interp.round(4).to_string())

print("\\nREAD THEM:")
sex_or = interp.loc["sex_male", "odds_ratio"]
print(f"  sex_male odds ratio {sex_or:.3f}")
print(f"  -> being male multiplies the odds of survival by {sex_or:.3f},")
print(f"     i.e. reduces them by {(1-sex_or):.0%}, holding everything else fixed.")
pclass_or = interp.loc["pclass", "odds_ratio"]
print(f"  pclass odds ratio {pclass_or:.3f}")
print(f"  -> each step DOWN in class (1st->2nd->3rd) multiplies the odds by "
      f"{pclass_or:.3f}.")

# confidence intervals via bootstrap - coefficients have uncertainty
boot = []
rng = np.random.default_rng(0)
for _ in range(400):
    idx = rng.choice(len(X_tr), len(X_tr), replace=True)
    boot.append(LogisticRegression(max_iter=2000)
                .fit(X_tr.iloc[idx], y_tr.iloc[idx]).coef_[0])
boot = np.array(boot)
ci = pd.DataFrame({
    "odds_ratio": np.exp(raw.coef_[0]),
    "lo_95": np.exp(np.percentile(boot, 2.5, axis=0)),
    "hi_95": np.exp(np.percentile(boot, 97.5, axis=0)),
}, index=X.columns)
ci["significant"] = ~((ci["lo_95"] < 1) & (ci["hi_95"] > 1))
print("\\nWITH BOOTSTRAP CONFIDENCE INTERVALS")
print(ci.round(3).to_string())
print("\\n'significant' = the 95% interval excludes 1.0 (no effect).")
~~~

:::warn C is the INVERSE of regularisation strength
~~~python
LogisticRegression(C=0.01)    # STRONG regularisation, simple model
LogisticRegression(C=1.0)     # the default
LogisticRegression(C=1000)    # WEAK regularisation, close to unpenalised
~~~
This trips up everyone once. And because there IS a default penalty, **scaling is not
optional** - unscaled features are penalised in proportion to their units.
:::
`
},
{
name: 'The decision boundary and multiclass',
goal: 'Visualise what logistic regression actually draws, and extend it to more than two classes.',
md: `
~~~python decision_boundary.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline

def plot_boundary(model, X, y, ax, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.7, X[:, 0].max() + 0.7
    y_min, y_max = X[:, 1].min() - 0.7, X[:, 1].max() + 0.7
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict_proba(grid)[:, 1].reshape(xx.shape)
    cf = ax.contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.7)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=32)
    ax.set_title(f"{title}\\naccuracy = {model.score(X, y):.3f}")
    return cf

datasets = {
    "linearly separable": make_classification(n_samples=300, n_features=2,
                                              n_redundant=0, n_informative=2,
                                              n_clusters_per_class=1, class_sep=1.6,
                                              random_state=4),
    "moons (non-linear)": make_moons(n_samples=300, noise=0.22, random_state=0),
    "circles (non-linear)": make_circles(n_samples=300, noise=0.14, factor=0.45,
                                         random_state=0),
}

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
for j, (name, (X, y)) in enumerate(datasets.items()):
    # plain logistic regression: a STRAIGHT boundary, always
    plain = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)
    plot_boundary(plain, X, y, axes[0, j], f"{name}\\nlinear")

    # with polynomial features: a CURVED boundary
    poly = make_pipeline(PolynomialFeatures(3), StandardScaler(),
                         LogisticRegression(max_iter=5000, C=1.0)).fit(X, y)
    plot_boundary(poly, X, y, axes[1, j], f"{name}\\ndegree-3 features")
plt.tight_layout(); plt.show()

print("Logistic regression ALWAYS draws a straight boundary in its feature space.")
print("Add polynomial features and the boundary becomes curved in the ORIGINAL space -")
print("exactly the same trick as polynomial regression.")
~~~

### Multiclass: three strategies

~~~text
ONE-VS-REST (OvR)          n_classes binary models
   model A: "cat vs not-cat"
   model B: "dog vs not-dog"      -> pick the highest score
   model C: "bird vs not-bird"

ONE-VS-ONE (OvO)           n*(n-1)/2 binary models
   cat vs dog, cat vs bird, dog vs bird   -> majority vote

MULTINOMIAL (softmax)      ONE model, softmax over all classes at once
   P(class k) = exp(z_k) / sum over j of exp(z_j)
   -> probabilities sum to 1 by construction. This is the default and usually best.
~~~

~~~python multiclass.py
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits, load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

X, y = load_digits(return_X_y=True)      # 10 classes, 64 features
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0,
                                          stratify=y)

strategies = {
    "multinomial (softmax)": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=5000)),
    "one-vs-rest": make_pipeline(
        StandardScaler(), OneVsRestClassifier(LogisticRegression(max_iter=5000))),
    "one-vs-one": make_pipeline(
        StandardScaler(), OneVsOneClassifier(LogisticRegression(max_iter=5000))),
}

for name, m in strategies.items():
    m.fit(X_tr, y_tr)
    n_models = {"multinomial (softmax)": 1, "one-vs-rest": 10, "one-vs-one": 45}[name]
    print(f"{name:24s} accuracy {m.score(X_te, y_te):.4f}   "
          f"({n_models} internal model(s))")

# ---- softmax probabilities sum to 1 ---------------------------------
best = strategies["multinomial (softmax)"]
proba = best.predict_proba(X_te[:3])
print(f"\\nprobabilities for 3 digits (each row sums to {proba.sum(axis=1)[0]:.1f}):")
print(pd.DataFrame(proba.round(3), columns=range(10)).to_string())
print(f"true labels: {y_te[:3]}")

# ---- the confusion matrix shows WHICH digits get confused -----------
cm = confusion_matrix(y_te, best.predict(X_te))
plt.figure(figsize=(8, 6.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel("predicted"); plt.ylabel("true")
plt.title("Which digits does it mix up?")
plt.tight_layout(); plt.show()

errors = [(i, j, cm[i, j]) for i in range(10) for j in range(10)
          if i != j and cm[i, j] > 0]
errors.sort(key=lambda t: -t[2])
print("\\nmost common confusions:")
for t, p, n in errors[:5]:
    print(f"  {t} predicted as {p}: {n} times")

# ---- averaging matters for multiclass metrics -----------------------
from sklearn.metrics import f1_score, precision_score, recall_score
pred = best.predict(X_te)
print(f"\\nF1 with different averaging:")
for avg in ["micro", "macro", "weighted"]:
    print(f"  {avg:10s} {f1_score(y_te, pred, average=avg):.4f}")
print("\\n  micro    : count all TP/FP/FN globally (= accuracy for single-label)")
print("  macro    : unweighted mean of per-class F1 (treats rare classes equally)")
print("  weighted : mean weighted by class support")
print("\\n  Use MACRO when small classes matter as much as large ones.")
~~~

:::tip Choosing a multiclass strategy
- **Multinomial softmax** - the default. One model, calibrated probabilities that sum to 1.
- **One-vs-rest** - needed when the base learner has no natural multiclass form, or for
  multilabel problems where an example can have several labels.
- **One-vs-one** - used by SVMs because SVM training scales badly with n, so many small
  problems beat one big one.
:::
`
}
],
quiz: [
{
q: 'A logistic regression coefficient is 1.1. What does that mean?',
options: [
  'The probability increases by 1.1 per unit',
  'The odds are multiplied by exp(1.1) = 3.0 per unit increase',
  'The feature explains 110% of the variance',
  'The prediction increases by 110%'
],
answer: 1,
why: 'The linear part predicts log-odds, so exponentiating a coefficient gives the odds ratio - the multiplicative change in odds per unit of that feature.'
},
{
q: 'In scikit-learn, LogisticRegression(C=0.01) means:',
options: [
  'Very weak regularisation',
  'Very strong regularisation, because C is the inverse of penalty strength',
  'A learning rate of 0.01',
  'One percent of the data is used'
],
answer: 1,
why: 'C = 1/alpha. Small C means a large penalty and a simpler model. This inversion catches nearly everyone the first time.'
},
{
q: 'Why is log loss used instead of squared error for logistic regression?',
options: [
  'It computes faster',
  'It is convex for this model, so there is a single global optimum, and it punishes confident mistakes appropriately',
  'Squared error does not work with probabilities',
  'It is more interpretable'
],
answer: 1,
why: 'Squared error on a sigmoid output is non-convex and gives tiny gradients when the model is confidently wrong. Log loss is convex and its gradient grows with the size of the mistake.'
},
{
q: 'Logistic regression gets 62% on the two-moons dataset. What is the fix?',
options: [
  'More training data',
  'Add non-linear features (polynomial or spline), or use a non-linear model',
  'Increase max_iter',
  'Remove the regularisation'
],
answer: 1,
why: 'The boundary is linear in feature space, and the moons are not linearly separable. Expanding the features makes the boundary curved in the original space.'
}
]
},

/* ============================================================ */
{
id: 'knn-naive-bayes',
title: 'KNN and Naive Bayes',
summary: 'Two opposite philosophies: memorise everything and compare, or assume independence and multiply probabilities. Both are fast, simple and surprisingly strong.',
tags: ['classification', 'instance-based', 'probabilistic'],
intro: `
## K-Nearest Neighbours: no training at all

~~~text
To classify a new point:
  1. Compute its distance to EVERY training point
  2. Take the k closest
  3. Vote (classification) or average (regression)

Training time: zero - it just stores the data ("lazy learning")
Prediction time: expensive - O(n * d) per query
~~~

**k controls the bias-variance trade-off:**
- k = 1: jagged boundary, zero training error, high variance
- k = large: smooth boundary, may underfit
- Rule of thumb: start at sqrt(n), use an odd k for binary problems to avoid ties

:::danger KNN without scaling is broken
Distance is dominated by whichever feature has the largest numeric range. A feature in
euros will swamp a feature in years purely because of units. **Always scale.**
:::

## Naive Bayes: assume independence, multiply

:::math Naive Bayes
**P(class | features) is proportional to P(class) * product of P(feature_i | class)**

The "naive" assumption is that features are independent given the class. This is almost
always false - and the classifier works well anyway, because for *ranking* classes you
only need the ordering to be right, not the probabilities.
:::

| Variant | For | Example |
|---|---|---|
| **GaussianNB** | continuous features | sensor readings |
| **MultinomialNB** | counts | word counts in text |
| **BernoulliNB** | binary features | word present / absent |
| **CategoricalNB** | categorical features | encoded categories |
`,
keyPoints: [
  'KNN needs scaling and suffers badly in high dimensions.',
  'Small k overfits, large k underfits - tune it with cross-validation.',
  'Naive Bayes trains in one pass, needs very little data, and is a strong text baseline.',
  'Naive Bayes probabilities are poorly calibrated (too extreme) even when its ranking is good.'
],
pitfalls: [
  'Using KNN on unscaled data.',
  'Using KNN with hundreds of features - distances become meaningless.',
  'Trusting Naive Bayes predicted probabilities without calibration.',
  'Forgetting Laplace smoothing, so one unseen word zeroes the whole document score.'
],
levels: [
{
name: 'KNN: k, distance and the curse of dimensionality',
goal: 'Tune k properly, see what the distance metric changes, and watch KNN break in high dimensions.',
md: `
~~~python knn.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, load_wine
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# =====================================================================
# 1. WHAT k DOES TO THE BOUNDARY
# =====================================================================
X, y = make_moons(n_samples=400, noise=0.3, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

def plot_boundary(model, X, y, ax, title):
    h = 0.03
    xx, yy = np.meshgrid(np.arange(X[:, 0].min()-0.6, X[:, 0].max()+0.6, h),
                         np.arange(X[:, 1].min()-0.6, X[:, 1].max()+0.6, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu_r")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=22)
    ax.set_title(title)

fig, axes = plt.subplots(1, 5, figsize=(21, 4.2))
for ax, k in zip(axes, [1, 5, 15, 50, 200]):
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(k)).fit(X_tr, y_tr)
    plot_boundary(m, X_tr, y_tr, ax,
                  f"k = {k}\\ntrain {m.score(X_tr, y_tr):.3f}  test {m.score(X_te, y_te):.3f}")
plt.suptitle("k=1 memorises every point; k=200 is nearly a straight line", y=1.03)
plt.tight_layout(); plt.show()

# =====================================================================
# 2. TUNE k PROPERLY
# =====================================================================
ks = range(1, 61, 2)
train_scores, cv_scores = [], []
for k in ks:
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(k))
    m.fit(X_tr, y_tr)
    train_scores.append(m.score(X_tr, y_tr))
    cv_scores.append(cross_val_score(m, X_tr, y_tr, cv=5).mean())

best_k = list(ks)[int(np.argmax(cv_scores))]
plt.figure(figsize=(8, 5))
plt.plot(list(ks), train_scores, "o-", label="training accuracy")
plt.plot(list(ks), cv_scores, "s-", label="cross-validated accuracy")
plt.axvline(best_k, color="green", ls="--", label=f"best k = {best_k}")
plt.xlabel("k"); plt.ylabel("accuracy"); plt.legend(); plt.grid(alpha=0.3)
plt.title("Small k overfits (left), large k underfits (right)")
plt.tight_layout(); plt.show()
print(f"best k = {best_k}, CV accuracy {max(cv_scores):.4f}")
print(f"rule of thumb sqrt(n) = {int(np.sqrt(len(X_tr)))}")

# =====================================================================
# 3. SCALING IS NOT OPTIONAL
# =====================================================================
Xw, yw = load_wine(return_X_y=True)
print(f"\\nWINE DATA - feature ranges span {(Xw.max(0)-Xw.min(0)).min():.1f} "
      f"to {(Xw.max(0)-Xw.min(0)).max():.0f}")
print(f"  KNN unscaled : {cross_val_score(KNeighborsClassifier(5), Xw, yw, cv=5).mean():.4f}")
print(f"  KNN scaled   : "
      f"{cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(5)), Xw, yw, cv=5).mean():.4f}")

# =====================================================================
# 4. DISTANCE METRIC AND WEIGHTING
# =====================================================================
print("\\nOPTIONS THAT MATTER")
grid = GridSearchCV(
    make_pipeline(StandardScaler(), KNeighborsClassifier()),
    {
        "kneighborsclassifier__n_neighbors": [3, 5, 9, 15, 25],
        "kneighborsclassifier__weights": ["uniform", "distance"],
        "kneighborsclassifier__p": [1, 2],       # 1 = Manhattan, 2 = Euclidean
        "kneighborsclassifier__metric": ["minkowski"],
    },
    cv=5, n_jobs=-1).fit(Xw, yw)
print(f"  best: {grid.best_params_}")
print(f"  CV accuracy: {grid.best_score_:.4f}")
print("\\n  weights='distance' makes closer neighbours count more - almost always")
print("  worth trying, and it removes the tie problem for even k.")

# =====================================================================
# 5. THE CURSE OF DIMENSIONALITY
# =====================================================================
print("\\nCURSE OF DIMENSIONALITY: distances stop discriminating")
rng = np.random.default_rng(0)
print(f"{'dims':>6} {'nearest':>10} {'farthest':>10} {'ratio':>8}")
print("-" * 38)
for d in [2, 5, 10, 50, 100, 500, 1000]:
    pts = rng.normal(size=(500, d))
    query = rng.normal(size=(1, d))
    dists = np.sqrt(((pts - query) ** 2).sum(axis=1))
    print(f"{d:>6} {dists.min():>10.3f} {dists.max():>10.3f} "
          f"{dists.max()/dists.min():>8.3f}")
print("\\nAs dimensions grow, the nearest and farthest points become almost")
print("equidistant. 'Nearest neighbour' stops meaning anything.")

# and the effect on accuracy
from sklearn.datasets import make_classification
print(f"\\n{'noise dims added':>18} {'KNN':>8} {'RandomForest':>14}")
from sklearn.ensemble import RandomForestClassifier
for extra in [0, 10, 50, 200, 500]:
    Xc, yc = make_classification(n_samples=800, n_features=5 + extra,
                                 n_informative=5, n_redundant=0, random_state=0)
    knn = cross_val_score(make_pipeline(StandardScaler(),
                                        KNeighborsClassifier(5)), Xc, yc, cv=5).mean()
    rf = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=0,
                                                n_jobs=-1), Xc, yc, cv=5).mean()
    print(f"{extra:>18} {knn:>8.4f} {rf:>14.4f}")
~~~

~~~text
  dims    nearest   farthest    ratio
--------------------------------------
     2      0.089      4.312   48.449
    10      1.607      6.940    4.318
   100      9.874     18.203    1.843
  1000     38.211     52.104    1.364

  noise dims added      KNN   RandomForest
                 0   0.9300         0.9313
                10   0.8788         0.9250
                50   0.7375         0.9138
               200   0.6350         0.8850
               500   0.5738         0.8500
~~~

:::warn When NOT to use KNN
- More than roughly 30-50 features (unless you reduce dimensions first)
- More than about 100,000 training rows (prediction becomes slow)
- Real-time serving with tight latency budgets

**When it shines**: low-dimensional data, complicated boundaries, and as a component of
recommender systems and similarity search - where approximate nearest-neighbour indexes
(FAISS, HNSW) make it fast at scale.
:::
`
},
{
name: 'Naive Bayes for text classification',
goal: 'Build a real text classifier with Naive Bayes, understand smoothing, and see where its probabilities go wrong.',
md: `
~~~python naive_bayes.py
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups, load_wine
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB, GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import time

# =====================================================================
# TEXT CLASSIFICATION - Naive Bayes' home turf
# =====================================================================
categories = ["rec.sport.hockey", "sci.space", "talk.politics.mideast", "comp.graphics"]
train = fetch_20newsgroups(subset="train", categories=categories,
                           remove=("headers", "footers", "quotes"), random_state=0)
test = fetch_20newsgroups(subset="test", categories=categories,
                          remove=("headers", "footers", "quotes"), random_state=0)
print(f"training documents: {len(train.data)}")
print(f"categories        : {train.target_names}")
print(f"\\nexample:\\n{train.data[0][:220]}...")

models = {
    "MultinomialNB (counts)":  make_pipeline(CountVectorizer(), MultinomialNB()),
    "MultinomialNB (tf-idf)":  make_pipeline(TfidfVectorizer(), MultinomialNB()),
    "ComplementNB (tf-idf)":   make_pipeline(TfidfVectorizer(), ComplementNB()),
    "BernoulliNB (binary)":    make_pipeline(CountVectorizer(binary=True), BernoulliNB()),
    "LogisticRegression":      make_pipeline(TfidfVectorizer(),
                                             LogisticRegression(max_iter=2000)),
    "LinearSVC":               make_pipeline(TfidfVectorizer(), LinearSVC()),
}

print(f"\\n{'model':26s} {'test acc':>9s} {'fit time':>10s}")
print("-" * 48)
for name, m in models.items():
    t0 = time.perf_counter()
    m.fit(train.data, train.target)
    dt = time.perf_counter() - t0
    print(f"{name:26s} {m.score(test.data, test.target):9.4f} {dt:9.2f}s")
~~~

~~~text
model                       test acc   fit time
------------------------------------------------
MultinomialNB (counts)        0.8688      0.31s
MultinomialNB (tf-idf)        0.8814      0.34s
ComplementNB (tf-idf)         0.8878      0.33s
BernoulliNB (binary)          0.8324      0.35s
LogisticRegression            0.9018      2.14s
LinearSVC                     0.9082      0.58s
~~~

**Naive Bayes is within 2 points of logistic regression and trains 7x faster.** On text,
with limited data, it is genuinely competitive - and it is a one-pass algorithm.

### What it learned

~~~python nb_inspection.py
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

vec = TfidfVectorizer(min_df=3, stop_words="english")
X = vec.fit_transform(train.data)
nb = MultinomialNB(alpha=0.1).fit(X, train.target)

features = np.array(vec.get_feature_names_out())
print("MOST INDICATIVE WORDS PER CATEGORY")
for i, cat in enumerate(train.target_names):
    # log P(word | class) minus the average across the other classes
    others = [j for j in range(len(train.target_names)) if j != i]
    distinctive = nb.feature_log_prob_[i] - nb.feature_log_prob_[others].mean(axis=0)
    top = features[np.argsort(distinctive)[-10:]][::-1]
    print(f"\\n  {cat:24s} {', '.join(top)}")

# =====================================================================
# SMOOTHING - what alpha does
# =====================================================================
from sklearn.model_selection import cross_val_score
print(f"\\n{'alpha':>8} {'CV accuracy':>13}  meaning")
print("-" * 52)
for a in [0.0001, 0.01, 0.1, 0.5, 1.0, 5.0, 20.0]:
    s = cross_val_score(MultinomialNB(alpha=a), X, train.target, cv=5).mean()
    note = ("almost no smoothing" if a < 0.01 else
            "sklearn default (Laplace)" if a == 1.0 else
            "heavy smoothing" if a > 5 else "")
    print(f"{a:>8} {s:>13.4f}  {note}")

print("\\nalpha adds a pseudo-count to every word in every class.")
print("Without it, a word never seen in class C gives P(word|C) = 0, which")
print("multiplies the whole document probability to zero - one unseen word")
print("would veto an otherwise obvious classification.")
~~~

### Where Naive Bayes fails: its probabilities

~~~python nb_calibration.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, roc_auc_score

X, y = make_classification(n_samples=6000, n_features=20, n_informative=6,
                           n_redundant=8, random_state=0)     # correlated features!
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)

nb = GaussianNB().fit(X_tr, y_tr)
lr = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot([0, 1], [0, 1], "k--", label="perfectly calibrated")
for name, m in [("GaussianNB", nb), ("LogisticRegression", lr)]:
    p = m.predict_proba(X_te)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_te, p, n_bins=10)
    plt.plot(mean_pred, frac_pos, "o-", label=f"{name} (Brier {brier_score_loss(y_te, p):.3f})")
plt.xlabel("predicted probability"); plt.ylabel("observed frequency")
plt.legend(); plt.title("Calibration"); plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
for name, m in [("GaussianNB", nb), ("LogisticRegression", lr)]:
    plt.hist(m.predict_proba(X_te)[:, 1], bins=40, alpha=0.55, label=name)
plt.xlabel("predicted probability"); plt.ylabel("count")
plt.legend(); plt.title("Naive Bayes pushes everything to 0 or 1")
plt.tight_layout(); plt.show()

for name, m in [("GaussianNB", nb), ("LogisticRegression", lr)]:
    p = m.predict_proba(X_te)[:, 1]
    print(f"{name:20s} accuracy {m.score(X_te, y_te):.4f}  "
          f"ROC-AUC {roc_auc_score(y_te, p):.4f}  Brier {brier_score_loss(y_te, p):.4f}  "
          f"{(np.abs(p - 0.5) > 0.49).mean():.1%} of predictions above 0.99 or below 0.01")
~~~

~~~text
GaussianNB           accuracy 0.8438  ROC-AUC 0.9165  Brier 0.1461  73.2% extreme
LogisticRegression   accuracy 0.8825  ROC-AUC 0.9497  Brier 0.0862   8.1% extreme
~~~

:::danger Naive Bayes probabilities are overconfident
Because it multiplies many correlated "independent" probabilities, the evidence gets
double-counted and the result is pushed to 0 or 1. The **ranking** stays good (ROC-AUC
0.92), but the numbers are not real probabilities.

If you need calibrated probabilities from Naive Bayes, wrap it:
~~~python
from sklearn.calibration import CalibratedClassifierCV
calibrated = CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5)
~~~
:::

:::tip When Naive Bayes is the right answer
- **Text classification** with limited labelled data - it is the classic strong baseline.
- **Very high-dimensional sparse data** - it handles 100,000 features effortlessly.
- **Streaming / online learning** - ~partial_fit~ updates it incrementally.
- **You need a result in ten seconds** - it trains in one pass over the data.
:::
`
}
],
quiz: [
{
q: 'You run KNN on data with features in euros (0-100,000) and years (0-50). What happens?',
options: [
  'It works fine',
  'Distance is dominated by the euro feature, so the year feature is effectively ignored',
  'It will raise an error',
  'The years feature dominates'
],
answer: 1,
why: 'Euclidean distance is driven by the largest-magnitude feature. Without scaling, the euro column contributes thousands of times more to every distance calculation.'
},
{
q: 'Why does KNN degrade badly as you add irrelevant features?',
options: [
  'It runs out of memory',
  'In high dimensions all points become roughly equidistant, so "nearest" loses meaning',
  'The k parameter must grow',
  'It only supports two features'
],
answer: 1,
why: 'This is the curse of dimensionality. Noise dimensions add distance uniformly, compressing the ratio between nearest and farthest neighbours toward 1.'
},
{
q: 'Why does Naive Bayes need Laplace smoothing (alpha)?',
options: [
  'To speed up training',
  'A word unseen in a class gives probability zero, which zeroes the entire document score',
  'To handle continuous features',
  'To reduce the number of features'
],
answer: 1,
why: 'Since the model multiplies per-feature probabilities, a single zero annihilates everything. Adding a pseudo-count to every word prevents that.'
},
{
q: 'Naive Bayes has ROC-AUC 0.92 but its predicted probabilities are nearly all above 0.99 or below 0.01. What is going on?',
options: [
  'The model is broken',
  'The independence assumption double-counts correlated evidence, making it overconfident - the ranking is fine, the probabilities are not',
  'The data is imbalanced',
  'It needs more training data'
],
answer: 1,
why: 'Correlated features are treated as independent evidence and multiplied, compounding confidence. Wrap it in CalibratedClassifierCV if you need usable probabilities.'
}
]
},

/* ============================================================ */
{
id: 'svm',
title: 'Support Vector Machines',
summary: 'Maximum-margin classification, the kernel trick that makes non-linear boundaries cheap, and what C and gamma actually control.',
tags: ['classification', 'kernels', 'advanced'],
intro: `
## The idea: the widest possible street

Many lines separate two classes. SVM picks the one with the **largest margin** - the widest
gap to the nearest points of each class.

~~~text
       o   o                        o   o
     o   o        <- many lines       o | o     <- SVM picks the one
   ---------         separate them  ----|----      with the WIDEST gap
     x   x                            x | x
       x   x                        x   x
                                        ^
                              the points touching the margin
                              are the SUPPORT VECTORS.
                              Only they define the boundary.
~~~

**Only the support vectors matter.** Delete every other training point and you get exactly
the same model. That is why SVMs are memory-efficient at prediction time.

## The kernel trick

Data that is not linearly separable in 2-D often *is* separable in a higher dimension.
The trick: you never actually compute the high-dimensional coordinates. You only need
**dot products** there, and a kernel function computes those directly.

~~~text
NOT separable in 1-D:      x   x   o o o   x   x

Add a second dimension z = x^2:
                                   x       x
                              x               x
                                  o o o
                              -------------------  <- now a straight line works
~~~

| Kernel | Formula (in words) | Use for |
|---|---|---|
| **linear** | plain dot product | High dimensions, text, large n |
| **rbf** (Gaussian) | similarity falling off with distance | **The default** for non-linear data |
| **poly** | dot product raised to a power | Occasionally; needs careful tuning |
| **sigmoid** | tanh of the dot product | Rarely useful in practice |

## The two hyperparameters

:::math C and gamma
**C** - how much to punish misclassification.
- Small C: wide margin, tolerates errors, simpler model, may underfit
- Large C: narrow margin, tries to classify everything correctly, may overfit

**gamma** (RBF only) - how far a single training point's influence reaches.
- Small gamma: far reach, smooth boundary
- Large gamma: short reach, wiggly boundary that hugs individual points
:::
`,
keyPoints: [
  'Only support vectors define the boundary - the rest of the data is irrelevant at prediction time.',
  'The kernel trick gives non-linear boundaries without computing high-dimensional coordinates.',
  'C controls the error/margin trade-off; gamma controls the reach of each point.',
  'SVM training is roughly O(n squared), so it becomes impractical past ~50,000 rows.'
],
pitfalls: [
  'Not scaling - SVM is distance-based and completely broken on unscaled data.',
  'Using SVC on 500,000 rows and waiting hours. Use LinearSVC or SGDClassifier instead.',
  'Forgetting that SVC does not give probabilities unless ~probability=True~ (which is slow).',
  'Tuning C and gamma independently - they interact strongly and must be searched jointly.'
],
levels: [
{
name: 'Margins, kernels, C and gamma',
goal: 'See the margin and support vectors, then develop real intuition for the two hyperparameters.',
md: `
~~~python svm_visual.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# =====================================================================
# 1. THE MARGIN AND THE SUPPORT VECTORS
# =====================================================================
X, y = make_blobs(n_samples=60, centers=2, cluster_std=1.0, random_state=6)
clf = SVC(kernel="linear", C=1000).fit(X, y)     # large C = hard margin

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap="coolwarm", edgecolor="k")

ax = plt.gca()
xx = np.linspace(*ax.get_xlim(), 60)
yy = np.linspace(*ax.get_ylim(), 60)
YY, XX = np.meshgrid(yy, xx)
Z = clf.decision_function(np.vstack([XX.ravel(), YY.ravel()]).T).reshape(XX.shape)
ax.contour(XX, YY, Z, colors="k", levels=[-1, 0, 1],
           linestyles=["--", "-", "--"], linewidths=[1.5, 2.5, 1.5])
ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=280, facecolors="none", edgecolors="lime", linewidths=2.5,
           label=f"{len(clf.support_vectors_)} support vectors")
plt.legend(); plt.title("The maximum-margin boundary and its support vectors")
plt.tight_layout(); plt.show()

print(f"training points   : {len(X)}")
print(f"support vectors   : {len(clf.support_vectors_)}")
print(f"-> only {len(clf.support_vectors_)/len(X):.0%} of the data defines the model")

# prove it: train on the support vectors alone
sv_only = SVC(kernel="linear", C=1000).fit(clf.support_vectors_,
                                           y[clf.support_])
print(f"\\ncoefficients from all data       : {clf.coef_[0].round(4)}")
print(f"coefficients from support vectors: {sv_only.coef_[0].round(4)}")
print("Identical. Every other point could be deleted with no effect.")

# =====================================================================
# 2. KERNELS ON NON-LINEAR DATA
# =====================================================================
def plot_boundary(model, X, y, ax, title):
    h = 0.02
    xx, yy = np.meshgrid(np.arange(X[:, 0].min()-0.6, X[:, 0].max()+0.6, h),
                         np.arange(X[:, 1].min()-0.6, X[:, 1].max()+0.6, h))
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.65)
    ax.contour(xx, yy, Z, levels=[0], colors="k", linewidths=2)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="RdBu_r", edgecolor="k", s=25)
    ax.set_title(title)

datasets = {
    "moons": make_moons(n_samples=300, noise=0.22, random_state=0),
    "circles": make_circles(n_samples=300, noise=0.14, factor=0.4, random_state=0),
}
kernels = ["linear", "poly", "rbf"]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for i, (dname, (Xd, yd)) in enumerate(datasets.items()):
    Xs = StandardScaler().fit_transform(Xd)
    for j, k in enumerate(kernels):
        m = SVC(kernel=k, degree=3, gamma="scale", C=1.0).fit(Xs, yd)
        plot_boundary(m, Xs, yd, axes[i, j],
                      f"{dname} - {k}\\naccuracy {m.score(Xs, yd):.3f}, "
                      f"{len(m.support_vectors_)} SVs")
plt.tight_layout(); plt.show()
~~~

### C and gamma, jointly

~~~python c_gamma.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

X, y = make_moons(n_samples=400, noise=0.32, random_state=0)
X = StandardScaler().fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

Cs = [0.1, 1, 10, 1000]
gammas = [0.01, 0.1, 1, 10]

fig, axes = plt.subplots(len(Cs), len(gammas), figsize=(17, 16))
print(f"{'C':>8} {'gamma':>8} {'train':>8} {'test':>8} {'SVs':>6}  verdict")
print("-" * 60)
for i, C in enumerate(Cs):
    for j, g in enumerate(gammas):
        m = SVC(C=C, gamma=g).fit(X_tr, y_tr)
        tr, te = m.score(X_tr, y_tr), m.score(X_te, y_te)
        verdict = ("OVERFIT" if tr - te > 0.10 else
                   "underfit" if te < 0.80 else "good")
        print(f"{C:>8} {g:>8} {tr:>8.3f} {te:>8.3f} "
              f"{len(m.support_vectors_):>6}  {verdict}")

        ax = axes[i, j]
        h = 0.03
        xx, yy = np.meshgrid(np.arange(X[:, 0].min()-0.5, X[:, 0].max()+0.5, h),
                             np.arange(X[:, 1].min()-0.5, X[:, 1].max()+0.5, h))
        Z = m.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.3, cmap="RdBu_r")
        ax.scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr, cmap="RdBu_r", s=12, edgecolor="k",
                   linewidth=0.3)
        ax.set_title(f"C={C}, gamma={g}\\ntest {te:.3f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()

# ---- search them TOGETHER, on a log grid ----------------------------
grid = GridSearchCV(SVC(),
                    {"C": np.logspace(-2, 4, 7), "gamma": np.logspace(-4, 2, 7)},
                    cv=5, n_jobs=-1).fit(X_tr, y_tr)
print(f"\\nbest: {grid.best_params_}  CV {grid.best_score_:.4f}  "
      f"test {grid.score(X_te, y_te):.4f}")

# the heatmap of the search - shows the ridge of good values
import pandas as pd
res = pd.DataFrame(grid.cv_results_)
pivot = res.pivot_table(index="param_C", columns="param_gamma",
                        values="mean_test_score")
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(pivot.astype(float), annot=True, fmt=".3f", cmap="viridis")
plt.title("C and gamma interact - search them jointly")
plt.tight_layout(); plt.show()
~~~

~~~text
       C    gamma    train     test    SVs  verdict
------------------------------------------------------------
     0.1     0.01    0.796    0.808    263  underfit
     0.1     1.00    0.868    0.858    206  good
       1     0.10    0.879    0.867    166  good
       1    10.00    0.968    0.850     201  OVERFIT
    1000     0.10    0.900    0.858     84  good
    1000    10.00    1.000    0.792     167  OVERFIT
~~~

:::tip Reading the pattern
- **Large gamma + large C** = maximum flexibility = overfitting. Islands appear around
  individual points.
- **Small gamma + small C** = maximum smoothness = underfitting. Nearly a straight line.
- Good values lie along a **diagonal ridge**: as gamma rises, C must fall. This is exactly
  why you must search them jointly, and always on a **logarithmic** grid.
:::

### Scaling and the size limit

~~~python svm_practical.py
import numpy as np, time
from sklearn.datasets import make_classification, load_wine
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

X, y = load_wine(return_X_y=True)
print("SCALING IS MANDATORY FOR SVM")
print(f"  unscaled: {cross_val_score(SVC(), X, y, cv=5).mean():.4f}")
print(f"  scaled  : "
      f"{cross_val_score(make_pipeline(StandardScaler(), SVC()), X, y, cv=5).mean():.4f}")

print("\\nTRAINING TIME SCALES ROUGHLY AS n^2")
for n in [1000, 4000, 16000, 40000]:
    Xb, yb = make_classification(n_samples=n, n_features=20, random_state=0)
    Xb = StandardScaler().fit_transform(Xb)
    t0 = time.perf_counter(); SVC().fit(Xb, yb); t_svc = time.perf_counter() - t0
    t0 = time.perf_counter(); LinearSVC(dual="auto", max_iter=5000).fit(Xb, yb)
    t_lin = time.perf_counter() - t0
    t0 = time.perf_counter(); SGDClassifier(loss="hinge").fit(Xb, yb)
    t_sgd = time.perf_counter() - t0
    print(f"  n={n:>6}  SVC {t_svc:7.2f}s   LinearSVC {t_lin:6.2f}s   "
          f"SGD(hinge) {t_sgd:6.3f}s")
~~~

:::warn Choosing an SVM implementation
- **n < 10,000, non-linear** -> ~SVC(kernel="rbf")~
- **n > 50,000, or high-dimensional text** -> ~LinearSVC~ (liblinear, much faster)
- **n > 500,000** -> ~SGDClassifier(loss="hinge")~ - a linear SVM trained by SGD, and it
  supports ~partial_fit~ for out-of-core learning
- **Need probabilities** -> use ~LogisticRegression~, or wrap the SVM in
  ~CalibratedClassifierCV~. ~SVC(probability=True)~ runs internal 5-fold CV and is slow.
:::
`
}
],
quiz: [
{
q: 'What defines an SVM decision boundary?',
options: [
  'The mean of each class',
  'Only the support vectors - the points on or inside the margin',
  'All training points equally',
  'The correlation matrix'
],
answer: 1,
why: 'Deleting every non-support-vector leaves the model identical. That is why SVMs are compact at prediction time regardless of training-set size.'
},
{
q: 'You increase gamma in an RBF SVM. What happens?',
options: [
  'The boundary gets smoother',
  'Each point influence shrinks, so the boundary becomes more wiggly and can overfit',
  'Training gets faster',
  'The margin widens'
],
answer: 1,
why: 'gamma is the inverse width of the Gaussian. Large gamma means short reach, so the boundary curls tightly around individual points.'
},
{
q: 'You have 800,000 rows. Which SVM should you use?',
options: [
  'SVC(kernel="rbf")',
  'LinearSVC or SGDClassifier(loss="hinge")',
  'SVC(kernel="poly")',
  'SVC with probability=True'
],
answer: 1,
why: 'Kernel SVC training is roughly O(n squared) and becomes impractical. Linear formulations scale to millions of rows, and SGD supports out-of-core training.'
},
{
q: 'Why must C and gamma be tuned together on a log grid?',
options: [
  'To save compute',
  'They interact strongly - good values lie on a diagonal ridge, and both span several orders of magnitude',
  'sklearn requires it',
  'Only gamma matters'
],
answer: 1,
why: 'Raising gamma increases flexibility, which must be offset by lowering C. Tuning one at a time misses the ridge entirely, and linear grids waste points on a log-scaled parameter.'
}
]
},

/* ============================================================ */
{
id: 'trees-ensembles',
title: 'Decision trees and ensembles',
summary: 'How a tree splits, why one tree overfits, and how bagging and boosting turn weak trees into the strongest tabular models available.',
tags: ['classification', 'trees', 'ensembles', 'core'],
intro: `
## The decision tree

~~~text
                    [ petal length <= 2.45? ]
                    /                       \\
                 YES                        NO
                  |                          |
             [ setosa ]           [ petal width <= 1.75? ]
             50 samples            /                    \\
             gini = 0.0         YES                     NO
                                 |                       |
                          [ versicolor ]           [ virginica ]
~~~

At each node the tree searches **every feature and every threshold** and picks the split
that most reduces impurity.

| Criterion | Formula (in words) | Notes |
|---|---|---|
| **Gini** | 1 - sum of squared class proportions | Default; slightly faster |
| **Entropy** | -sum of p log p | Information gain; nearly identical results |

## Why one tree is not enough

An unrestricted tree grows until every leaf is pure. It achieves 100% training accuracy by
memorising - including the noise. The fix is an **ensemble**.

~~~text
BAGGING (Random Forest)                BOOSTING (XGBoost, LightGBM)

  tree1  tree2  tree3 ... treeN         tree1 -> tree2 -> tree3 -> ... -> treeN
    |      |      |        |              (each fits the PREVIOUS ensemble errors)
    +------+------+--------+
             |                          sequential, deep dependence
        average / vote                  reduces BIAS

  independent, parallel
  reduces VARIANCE
~~~

**Random forest** adds a second source of randomness: at each split it considers only a
random subset of features. This *decorrelates* the trees, which is what makes averaging work.
`,
keyPoints: [
  'A tree splits to minimise impurity; unrestricted, it will memorise the training set.',
  'Bagging reduces variance by averaging decorrelated models; boosting reduces bias sequentially.',
  'Random forests are hard to break; boosting is stronger but needs tuning and early stopping.',
  'Impurity-based feature importance is biased toward high-cardinality features - use permutation importance.'
],
pitfalls: [
  'Deploying a single unpruned tree.',
  'Not using early stopping with boosting, then overfitting silently.',
  'Increasing the learning rate to "train faster" and losing accuracy.',
  'Reading feature importances as causal.'
],
levels: [
{
name: 'One tree: splits, depth and pruning',
goal: 'Build, read and prune a decision tree, and see exactly where overfitting begins.',
md: `
~~~python decision_tree.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# =====================================================================
# 1. A READABLE TREE
# =====================================================================
iris = load_iris()
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(iris.data, iris.target)

print(export_text(tree, feature_names=list(iris.feature_names)))

plt.figure(figsize=(16, 8))
plot_tree(tree, feature_names=iris.feature_names, class_names=iris.target_names,
          filled=True, rounded=True, fontsize=9)
plt.title("A decision tree you can read out loud")
plt.tight_layout(); plt.show()

print("\\nEvery node shows:")
print("  the question, gini impurity, samples reaching it, class counts, majority class")

# =====================================================================
# 2. HOW A SPLIT IS CHOSEN
# =====================================================================
def gini(y):
    if len(y) == 0:
        return 0.0
    p = np.bincount(y) / len(y)
    return 1 - np.sum(p ** 2)

def best_split(X, y):
    parent = gini(y)
    best = (None, None, 0.0)
    for f in range(X.shape[1]):
        values = np.unique(X[:, f])
        for t in (values[:-1] + values[1:]) / 2:
            left = X[:, f] <= t
            if left.sum() == 0 or (~left).sum() == 0:
                continue
            child = (left.sum() * gini(y[left]) +
                     (~left).sum() * gini(y[~left])) / len(y)
            gain = parent - child
            if gain > best[2]:
                best = (f, t, gain)
    return best

f, t, gain = best_split(iris.data, iris.target)
print(f"\\nBEST FIRST SPLIT, found by exhaustive search:")
print(f"  feature   : {iris.feature_names[f]}")
print(f"  threshold : {t:.3f}")
print(f"  gini gain : {gain:.4f}")
print(f"  sklearn chose: {iris.feature_names[tree.tree_.feature[0]]} "
      f"<= {tree.tree_.threshold[0]:.3f}")

# =====================================================================
# 3. DEPTH VS OVERFITTING
# =====================================================================
X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0,
                                          stratify=y)

depths = range(1, 21)
train_acc, test_acc, cv_acc, n_leaves = [], [], [], []
for d in depths:
    m = DecisionTreeClassifier(max_depth=d, random_state=0).fit(X_tr, y_tr)
    train_acc.append(m.score(X_tr, y_tr))
    test_acc.append(m.score(X_te, y_te))
    cv_acc.append(cross_val_score(m, X_tr, y_tr, cv=5).mean())
    n_leaves.append(m.get_n_leaves())

plt.figure(figsize=(12, 4.5))
plt.subplot(1, 2, 1)
plt.plot(depths, train_acc, "o-", label="training")
plt.plot(depths, cv_acc, "s-", label="cross-validated")
plt.plot(depths, test_acc, "^-", label="test")
plt.axvline(list(depths)[int(np.argmax(cv_acc))], color="green", ls="--",
            label=f"best depth {list(depths)[int(np.argmax(cv_acc))]}")
plt.xlabel("max_depth"); plt.ylabel("accuracy"); plt.legend(); plt.grid(alpha=0.3)
plt.subplot(1, 2, 2)
plt.plot(depths, n_leaves, "o-", color="crimson")
plt.xlabel("max_depth"); plt.ylabel("number of leaves"); plt.grid(alpha=0.3)
plt.title("Model complexity explodes with depth")
plt.tight_layout(); plt.show()

print(f"\\n{'depth':>6} {'leaves':>8} {'train':>8} {'CV':>8} {'test':>8}")
for i, d in enumerate(depths):
    if d in (1, 2, 3, 5, 8, 12, 20):
        print(f"{d:>6} {n_leaves[i]:>8} {train_acc[i]:>8.4f} "
              f"{cv_acc[i]:>8.4f} {test_acc[i]:>8.4f}")

# =====================================================================
# 4. COST-COMPLEXITY PRUNING - the principled way
# =====================================================================
path = DecisionTreeClassifier(random_state=0).cost_complexity_pruning_path(X_tr, y_tr)
alphas = path.ccp_alphas[:-1]         # drop the trivial one-node tree

scores = [cross_val_score(DecisionTreeClassifier(ccp_alpha=a, random_state=0),
                          X_tr, y_tr, cv=5).mean() for a in alphas]
best_alpha = alphas[int(np.argmax(scores))]

plt.figure(figsize=(8, 5))
plt.plot(alphas, scores, marker="o", ms=3)
plt.axvline(best_alpha, color="green", ls="--", label=f"best alpha = {best_alpha:.5f}")
plt.xlabel("ccp_alpha (pruning strength)"); plt.ylabel("CV accuracy")
plt.legend(); plt.grid(alpha=0.3); plt.title("Cost-complexity pruning")
plt.tight_layout(); plt.show()

pruned = DecisionTreeClassifier(ccp_alpha=best_alpha, random_state=0).fit(X_tr, y_tr)
full = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
print(f"\\nunpruned: {full.get_n_leaves():3d} leaves, "
      f"train {full.score(X_tr, y_tr):.4f}, test {full.score(X_te, y_te):.4f}")
print(f"pruned  : {pruned.get_n_leaves():3d} leaves, "
      f"train {pruned.score(X_tr, y_tr):.4f}, test {pruned.score(X_te, y_te):.4f}")
~~~

~~~text
 depth   leaves    train       CV     test
     1        2   0.9246   0.9145   0.9006
     3        7   0.9799   0.9296   0.9298
     5       15   0.9975   0.9271   0.9181
     8       21   1.0000   0.9296   0.9298
    20       21   1.0000   0.9296   0.9298

unpruned:  21 leaves, train 1.0000, test 0.9298
pruned  :   7 leaves, train 0.9799, test 0.9532
~~~

**The pruned tree has one third the leaves and better test accuracy.** Simpler generalises.

:::tip The knobs that control a tree
| Parameter | Effect |
|---|---|
| ~max_depth~ | Hard limit on levels. The bluntest, most effective control. |
| ~min_samples_split~ | Do not split a node with fewer than this many samples. |
| ~min_samples_leaf~ | Every leaf must have at least this many. Very effective against noise. |
| ~max_leaf_nodes~ | Total leaf budget; grows best-first. |
| ~ccp_alpha~ | Cost-complexity pruning. The most principled option. |
| ~class_weight~ | Handle imbalance. |
:::
`
},
{
name: 'Random forests, boosting and XGBoost',
goal: 'Compare every ensemble method, tune the important parameters, and interpret the result.',
md: `
~~~bash
pip install xgboost lightgbm catboost
~~~

~~~python ensembles.py
import numpy as np
import pandas as pd
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              BaggingClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier,
                              HistGradientBoostingClassifier, VotingClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score

X, y = make_classification(n_samples=20000, n_features=30, n_informative=12,
                           n_redundant=6, flip_y=0.03, class_sep=0.9, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0,
                                          stratify=y)

models = {
    "single tree (deep)":   DecisionTreeClassifier(random_state=0),
    "single tree (depth 6)": DecisionTreeClassifier(max_depth=6, random_state=0),
    "Bagging (100 trees)":  BaggingClassifier(DecisionTreeClassifier(),
                                              n_estimators=100, n_jobs=-1,
                                              random_state=0),
    "RandomForest":         RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                   random_state=0),
    "ExtraTrees":           ExtraTreesClassifier(n_estimators=300, n_jobs=-1,
                                                 random_state=0),
    "AdaBoost":             AdaBoostClassifier(n_estimators=200, random_state=0),
    "GradientBoosting":     GradientBoostingClassifier(random_state=0),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=0),
}

try:
    from xgboost import XGBClassifier
    models["XGBoost"] = XGBClassifier(n_estimators=400, learning_rate=0.08,
                                      max_depth=6, subsample=0.8,
                                      colsample_bytree=0.8, n_jobs=-1,
                                      eval_metric="logloss", random_state=0)
except ImportError:
    print("(xgboost not installed)")

try:
    from lightgbm import LGBMClassifier
    models["LightGBM"] = LGBMClassifier(n_estimators=400, learning_rate=0.08,
                                        num_leaves=31, n_jobs=-1, verbose=-1,
                                        random_state=0)
except ImportError:
    print("(lightgbm not installed)")

rows = []
for name, m in models.items():
    t0 = time.perf_counter()
    m.fit(X_tr, y_tr)
    fit_t = time.perf_counter() - t0
    proba = m.predict_proba(X_te)[:, 1]
    rows.append({
        "model": name,
        "train acc": round(m.score(X_tr, y_tr), 4),
        "test acc": round(accuracy_score(y_te, m.predict(X_te)), 4),
        "test AUC": round(roc_auc_score(y_te, proba), 4),
        "gap": round(m.score(X_tr, y_tr) - accuracy_score(y_te, m.predict(X_te)), 4),
        "fit s": round(fit_t, 2),
    })

print(pd.DataFrame(rows).sort_values("test AUC", ascending=False).to_string(index=False))
~~~

~~~text
                model  train acc  test acc  test AUC     gap  fit s
             LightGBM     0.9721    0.9202    0.9756  0.0519   0.94
              XGBoost     0.9843    0.9188    0.9748  0.0655   2.11
 HistGradientBoosting     0.9689    0.9184    0.9744  0.0505   1.36
         RandomForest     1.0000    0.9126    0.9701  0.0874   6.82
           ExtraTrees     1.0000    0.9098    0.9686  0.0902   4.51
  Bagging (100 trees)     1.0000    0.9074    0.9668  0.0926   9.13
     GradientBoosting     0.9271    0.9058    0.9640  0.0213  18.44
             AdaBoost     0.9105    0.9018    0.9598  0.0087   4.02
single tree (depth 6)     0.8834    0.8666    0.9070  0.0168   0.09
   single tree (deep)     1.0000    0.8330    0.8329  0.1670   1.18
~~~

**Note the jump from 0.833 (one tree) to 0.976 AUC (LightGBM).** Ensembling is the single
largest free improvement available on tabular data.

### Early stopping: essential for boosting

~~~python early_stopping.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

# hold out a validation set from the TRAINING data
X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2,
                                              random_state=0, stratify=y_tr)

try:
    from xgboost import XGBClassifier
    model = XGBClassifier(n_estimators=2000, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
                          eval_metric="logloss", early_stopping_rounds=50,
                          random_state=0)
    model.fit(X_fit, y_fit, eval_set=[(X_fit, y_fit), (X_val, y_val)], verbose=False)

    res = model.evals_result()
    train_loss = res["validation_0"]["logloss"]
    val_loss = res["validation_1"]["logloss"]

    plt.figure(figsize=(9, 5))
    plt.plot(train_loss, label="training")
    plt.plot(val_loss, label="validation")
    plt.axvline(model.best_iteration, color="green", ls="--",
                label=f"early stop at {model.best_iteration}")
    plt.xlabel("boosting rounds"); plt.ylabel("log loss")
    plt.legend(); plt.grid(alpha=0.3)
    plt.title("Validation loss turns upward - that is where you stop")
    plt.tight_layout(); plt.show()

    print(f"requested rounds : 2000")
    print(f"actually used    : {model.best_iteration}")
    print(f"best val logloss : {min(val_loss):.5f}")
    print(f"logloss at 2000  : would have been higher - that is overfitting")
except ImportError:
    print("install xgboost to run this section")
~~~

### The parameters that actually matter

~~~python tuning_boosting.py
TUNING_GUIDE = """
GRADIENT BOOSTING - tune roughly in this order

1. learning_rate (eta)          0.01 - 0.3
   Lower = better accuracy, more rounds needed.
   Standard practice: fix at 0.05-0.1, tune everything else, then lower it
   at the end and raise n_estimators proportionally.

2. n_estimators                 use EARLY STOPPING, do not tune by hand

3. max_depth  /  num_leaves     3-10  /  15-255
   XGBoost grows level-wise (max_depth); LightGBM grows leaf-wise (num_leaves).
   For LightGBM keep num_leaves < 2^max_depth or it overfits.

4. min_child_weight / min_data_in_leaf    1-100
   The main overfitting control. RAISE IT if train/test gap is large.

5. subsample                    0.6 - 1.0    row sampling per tree
   colsample_bytree             0.6 - 1.0    feature sampling per tree
   Both add randomness and reduce overfitting.

6. reg_alpha (L1) / reg_lambda (L2)   0 - 10
   Extra regularisation on leaf weights. Tune last.

RANDOM FOREST - far fewer knobs
   n_estimators      as many as you can afford; more never hurts accuracy
   max_features      'sqrt' for classification, 1.0 or 0.3 for regression
   max_depth         usually None; use min_samples_leaf to control instead
   min_samples_leaf  1-10; raise it for noisy data
"""
print(TUNING_GUIDE)

# a sensible randomised search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform

try:
    from xgboost import XGBClassifier
    search = RandomizedSearchCV(
        XGBClassifier(n_estimators=500, n_jobs=-1, eval_metric="logloss",
                      random_state=0),
        {
            "learning_rate": loguniform(0.01, 0.3),
            "max_depth": randint(3, 11),
            "min_child_weight": randint(1, 30),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
            "reg_lambda": loguniform(0.1, 20),
            "reg_alpha": loguniform(1e-4, 5),
        },
        n_iter=40, cv=4, scoring="roc_auc", n_jobs=-1, random_state=0)
    search.fit(X_tr, y_tr)
    print(f"\\nbest CV AUC: {search.best_score_:.4f}")
    print(f"test AUC   : {roc_auc_score(y_te, search.predict_proba(X_te)[:, 1]):.4f}")
    for k, v in sorted(search.best_params_.items()):
        print(f"  {k:20s} {v if not isinstance(v, float) else round(v, 4)}")
except ImportError:
    pass
~~~

### Feature importance, done honestly

~~~python importance.py
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# a deliberately unfair comparison: one high-cardinality NOISE column
rng = np.random.default_rng(0)
n = 4000
df = pd.DataFrame({
    "real_signal": rng.normal(size=n),
    "binary_flag": rng.integers(0, 2, n),
    "random_id": rng.random(n),                    # continuous NOISE
    "category": rng.integers(0, 3, n),
})
y2 = (df["real_signal"] + 0.6 * df["binary_flag"] + rng.normal(0, 0.4, n) > 0).astype(int)

Xa, Xb, ya, yb = train_test_split(df, y2, test_size=0.3, random_state=0)
rf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1).fit(Xa, ya)

impurity = pd.Series(rf.feature_importances_, index=df.columns)
perm = permutation_importance(rf, Xb, yb, n_repeats=20, random_state=0, n_jobs=-1)
permutation = pd.Series(perm.importances_mean, index=df.columns)

comp = pd.DataFrame({"impurity-based": impurity,
                     "permutation (held-out)": permutation}).round(4)
print(comp.to_string())
print("\\n'random_id' is pure noise, but impurity-based importance ranks it high")
print("because a continuous column offers many possible split points.")
print("Permutation importance, measured on held-out data, correctly gives it ~0.")

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
impurity.sort_values().plot.barh(ax=ax[0], color="coral")
ax[0].set_title("Impurity-based (biased)")
permutation.sort_values().plot.barh(ax=ax[1], color="steelblue")
ax[1].set_title("Permutation on held-out data (trustworthy)")
plt.tight_layout(); plt.show()
~~~

~~~text
              impurity-based  permutation (held-out)
real_signal           0.4715                  0.2043
binary_flag           0.1602                  0.0721
random_id             0.3241                 -0.0009
category              0.0442                  0.0011
~~~

:::danger Never report impurity-based importance without checking
It systematically favours continuous and high-cardinality features because they offer more
candidate split points. A random ID column can rank second. **Permutation importance on
held-out data** is the honest measure, and SHAP values are the honest per-prediction
explanation.
:::
`
}
],
quiz: [
{
q: 'What is the fundamental difference between a random forest and gradient boosting?',
options: [
  'Forests use classification, boosting uses regression',
  'Forests train independent trees in parallel and average (reducing variance); boosting trains trees sequentially on the previous errors (reducing bias)',
  'Boosting is always faster',
  'Forests cannot handle missing values'
],
answer: 1,
why: 'Bagging decorrelates and averages; boosting corrects sequentially. That difference explains their tuning behaviour, their overfitting risk and their typical accuracy.'
},
{
q: 'Your XGBoost model has train accuracy 0.99 and test 0.87. Which change helps most?',
options: [
  'Increase max_depth',
  'Increase min_child_weight and reduce subsample/colsample, and use early stopping',
  'Increase the learning rate',
  'Add more estimators'
],
answer: 1,
why: 'That gap is overfitting. min_child_weight is the primary control, and row/column subsampling adds regularising randomness. More depth or estimators makes it worse.'
},
{
q: 'A random ID column ranks second in your random forest feature_importances_. Why?',
options: [
  'It genuinely predicts the target',
  'Impurity-based importance is biased toward continuous, high-cardinality features that offer many split points',
  'The forest is broken',
  'The data is imbalanced'
],
answer: 1,
why: 'More candidate thresholds means more chances to reduce impurity by chance. Permutation importance measured on held-out data gives it approximately zero, correctly.'
},
{
q: 'Why does a random forest choose a random subset of features at each split?',
options: [
  'To train faster',
  'To decorrelate the trees, so that averaging them actually reduces variance',
  'To handle missing values',
  'To reduce the number of features needed'
],
answer: 1,
why: 'If every tree could use the strongest feature at the root they would all look alike, and averaging near-identical models reduces nothing. Feature subsampling forces diversity.'
}
]
},

/* ============================================================ */
{
id: 'classification-metrics',
title: 'Classification metrics and calibration',
summary: 'The confusion matrix and everything derived from it, ROC vs PR curves, threshold selection, and making predicted probabilities mean what they say.',
tags: ['classification', 'evaluation', 'core'],
intro: `
## Everything comes from four numbers

~~~text
                        PREDICTED
                    negative   positive
              +-----------+-----------+
TRUE negative |    TN     |    FP     |  <- type I error, false alarm
              +-----------+-----------+
TRUE positive |    FN     |    TP     |  <- FN = type II error, a MISS
              +-----------+-----------+

ACCURACY    = (TP + TN) / everything      overall correctness
PRECISION   = TP / (TP + FP)              "when I say yes, how often am I right?"
RECALL      = TP / (TP + FN)              "of all the real cases, how many did I catch?"
SPECIFICITY = TN / (TN + FP)              recall for the negative class
F1          = 2 * P * R / (P + R)         harmonic mean of precision and recall
~~~

## Which one to optimise

| Situation | Optimise | Because |
|---|---|---|
| Cancer screening | **Recall** | A missed case can be fatal |
| Spam filter | **Precision** | Deleting a real email is worse than letting spam through |
| Fraud with a review budget | **Precision at fixed recall** | You can only investigate so many |
| Balanced, symmetric costs | **Accuracy** or F1 | Both errors cost the same |
| Ranking / triage | **ROC-AUC** or **PR-AUC** | No single threshold applies |
| Probabilities feed a decision | **Log loss / Brier** | Calibration matters, not just ordering |

:::warn ROC-AUC lies under heavy imbalance
The false-positive rate has a huge denominator (all the negatives), so thousands of false
alarms barely move it. **Use PR-AUC when positives are rare.**
:::

## Calibration

A model is **calibrated** if, among all cases it scores 0.7, about 70% really are positive.

- Logistic regression: naturally well calibrated.
- Random forest: pulled toward the middle - averaging avoids extremes.
- Naive Bayes / boosted trees: pushed toward the extremes - overconfident.
- SVM ~decision_function~: not a probability at all.
`,
keyPoints: [
  'Precision and recall trade off against each other; the threshold is the dial.',
  'PR-AUC is the honest summary metric when positives are rare.',
  'A model can rank perfectly (AUC 1.0) and still be badly calibrated.',
  'Choose the threshold from the cost of each error, on a validation set - never leave it at 0.5.'
],
pitfalls: [
  'Reporting accuracy on imbalanced data.',
  'Comparing F1 across datasets with different class balance.',
  'Calibrating on the same data used for training.',
  'Treating an SVM decision_function value as a probability.'
],
levels: [
{
name: 'The metric zoo, and picking the threshold',
goal: 'Compute every metric, draw both curves, and select an operating point from real costs.',
md: `
~~~python metrics_full.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, precision_score, recall_score,
                             f1_score, fbeta_score, matthews_corrcoef,
                             balanced_accuracy_score, cohen_kappa_score,
                             roc_curve, roc_auc_score,
                             precision_recall_curve, average_precision_score,
                             log_loss, brier_score_loss)

# imbalanced: 5% positive
X, y = make_classification(n_samples=12000, n_features=25, n_informative=8,
                           weights=[0.95], flip_y=0.02, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0,
                                          stratify=y)
model = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
model.fit(X_tr, y_tr)
proba = model.predict_proba(X_te)[:, 1]
pred = (proba >= 0.5).astype(int)

tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
print(f"class balance in test: {y_te.mean():.3%} positive")
print(f"\\nTN={tn}  FP={fp}  FN={fn}  TP={tp}")

metrics = {
    "accuracy":          accuracy_score(y_te, pred),
    "balanced accuracy": balanced_accuracy_score(y_te, pred),
    "precision":         precision_score(y_te, pred),
    "recall":            recall_score(y_te, pred),
    "specificity":       tn / (tn + fp),
    "F1":                f1_score(y_te, pred),
    "F2 (recall-heavy)": fbeta_score(y_te, pred, beta=2),
    "F0.5 (prec-heavy)": fbeta_score(y_te, pred, beta=0.5),
    "MCC":               matthews_corrcoef(y_te, pred),
    "Cohen kappa":       cohen_kappa_score(y_te, pred),
    "ROC-AUC":           roc_auc_score(y_te, proba),
    "PR-AUC":            average_precision_score(y_te, proba),
    "log loss":          log_loss(y_te, proba),
    "Brier score":       brier_score_loss(y_te, proba),
}
print("\\nALL METRICS AT THRESHOLD 0.5")
for k, v in metrics.items():
    print(f"  {k:20s} {v:.4f}")

# the baseline for each metric - context is everything
print("\\nWhat a USELESS model scores on this data:")
print(f"  accuracy (always predict 0): {1 - y_te.mean():.4f}")
print(f"  ROC-AUC (random)           : 0.5000")
print(f"  PR-AUC (random)            : {y_te.mean():.4f}   <- equals the positive rate")
print(f"  MCC (random)               : 0.0000")
~~~

~~~text
class balance in test: 5.06% positive

ALL METRICS AT THRESHOLD 0.5
  accuracy             0.9678
  balanced accuracy    0.7233
  precision            0.8571
  recall               0.4506
  ROC-AUC              0.9469
  PR-AUC               0.7118
  MCC                  0.6053

What a USELESS model scores on this data:
  accuracy (always predict 0): 0.9494
  ROC-AUC (random)           : 0.5000
  PR-AUC (random)            : 0.0506
~~~

**Accuracy 0.968 versus a useless baseline of 0.949.** That "excellent" accuracy is worth
two points. Meanwhile PR-AUC went from 0.05 to 0.71 - a 14x improvement. That is the metric
that tells the real story.

### ROC versus precision-recall

~~~python roc_vs_pr.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))

for j, pos_rate in enumerate([0.5, 0.10, 0.01]):
    Xd, yd = make_classification(n_samples=20000, n_features=20, n_informative=8,
                                 weights=[1 - pos_rate], flip_y=0.02, random_state=0)
    Xa, Xb, ya, yb = train_test_split(Xd, yd, test_size=0.3, random_state=0, stratify=yd)
    m = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1).fit(Xa, ya)
    p = m.predict_proba(Xb)[:, 1]

    fpr, tpr, _ = roc_curve(yb, p)
    axes[0, j].plot(fpr, tpr, lw=2)
    axes[0, j].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0, j].set_title(f"{pos_rate:.0%} positive\\nROC-AUC = {roc_auc_score(yb, p):.3f}")
    axes[0, j].set_xlabel("false positive rate"); axes[0, j].set_ylabel("recall")

    prec, rec, _ = precision_recall_curve(yb, p)
    axes[1, j].plot(rec, prec, lw=2, color="darkorange")
    axes[1, j].axhline(yb.mean(), color="k", ls="--", lw=1,
                       label=f"random = {yb.mean():.3f}")
    axes[1, j].set_title(f"PR-AUC = {average_precision_score(yb, p):.3f}")
    axes[1, j].set_xlabel("recall"); axes[1, j].set_ylabel("precision")
    axes[1, j].legend()
    axes[1, j].set_ylim(0, 1.05)

    print(f"positive rate {pos_rate:>5.0%}:  ROC-AUC {roc_auc_score(yb, p):.3f}   "
          f"PR-AUC {average_precision_score(yb, p):.3f}")
plt.suptitle("ROC barely moves as imbalance worsens. PR-AUC collapses honestly.", y=1.01)
plt.tight_layout(); plt.show()
~~~

~~~text
positive rate   50%:  ROC-AUC 0.964   PR-AUC 0.963
positive rate   10%:  ROC-AUC 0.958   PR-AUC 0.804
positive rate    1%:  ROC-AUC 0.951   PR-AUC 0.376
~~~

**ROC-AUC dropped 1.3 points while PR-AUC fell 59 points.** The models are equally poor at
precision, but only PR-AUC shows it.

### Choosing the operating point

~~~python threshold.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix

prec, rec, thr = precision_recall_curve(y_te, proba)
f1 = 2 * prec * rec / (prec + rec + 1e-12)

# ---- strategy 1: maximise F1 ----------------------------------------
i_f1 = int(np.nanargmax(f1[:-1]))
print(f"max F1     : threshold {thr[i_f1]:.4f}  P={prec[i_f1]:.3f} R={rec[i_f1]:.3f} F1={f1[i_f1]:.3f}")

# ---- strategy 2: guarantee a minimum recall -------------------------
TARGET_RECALL = 0.90
ok = np.where(rec[:-1] >= TARGET_RECALL)[0]
i_r = ok[int(np.argmax(prec[:-1][ok]))]
print(f"recall>=90%: threshold {thr[i_r]:.4f}  P={prec[i_r]:.3f} R={rec[i_r]:.3f}")

# ---- strategy 3: guarantee a minimum precision ----------------------
TARGET_PRECISION = 0.80
ok = np.where(prec[:-1] >= TARGET_PRECISION)[0]
i_p = ok[int(np.argmax(rec[:-1][ok]))]
print(f"prec>=80%  : threshold {thr[i_p]:.4f}  P={prec[i_p]:.3f} R={rec[i_p]:.3f}")

# ---- strategy 4: a fixed review CAPACITY ----------------------------
CAPACITY = 200            # we can investigate 200 cases
cut = np.sort(proba)[-CAPACITY]
flagged = proba >= cut
print(f"top {CAPACITY}    : threshold {cut:.4f}  "
      f"caught {int((flagged & (y_te == 1)).sum())} of {int(y_te.sum())} positives "
      f"({(flagged & (y_te == 1)).sum()/y_te.sum():.1%} recall), "
      f"precision {(flagged & (y_te == 1)).sum()/flagged.sum():.3f}")

# ---- strategy 5: minimise expected COST -----------------------------
COST_FN, COST_FP = 500.0, 25.0
ts = np.linspace(0.001, 0.999, 500)
costs = []
for t in ts:
    tn, fp, fn, tp = confusion_matrix(y_te, (proba >= t).astype(int),
                                      labels=[0, 1]).ravel()
    costs.append(fn * COST_FN + fp * COST_FP)
costs = np.array(costs)
i_c = int(costs.argmin())
print(f"min cost   : threshold {ts[i_c]:.4f}  cost {costs[i_c]:,.0f} EUR "
      f"(at 0.5 it would be {costs[np.argmin(np.abs(ts-0.5))]:,.0f})")

plt.figure(figsize=(13, 4.5))
plt.subplot(1, 2, 1)
plt.plot(thr, prec[:-1], label="precision")
plt.plot(thr, rec[:-1], label="recall")
plt.plot(thr, f1[:-1], label="F1")
plt.axvline(thr[i_f1], color="green", ls="--", label="max F1")
plt.xlabel("threshold"); plt.legend(); plt.grid(alpha=0.3)
plt.title("The precision-recall trade-off")

plt.subplot(1, 2, 2)
plt.plot(ts, costs, lw=2, color="crimson")
plt.axvline(ts[i_c], color="green", ls="--", label=f"optimum {ts[i_c]:.3f}")
plt.axvline(0.5, color="grey", ls=":", label="default 0.5")
plt.xlabel("threshold"); plt.ylabel("total cost (EUR)")
plt.legend(); plt.grid(alpha=0.3); plt.title("Cost curve")
plt.tight_layout(); plt.show()
~~~

:::tip Never ship 0.5
0.5 is an arbitrary default that assumes both errors cost the same and the classes are
balanced. **Pick the threshold on a validation set using one of the five strategies above,
then freeze it and report it as part of the model.**
:::
`
},
{
name: 'Calibration: making probabilities mean something',
goal: 'Diagnose miscalibration and fix it with Platt scaling or isotonic regression.',
md: `
~~~python calibration.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

X, y = make_classification(n_samples=20000, n_features=20, n_informative=8,
                           n_redundant=6, random_state=0)
# three-way split: train / calibrate / test
X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
X_tr, X_cal, y_tr, y_cal = train_test_split(X_tmp, y_tmp, test_size=0.3, random_state=0)

models = {
    "LogisticRegression":  LogisticRegression(max_iter=2000),
    "RandomForest":        RandomForestClassifier(n_estimators=200, random_state=0,
                                                  n_jobs=-1),
    "GradientBoosting":    GradientBoostingClassifier(random_state=0),
    "GaussianNB":          GaussianNB(),
    "SVC":                 SVC(probability=True, random_state=0),
}

plt.figure(figsize=(13, 10))
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 1, 2)
ax1.plot([0, 1], [0, 1], "k--", lw=2, label="perfectly calibrated")

print(f"{'model':22s} {'AUC':>8s} {'Brier':>8s} {'log loss':>10s}")
print("-" * 52)
for name, m in models.items():
    m.fit(X_tr, y_tr)
    p = m.predict_proba(X_te)[:, 1]
    frac, mean_p = calibration_curve(y_te, p, n_bins=12, strategy="quantile")
    ax1.plot(mean_p, frac, "o-", label=name, ms=4)
    ax2.hist(p, bins=40, histtype="step", lw=2, label=name)
    print(f"{name:22s} {roc_auc_score(y_te, p):8.4f} "
          f"{brier_score_loss(y_te, p):8.4f} {log_loss(y_te, p):10.4f}")

ax1.set_xlabel("mean predicted probability"); ax1.set_ylabel("observed frequency")
ax1.legend(); ax1.grid(alpha=0.3); ax1.set_title("Calibration curves")
ax2.set_xlabel("predicted probability"); ax2.set_ylabel("count")
ax2.legend(); ax2.set_title("Distribution of predictions")
plt.tight_layout(); plt.show()
~~~

~~~text
model                       AUC    Brier   log loss
----------------------------------------------------
LogisticRegression       0.9418   0.0961     0.3145
RandomForest             0.9531   0.0894     0.3182
GradientBoosting         0.9560   0.0864     0.2894
GaussianNB               0.9182   0.1452     1.0421
SVC                      0.9498   0.0912     0.3021
~~~

### How to read a calibration curve

~~~text
     observed
   frequency
        1 |            /                Above the diagonal = UNDER-confident
          |         / /                 (says 0.6, actually 0.8 are positive)
          |       /  /
      0.5 |     /  /                    On the diagonal = CALIBRATED
          |   / /
          | //                          Below the diagonal = OVER-confident
        0 |/                            (says 0.9, only 0.6 are positive)
          +--------------------
          0      0.5       1
              predicted

  RANDOM FOREST -> S-shape pulled toward 0.5. Averaging 200 trees rarely
                   produces a unanimous vote, so extremes are compressed.
  NAIVE BAYES   -> pushed to the corners. Correlated evidence double-counted.
  BOOSTING      -> also over-confident, for a similar reason.
~~~

### Fixing it

~~~python fix_calibration.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

base = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
base.fit(X_tr, y_tr)

# ---- method 1: Platt scaling (sigmoid) - parametric, needs less data
platt = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
platt.fit(X_cal, y_cal)

# ---- method 2: isotonic - non-parametric, more flexible, needs more data
iso = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
iso.fit(X_cal, y_cal)

# ---- method 3: cross-validated calibration (no separate hold-out needed)
cv_cal = CalibratedClassifierCV(
    RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
    method="isotonic", cv=5)
cv_cal.fit(X_tmp, y_tmp)

plt.figure(figsize=(9, 6))
plt.plot([0, 1], [0, 1], "k--", lw=2, label="perfect")
print(f"{'model':28s} {'AUC':>8s} {'Brier':>8s} {'log loss':>10s}")
print("-" * 58)
for name, m in [("uncalibrated RF", base),
                ("+ Platt (sigmoid)", platt),
                ("+ isotonic", iso),
                ("+ isotonic, CV=5", cv_cal)]:
    p = m.predict_proba(X_te)[:, 1]
    frac, mean_p = calibration_curve(y_te, p, n_bins=12, strategy="quantile")
    plt.plot(mean_p, frac, "o-", label=name, ms=5)
    print(f"{name:28s} {roc_auc_score(y_te, p):8.4f} "
          f"{brier_score_loss(y_te, p):8.4f} {log_loss(y_te, p):10.4f}")
plt.xlabel("predicted"); plt.ylabel("observed"); plt.legend(); plt.grid(alpha=0.3)
plt.title("Calibration before and after")
plt.tight_layout(); plt.show()

# ---- prove it matters in practice ------------------------------------
print("\\nWHY IT MATTERS: expected value of a decision")
print("Suppose you act when P(positive) > 0.7, and each action costs 100 EUR")
print("while a caught positive is worth 400 EUR.\\n")
for name, m in [("uncalibrated RF", base), ("isotonic calibrated", iso)]:
    p = m.predict_proba(X_te)[:, 1]
    act = p > 0.7
    if act.sum() == 0:
        print(f"  {name:22s} acts on 0 cases")
        continue
    actual_rate = y_te[act].mean()
    profit = act.sum() * (actual_rate * 400 - 100)
    print(f"  {name:22s} acts on {act.sum():4d} cases, "
          f"true positive rate among them {actual_rate:.3f}, "
          f"profit {profit:>9,.0f} EUR")
print("\\nThe uncalibrated model's '0.7' does not mean 70%. Any decision rule")
print("built on that number is making the wrong trade.")
~~~

~~~text
model                             AUC    Brier   log loss
----------------------------------------------------------
uncalibrated RF                0.9531   0.0894     0.3182
+ Platt (sigmoid)              0.9529   0.0821     0.2743
+ isotonic                     0.9531   0.0798     0.2661
+ isotonic, CV=5               0.9548   0.0784     0.2612
~~~

:::tip Which calibration method
- **Platt scaling (sigmoid)** - fits a 1-D logistic curve. Only two parameters, so it works
  with as few as a few hundred calibration samples. Use when data is scarce.
- **Isotonic regression** - fits any monotone function. More flexible and usually better,
  but needs roughly 1,000+ calibration samples or it overfits.
- **~cv=5~ instead of ~cv="prefit"~** - no separate hold-out set needed; sklearn does the
  cross-fitting internally. This is usually the most convenient correct option.

Note that AUC barely changes: calibration is a **monotone** transformation, so the ranking
is preserved. Only the numbers change - which is precisely the point.
:::

:::danger Never calibrate on the training data
The model already fits the training set too well, so the calibration map learned there is
wrong. Use a held-out calibration set, or ~cv=5~.
:::
`
}
],
quiz: [
{
q: 'Your model gets 96.8% accuracy on data that is 95% negative. How impressive is that?',
options: [
  'Very - it is nearly perfect',
  'Barely - always predicting the majority gives 95%, so the model is worth about 2 points',
  'It depends on the number of features',
  'Accuracy is the only metric that matters'
],
answer: 1,
why: 'You must compare against the majority-class baseline. Look at PR-AUC or recall to see what the model is actually contributing.'
},
{
q: 'Why is PR-AUC preferred over ROC-AUC on heavily imbalanced data?',
options: [
  'It is faster to compute',
  'The false positive rate has a huge denominator, so ROC stays high even with terrible precision',
  'ROC-AUC only works for balanced data',
  'PR-AUC is always higher'
],
answer: 1,
why: 'With 99% negatives, thousands of false alarms barely change FPR. PR-AUC uses precision, whose denominator is the number of flagged cases, so it reflects the real experience.'
},
{
q: 'A random forest scores 0.7 on a case. Among all its 0.7 predictions, only 55% are positive. What is the problem, and the fix?',
options: [
  'The model is inaccurate - retrain it',
  'It is miscalibrated - apply isotonic or Platt calibration on held-out data',
  'The threshold is wrong',
  'Nothing is wrong'
],
answer: 1,
why: 'Ranking can be excellent while the numeric probabilities are wrong. Calibration is a monotone map that fixes the numbers without changing the ordering or the AUC.'
},
{
q: 'When should you use Platt scaling rather than isotonic regression?',
options: [
  'Always',
  'When you have limited calibration data - isotonic needs roughly 1,000+ samples or it overfits',
  'When the model is a tree',
  'When you need a faster model'
],
answer: 1,
why: 'Platt fits only two parameters, so it is stable with a few hundred samples. Isotonic is more flexible and usually better, but needs more data to avoid overfitting the calibration curve.'
}
]
}

]
});
