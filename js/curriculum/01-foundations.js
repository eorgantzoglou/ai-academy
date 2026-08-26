/* Track 01 - Foundations of Data Science, ML and AI */
(window.CURRICULUM = window.CURRICULUM || []).push({
id: 'foundations',
title: 'Foundations',
icon: 'A',
level: 'Beginner',
blurb: 'What AI, ML and Data Science actually are, how learning works, the end-to-end workflow, and your very first working model.',
intro: `
Start here if you are new. This track gives you the **mental model** that everything
else hangs off. Nothing here is hard, but skipping it is the single most common reason
people get lost later: they can write ~model.fit(X, y)~ but cannot say what ~X~, ~y~ or
"fit" really mean.

By the end of this track you will have trained, evaluated and interpreted a real model.
`,
topics: [

/* ============================================================ */
{
id: 'what-is-ai',
title: 'What AI, ML, DL and Data Science actually are',
summary: 'The four words everyone mixes up, drawn as nested circles, with the one idea that separates them from ordinary programming.',
tags: ['theory', 'orientation'],
intro: `
## The one-sentence version

**Ordinary programming**: you write the rules, the computer applies them to data to produce answers.

**Machine learning**: you give the computer data *and* answers, and it works out the rules.

That inversion is the whole idea. Everything else is engineering detail.

~~~text
CLASSICAL PROGRAMMING          MACHINE LEARNING
    rules  ---+                    data    ---+
              |--> [computer] -> answers      |--> [computer] -> rules
    data   ---+                    answers ---+
~~~

## Why the inversion matters

Imagine writing a spam filter by hand. You start with ~if "free money" in subject: spam~.
Then spammers write "fr33 m0ney". You add a rule. Then "FREE M0NEY!!!". Another rule.
After 300 rules the system is unmaintainable and still loses.

Now imagine instead you collect 50,000 emails that humans already labelled spam / not-spam,
and you let an algorithm find the statistical patterns. It discovers thousands of subtle
signals you would never have written by hand, and when spammers change tactics you retrain
rather than rewrite.

**Machine learning is what you reach for when the rules are too many, too subtle, or
too fast-changing for a human to write down.**

## The nested circles

~~~text
+------------------------------------------------------------+
| ARTIFICIAL INTELLIGENCE                                     |
| Any technique that makes machines act intelligently.        |
| Includes search, logic, planning, expert systems, ML.       |
|                                                             |
|   +------------------------------------------------------+  |
|   | MACHINE LEARNING                                     |  |
|   | Systems that improve at a task from data,            |  |
|   | without being explicitly programmed for it.          |  |
|   |                                                      |  |
|   |   +----------------------------------------------+   |  |
|   |   | DEEP LEARNING                                |   |  |
|   |   | ML with many-layered neural networks that    |   |  |
|   |   | learn their own features from raw data.      |   |  |
|   |   |   -> CNNs, RNNs, LSTMs, Transformers, LLMs   |   |  |
|   |   +----------------------------------------------+   |  |
|   +------------------------------------------------------+  |
+------------------------------------------------------------+

DATA SCIENCE overlaps all of the above but is a *different axis*:
it is the practice of getting value out of data - which includes
statistics, cleaning, visualisation, experimentation, communication,
and yes, sometimes ML.
~~~

## The distinction people actually get wrong

Data Science is **not** a subset of AI, and AI is **not** a subset of Data Science.
They overlap. A data scientist who spends the week building dashboards and running an
A/B test did no AI at all - and did excellent data science.

| Field | Core question | Typical output |
|---|---|---|
| Data Science | What is going on, and what should we do? | Analysis, dashboard, experiment, sometimes a model |
| Machine Learning | Can a system learn this mapping from data? | A trained model with measured accuracy |
| Deep Learning | Can a neural network learn the features too? | A network + weights |
| AI (broad) | Can a machine behave intelligently? | An agent, planner, solver, or model |

:::tip A useful test
If your solution would still work with 50 handwritten ~if~ statements, you probably do
not need machine learning. Use ML when the rules are unknown, numerous, or unstable.
:::

## Where "AI" sits today

The phrase "AI" in 2020s industry usually means one of three things:

1. **Predictive ML** - a model outputs a number or a class (churn risk, price, tumour / no tumour).
2. **Generative AI** - a model outputs new content (text, image, audio, code). Mostly Transformers.
3. **Agentic AI** - a model that plans, calls tools, and takes multi-step actions.

You will build all three in this course.
`,
keyPoints: [
  'ML flips programming: data + answers in, rules out.',
  'DL is a subset of ML; ML is a subset of AI; Data Science overlaps all three but is its own discipline.',
  'Reach for ML when rules are unknown, too numerous, or changing.',
  'Deep learning learns its own features; classical ML usually needs you to engineer them.'
],
pitfalls: [
  'Calling every ~if~ statement "AI" - or refusing to call a linear model "AI" out of purism.',
  'Jumping to deep learning for a 500-row spreadsheet. Small tabular data usually wants gradient boosting.',
  'Believing more data always beats better features. Early on, features win.'
],
levels: [
{
name: 'See the inversion',
goal: 'Watch a hand-written rule system lose to a learned one on the same problem.',
md: `
We will build a tiny spam detector twice: once with rules you write, once learned from data.
This is the shortest honest demonstration of what machine learning buys you.

### 1. The hand-written version

~~~python rules_spam.py
# The "classical programming" approach: a human writes the rules.

SPAM_WORDS = ["free", "winner", "prize", "click here", "viagra", "lottery"]

def is_spam_by_rules(text):
    """Return True if the message looks like spam, using hand-written rules."""
    t = text.lower()                       # normalise so 'FREE' matches 'free'
    hits = sum(word in t for word in SPAM_WORDS)   # count how many spam words appear
    return hits >= 1                       # one hit is enough -> call it spam


tests = [
    ("Congratulations, you are a WINNER! Click here for your prize", True),
    ("Hi Maria, are we still on for lunch tomorrow?",                False),
    ("Fr33 m0ney, claim n0w!!!",                                     True),   # obfuscated
    ("The free trial of our library ends Friday - team update",      False),  # innocent 'free'
]

for text, truth in tests:
    pred = is_spam_by_rules(text)
    mark = "OK " if pred == truth else "MISS"
    print(mark, "|", pred, "| expected", truth, "|", text[:45])
~~~

Run it and you will see two failures:

~~~text
OK   | True  | expected True  | Congratulations, you are a WINNER! Click h
OK   | False | expected False | Hi Maria, are we still on for lunch tomorr
MISS | False | expected True  | Fr33 m0ney, claim n0w!!!
MISS | True  | expected False | The free trial of our library ends Friday
~~~

The obfuscated spam slips through, and an innocent email gets flagged.
Your instinct is to add more rules. That instinct is the trap.

### 2. The learned version

~~~python learned_spam.py
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# ---- the data: messages + the answer for each one -------------------
messages = [
    "Congratulations you are a winner claim your prize now",
    "Free money click here limited offer act now",
    "Fr33 m0ney claim n0w winner winner",
    "You have won the lottery send your bank details",
    "Cheap pills no prescription needed buy now",
    "Hi Maria are we still on for lunch tomorrow",
    "The free trial of our library ends Friday team update",
    "Please review the attached quarterly report before Monday",
    "Can you send me the notes from yesterday's lecture",
    "Reminder: dentist appointment on Thursday at 3pm",
]
labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]     # 1 = spam, 0 = not spam

# ---- the model: turn text into counts, then learn word probabilities -
model = make_pipeline(
    CountVectorizer(),      # "free money now" -> {free:1, money:1, now:1}
    MultinomialNB()         # learns P(word | spam) vs P(word | not spam)
)

model.fit(messages, labels)  # <-- this is where the rules get discovered

# ---- try it -----------------------------------------------------------
new = [
    "winner claim your free prize now",
    "are we still on for lunch",
    "the free library trial ends Friday",
]
for text, pred in zip(new, model.predict(new)):
    print(("SPAM    " if pred == 1 else "NOT SPAM"), "|", text)
~~~

~~~text
SPAM     | winner claim your free prize now
NOT SPAM | are we still on for lunch
NOT SPAM | the free library trial ends Friday
~~~

Notice: **you never told it that "free" near "prize" is worse than "free" near "library".**
It worked that out from the ten examples. Give it 50,000 and it gets genuinely good.

:::note What just happened
~CountVectorizer~ turned each message into a vector of word counts.
~MultinomialNB~ estimated, for every word, how much more often it appears in spam than
in normal mail, and combined those into a probability. That combination *is* the learned
rule set - thousands of soft rules instead of six hard ones.
:::
`
},
{
name: 'Draw the boundary',
goal: 'Make the abstract idea of "learning a rule" visible as a line on a plot.',
md: `
Learning is easier to believe when you can see it. Here a model learns to separate two
groups of points - and we draw the rule it invented.

~~~python see_the_rule.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

# 1. Make 200 points in two clusters. y is the answer (0 or 1) for each point.
X, y = make_blobs(n_samples=200, centers=2, cluster_std=1.6, random_state=7)

# 2. Learn a rule that separates them.
clf = LogisticRegression().fit(X, y)

# 3. Ask the model about every point on a fine grid, so we can paint its decision.
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]         # shape (90000, 2): every grid point
Z = clf.predict(grid).reshape(xx.shape)      # the model's answer everywhere

# 4. Plot the learned region + the training points.
plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, Z, alpha=0.25, cmap="coolwarm")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=45)
plt.title("The rule the model invented from the data")
plt.xlabel("feature 1"); plt.ylabel("feature 2")
plt.tight_layout(); plt.show()

# The rule in numbers: w1*x1 + w2*x2 + b = 0 is the boundary line
w1, w2 = clf.coef_[0]
b = clf.intercept_[0]
print(f"Learned rule: predict class 1 when {w1:.2f}*x1 + {w2:.2f}*x2 + {b:.2f} > 0")
~~~

You get a coloured plane split by a straight line, and a printed rule such as:

~~~text
Learned rule: predict class 1 when -1.83*x1 + 2.41*x2 + 1.07 > 0
~~~

**That printed line is the model.** Three numbers. Nobody wrote them; they were found by
minimising mistakes on the 200 examples.

:::tip Do this
Change ~cluster_std=1.6~ to ~4.0~ and rerun. The clusters overlap, the line can no longer
separate them cleanly, and some points end up on the wrong side. That is your first
encounter with *irreducible error* - noise no model can remove.
:::
`
},
{
name: 'Where each field is used',
goal: 'Map the four terms onto real jobs and real deliverables so you can tell them apart in the wild.',
md: `
## Four days in four jobs

**Data Analyst / Data Scientist (analysis mode)**
Question: "Why did revenue drop 8% in the Athens region last quarter?"
Tools: SQL, pandas, a plotting library, a statistics test.
Deliverable: a notebook and a five-slide answer. No model is trained.

**Machine Learning Engineer**
Question: "Can we predict which customers will cancel next month?"
Tools: pandas, scikit-learn / XGBoost, cross-validation, a serving API.
Deliverable: a model with a measured recall, deployed behind an endpoint, monitored for drift.

**Deep Learning / Research Engineer**
Question: "Can we read the meter number from a photo of a utility meter?"
Tools: PyTorch, a CNN or a vision transformer, GPUs, data augmentation.
Deliverable: a network, its weights, and an accuracy number on a held-out test set.

**AI Engineer (the 2020s job)**
Question: "Can support agents ask our documentation in plain language?"
Tools: an LLM, embeddings, a vector store, retrieval, evaluation harness, guardrails.
Deliverable: a retrieval-augmented system with latency and quality measured.

## The same problem, four ways

Take "detect fraudulent transactions":

| Approach | What it looks like | When it wins |
|---|---|---|
| Rules / expert system | ~if amount > 5000 and country != home_country~ | Regulation demands explainability; very little data |
| Classical ML | Gradient boosting on 60 engineered features | Tabular data, 10k-10M rows. **Usually the winner.** |
| Deep learning | An LSTM over the user's transaction sequence | Order and timing matter a lot; lots of data |
| LLM / agent | A model reasoning over the case file and calling tools | Messy unstructured evidence, human-in-the-loop |

:::warn The most common career mistake
Beginners skip straight to deep learning because it sounds impressive. In industry,
**most tabular problems are still won by gradient boosting**, and most value comes from
better data, not fancier models. Learn the classical track properly - it is what you will
actually be paid to do first.
:::

## What this course does with that

- Tracks 1-4 build the foundation everyone needs: Python, math intuition, data engineering.
- Tracks 5-8 make you genuinely good at classical ML - the bread and butter.
- Tracks 9-14 take you through deep learning, vision, sequences, transformers and generative models.
- Track 15 covers AI fundamentals proper: search, logic, planning, reinforcement learning, agents.
- Track 16 gets it into production; Track 17 gives you portfolio projects.
`
}
],
quiz: [
{
q: 'You have 800 rows of customer data in a spreadsheet and want to predict churn. What is the sensible first model?',
options: [
  'A deep neural network with 6 hidden layers',
  'Logistic regression or gradient boosting',
  'A large language model with a custom prompt',
  'A hand-written rule system'
],
answer: 1,
why: 'Small tabular data is classical-ML territory. Deep nets need far more rows to beat gradient boosting here, and starting simple gives you a baseline to beat.'
},
{
q: 'What makes machine learning different from ordinary programming?',
options: [
  'It runs on GPUs instead of CPUs',
  'It uses statistics instead of logic',
  'You supply data and answers, and the rules are learned rather than written',
  'It is always more accurate'
],
answer: 2,
why: 'The inversion is the definition: rules come out rather than going in. GPUs and statistics are implementation details, and ML is frequently less accurate than a good rule when the rule is knowable.'
},
{
q: 'Which statement is true?',
options: [
  'Deep learning is a subset of machine learning, which is a subset of AI',
  'AI is a subset of machine learning',
  'Data science is a subset of deep learning',
  'Machine learning and AI are exactly the same thing'
],
answer: 0,
why: 'The nesting runs AI > ML > DL. Data science overlaps all of them but sits on a different axis - it is about extracting value from data, model or no model.'
}
]
},

/* ============================================================ */
{
id: 'types-of-learning',
title: 'The four types of learning',
summary: 'Supervised, unsupervised, semi-supervised and reinforcement learning - what the data looks like in each, and how to recognise which one your problem is.',
tags: ['theory', 'taxonomy'],
intro: `
## The question that decides everything

Before choosing an algorithm, answer one question:

> **Do I have the answers?**

That single question splits machine learning into its main families.

~~~text
Do you have labelled answers for your examples?

  YES, for all of them ........... SUPERVISED LEARNING
     |-- answer is a number ......... regression
     |-- answer is a category ....... classification

  NO, none at all ................ UNSUPERVISED LEARNING
     |-- group similar things ....... clustering
     |-- compress / simplify ........ dimensionality reduction
     |-- find odd ones out .......... anomaly detection

  SOME of them ................... SEMI-SUPERVISED / SELF-SUPERVISED
     |-- use structure in the unlabelled data to help

  NO answers, but a reward signal  REINFORCEMENT LEARNING
     |-- agent acts, world responds, agent learns from consequences
~~~

## 1. Supervised learning

You have ~X~ (inputs) and ~y~ (the correct answers). The model learns the mapping X -> y.

- **Regression**: y is a continuous number. House price, temperature, delivery time.
- **Classification**: y is a category. Spam / not spam, tumour type, which of 1000 objects.

This is 90% of deployed machine learning, because labels are what businesses actually have:
past sales, past churn, past diagnoses.

## 2. Unsupervised learning

You have ~X~ only. There is no answer key, so the model looks for structure.

- **Clustering**: "which of these customers behave alike?" (K-Means, DBSCAN)
- **Dimensionality reduction**: "can I describe these 200 columns with 10?" (PCA, t-SNE)
- **Anomaly detection**: "which transactions look nothing like the rest?"

The catch: **there is no objectively right answer**, so evaluation is judgement-heavy.

## 3. Semi-supervised and self-supervised

Labelling is expensive; raw data is cheap. So:

- **Semi-supervised**: 1,000 labelled X-rays + 100,000 unlabelled ones. Use the unlabelled
  ones to learn the shape of the data, the labelled ones to attach meaning.
- **Self-supervised**: invent labels from the data itself. Hide a word and predict it.
  Hide a patch of an image and reconstruct it. **This is how GPT and BERT are trained** -
  and it is why they can use the entire internet without a single human annotator.

## 4. Reinforcement learning

No dataset. An **agent** takes actions in an **environment**, receives **rewards**, and
learns a **policy** that maximises reward over time.

~~~text
        +-----------+   action a_t    +--------------+
        |   AGENT   | --------------> | ENVIRONMENT  |
        |  policy   | <-------------- |              |
        +-----------+  state s_t+1    +--------------+
                       reward r_t+1
~~~

Game playing, robotics, recommendation ordering, and the RLHF step that makes chat models
helpful. Powerful, sample-hungry, and fiddly - covered properly in the AI Fundamentals track.
`,
keyPoints: [
  '"Do I have the answers?" is the first question, and it picks the family.',
  'Regression predicts a number; classification predicts a category. Same family, different output.',
  'Unsupervised learning has no ground truth, so evaluating it always involves judgement.',
  'Self-supervised learning invents its own labels - it is why LLMs can train on raw internet text.'
],
pitfalls: [
  'Treating a ranked category (small/medium/large) as plain classification and throwing away the order.',
  'Using clustering when you actually have labels - if you have y, use it.',
  'Reaching for reinforcement learning for a problem that is really just supervised prediction.'
],
levels: [
{
name: 'One dataset, three lenses',
goal: 'Run supervised, unsupervised and anomaly detection on the SAME data and see how the questions differ.',
md: `
We will use the classic iris flowers dataset: 150 flowers, 4 measurements each, 3 species.

~~~python three_lenses.py
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data = load_iris()
X, y = data.data, data.target        # X: 150x4 measurements, y: 0/1/2 species
print("X shape:", X.shape, " y shape:", y.shape)

# ---------------------------------------------------------------
# LENS 1 - SUPERVISED: "given the measurements, which species is it?"
#          We USE y. Success = accuracy on flowers it never saw.
# ---------------------------------------------------------------
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
clf = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
print("Supervised accuracy:", round(accuracy_score(y_te, clf.predict(X_te)), 3))

# ---------------------------------------------------------------
# LENS 2 - UNSUPERVISED: "are there natural groups here at all?"
#          We HIDE y. The model never sees a species name.
# ---------------------------------------------------------------
km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X)
print("Cluster sizes:", np.bincount(km.labels_))

# Cluster ids are arbitrary (cluster 0 is not species 0), so to check
# quality we ask: within each cluster, how pure is the true species mix?
for c in range(3):
    species_in_cluster = y[km.labels_ == c]
    counts = np.bincount(species_in_cluster, minlength=3)
    purity = counts.max() / counts.sum()
    print(f"  cluster {c}: n={counts.sum():3d}  species mix={counts}  purity={purity:.2f}")

# ---------------------------------------------------------------
# LENS 3 - ANOMALY DETECTION: "which flowers are weird?"
#          No labels, no groups - just typical vs unusual.
# ---------------------------------------------------------------
iso = IsolationForest(contamination=0.05, random_state=0).fit(X)
outliers = np.where(iso.predict(X) == -1)[0]
print("Flagged as unusual:", outliers)
print("Their measurements:\\n", X[outliers])
~~~

Typical output:

~~~text
X shape: (150, 4)  y shape: (150,)
Supervised accuracy: 0.978
Cluster sizes: [50 62 38]
  cluster 0: n= 50  species mix=[50  0  0]  purity=1.00
  cluster 1: n= 62  species mix=[ 0 48 14]  purity=0.77
  cluster 2: n= 38  species mix=[ 0  2 36]  purity=0.95
Flagged as unusual: [ 13  14  15  22  32  33  41 109 117 118]
~~~

**Read that carefully.** The unsupervised model, with *no labels at all*, recovered one
species perfectly and mostly separated the other two. That is real structure in the data -
and it is exactly why clustering is useful when you have no answer key.

:::note The honest limitation
Cluster 1 mixes two species because those two genuinely overlap in these four measurements.
A supervised model gets 97.8% because the labels tell it where to draw the tricky line.
**Labels are information.** Never throw them away.
:::
`
},
{
name: 'Regression vs classification',
goal: 'Feel the difference between predicting a number and predicting a category, including the trap of converting one to the other.',
md: `
Same family, different output type - and the metrics, loss functions and mistakes all differ.

~~~python reg_vs_clf.py
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, classification_report

hous = fetch_california_housing()
X, y_price = hous.data, hous.target          # y_price = median house value ($100k units)
X_tr, X_te, y_tr, y_te = train_test_split(X, y_price, test_size=0.2, random_state=0)

# ---------------- REGRESSION: predict the actual number -------------
reg = LinearRegression().fit(X_tr, y_tr)
pred = reg.predict(X_te)
print("REGRESSION")
print("  mean absolute error:", round(mean_absolute_error(y_te, pred), 3), "(in $100k)")
print("  R2 score           :", round(r2_score(y_te, pred), 3))
print("  example: true 2.15 -> predicted", round(pred[0], 2))

# ---------------- CLASSIFICATION: predict a bucket ------------------
# Turn the same target into a category: is this house 'expensive'?
threshold = np.median(y_price)
y_class = (y_price > threshold).astype(int)      # 1 = above median, 0 = below
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X, y_class, test_size=0.2, random_state=0)

clf = LogisticRegression(max_iter=1000).fit(Xc_tr, yc_tr)
print("\\nCLASSIFICATION")
print("  accuracy:", round(accuracy_score(yc_te, clf.predict(Xc_te)), 3))
print(classification_report(yc_te, clf.predict(Xc_te),
                            target_names=["below median", "above median"]))

# The probability, not just the label - often what you actually want
proba = clf.predict_proba(Xc_te[:5])[:, 1]
print("  P(expensive) for 5 houses:", np.round(proba, 3))
~~~

### The trap you just walked into

We turned a number into a category by thresholding at the median. That is sometimes right
and often a mistake:

- A house at 1.99 and a house at 2.01 are nearly identical, but land in opposite classes.
- We threw away the difference between "slightly above median" and "a mansion".
- We made the problem *look* easier (accuracy 0.83 sounds better than R2 0.58) without
  making the prediction more useful.

:::warn Rule of thumb
**If the target is naturally a number, predict the number.** Bucket it afterwards if the
business needs a yes/no. Bucketing before modelling destroys information permanently.
:::

### Which metric for which

| | Regression | Classification |
|---|---|---|
| Output | continuous number | class label or probability |
| Typical loss | Mean Squared Error | Cross-entropy (log loss) |
| Typical metrics | MAE, RMSE, R2 | Accuracy, precision, recall, F1, ROC-AUC |
| "How wrong" means | distance | right or wrong, plus confidence |
| Watch out for | outliers dragging MSE | class imbalance faking accuracy |
`
},
{
name: 'Self-supervised: how LLMs learn without labels',
goal: 'Build the actual training objective behind GPT-style models, in 30 lines, so the idea stops being magic.',
md: `
Self-supervised learning is the trick that unlocked modern AI: **create labels from the
data itself**. For text, the label for a sequence is simply *the next word*.

~~~python next_word.py
"""
The GPT objective, stripped to its bones:
  input  = all words up to position t
  label  = the word at position t+1
No human ever labelled anything. The text labels itself.
"""
from collections import defaultdict, Counter
import random

text = """
the cat sat on the mat the cat ate the fish the dog sat on the rug
the dog ate the bone the cat sat on the rug the bird sat on the fence
""".split()

# ---- build the "dataset": every (context -> next word) pair ----------
CONTEXT = 2                      # how many previous words we look at
pairs = []
for i in range(CONTEXT, len(text)):
    context = tuple(text[i - CONTEXT:i])   # the input  (X)
    target = text[i]                       # the label  (y)  <- free!
    pairs.append((context, target))

print("First 5 training examples created from raw text with no annotator:")
for ctx, tgt in pairs[:5]:
    print("   ", ctx, "->", tgt)

# ---- 'train': count how often each context is followed by each word --
model = defaultdict(Counter)
for context, target in pairs:
    model[context][target] += 1

# ---- generate: repeatedly predict the next word ----------------------
def generate(seed, n_words=12):
    out = list(seed)
    for _ in range(n_words):
        ctx = tuple(out[-CONTEXT:])
        if ctx not in model:
            break
        # sample proportionally to how often each word followed this context
        words, counts = zip(*model[ctx].items())
        out.append(random.choices(words, weights=counts)[0])
    return " ".join(out)

random.seed(0)
print("\\nGenerated:", generate(["the", "cat"]))
print("Generated:", generate(["the", "dog"]))
~~~

~~~text
First 5 training examples created from raw text with no annotator:
    ('the', 'cat') -> sat
    ('cat', 'sat') -> on
    ('sat', 'on') -> the
    ('on', 'the') -> mat
    ('the', 'mat') -> the

Generated: the cat sat on the rug the dog ate the bone the cat
Generated: the dog sat on the rug the cat ate the fish the dog
~~~

**That is a language model.** A real one differs in three ways only:

1. It uses a neural network (a Transformer) instead of a counting table, so it can
   generalise to contexts it never saw.
2. Its context is thousands of tokens, not two.
3. It is trained on trillions of tokens instead of thirty.

The *objective* - predict the next token - is identical. You will build the real version
in the Transformers track.

:::tip Why this changed everything
Supervised learning is bottlenecked by how many labels humans can produce.
Self-supervised learning removed the bottleneck: every sentence ever written is training
data. That is the entire reason 2020s AI happened when it did.
:::
`
}
],
quiz: [
{
q: 'You want to group 40,000 customers into segments for a marketing campaign. Nobody has labelled them. Which family?',
options: ['Supervised classification', 'Unsupervised clustering', 'Reinforcement learning', 'Regression'],
answer: 1,
why: 'No labels + "find natural groups" is exactly clustering. If marketing later hand-labels segments, you could switch to supervised classification for new customers.'
},
{
q: 'Predicting how many minutes a delivery will take is:',
options: ['Classification', 'Clustering', 'Regression', 'Anomaly detection'],
answer: 2,
why: 'The target is a continuous number, so it is regression. Bucketing it into fast/slow first would throw away information.'
},
{
q: 'Why is self-supervised learning so important for large language models?',
options: [
  'It is faster to compute than supervised learning',
  'It generates labels from the data itself, removing the human-annotation bottleneck',
  'It needs less data than supervised learning',
  'It guarantees the model tells the truth'
],
answer: 1,
why: 'Labels come free from the text (predict the next token), so training can scale to trillions of tokens. It needs far MORE data, not less - but that data is essentially free.'
}
]
},

/* ============================================================ */
{
id: 'ml-workflow',
title: 'The end-to-end ML workflow',
summary: 'The seven stages every real project passes through, why modelling is the smallest of them, and a runnable skeleton you can reuse for any problem.',
tags: ['process', 'practical'],
intro: `
## Where the time actually goes

Beginners imagine a machine learning project is mostly modelling. It is not.

~~~text
  Framing the problem     ####                        8%
  Getting the data        ########                   15%
  Cleaning + EDA          ##################         35%   <-- the real job
  Feature engineering     ##########                 20%
  Modelling               #####                      10%
  Evaluation              ####                        7%
  Deployment + monitoring #####                       5%+ (forever)
~~~

## The seven stages

**1. Frame the problem.**
Turn a business question into a prediction task with a measurable target.
"Reduce churn" is not a task. "Predict, for each active subscriber, the probability they
cancel within 30 days, so we can offer the top 5% a discount" is.

Decide *now*: what is X, what is y, what metric decides success, and what a useful
model must beat (the baseline).

**2. Get the data.**
CSV, database, API, scraping, logs. Ask immediately: *would this column have been
available at prediction time?* If not, it is leakage and must go.

**3. Explore and clean (EDA).**
Look at every column. Distributions, missing values, outliers, duplicates, wrong types,
impossible values (age = 200, price = -5). Plot things. This is where you find the bugs
that would otherwise silently ruin your model.

**4. Engineer features.**
Encode categories, scale numbers, extract date parts, create ratios, handle text.
Good features on a simple model beat bad features on a fancy one, essentially always.

**5. Model.**
Start with a dumb baseline. Then a simple model. Then a strong one. Compare honestly
using cross-validation.

**6. Evaluate.**
On data the model has never seen, with a metric that matches the business cost of being
wrong. Look at *which* examples it gets wrong, not just the score.

**7. Deploy and monitor.**
Save the model, serve it, watch for drift, retrain. A model that is never used has zero value.

:::warn The golden rule
The test set is touched **once**, at the very end. Every time you look at it and then
change something, you leak a little information into your model and your reported score
becomes a lie.
:::
`,
keyPoints: [
  'Framing the problem well is worth more than any algorithm choice.',
  'Cleaning and EDA take more time than modelling, and prevent more errors.',
  'Always build a dumb baseline first - it tells you whether the model is doing anything.',
  'The test set is opened once, at the end.'
],
pitfalls: [
  'Scaling or imputing before splitting - the test set then influences training. This is leakage.',
  'Tuning on the test set, then reporting the test score as if it were unbiased.',
  'Optimising accuracy when the business cares about recall (or cost in euros).',
  'Including a column that would not exist at prediction time (e.g. ~cancellation_date~ when predicting churn).'
],
levels: [
{
name: 'The seven-stage skeleton',
goal: 'A single runnable script that walks all seven stages on a real dataset - your reusable template.',
md: `
Save this as your project template. Every future project is a variation of it.

~~~python workflow_skeleton.py
"""
END-TO-END ML WORKFLOW - the reusable skeleton.
Task: predict whether a passenger survived the Titanic (binary classification).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =====================================================================
# STAGE 1 - FRAME
#   X = passenger details available at boarding
#   y = survived (1) or not (0)
#   metric = accuracy, but we will also look at recall for survivors
#   baseline to beat = always predict the majority class
# =====================================================================

# =====================================================================
# STAGE 2 - GET THE DATA
# =====================================================================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("Loaded:", df.shape)

# =====================================================================
# STAGE 3 - EXPLORE AND CLEAN
# =====================================================================
print("\\n--- first look ---")
print(df.head(3))
print("\\n--- types and missing ---")
print(pd.DataFrame({
    "dtype": df.dtypes,
    "missing": df.isna().sum(),
    "missing_%": (df.isna().mean() * 100).round(1),
    "unique": df.nunique(),
}))
print("\\n--- target balance ---")
print(df["Survived"].value_counts(normalize=True).round(3))

# Drop columns that are identifiers or almost entirely missing
df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# =====================================================================
# STAGE 4 - FEATURES
#   Note: we DEFINE the transformations here but do NOT apply them yet.
#   They go inside a Pipeline so they are fitted on training data only.
# =====================================================================
y = df["Survived"]
X = df.drop(columns=["Survived"])

num_cols = ["Age", "SibSp", "Parch", "Fare"]
cat_cols = ["Pclass", "Sex", "Embarked"]

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # Age has gaps
        ("scale",  StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

# =====================================================================
# STAGE 5 - MODEL  (split FIRST, always)
# =====================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y   # stratify keeps class balance
)

models = {
    "baseline (majority)": DummyClassifier(strategy="most_frequent"),
    "logistic regression": LogisticRegression(max_iter=1000),
    "random forest":       RandomForestClassifier(n_estimators=300, random_state=42),
}

print("\\n--- cross-validated accuracy on TRAINING data ---")
fitted = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", clf)])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    print(f"  {name:22s}: {scores.mean():.3f} (+/- {scores.std():.3f})")
    fitted[name] = pipe.fit(X_train, y_train)

# =====================================================================
# STAGE 6 - EVALUATE  (test set opened exactly once)
# =====================================================================
best = fitted["random forest"]
y_pred = best.predict(X_test)
print("\\n--- FINAL test-set result ---")
print("accuracy:", round(accuracy_score(y_test, y_pred), 3))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["died", "survived"]))

# =====================================================================
# STAGE 7 - SAVE FOR DEPLOYMENT
# =====================================================================
import joblib
joblib.dump(best, "titanic_model.joblib")
print("\\nSaved titanic_model.joblib - the whole pipeline, preprocessing included.")
~~~

Expected output (abridged):

~~~text
--- cross-validated accuracy on TRAINING data ---
  baseline (majority)   : 0.616 (+/- 0.003)
  logistic regression   : 0.792 (+/- 0.031)
  random forest         : 0.818 (+/- 0.028)

--- FINAL test-set result ---
accuracy: 0.816
[[92 13]
 [20 54]]
~~~

:::tip Read the baseline first
Always predicting "died" gets 61.6%. Your model gets 81.6%. **The model is worth 20 points
of accuracy** - that is the real headline, not the 81.6.
:::

### The two things this skeleton gets right

1. **The split happens before any fitting.** Imputation and scaling live inside the
   ~Pipeline~, so during cross-validation they are refitted on each fold's training part only.
2. **Saving the pipeline, not the model.** ~titanic_model.joblib~ contains the imputer, the
   scaler, the encoder *and* the forest. At serving time you feed it a raw row and it just works.
`
},
{
name: 'Framing: turning a business question into a task',
goal: 'Practise the stage everyone skips - converting vague requests into a specific, measurable prediction problem.',
md: `
## The framing checklist

For every project, write these seven lines *before* opening a notebook:

~~~text
1. Business question :
2. Prediction target (y) :
3. Unit of prediction (one row = ?) :
4. Features (X) available AT PREDICTION TIME :
5. Metric, and why that metric :
6. Baseline to beat :
7. What action is taken with the prediction :
~~~

If you cannot fill in line 7, stop. A prediction nobody acts on is a hobby, not a project.

## Worked example: "reduce customer churn"

~~~text
1. Business question : Too many subscribers cancel; can we intervene before they do?
2. Prediction target : will this subscriber cancel within the next 30 days? (binary)
3. Unit of prediction : one active subscriber, evaluated on the 1st of each month
4. Features           : tenure, plan, monthly spend, logins last 30d, support tickets,
                        payment failures, NPS score.
                        NOT: cancellation_date, refund_issued  <-- these leak the answer
5. Metric             : recall at the top 5% of predicted risk. We can only afford to
                        contact 5% of the base, so we care about catching real churners
                        inside that budget - not overall accuracy.
6. Baseline           : contact the 5% with the lowest login count. Currently catches 11%
                        of churners. Model must beat that.
7. Action             : send those 5% a retention offer.
~~~

Notice line 5. **Accuracy would be a terrible metric here**: if 3% of subscribers churn,
predicting "nobody churns" is 97% accurate and completely useless.

## The leakage test, in code

The single most expensive bug in applied ML is a feature that would not exist at
prediction time. Here is how it looks when it bites you:

~~~python leakage_demo.py
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

rng = np.random.default_rng(0)
n = 2000

tenure   = rng.integers(1, 60, n)
logins   = rng.poisson(12, n)
# true churn depends on tenure and logins, plus noise
risk     = 1 / (1 + np.exp(-(-2.0 + 0.06*(24 - tenure) + 0.10*(8 - logins))))
churned  = (rng.random(n) < risk).astype(int)

# A leaky column: refunds are only issued AFTER someone cancels.
refund_issued = np.where(churned == 1, rng.random(n) < 0.8, rng.random(n) < 0.02).astype(int)

df = pd.DataFrame({"tenure": tenure, "logins": logins,
                   "refund_issued": refund_issued, "churned": churned})

def run(cols, label):
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[cols], df["churned"], test_size=0.3, random_state=0)
    m = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
    print(f"{label:32s} accuracy = {accuracy_score(y_te, m.predict(X_te)):.3f}")

run(["tenure", "logins"],                  "honest features")
run(["tenure", "logins", "refund_issued"], "WITH leaky feature")
~~~

~~~text
honest features                  accuracy = 0.742
WITH leaky feature               accuracy = 0.915
~~~

The leaky model looks spectacular in your notebook and is worthless in production,
because on the 1st of the month, for an active subscriber, ~refund_issued~ is always 0.
You will have shipped a model that has learned "people who already got a refund have churned".

:::danger How to catch leakage
For every feature ask: *if I froze the world at prediction time, could I compute this?*
Any column whose value is determined by the outcome - or recorded after it - is leakage.
Suspiciously high accuracy is the symptom; this question is the diagnosis.
:::
`
},
{
name: 'Baselines: what "good" even means',
goal: 'Build the four standard baselines so you never again report a score without context.',
md: `
A score with no baseline is meaningless. Is 0.83 R2 good? Depends entirely on what
predicting the mean would have got you.

~~~python baselines.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

# ================= CLASSIFICATION BASELINES =========================
Xc, yc = load_breast_cancer(return_X_y=True)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.25,
                                              random_state=0, stratify=yc)

print("CLASSIFICATION  (breast cancer, 2 classes)")
print(f"  class balance in test set: {np.bincount(yc_te) / len(yc_te)}")

candidates = {
    "1. always majority class": DummyClassifier(strategy="most_frequent"),
    "2. random, matching rates": DummyClassifier(strategy="stratified", random_state=0),
    "3. logistic regression":    LogisticRegression(max_iter=5000),
    "4. random forest":          RandomForestClassifier(n_estimators=300, random_state=0),
}
for name, m in candidates.items():
    m.fit(Xc_tr, yc_tr)
    p = m.predict(Xc_te)
    print(f"  {name:26s} acc={accuracy_score(yc_te, p):.3f}  f1={f1_score(yc_te, p):.3f}")

# ================= REGRESSION BASELINES =============================
Xr, yr = fetch_california_housing(return_X_y=True)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.25, random_state=0)

print("\\nREGRESSION  (California housing, target in $100k)")
regs = {
    "1. always predict the mean":   DummyRegressor(strategy="mean"),
    "2. always predict the median": DummyRegressor(strategy="median"),
    "3. linear regression":         LinearRegression(),
}
for name, m in regs.items():
    m.fit(Xr_tr, yr_tr)
    p = m.predict(Xr_te)
    print(f"  {name:29s} MAE={mean_absolute_error(yr_te, p):.3f}  R2={r2_score(yr_te, p):6.3f}")
~~~

~~~text
CLASSIFICATION  (breast cancer, 2 classes)
  class balance in test set: [0.371 0.629]
  1. always majority class   acc=0.629  f1=0.772
  2. random, matching rates  acc=0.545  f1=0.629
  3. logistic regression     acc=0.958  f1=0.967
  4. random forest           acc=0.965  f1=0.972

REGRESSION  (California housing, target in $100k)
  1. always predict the mean    MAE=0.911  R2=-0.000
  2. always predict the median  MAE=0.869  R2=-0.033
  3. linear regression          MAE=0.531  R2= 0.596
~~~

### How to read this

- **R2 = 0 means "no better than predicting the mean".** That is what R2 is defined against.
  A negative R2 means you are *worse* than the mean - which the median baseline is here,
  because the target is skewed.
- Accuracy 0.629 for a model that has learned nothing tells you the true bar. Reporting
  "our model is 96% accurate" is only impressive against that 63%.
- The F1 of 0.772 for the majority-class dummy is the strongest warning in this output:
  **on imbalanced data even F1 can look respectable for a useless model.**

:::tip The four baselines to always run
1. Majority class / mean - the floor.
2. A single simple feature ("predict churn = lowest logins") - the business's current heuristic.
3. Linear or logistic regression - the "is this problem even non-linear?" probe.
4. Last year's model, if one exists - the only baseline your manager cares about.
:::
`
}
],
quiz: [
{
q: 'Where should scaling and imputation live in a cross-validated workflow?',
options: [
  'Applied to the whole dataset before splitting, for consistency',
  'Inside a Pipeline, so they are refitted on each training fold only',
  'Applied to the test set separately with its own statistics',
  'Skipped entirely when using tree models'
],
answer: 1,
why: 'Fitting transformers on all data lets test-set statistics influence training - leakage. A Pipeline refits them per fold. (Trees do not need scaling, but they still need imputation handled correctly.)'
},
{
q: 'Your churn model reports 97% accuracy. Only 3% of customers actually churn. What is the most likely explanation?',
options: [
  'The model is excellent',
  'The model predicts "no churn" for everyone',
  'There is too much data',
  'The learning rate is too high'
],
answer: 1,
why: 'Predicting the majority class always gives 97% here. This is exactly why you compute a DummyClassifier baseline and look at recall or precision instead of accuracy.'
},
{
q: 'Which feature is leakage when predicting whether a subscriber will churn next month?',
options: ['Tenure in months', 'Number of logins in the last 30 days', 'Whether a cancellation refund was issued', 'Monthly spend'],
answer: 2,
why: 'Refunds are issued after cancellation, so the column encodes the answer. At real prediction time it would always be 0 for active subscribers, and the model would collapse.'
}
]
},

/* ============================================================ */
{
id: 'data-basics',
title: 'Data, features and labels',
summary: 'The vocabulary of every dataset - rows, features, targets, dtypes - and the four shapes of data you will meet: tabular, image, text and time series.',
tags: ['theory', 'data'],
intro: `
## The vocabulary

~~~text
                     FEATURES (X)                       TARGET (y)
        +---------------------------------------+     +-----------+
        | age | income | city    | has_car      |     | churned   |
  row 0 |  34 |  28000 | Athens  | True         |     |     0     |  <- one sample /
  row 1 |  51 |  61000 | Patras  | False        |     |     1     |     example /
  row 2 |  22 |  15000 | Athens  | True         |     |     0     |     observation
        +---------------------------------------+     +-----------+
           |       |       |         |
        numeric numeric categorical boolean          <- dtypes matter enormously

  shape: (n_samples, n_features)  =  (3, 4)
~~~

Words that mean the same thing, depending on who is talking:

| Term | Also called |
|---|---|
| Sample | row, record, observation, example, instance, data point |
| Feature | column, variable, predictor, attribute, covariate, input, X |
| Target | label, response, outcome, dependent variable, ground truth, y |

## Feature types, and why they decide your preprocessing

**Numerical**
- *Continuous*: height, price, temperature. Any value in a range.
- *Discrete*: number of children, page views. Whole counts.

**Categorical**
- *Nominal*: no order. City, colour, browser. -> one-hot encode.
- *Ordinal*: has order. small < medium < large, or 1-5 star rating. -> ordinal encode, keep the order.
- *Binary*: two values. -> a single 0/1 column.

**Datetime** - never feed a raw timestamp. Extract year, month, day-of-week, hour,
is_weekend, days_since_signup. The *parts* carry the signal.

**Text** - needs vectorising (bag of words, TF-IDF, embeddings).

**Image** - a 3-D array of pixel intensities: (height, width, channels).

:::warn The most common beginner bug
A column of ~"1", "2", "3"~ read as strings, or a category encoded as ~1, 2, 3~ and treated
as numeric. In the second case the model believes category 3 is "three times" category 1,
and that Athens + Patras = Volos. Always check ~df.dtypes~ first.
:::

## The four shapes of data

~~~text
TABULAR      (n_samples, n_features)          -> sklearn, XGBoost
             1000 customers x 20 columns

IMAGE        (n_samples, H, W, C)             -> CNNs, ViT
             50000 photos x 32 x 32 x 3

TEXT         (n_samples, sequence_length)     -> RNN/LSTM, Transformers
             10000 reviews x up to 512 tokens

TIME SERIES  (n_timesteps, n_features)        -> ARIMA, LSTM, temporal CNN
             3 years of daily sales x 5 signals   ** order matters, so never shuffle **
~~~
`,
keyPoints: [
  'X is (n_samples, n_features); y is (n_samples,). Getting shapes right prevents most errors.',
  'Nominal categories need one-hot; ordinal categories should keep their order.',
  'Never feed a raw datetime - extract its parts.',
  'Time series must not be shuffled: the split has to respect time.'
],
pitfalls: [
  'Encoding nominal categories as 1,2,3 - the model then invents an ordering that does not exist.',
  'Leaving IDs (customer_id, row_number) in X. The model memorises them and learns nothing.',
  'Forgetting that y must not appear inside X in any disguised form.'
],
levels: [
{
name: 'Anatomy of a dataset',
goal: 'Load a messy dataset and produce a complete profile of every column - the first thing you do on any project.',
md: `
~~~python profile.py
import numpy as np
import pandas as pd

# A deliberately messy dataset with one of every problem
df = pd.DataFrame({
    "customer_id": [101, 102, 103, 104, 105, 106, 107, 108],
    "age":         [34, 51, 22, np.nan, 45, 29, 200, 38],       # missing + impossible
    "income":      ["28000", "61000", "15000", "42000",         # numbers stored as text!
                    "39000", "33000", "88000", None],
    "city":        ["Athens", "Patras", "Athens", "Volos",
                    "Athens", "Patras", "Volos", "Athens"],     # nominal
    "plan":        ["small", "large", "small", "medium",
                    "medium", "small", "large", "medium"],      # ORDINAL
    "signup_date": pd.to_datetime(["2023-01-15", "2022-06-02", "2023-11-30", "2021-03-19",
                                   "2023-05-05", "2024-02-14", "2020-08-08", "2023-09-01"]),
    "churned":     [0, 1, 0, 1, 0, 0, 1, 0],                    # the TARGET
})

def profile(df):
    """The report you should generate for every new dataset."""
    out = pd.DataFrame({
        "dtype":      df.dtypes.astype(str),
        "non_null":   df.notna().sum(),
        "missing":    df.isna().sum(),
        "missing_%":  (df.isna().mean() * 100).round(1),
        "unique":     df.nunique(),
        "sample":     [df[c].dropna().iloc[0] if df[c].notna().any() else None
                       for c in df.columns],
    })
    return out

print("SHAPE:", df.shape, " (rows, columns)")
print("\\n=== COLUMN PROFILE ===")
print(profile(df))

print("\\n=== NUMERIC SUMMARY ===")
print(df.describe().T.round(2))

print("\\n=== SUSPICIOUS THINGS ===")
print("  income dtype is", df["income"].dtype, "-> it is TEXT, must be converted")
print("  max age is", df["age"].max(), "-> impossible, needs handling")
print("  customer_id has", df["customer_id"].nunique(), "unique values out of",
      len(df), "-> it is an ID, must be DROPPED from features")
~~~

~~~text
SHAPE: (8, 7)  (rows, columns)

=== COLUMN PROFILE ===
               dtype  non_null  missing  missing_%  unique               sample
customer_id    int64         8        0        0.0       8                  101
age          float64         7        1       12.5       7                 34.0
income        object         7        1       12.5       7                28000
city          object         8        0        0.0       3               Athens
plan          object         8        0        0.0       3                small
signup_date   datetime64[ns] 8       0        0.0       8  2023-01-15 00:00:00
churned        int64         8        0        0.0       2                    0
~~~

### Now fix each problem

~~~python fix_columns.py
# 1. Text that should be numeric
df["income"] = pd.to_numeric(df["income"], errors="coerce")   # bad values -> NaN

# 2. Impossible values -> treat as missing, do not silently keep them
df.loc[df["age"] > 120, "age"] = np.nan

# 3. Ordinal category keeps its order
plan_order = {"small": 0, "medium": 1, "large": 2}
df["plan_ord"] = df["plan"].map(plan_order)

# 4. Nominal category -> one-hot (no invented ordering)
city_dummies = pd.get_dummies(df["city"], prefix="city", dtype=int)
df = pd.concat([df, city_dummies], axis=1)

# 5. Datetime -> useful parts
ref = pd.Timestamp("2024-06-01")
df["tenure_days"] = (ref - df["signup_date"]).dt.days
df["signup_month"] = df["signup_date"].dt.month
df["signup_dow"] = df["signup_date"].dt.dayofweek
df["signup_is_weekend"] = (df["signup_dow"] >= 5).astype(int)

# 6. Build X and y properly: drop the ID, the raw versions, and the target
drop = ["customer_id", "city", "plan", "signup_date", "churned"]
X = df.drop(columns=drop)
y = df["churned"]

print("X columns:", list(X.columns))
print("X shape:", X.shape, " y shape:", y.shape)
print(X.head())
~~~

~~~text
X columns: ['age', 'income', 'plan_ord', 'city_Athens', 'city_Patras', 'city_Volos',
            'tenure_days', 'signup_month', 'signup_dow', 'signup_is_weekend']
X shape: (8, 10)  y shape: (8,)
~~~

:::tip Why drop ~customer_id~
It is unique per row, so a flexible model can memorise "customer 107 churned" and score
perfectly in training while learning absolutely nothing generalisable. Any column with
~nunique() == len(df)~ is an ID until proven otherwise.
:::
`
},
{
name: 'The four data shapes in code',
goal: 'See tabular, image, text and time-series data side by side, with the shape each model expects.',
md: `
~~~python four_shapes.py
import numpy as np
import pandas as pd

# ============================ 1. TABULAR ============================
tab = pd.DataFrame({
    "age":    [34, 51, 22],
    "income": [28000, 61000, 15000],
    "score":  [0.71, 0.35, 0.88],
})
X_tab = tab.values                       # -> numpy for sklearn
print("TABULAR     ", X_tab.shape, "= (n_samples, n_features)")

# ============================ 2. IMAGE ==============================
# 4 colour images, 32x32 pixels, 3 channels (R,G,B)
imgs = np.random.randint(0, 256, size=(4, 32, 32, 3), dtype=np.uint8)
print("IMAGE (TF)  ", imgs.shape, "= (n, height, width, channels)")

# PyTorch wants channels FIRST - a constant source of bugs
imgs_torch = imgs.transpose(0, 3, 1, 2)
print("IMAGE (torch)", imgs_torch.shape, "= (n, channels, height, width)")

# Models want floats in [0,1], not ints in [0,255]
imgs_float = imgs.astype("float32") / 255.0
print("             pixel range now:", imgs_float.min(), "to", imgs_float.max())

# A CNN sees this per image:
print("             one image is a", imgs[0].shape, "array of intensities")
print("             top-left pixel RGB:", imgs[0, 0, 0])

# ============================ 3. TEXT ===============================
corpus = ["the food was great", "terrible service", "great food great service"]

# Step 1: build a vocabulary
vocab = sorted({w for doc in corpus for w in doc.split()})
word2id = {w: i for i, w in enumerate(vocab)}
print("\\nTEXT vocabulary:", word2id)

# Step 2a: as token id sequences (what an RNN or Transformer eats)
seqs = [[word2id[w] for w in doc.split()] for doc in corpus]
maxlen = max(len(s) for s in seqs)
padded = np.array([s + [0] * (maxlen - len(s)) for s in seqs])   # pad to equal length
print("TEXT (seq)  ", padded.shape, "= (n_docs, sequence_length)")
print(padded)

# Step 2b: as a bag of words (what classical ML eats)
bow = np.zeros((len(corpus), len(vocab)), dtype=int)
for i, doc in enumerate(corpus):
    for w in doc.split():
        bow[i, word2id[w]] += 1
print("TEXT (bow)  ", bow.shape, "= (n_docs, vocab_size)")
print(bow)

# ============================ 4. TIME SERIES ========================
dates = pd.date_range("2024-01-01", periods=10, freq="D")
sales = pd.Series([120, 135, 128, 150, 165, 142, 138, 171, 180, 176], index=dates)
print("\\nTIME SERIES")
print(sales.head(4))

# Supervised framing: use the last 3 days to predict the next one (windowing)
def make_windows(series, lookback=3):
    values = series.values
    Xs, ys = [], []
    for i in range(len(values) - lookback):
        Xs.append(values[i:i + lookback])      # the window
        ys.append(values[i + lookback])        # the next value
    return np.array(Xs), np.array(ys)

Xt, yt = make_windows(sales, lookback=3)
print("windowed X  ", Xt.shape, "= (n_windows, lookback)")
print("windowed y  ", yt.shape)
print(np.c_[Xt, yt][:4], "  <- last column is the target")
~~~

~~~text
TABULAR      (3, 3) = (n_samples, n_features)
IMAGE (TF)   (4, 32, 32, 3) = (n, height, width, channels)
IMAGE (torch) (4, 3, 32, 32) = (n, channels, height, width)
             pixel range now: 0.0 to 1.0
             one image is a (32, 32, 3) array of intensities
             top-left pixel RGB: [ 45 201  17]

TEXT vocabulary: {'food': 0, 'great': 1, 'service': 2, 'terrible': 3, 'the': 4, 'was': 5}
TEXT (seq)   (3, 4) = (n_docs, sequence_length)
[[4 5 1 0]
 [3 2 0 0]
 [1 0 1 2]]
TEXT (bow)   (3, 6) = (n_docs, vocab_size)
[[1 1 0 0 1 1]
 [0 0 1 1 0 0]
 [1 2 1 0 0 0]]

TIME SERIES
2024-01-01    120
2024-01-02    135
...
windowed X   (7, 3) = (n_windows, lookback)
windowed y   (7,)
[[120 135 128 150]
 [135 128 150 165]
 [128 150 165 142]
 [150 165 142 138]]   <- last column is the target
~~~

:::danger Time series and shuffling
Look at the windowed data: window 1 and window 2 overlap. If you shuffle and split randomly,
a nearly identical window ends up in both train and test, and your score is fantasy.
**Time series always split by time**: train on the past, test on the future.
:::

:::note Channels-first vs channels-last
TensorFlow/Keras defaults to ~(N, H, W, C)~; PyTorch to ~(N, C, H, W)~. Mixing them up
produces a shape error at best and a silently wrong model at worst. When a vision model
misbehaves, print the shape first.
:::
`
}
],
quiz: [
{
q: 'A column "size" contains small, medium, large. How should it be encoded?',
options: ['One-hot encoding', 'Ordinal encoding preserving small < medium < large', 'Leave it as text', 'Drop it'],
answer: 1,
why: 'It is an ordinal category - the order carries real information. One-hot would work but throws the ordering away; leaving it as text will crash most models.'
},
{
q: 'What is the shape of a batch of 64 RGB images of size 224x224 in PyTorch convention?',
options: ['(64, 224, 224, 3)', '(64, 3, 224, 224)', '(3, 64, 224, 224)', '(224, 224, 3, 64)'],
answer: 1,
why: 'PyTorch is channels-first: (N, C, H, W). TensorFlow/Keras uses (N, H, W, C) - option A.'
},
{
q: 'Why must you not shuffle time-series data before splitting?',
options: [
  'Shuffling is slow on large datasets',
  'Overlapping windows would put nearly identical rows in both train and test, leaking the future',
  'Time series cannot be used with neural networks',
  'It changes the units of the target'
],
answer: 1,
why: 'Random splits let the model see the future during training. Time-series splits must be chronological - train on the past, evaluate on the future.'
}
]
},

/* ============================================================ */
{
id: 'overfitting',
title: 'Generalisation, overfitting and underfitting',
summary: 'The central tension of machine learning: fitting the data you have versus working on the data you have not seen. With plots you can generate yourself.',
tags: ['theory', 'core'],
intro: `
## The only thing that matters

A model that scores 100% on its training data and 60% on new data is worse than useless -
it is *confidently* wrong. The goal is never to fit the training set. The goal is
**generalisation**: performance on data the model has never seen.

~~~text
      UNDERFIT              GOOD FIT               OVERFIT
   (high bias)                                 (high variance)

   y|    o  o                y|    o  o           y|    o  o
    |  ______   o             |   _--~-_  o        |   /\\  /\\ o
    | /      \\                |  /      \\_         |  /  \\/  \\
    |o        o o             | o        o o       | o        o o
    +-------------- x         +-------------- x    +-------------- x

  too simple to capture     captures the trend    memorised the noise
  the real pattern          not the noise         including every wiggle

  train error: HIGH         train error: low      train error: ~0
  test  error: HIGH         test  error: low      test  error: HIGH
~~~

## Diagnosing from two numbers

| Train score | Test score | Diagnosis | Fix |
|---|---|---|---|
| Low | Low | **Underfitting** | More complex model, better features, train longer, less regularisation |
| High | High | **Just right** | Ship it |
| High | Low | **Overfitting** | More data, simpler model, more regularisation, dropout, early stopping |
| Low | High | Something is broken | Check your split, check for a bug |

## Why overfitting happens

Every dataset is *signal + noise*. A flexible enough model cannot tell them apart, so it
fits both. The noise is different in new data, so the memorised noise becomes error.

The more parameters relative to data points, the easier this is. A 15-degree polynomial
through 20 points can pass through every single one - and predict nonsense between them.

## The bias-variance decomposition

For squared error, the expected test error of a model decomposes exactly:

:::math Expected test error
**Error = Bias squared + Variance + Irreducible noise**

- **Bias**: error from wrong assumptions. A straight line fitting a curve. Underfitting.
- **Variance**: how much the model changes if you resample the training data. Overfitting.
- **Irreducible noise**: randomness in the world. No model removes it. It is the floor.
:::

Simple models: high bias, low variance. Complex models: low bias, high variance.
The best model minimises the sum - which is why "the most powerful model" is often not the best one.
`,
keyPoints: [
  'Only test-set performance counts. Training score is diagnostics, never a result.',
  'High train + low test = overfitting. Low + low = underfitting.',
  'More data is the most reliable cure for overfitting; more capacity is the cure for underfitting.',
  'Irreducible noise sets a floor no model can go below - chasing it is how you overfit.'
],
pitfalls: [
  'Reporting training accuracy as if it were performance.',
  'Adding capacity to fix a high test error that was actually caused by overfitting - it makes it worse.',
  'Trying dozens of models against the test set. That tunes you to the test set; use a validation set.'
],
levels: [
{
name: 'Watch it happen',
goal: 'Fit polynomials of increasing degree and watch the exact moment fitting turns into memorising.',
md: `
~~~python overfit_demo.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

rng = np.random.default_rng(0)

# ---- the TRUTH we are trying to recover -----------------------------
def true_function(x):
    return np.sin(1.5 * np.pi * x)

# ---- the DATA we actually observe: truth + noise --------------------
n = 25
X_train = np.sort(rng.uniform(0, 1, n)).reshape(-1, 1)
y_train = true_function(X_train).ravel() + rng.normal(0, 0.25, n)

X_test = np.sort(rng.uniform(0, 1, 200)).reshape(-1, 1)
y_test = true_function(X_test).ravel() + rng.normal(0, 0.25, 200)

X_smooth = np.linspace(0, 1, 400).reshape(-1, 1)   # for drawing curves

degrees = [1, 3, 9, 20]
fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))

print(f"{'degree':>7} | {'train MSE':>10} | {'test MSE':>10} | verdict")
print("-" * 56)

for ax, d in zip(axes, degrees):
    model = make_pipeline(PolynomialFeatures(d), LinearRegression())
    model.fit(X_train, y_train)

    tr = mean_squared_error(y_train, model.predict(X_train))
    te = mean_squared_error(y_test,  model.predict(X_test))

    if tr > 0.1 and te > 0.1:      verdict = "UNDERFIT"
    elif te > 3 * tr:              verdict = "OVERFIT"
    else:                          verdict = "good"
    print(f"{d:>7} | {tr:>10.4f} | {te:>10.4f} | {verdict}")

    ax.scatter(X_train, y_train, c="crimson", s=35, zorder=3, label="training data")
    ax.plot(X_smooth, true_function(X_smooth), "k--", lw=2, label="true function")
    ax.plot(X_smooth, model.predict(X_smooth), "b-", lw=2, label="model")
    ax.set_ylim(-2, 2)
    ax.set_title(f"degree {d}  |  train {tr:.3f}  test {te:.3f}")
    if d == 1:
        ax.legend(fontsize=8)

plt.tight_layout(); plt.show()
~~~

~~~text
 degree |  train MSE |   test MSE | verdict
--------------------------------------------------------
      1 |     0.3529 |     0.3872 | UNDERFIT
      3 |     0.0561 |     0.0741 | good
      9 |     0.0398 |     0.1104 | OVERFIT
     20 |     0.0089 |     4.7213 | OVERFIT
~~~

**Look at degree 20.** Training error is almost zero - the curve passes through nearly every
red dot. Test error is 4.72, more than ten times worse than the straight line. The model
learned the noise perfectly and the signal not at all.

:::note The tell
Train error going *down* while test error goes *up* is the signature of overfitting.
It is why you must always track both.
:::

### The validation curve: find the sweet spot automatically

~~~python validation_curve.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

degrees = np.arange(1, 16)
pipe = make_pipeline(PolynomialFeatures(), LinearRegression())

train_scores, val_scores = validation_curve(
    pipe, X_train, y_train,
    param_name="polynomialfeatures__degree",
    param_range=degrees,
    cv=5, scoring="neg_mean_squared_error",
)
train_mse = -train_scores.mean(axis=1)
val_mse   = -val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(degrees, train_mse, "o-", label="training error")
plt.plot(degrees, val_mse,  "s-", label="validation error")
plt.axvline(degrees[val_mse.argmin()], color="green", ls="--",
            label=f"best degree = {degrees[val_mse.argmin()]}")
plt.yscale("log"); plt.xlabel("polynomial degree (model complexity)")
plt.ylabel("mean squared error (log scale)"); plt.legend()
plt.title("The classic U-curve: underfit on the left, overfit on the right")
plt.tight_layout(); plt.show()

print("best degree:", degrees[val_mse.argmin()])
~~~

You get the canonical U-shape. The left arm is bias, the right arm is variance, and the
bottom of the U is the model you should ship.
`
},
{
name: 'Learning curves: do I need more data or a better model?',
goal: 'Answer the single most valuable practical question in ML using one plot.',
md: `
When your model is not good enough, there are two very different fixes, and picking wrong
wastes weeks. A **learning curve** - score versus training-set size - tells you which.

~~~text
  UNDERFITTING (high bias)            OVERFITTING (high variance)

  score                               score
   |  ....................             |            ..........train
   |  train ---------------            |        ....
   |  val   ---------------            |    ....
   |                                   |  ..
   |                                   |         ---------------val
   +----------------------- n          +----------------------- n
                                                    ^ big gap
   Curves MEET, both LOW.              Curves DIVERGE, gap persists.
   => more data will NOT help.         => more data WILL help.
      Get a better model/features.        Or regularise / simplify.
~~~

~~~python learning_curves.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

X, y = make_classification(n_samples=3000, n_features=20, n_informative=6,
                           n_redundant=2, class_sep=0.9, random_state=0)

def plot_learning_curve(model, title, ax):
    sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.05, 1.0, 12),
        cv=5, scoring="accuracy", n_jobs=-1, random_state=0, shuffle=True,
    )
    tr_m, tr_s = train_scores.mean(1), train_scores.std(1)
    va_m, va_s = val_scores.mean(1),   val_scores.std(1)

    ax.plot(sizes, tr_m, "o-", color="steelblue", label="training score")
    ax.fill_between(sizes, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color="steelblue")
    ax.plot(sizes, va_m, "s-", color="darkorange", label="validation score")
    ax.fill_between(sizes, va_m - va_s, va_m + va_s, alpha=0.15, color="darkorange")

    gap = tr_m[-1] - va_m[-1]
    ax.set_title(f"{title}\\nfinal gap = {gap:.3f}")
    ax.set_xlabel("training examples"); ax.set_ylabel("accuracy")
    ax.set_ylim(0.5, 1.02); ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    return gap

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
g1 = plot_learning_curve(LogisticRegression(max_iter=2000, C=0.01),
                         "Too simple (strong regularisation)", axes[0])
g2 = plot_learning_curve(DecisionTreeClassifier(random_state=0),
                         "Too complex (unpruned tree)", axes[1])
g3 = plot_learning_curve(DecisionTreeClassifier(max_depth=6, min_samples_leaf=10,
                                                random_state=0),
                         "Balanced (pruned tree)", axes[2])
plt.tight_layout(); plt.show()

for name, gap in [("underfit", g1), ("overfit", g2), ("balanced", g3)]:
    if gap > 0.08:
        print(f"{name:9s} gap={gap:.3f} -> HIGH VARIANCE: get more data, or regularise")
    elif gap < 0.03:
        print(f"{name:9s} gap={gap:.3f} -> low variance; if the score is also low, "
              f"you are UNDERFITTING: use a stronger model")
~~~

### How to act on the plot

| What you see | Meaning | What to do |
|---|---|---|
| Both curves low, touching | Underfitting | Stronger model, better features, less regularisation |
| Big gap, validation still rising | Overfitting, but data helps | Collect more data |
| Big gap, validation flat | Overfitting, data will not help | Regularise, simplify, add dropout |
| Validation above training | Bug, or heavy regularisation only at train time (dropout) | Check the split |

:::tip Save yourself a month
Before you argue for a data-collection budget, plot the learning curve. If the validation
curve has already flattened, **more data will not help you** and the money should go into
features or model class instead.
:::
`
},
{
name: 'Regularisation: making a model choose to be simpler',
goal: 'Use L1, L2 and early stopping to control overfitting deliberately rather than by luck.',
md: `
Regularisation adds a penalty for complexity to the loss, so the optimiser trades a little
training fit for a lot of generalisation.

:::math The two classic penalties
Ordinary least squares minimises: **sum of (y - prediction) squared**

- **Ridge (L2)** adds **alpha * sum of (w squared)** -> shrinks all weights toward zero, keeps them all.
- **Lasso (L1)** adds **alpha * sum of |w|** -> drives some weights *exactly* to zero, selecting features.

Larger alpha = stronger penalty = simpler model = more bias, less variance.
:::

~~~python regularisation.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error

# 60 features but only 8 actually matter -> a perfect overfitting setup
X, y, true_coef = make_regression(n_samples=120, n_features=60, n_informative=8,
                                  noise=12.0, coef=True, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0)

def evaluate(model, name):
    m = make_pipeline(StandardScaler(), model).fit(X_tr, y_tr)
    tr = mean_squared_error(y_tr, m.predict(X_tr))
    te = mean_squared_error(y_te, m.predict(X_te))
    coefs = m[-1].coef_
    nonzero = int(np.sum(np.abs(coefs) > 1e-6))
    print(f"{name:26s} train={tr:8.1f}  test={te:8.1f}  nonzero weights={nonzero:3d}/60")
    return coefs

print("120 samples, 60 features, only 8 informative\\n")
c_ols   = evaluate(LinearRegression(),  "plain linear regression")
c_ridge = evaluate(Ridge(alpha=10.0),   "ridge  (L2, alpha=10)")
c_lasso = evaluate(Lasso(alpha=2.0),    "lasso  (L1, alpha=2)")

# ---- see WHAT the penalties did to the weights ----------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, c, name in zip(axes, [c_ols, c_ridge, c_lasso],
                       ["Linear (no penalty)", "Ridge L2", "Lasso L1"]):
    ax.stem(c, markerfmt=" ", basefmt=" ")
    ax.set_title(f"{name}\\n{int(np.sum(np.abs(c) > 1e-6))} non-zero weights")
    ax.set_xlabel("feature index"); ax.set_ylabel("weight")
plt.tight_layout(); plt.show()
~~~

~~~text
120 samples, 60 features, only 8 informative

plain linear regression    train=    83.4  test=  4021.7  nonzero weights= 60/60
ridge  (L2, alpha=10)      train=   152.9  test=  1188.3  nonzero weights= 60/60
lasso  (L1, alpha=2)       train=   174.6  test=   329.5  nonzero weights= 11/60
~~~

Read the three columns together:

- Plain regression has the **best training error and the worst test error** by a factor of 12.
- Ridge trades 70 points of training error for 2800 points of test error. A bargain.
- Lasso keeps only 11 of 60 weights, close to the true 8, and wins outright - because the
  true model really is sparse.

### Choosing alpha properly

~~~python tune_alpha.py
from sklearn.linear_model import RidgeCV, LassoCV
import numpy as np

alphas = np.logspace(-3, 3, 60)

ridge_cv = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas, cv=5)).fit(X_tr, y_tr)
lasso_cv = make_pipeline(StandardScaler(), LassoCV(alphas=alphas, cv=5,
                                                   max_iter=20000, random_state=0)).fit(X_tr, y_tr)

print("ridge best alpha:", round(ridge_cv[-1].alpha_, 4),
      " test MSE:", round(mean_squared_error(y_te, ridge_cv.predict(X_te)), 1))
print("lasso best alpha:", round(lasso_cv[-1].alpha_, 4),
      " test MSE:", round(mean_squared_error(y_te, lasso_cv.predict(X_te)), 1))
~~~

:::warn Scaling is not optional here
Both penalties act on the raw size of the weights. A feature measured in euros gets a tiny
weight; the same feature in thousands of euros gets a weight 1000x larger and is penalised
1000x harder. **Always scale before Ridge or Lasso.** That is why every model above is
wrapped in ~make_pipeline(StandardScaler(), ...)~.
:::

### The other regularisers you will meet

| Technique | Where | What it does |
|---|---|---|
| L2 / weight decay | Linear models, neural nets | Shrinks weights smoothly |
| L1 | Linear models | Zeroes weights - built-in feature selection |
| Elastic Net | Linear models | Mix of both; good with correlated features |
| Early stopping | Boosting, neural nets | Stop when validation stops improving |
| Dropout | Neural nets | Randomly disables neurons during training |
| Max depth / min samples leaf | Trees | Limits how finely the tree can carve the space |
| Data augmentation | Vision, audio, text | Manufactures more effective data |
`
}
],
quiz: [
{
q: 'Training accuracy 0.99, validation accuracy 0.71. What is happening and what helps most?',
options: [
  'Underfitting - use a bigger model',
  'Overfitting - more data, regularisation or a simpler model',
  'The learning rate is too low',
  'Nothing is wrong'
],
answer: 1,
why: 'A large gap with high training score is the definition of overfitting. Adding capacity would widen the gap further.'
},
{
q: 'Your learning curve shows training and validation accuracy both at 0.68 and touching. What should you do?',
options: [
  'Collect more data',
  'Add dropout and L2 regularisation',
  'Use a more expressive model or better features',
  'Reduce the training set size'
],
answer: 2,
why: 'Converged, low curves mean high bias - underfitting. More data cannot help when the model is not capable of fitting even the data it has.'
},
{
q: 'What does Lasso (L1) do that Ridge (L2) does not?',
options: [
  'It works without scaling the features',
  'It drives some coefficients exactly to zero, performing feature selection',
  'It always achieves lower training error',
  'It can only be used for classification'
],
answer: 1,
why: 'The absolute-value penalty has a corner at zero, so the optimum often lands exactly on zero. Ridge shrinks smoothly and keeps every feature. Both require scaling.'
}
]
},

/* ============================================================ */
{
id: 'setup',
title: 'Setting up your environment',
summary: 'A clean, reproducible Python setup for data science on Windows, macOS or Linux - virtual environments, the core libraries, Jupyter, and how to avoid dependency hell.',
tags: ['practical', 'tooling'],
intro: `
## What you need

~~~text
Python 3.11 or 3.12         <- not the newest; libraries lag by a few months
  |
  +-- a VIRTUAL ENVIRONMENT per project   <- the single most important habit
        |
        +-- numpy        arrays and maths
        +-- pandas       tables
        +-- matplotlib   plots
        +-- seaborn      nicer statistical plots
        +-- scikit-learn classical machine learning
        +-- jupyter      notebooks
        |
        +-- (later) torch, torchvision      deep learning
        +-- (later) transformers, datasets  NLP / LLMs
        +-- (later) opencv-python           computer vision
        +-- (later) xgboost, lightgbm       gradient boosting
~~~

## Why virtual environments matter

Project A needs numpy 1.26. Project B needs numpy 2.1. Install both globally and one of
them breaks - usually the one with the deadline. A virtual environment is a private
folder of packages per project. It costs 20 seconds to create and saves entire evenings.
`,
keyPoints: [
  'One virtual environment per project. Always. No exceptions.',
  'Pin your versions in requirements.txt so the project still runs next year.',
  'Set a random seed everywhere you want reproducible results.',
  'Prefer Python 3.11/3.12 over the very newest release - library support lags.'
],
pitfalls: [
  'Installing everything globally with pip, then wondering why an old project broke.',
  'Forgetting to activate the venv, then being confused that an import fails.',
  'Committing the venv folder to git. Commit requirements.txt instead.'
],
levels: [
{
name: 'Local setup, step by step',
goal: 'Get a working, isolated data science environment on your machine in five minutes.',
md: `
### Windows (PowerShell)

~~~bash setup_windows.ps1
# 1. Check your Python version (want 3.11 or 3.12)
python --version

# 2. Make a project folder and go into it
mkdir C:\\Users\\you\\projects\\ml-playground
cd C:\\Users\\you\\projects\\ml-playground

# 3. Create a virtual environment named .venv
python -m venv .venv

# 4. Activate it  (your prompt should now start with (.venv))
.venv\\Scripts\\Activate.ps1
# If PowerShell blocks the script, run once as admin:
#   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 5. Upgrade pip, then install the core stack
python -m pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn jupyter

# 6. Freeze exactly what you installed
pip freeze > requirements.txt
~~~

### macOS / Linux

~~~bash setup_unix.sh
python3 --version
mkdir -p ~/projects/ml-playground && cd ~/projects/ml-playground
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
pip freeze > requirements.txt
~~~

### Verify everything works

~~~python check_env.py
"""Run this first. If it prints a version for every library, you are ready."""
import sys
print("Python", sys.version.split()[0])
print("-" * 40)

libs = [
    ("numpy",        "np"),
    ("pandas",       "pd"),
    ("matplotlib",   None),
    ("seaborn",      "sns"),
    ("sklearn",      None),
]
for name, _ in libs:
    try:
        mod = __import__(name)
        print(f"  {name:12s} {getattr(mod, '__version__', 'installed')}")
    except ImportError:
        print(f"  {name:12s} MISSING  ->  pip install {name}")

# A real 5-line end-to-end check
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
X, y = load_iris(return_X_y=True)
score = cross_val_score(RandomForestClassifier(random_state=0), X, y, cv=5).mean()
print("-" * 40)
print(f"Smoke test: random forest on iris = {score:.3f} accuracy")
print("Environment is working." if score > 0.9 else "Something is wrong.")
~~~

~~~text
Python 3.12.3
----------------------------------------
  numpy        2.1.3
  pandas       2.2.3
  matplotlib   3.9.2
  seaborn      0.13.2
  sklearn      1.5.2
----------------------------------------
Smoke test: random forest on iris = 0.967 accuracy
Environment is working.
~~~

### Start Jupyter

~~~bash
jupyter lab          # modern interface, recommended
# or
jupyter notebook     # the classic one
~~~

:::tip No installation at all
If you want to start learning *right now*, use **Google Colab** (colab.research.google.com).
Free, browser-based, most libraries preinstalled, and free GPU access under Runtime ->
Change runtime type -> T4 GPU. You will want a local setup eventually, but do not let
setup block you on day one.
:::
`
},
{
name: 'Reproducibility: getting the same numbers twice',
goal: 'Lock down randomness and dependencies so your results are repeatable by you and by others.',
md: `
"It worked yesterday" is the most expensive sentence in data science. Three things cause it:
unpinned versions, unseeded randomness, and undocumented data.

### 1. Seed everything

~~~python seeds.py
"""Call set_all_seeds(42) at the top of every experiment."""
import os
import random
import numpy as np

def set_all_seeds(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)   # affects dict/set ordering in some cases
    random.seed(seed)                          # Python's own RNG
    np.random.seed(seed)                       # legacy numpy global RNG

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # These two make GPU work deterministic - at a small speed cost
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    print(f"All random seeds set to {seed}")


set_all_seeds(42)

# Modern numpy style - prefer this for new code, it avoids global state
rng = np.random.default_rng(42)
print("reproducible draws:", rng.normal(size=3).round(4))

# And in scikit-learn, pass random_state to EVERYTHING that accepts it
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# train_test_split(..., random_state=42)
# RandomForestClassifier(random_state=42)
~~~

:::warn Seeding does not make results *correct*
A single seed gives you one sample of performance. If your score swings by 4 points across
seeds, your model is unstable and one seed is hiding that. For small datasets, report the
mean and standard deviation over several seeds.
:::

### 2. Pin your dependencies

~~~bash
# Capture the exact working state
pip freeze > requirements.txt

# Anyone (including future you) recreates it exactly
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
~~~

A minimal, readable ~requirements.txt~ for this course:

~~~text requirements.txt
numpy==2.1.3
pandas==2.2.3
matplotlib==3.9.2
seaborn==0.13.2
scikit-learn==1.5.2
jupyter==1.1.1
# deep learning (install when you reach that track)
# torch==2.5.1
# torchvision==0.20.1
# transformers==4.46.3
# xgboost==2.1.3
# opencv-python==4.10.0.84
~~~

### 3. A project layout that scales

~~~text
ml-playground/
  .venv/                 <- never commit this
  .gitignore             <- add: .venv/, data/raw/, *.joblib, .ipynb_checkpoints/
  requirements.txt
  README.md              <- what the project does, how to run it
  data/
    raw/                 <- original files, NEVER edited
    processed/           <- outputs of your cleaning scripts
  notebooks/
    01-eda.ipynb
    02-modelling.ipynb
  src/
    data.py              <- loading + cleaning functions
    features.py          <- feature engineering
    train.py             <- training entry point
  models/
    model.joblib
~~~

:::tip The rule that saves projects
**Raw data is immutable.** Never edit a file in ~data/raw/~. Every transformation is code
that turns raw into processed. If a cleaning step turns out to be wrong, you rerun the
script - you do not try to remember what you clicked in Excel three weeks ago.
:::
`
}
],
quiz: [
{
q: 'Why create a virtual environment per project?',
options: [
  'It makes Python run faster',
  'It isolates package versions so projects cannot break each other',
  'It is required to use Jupyter',
  'It compresses your dependencies'
],
answer: 1,
why: 'Isolation is the whole point. Different projects need different, sometimes incompatible, library versions.'
},
{
q: 'Which should you commit to git?',
options: ['The .venv folder', 'requirements.txt', 'Both', 'Neither'],
answer: 1,
why: 'requirements.txt is small, portable and recreates the environment anywhere. The .venv folder is large, platform-specific and useless to anyone else.'
}
]
},

/* ============================================================ */
{
id: 'first-model',
title: 'Your first model, four times over',
summary: 'Train a working model in ten lines, then rebuild it four times - each version adding one real-world concern - so you see exactly what each addition buys you.',
tags: ['practical', 'hands-on', 'capstone'],
intro: `
## The point of this lesson

Most tutorials show you ~model.fit(X, y)~ and stop. You end up able to copy code but
unable to say why any line is there.

So we will solve the *same* problem five times. Each version adds exactly one thing, and
each time we measure what it was worth. By the end you will have the full professional
recipe - and, more importantly, a reason for every line in it.

The problem: **predict whether a breast tumour is malignant or benign** from 30 measurements
taken from a digitised image of a cell sample. Real data, 569 patients, from scikit-learn.

~~~text
Level 1  ten lines, no thought          -> works, but the score is a lie
Level 2  + train/test split             -> now the score is honest
Level 3  + scaling and pipelines        -> now it is correct and safe
Level 4  + cross-validation and tuning  -> now the score is reliable
Level 5  + the right metric, threshold  -> now it is actually useful to a doctor
~~~
`,
keyPoints: [
  'Every professional habit exists because a specific mistake is expensive.',
  'A pipeline is not bureaucracy - it is what prevents leakage during cross-validation.',
  'Accuracy is rarely the metric that matters. Ask what a false negative costs.',
  'The decision threshold is a business choice, not a modelling detail. 0.5 is just a default.'
],
pitfalls: [
  'Evaluating on training data and being delighted.',
  'Scaling the whole dataset before splitting.',
  'Tuning hyperparameters against the test set.',
  'Shipping the default 0.5 threshold when the cost of the two errors is wildly different.'
],
levels: [
{
name: 'Ten lines that work',
goal: 'Get a model running end-to-end immediately, then discover why its score cannot be trusted.',
md: `
~~~python level1.py
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)      # X: 569x30 measurements, y: 0=malignant 1=benign

model = LogisticRegression(max_iter=10000)
model.fit(X, y)                                  # learn

print("Accuracy:", model.score(X, y))            # measure
print("Prediction for the first patient:", model.predict(X[:1]))
print("Probability benign:", model.predict_proba(X[:1])[0, 1].round(4))
~~~

~~~text
Accuracy: 0.9666080843585237
Prediction for the first patient: [0]
Probability benign: 0.0
~~~

**96.7% accuracy!** And it is meaningless.

We measured the model on the exact same 569 patients it learned from. That is an open-book
exam where the student wrote the questions. The number tells us the model can memorise;
it tells us nothing about a patient walking in tomorrow.

:::danger The cardinal sin
~model.score(X, y)~ on the training data is not a result. It is a sanity check at best.
If you ever see a report with only a training score, distrust everything else in it.
:::

Let us prove the number is inflated, using a model that can memorise perfectly:

~~~python proof_of_inflation.py
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

tree = DecisionTreeClassifier(random_state=0).fit(X, y)
print("Unpruned tree, scored on its own training data:", tree.score(X, y))

# now score it honestly
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
tree2 = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
print("Same tree, scored on unseen patients        :", round(tree2.score(X_te, y_te), 4))
~~~

~~~text
Unpruned tree, scored on its own training data: 1.0
Same tree, scored on unseen patients        : 0.9298
~~~

100% versus 93%. The tree memorised all 569 patients. That is what a training score
measures: memory, not skill.
`
},
{
name: 'Add an honest split',
goal: 'Hold out data the model never sees, so the reported number means something.',
md: `
~~~python level2.py
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

# ---------------------------------------------------------------
# THE SPLIT
#   test_size=0.2  -> 20% held back, never seen during training
#   random_state   -> the same split every run, so results compare
#   stratify=y     -> keep the malignant/benign ratio identical in
#                     both halves. Without it, an unlucky split can
#                     leave you with too few of the rare class.
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("train:", X_train.shape, " test:", X_test.shape)
print("class balance train:", np.bincount(y_train) / len(y_train))
print("class balance test :", np.bincount(y_test) / len(y_test))

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_train)                # learn from training only

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc  = accuracy_score(y_test,  model.predict(X_test))

print(f"\\ntraining accuracy: {train_acc:.4f}   <- diagnostics only")
print(f"test accuracy    : {test_acc:.4f}   <- THE result")
print(f"gap              : {train_acc - test_acc:+.4f}")

print("\\nconfusion matrix (rows = truth, cols = prediction):")
print(confusion_matrix(y_test, model.predict(X_test)))
print("\\n", classification_report(y_test, model.predict(X_test),
                                   target_names=["malignant", "benign"]))
~~~

~~~text
train: (455, 30)  test: (114, 30)
class balance train: [0.373 0.627]
class balance test : [0.377 0.623]

training accuracy: 0.9670   <- diagnostics only
test accuracy    : 0.9737   <- THE result
gap              : -0.0067

confusion matrix (rows = truth, cols = prediction):
[[41  2]
 [ 1 70]]
~~~

### How to read a confusion matrix

~~~text
                        PREDICTED
                  malignant   benign
              +-----------+-----------+
TRUE malignant|    41     |     2     |  <- 2 malignant tumours called benign
              |    TN     |    FP     |     ** these are the dangerous ones **
              +-----------+-----------+
TRUE benign   |     1     |    70     |  <- 1 benign called malignant
              |    FN     |    TP     |     (an unnecessary biopsy - unpleasant, not fatal)
              +-----------+-----------+
~~~

The two errors are **not equally bad**. Missing a malignant tumour can cost a life;
a false alarm costs a biopsy. Accuracy treats them as identical. We will fix that in Level 5.

:::note Why the gap is negative
Test accuracy came out slightly *above* training accuracy here. On a 114-row test set that
is ordinary noise - a couple of easy cases landed in the test half. Do not over-read small
gaps on small test sets; that is exactly what cross-validation is for.
:::
`
},
{
name: 'Add scaling, inside a pipeline',
goal: 'Fix the silent bug caused by features on wildly different scales - without introducing leakage.',
md: `
### The problem

~~~python why_scale.py
from sklearn.datasets import load_breast_cancer
import numpy as np

data = load_breast_cancer()
X = data.data
for i in [0, 3, 23]:
    col = X[:, i]
    print(f"{data.feature_names[i]:24s} min={col.min():9.3f}  max={col.max():9.3f}  "
          f"mean={col.mean():9.3f}")
~~~

~~~text
mean radius              min=    6.981  max=   28.110  mean=   14.127
mean area                min=  143.500  max= 2501.000  mean=  654.889
worst area               min=  185.200  max= 4254.000  mean=  880.583
~~~

~mean area~ spans thousands; ~mean radius~ spans tens. Any algorithm that measures distance
(KNN, SVM) or penalises weight size (Ridge, Lasso, logistic regression with regularisation)
is now dominated by area purely because of its **units**. That is not information - it is
an accident of measurement.

### The naive fix, and why it is wrong

~~~python leaky_scaling.py
# ---------- WRONG: scaler sees the test set ----------
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)                      # <-- fitted on ALL the data
X_tr, X_te, y_tr, y_te = train_test_split(X_scaled, y)  # split afterwards
# The mean and std used to scale training rows were computed partly from test rows.
# Information has leaked backwards. The test score is now optimistic.
~~~

### The right fix

~~~python level3.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# A Pipeline chains steps. When you call .fit(), each step's fit_transform runs
# on TRAINING data only; when you call .predict(), only .transform() runs.
# That single property is what makes scaling safe.
models = {
    "logistic regression": LogisticRegression(max_iter=10000),
    "k-nearest neighbours": KNeighborsClassifier(n_neighbors=5),
    "support vector machine": SVC(),
}

print(f"{'model':24s} {'unscaled':>10s} {'scaled':>10s} {'gain':>8s}")
print("-" * 56)
for name, clf in models.items():
    # without scaling
    raw = clf.__class__(**clf.get_params()).fit(X_train, y_train)
    a_raw = accuracy_score(y_test, raw.predict(X_test))

    # with scaling, safely
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", clf.__class__(**clf.get_params())),
    ]).fit(X_train, y_train)
    a_scaled = accuracy_score(y_test, pipe.predict(X_test))

    print(f"{name:24s} {a_raw:10.4f} {a_scaled:10.4f} {a_scaled - a_raw:+8.4f}")
~~~

~~~text
model                      unscaled     scaled     gain
--------------------------------------------------------
logistic regression          0.9737     0.9825  +0.0088
k-nearest neighbours         0.9298     0.9649  +0.0351
support vector machine       0.9211     0.9825  +0.0614
~~~

The SVM gained **six points** from one line of preprocessing. That is a bigger improvement
than you will usually get from switching algorithms.

### Which models need scaling?

| Needs scaling | Does not care |
|---|---|
| KNN, SVM (distance-based) | Decision trees |
| Ridge, Lasso, regularised logistic regression | Random forests |
| Neural networks | Gradient boosting (XGBoost, LightGBM) |
| PCA, K-Means | Naive Bayes (mostly) |

Trees split on one feature at a time using thresholds, so monotonic rescaling changes
nothing. Everything that measures a distance or penalises a weight is affected.

:::tip Scale by default
Putting ~StandardScaler~ in your pipeline never hurts a tree and often rescues everything
else. Make it your default and only remove it if you have a reason.
:::
`
},
{
name: 'Add cross-validation and tuning',
goal: 'Stop trusting a single split, and choose hyperparameters without ever touching the test set.',
md: `
### Why one split is not enough

~~~python split_lottery.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np

X, y = load_breast_cancer(return_X_y=True)
scores = []
for seed in range(15):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                              random_state=seed, stratify=y)
    m = make_pipeline(StandardScaler(), SVC()).fit(X_tr, y_tr)
    scores.append(m.score(X_te, y_te))

scores = np.array(scores)
print("15 different random splits, same model and data:")
print(np.round(scores, 4))
print(f"\\nlowest  : {scores.min():.4f}")
print(f"highest : {scores.max():.4f}")
print(f"spread  : {scores.max() - scores.min():.4f}  <- pure luck of the draw")
print(f"mean    : {scores.mean():.4f} +/- {scores.std():.4f}")
~~~

~~~text
15 different random splits, same model and data:
[0.9825 0.9737 0.9649 1.     0.9825 0.9561 0.9912 0.9737 0.9649 0.9825
 0.9737 0.9912 0.9649 0.9825 0.9737]

lowest  : 0.9561
highest : 1.0000
spread  : 0.0439  <- pure luck of the draw
mean    : 0.9772 +/- 0.0109
~~~

Four and a half points of swing from nothing but the random seed. If you report the lucky
split you are fooling yourself, and if a colleague reports theirs you cannot compare.

### K-fold cross-validation

~~~text
5-FOLD CV: every row is used for testing exactly once.

fold 1  [ TEST ][ train ][ train ][ train ][ train ]  -> score 1
fold 2  [ train][ TEST  ][ train ][ train ][ train ]  -> score 2
fold 3  [ train][ train ][ TEST  ][ train ][ train ]  -> score 3
fold 4  [ train][ train ][ train ][ TEST  ][ train ]  -> score 4
fold 5  [ train][ train ][ train ][ train ][ TEST  ]  -> score 5

report  mean +/- standard deviation
~~~

~~~python level4.py
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

X, y = load_breast_cancer(return_X_y=True)

# The test set is created FIRST and then locked in a drawer.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([("scaler", StandardScaler()), ("svm", SVC())])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# --- 1. Honest estimate of the default model ------------------------
base = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
print(f"default SVC, 5-fold CV: {base.mean():.4f} +/- {base.std():.4f}")
print("  per fold:", np.round(base, 4))

# --- 2. Search hyperparameters, still only on training data ---------
grid = {
    "svm__C":      [0.1, 1, 10, 100],        # how much slack the margin allows
    "svm__gamma":  ["scale", 0.001, 0.01, 0.1],
    "svm__kernel": ["rbf", "linear"],
}
search = GridSearchCV(pipe, grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=0)
search.fit(X_train, y_train)

print(f"\\nbest CV score : {search.best_score_:.4f}")
print("best params   :", search.best_params_)

# --- 3. NOW open the test set. Once. --------------------------------
final = search.best_estimator_
test_acc = accuracy_score(y_test, final.predict(X_test))
print(f"\\nFINAL test accuracy: {test_acc:.4f}")
print(classification_report(y_test, final.predict(X_test),
                            target_names=["malignant", "benign"]))

# --- see the top of the search --------------------------------------
import pandas as pd
res = pd.DataFrame(search.cv_results_)
cols = ["param_svm__C", "param_svm__gamma", "param_svm__kernel",
        "mean_test_score", "std_test_score"]
print(res.sort_values("mean_test_score", ascending=False)[cols].head(5).to_string(index=False))
~~~

~~~text
default SVC, 5-fold CV: 0.9736 +/- 0.0140
  per fold: [0.978  0.978  0.9670 0.9560 0.9890]

best CV score : 0.9824
best params   : {'svm__C': 10, 'svm__gamma': 'scale', 'svm__kernel': 'rbf'}

FINAL test accuracy: 0.9825
~~~

### The three-way split, stated clearly

~~~text
  ALL DATA
  |
  +-- TRAINING SET (80%)              +-- TEST SET (20%)
       |                                   |
       | cross-validation splits this      | opened ONCE, at the end,
       | into train/validation folds       | after every decision is made
       | as many times as you like         |
       |                                   |
       v                                   v
  choose model, features,             report this number and
  hyperparameters, threshold          nothing else
~~~

:::danger The subtle leak nobody notices
If you run GridSearchCV, look at the test score, then go back and widen the grid, you have
used the test set to make a decision. Do that ten times and your test score is now an
optimistic training score in disguise. **Decide, then measure, then stop.**
:::
`
},
{
name: 'Add the metric that actually matters',
goal: 'Move from accuracy to a decision that reflects the real cost of each kind of mistake.',
md: `
### Restating the problem honestly

Two errors, wildly different costs:

- **False negative**: a malignant tumour predicted benign. The patient goes home. Potentially fatal.
- **False positive**: a benign tumour predicted malignant. An unnecessary biopsy. Unpleasant, survivable.

Accuracy weighs these equally. No clinician would.

~~~text
PRECISION = TP / (TP + FP)   "when I raise the alarm, how often am I right?"
                             -> optimise when false alarms are expensive

RECALL    = TP / (TP + FN)   "of all the real cases, how many did I catch?"
                             -> optimise when misses are expensive   ** our case **

F1        = harmonic mean of precision and recall
                             -> use when you need one number and both matter

ROC-AUC   = probability the model ranks a random positive above a random negative
                             -> threshold-free measure of ranking quality
~~~

~~~python level5.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report,
                             precision_recall_curve, roc_curve, roc_auc_score,
                             average_precision_score)

data = load_breast_cancer()
X, y_raw = data.data, data.target

# IMPORTANT: sklearn ships 0 = malignant, 1 = benign.
# The "positive" class should be the thing we care about detecting.
# So we flip it: 1 = MALIGNANT.
y = 1 - y_raw

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

pipe = Pipeline([("scaler", StandardScaler()),
                 ("lr", LogisticRegression(max_iter=10000))])
cv = StratifiedKFold(5, shuffle=True, random_state=42)

# Tune for RECALL, not accuracy - we care about not missing cancer
search = GridSearchCV(pipe, {"lr__C": [0.01, 0.1, 1, 10, 100]},
                      cv=cv, scoring="recall", n_jobs=-1).fit(X_train, y_train)
model = search.best_estimator_
print("best C:", search.best_params_["lr__C"], " CV recall:", round(search.best_score_, 4))

# ---- probabilities, not labels --------------------------------------
proba = model.predict_proba(X_test)[:, 1]        # P(malignant)

print("\\nROC-AUC :", round(roc_auc_score(y_test, proba), 4))
print("PR-AUC  :", round(average_precision_score(y_test, proba), 4))

# ---- the threshold is a CHOICE --------------------------------------
print(f"\\n{'threshold':>10} {'precision':>10} {'recall':>8} {'FN':>4} {'FP':>4}  meaning")
print("-" * 74)
for t in [0.20, 0.35, 0.50, 0.65, 0.80]:
    pred = (proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    note = ("catches everything, more biopsies" if t <= 0.35 else
            "sklearn default" if t == 0.50 else
            "misses real cancers - unacceptable")
    print(f"{t:>10.2f} {prec:>10.3f} {rec:>8.3f} {fn:>4} {fp:>4}  {note}")

# ---- pick the threshold by a rule, not by eye -----------------------
prec, rec, thr = precision_recall_curve(y_test, proba)
TARGET_RECALL = 0.99
ok = np.where(rec[:-1] >= TARGET_RECALL)[0]
best_i = ok[np.argmax(prec[:-1][ok])]        # among those, best precision
chosen = thr[best_i]
print(f"\\nTo guarantee {TARGET_RECALL:.0%} recall, use threshold {chosen:.4f}")
print(f"  precision there: {prec[best_i]:.3f}")

final_pred = (proba >= chosen).astype(int)
print("\\nFinal confusion matrix at the chosen threshold:")
print(confusion_matrix(y_test, final_pred))
print(classification_report(y_test, final_pred,
                            target_names=["benign", "MALIGNANT"], digits=3))

# ---- the curves ------------------------------------------------------
fpr, tpr, _ = roc_curve(y_test, proba)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_test, proba):.3f}")
ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="random")
ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate (recall)")
ax[0].set_title("ROC curve"); ax[0].legend()

ax[1].plot(rec, prec, lw=2)
ax[1].axvline(TARGET_RECALL, color="red", ls="--", label=f"target recall {TARGET_RECALL}")
ax[1].set_xlabel("recall"); ax[1].set_ylabel("precision")
ax[1].set_title("Precision-Recall curve"); ax[1].legend()
plt.tight_layout(); plt.show()
~~~

~~~text
best C: 100  CV recall: 0.9765

ROC-AUC : 0.9977
PR-AUC  : 0.9954

 threshold  precision   recall   FN   FP  meaning
--------------------------------------------------------------------------
      0.20      0.911    0.976    1    4  catches everything, more biopsies
      0.35      0.930    0.976    1    3  catches everything, more biopsies
      0.50      0.951    0.929    3    2  sklearn default
      0.65      0.972    0.833    7    1  misses real cancers - unacceptable
      0.80      1.000    0.762   10    0  misses real cancers - unacceptable

To guarantee 99% recall, use threshold 0.0871
  precision there: 0.860
~~~

### The lesson

At the default threshold of 0.5 this excellent model (AUC 0.998) **misses three malignant
tumours**. Moving the threshold to 0.087 catches all of them, at the cost of seven extra
biopsies. In this domain that is obviously the right trade - and no algorithm could have
made that decision for you. It is a human, contextual, ethical judgement.

:::tip Take this to every project
1. Choose the positive class to be **the thing you want to detect**.
2. Get **probabilities**, not just labels.
3. Report **ROC-AUC / PR-AUC** for threshold-free quality.
4. Set the **threshold from the cost of each error**, not from the default.
5. On imbalanced data, prefer the **precision-recall curve** to ROC - it is more honest
   when positives are rare.
:::

### Everything, assembled

~~~python final_recipe.py
"""The complete professional recipe, in the order it should be run."""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np, joblib

# 1. data, with the positive class chosen deliberately
d = load_breast_cancer(); X, y = d.data, 1 - d.target        # 1 = malignant

# 2. lock away the test set before anything else happens
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)

# 3. preprocessing + model as one object
pipe = Pipeline([("scaler", StandardScaler()),
                 ("clf", LogisticRegression(max_iter=10000, class_weight="balanced"))])

# 4. tune with cross-validation on training data only, for the right metric
search = GridSearchCV(pipe, {"clf__C": np.logspace(-3, 3, 13)},
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      scoring="average_precision", n_jobs=-1).fit(X_tr, y_tr)

# 5. evaluate once
proba = search.best_estimator_.predict_proba(X_te)[:, 1]
print("test ROC-AUC:", round(roc_auc_score(y_te, proba), 4))
print(classification_report(y_te, (proba >= 0.1).astype(int),
                            target_names=["benign", "MALIGNANT"], digits=3))

# 6. save the WHOLE pipeline, so serving needs no preprocessing code
joblib.dump(search.best_estimator_, "tumour_model.joblib")
print("saved tumour_model.joblib")
~~~

You now have the complete loop. Every remaining track in this course deepens one of these
six steps.
`
}
],
quiz: [
{
q: 'You scale your features with StandardScaler fitted on the full dataset, then split into train and test. What is wrong?',
options: [
  'Nothing, this is standard practice',
  'The scaler saw test-set statistics, so information leaked and the test score is optimistic',
  'StandardScaler cannot be used with logistic regression',
  'It will raise an error'
],
answer: 1,
why: 'The mean and standard deviation used on training rows were partly computed from test rows. Put the scaler in a Pipeline so it is fitted per training fold.'
},
{
q: 'For cancer screening, which threshold behaviour do you want?',
options: [
  'A high threshold, to maximise precision',
  'A low threshold, to maximise recall and avoid missing malignant cases',
  'Always 0.5, the default',
  'Whatever maximises accuracy'
],
answer: 1,
why: 'A missed cancer is far more costly than an unnecessary biopsy, so you accept more false positives to drive false negatives toward zero.'
},
{
q: 'Why use cross-validation instead of a single train/validation split?',
options: [
  'It trains faster',
  'It uses less memory',
  'It averages over several splits, so the estimate is far less dependent on lucky partitioning',
  'It removes the need for a test set'
],
answer: 2,
why: 'A single split can swing several accuracy points on seed alone. CV gives a mean and a standard deviation. You still need a separate untouched test set.'
}
]
}

]
});
