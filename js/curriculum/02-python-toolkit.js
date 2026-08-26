/* Track 02 - The Python toolkit: NumPy, pandas, matplotlib, seaborn */
(window.CURRICULUM = window.CURRICULUM || []).push({
id: 'toolkit',
title: 'The Python Toolkit',
icon: 'Py',
level: 'Beginner',
blurb: 'NumPy, pandas, matplotlib and seaborn - the four libraries every single thing in this course is built on. From first array to production-grade data wrangling.',
intro: `
Every model you will ever train is preceded by an hour of moving data around. These four
libraries are how that hour is spent:

| Library | What it is | Mental model |
|---|---|---|
| **NumPy** | Fast numeric arrays | A spreadsheet of numbers, all the same type, with maths that applies to everything at once |
| **pandas** | Labelled tables | Excel, but scriptable, and built on NumPy |
| **matplotlib** | Plotting | A drawing canvas. Verbose, total control |
| **seaborn** | Statistical plots | matplotlib with taste, one line per chart |

Learn NumPy properly. Almost every strange error in scikit-learn, PyTorch or TensorFlow is
really a NumPy shape error wearing a costume.
`,
topics: [

/* ============================================================ */
{
id: 'python-essentials',
title: 'The Python you actually need',
summary: 'Comprehensions, unpacking, f-strings, functions with defaults, and the handful of standard-library tools that appear in every data science script.',
tags: ['python', 'fundamentals'],
intro: `
## Assumed, and quickly re-taught

You do not need to be a software engineer to do data science, but a few Python constructs
appear on nearly every line of real code. If these are automatic for you, this lesson is a
ten-minute skim. If not, it is the highest-value ten minutes in the track.

The list:

1. **List / dict comprehensions** - the loop replacement
2. **Unpacking** - ~a, b = b, a~ and ~*args~ / ~**kwargs~
3. **f-strings** - formatting numbers for output
4. **Functions with default and keyword arguments**
5. **~enumerate~, ~zip~, ~sorted~ with a key**
6. **~collections.Counter~ and ~defaultdict~**
7. **Slicing** - the same syntax you will use on NumPy arrays and DataFrames
8. **~with~ blocks** for files
9. **A minimal class** - because scikit-learn estimators are classes

:::tip What you do NOT need
Metaclasses, decorators beyond copying them, async, threading, inheritance hierarchies.
Data science Python is a small, practical dialect.
:::
`,
keyPoints: [
  'A comprehension is a loop that returns a list - shorter and faster than append in a for loop.',
  'Slicing syntax start:stop:step carries over identically to NumPy and pandas.',
  'f-strings with format specs are how you print readable numbers: ~f"{x:.3f}"~.',
  'Mutable default arguments are a real trap - use ~None~ and create inside the function.'
],
pitfalls: [
  'Using a mutable default like ~def f(items=[])~ - the list is shared across all calls.',
  'Confusing ~is~ with ~==~. Use ~==~ for values, ~is~ only for ~None~.',
  'Modifying a list while iterating over it.',
  'Forgetting that Python slices exclude the stop index: ~x[0:3]~ gives three items.'
],
levels: [
{
name: 'The core constructs',
goal: 'Cover every Python construct that appears in the rest of this course, with a runnable example each.',
md: `
~~~python essentials.py
# =====================================================================
# 1. COMPREHENSIONS - the single most-used construct in data science
# =====================================================================
nums = [1, 2, 3, 4, 5, 6, 7, 8]

squares = [n ** 2 for n in nums]                        # map
evens = [n for n in nums if n % 2 == 0]                 # filter
labelled = ["big" if n > 4 else "small" for n in nums]  # conditional value
print(squares, evens, labelled, sep="\\n")

# dict comprehension
name_len = {w: len(w) for w in ["athens", "patras", "volos"]}
print(name_len)                      # {'athens': 6, 'patras': 6, 'volos': 5}

# set comprehension (unique values)
first_letters = {w[0] for w in ["apple", "avocado", "banana"]}
print(first_letters)                 # {'a', 'b'}

# nested - flatten a list of lists
matrix = [[1, 2], [3, 4], [5, 6]]
flat = [x for row in matrix for x in row]
print(flat)                          # [1, 2, 3, 4, 5, 6]

# =====================================================================
# 2. UNPACKING
# =====================================================================
point = (3, 7)
x, y = point                                   # tuple unpacking
a, b = b, a = 1, 2                             # (works, but write it simply)
first, *rest = [10, 20, 30, 40]                # star catches the remainder
print(first, rest)                             # 10 [20, 30, 40]

def describe(name, *scores, **options):
    """*scores collects extra positionals, **options collects keywords."""
    avg = sum(scores) / len(scores) if scores else 0
    unit = options.get("unit", "points")
    return f"{name}: {avg:.1f} {unit} over {len(scores)} tests"

print(describe("Maria", 88, 92, 79, unit="marks"))

# =====================================================================
# 3. F-STRINGS - how you format model output
# =====================================================================
acc, n, name = 0.9317483, 1204, "random forest"
print(f"{name:>18s} | acc={acc:.3f} | n={n:,} | pct={acc:.1%}")
print(f"{'model':<18s} | {'score':>8s}")
print(f"{acc:8.4f}  fixed width")
print(f"{1234567.891:,.2f}  thousands separator")
print(f"{0.000031:.2e}     scientific")

# =====================================================================
# 4. FUNCTIONS: defaults, keywords, type hints, docstrings
# =====================================================================
def train(data, epochs=10, lr=0.01, verbose=True):
    """One line saying what it does.

    Args:
        data: the training data
        epochs: how many passes over the data
        lr: learning rate
    Returns:
        a dict of results
    """
    if verbose:
        print(f"training for {epochs} epochs at lr={lr}")
    return {"epochs": epochs, "lr": lr}

train([1, 2, 3])                     # all defaults
train([1, 2, 3], lr=0.1)             # keyword argument - readable and safe
train([1, 2, 3], 50, 0.001, False)   # positional - fragile, avoid

# THE CLASSIC TRAP
def bad(item, bucket=[]):            # the list is created ONCE, at definition
    bucket.append(item)
    return bucket
print(bad(1), bad(2), bad(3))        # [1] [1,2] [1,2,3]  <- surprise!

def good(item, bucket=None):         # the fix
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket
print(good(1), good(2), good(3))     # [1] [2] [3]

# =====================================================================
# 5. ITERATION HELPERS
# =====================================================================
models = ["logistic", "forest", "svm"]
scores = [0.81, 0.86, 0.84]

for i, m in enumerate(models, start=1):          # index + value
    print(f"  {i}. {m}")

for m, s in zip(models, scores):                 # walk two lists together
    print(f"  {m:10s} {s:.2f}")

best = max(zip(models, scores), key=lambda pair: pair[1])
print("best:", best)

ranked = sorted(zip(models, scores), key=lambda p: p[1], reverse=True)
print("ranked:", ranked)

# =====================================================================
# 6. COLLECTIONS
# =====================================================================
from collections import Counter, defaultdict

words = "the cat sat on the mat the cat".split()
counts = Counter(words)
print(counts)                        # Counter({'the': 3, 'cat': 2, ...})
print(counts.most_common(2))         # [('the', 3), ('cat', 2)]

groups = defaultdict(list)           # no KeyError on a missing key
for w in words:
    groups[len(w)].append(w)
print(dict(groups))                  # {3: ['the','cat','sat','the','cat','mat'], 2: ['on']}

# =====================================================================
# 7. SLICING - identical syntax in NumPy and pandas
# =====================================================================
xs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(xs[2:5])      # [2, 3, 4]     start inclusive, stop EXCLUSIVE
print(xs[:3])       # [0, 1, 2]     from the beginning
print(xs[7:])       # [7, 8, 9]     to the end
print(xs[-3:])      # [7, 8, 9]     last three
print(xs[::2])      # [0,2,4,6,8]   every second
print(xs[::-1])     # reversed

# =====================================================================
# 8. FILES
# =====================================================================
with open("notes.txt", "w", encoding="utf-8") as f:      # 'with' closes for you
    f.write("first line\\nsecond line\\n")

with open("notes.txt", encoding="utf-8") as f:
    for line in f:
        print("read:", line.rstrip())

# =====================================================================
# 9. A MINIMAL CLASS - the shape of every sklearn estimator
# =====================================================================
class MeanPredictor:
    """Predicts the training mean for everything. The dumbest useful baseline."""

    def __init__(self, verbose=False):
        self.verbose = verbose        # hyperparameters go in __init__
        self.mean_ = None             # learned things end with _ , by convention

    def fit(self, X, y):
        self.mean_ = sum(y) / len(y)
        if self.verbose:
            print(f"learned mean = {self.mean_:.3f}")
        return self                   # fit returns self, so you can chain

    def predict(self, X):
        if self.mean_ is None:
            raise RuntimeError("call fit before predict")
        return [self.mean_] * len(X)

m = MeanPredictor(verbose=True).fit([[1], [2], [3]], [10, 20, 30])
print(m.predict([[9], [9]]))          # [20.0, 20.0]
~~~

:::note Why the trailing underscore
scikit-learn's convention: ~self.alpha~ is something *you* set; ~self.coef_~ is something
the model *learned*. When you see a trailing underscore, it only exists after ~fit~.
:::
`
},
{
name: 'Writing code you can still read next month',
goal: 'The small habits - type hints, docstrings, pure functions, guard clauses - that turn notebook spaghetti into reusable code.',
md: `
Notebooks encourage a style that stops working at about 200 lines. These habits cost
nothing and scale.

~~~python clean_style.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# =====================================================================
# 1. TYPE HINTS - documentation your editor can check
# =====================================================================
def normalise(values: Sequence[float]) -> np.ndarray:
    """Scale values to mean 0, standard deviation 1.

    Args:
        values: any sequence of numbers.
    Returns:
        A numpy array with mean 0 and std 1.
    Raises:
        ValueError: if all values are identical (std would be 0).
    """
    arr = np.asarray(values, dtype=float)
    std = arr.std()
    if std == 0:                             # guard clause: handle the bad case FIRST
        raise ValueError("cannot normalise constant values")
    return (arr - arr.mean()) / std


print(normalise([10, 20, 30, 40]).round(3))

# =====================================================================
# 2. PURE FUNCTIONS - no hidden state, no surprises
# =====================================================================
# BAD: modifies its input, so the caller's data silently changes
def add_ratio_bad(df):
    df["ratio"] = df["a"] / df["b"]
    return df

# GOOD: returns a new frame, leaves the caller's data alone
def add_ratio(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ratio"] = out["a"] / out["b"]
    return out

df = pd.DataFrame({"a": [10, 20], "b": [2, 5]})
new = add_ratio(df)
print("original untouched:", list(df.columns))     # ['a', 'b']
print("new has the column:", list(new.columns))    # ['a', 'b', 'ratio']

# =====================================================================
# 3. CONFIG IN ONE PLACE, not scattered as magic numbers
# =====================================================================
@dataclass(frozen=True)
class Config:
    data_path: Path = Path("data/raw/train.csv")
    test_size: float = 0.2
    random_state: int = 42
    n_estimators: int = 300
    target: str = "churned"

CFG = Config()
print(CFG)
print("one place to change everything:", CFG.random_state)

# =====================================================================
# 4. SMALL FUNCTIONS THAT DO ONE THING
# =====================================================================
def load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.drop_duplicates()
    out.columns = [c.strip().lower().replace(" ", "_") for c in out.columns]
    return out

def split_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=[target]), df[target]

# The whole pipeline reads like a sentence:
# X, y = split_xy(clean(load(CFG.data_path)), CFG.target)

# =====================================================================
# 5. FAIL LOUDLY AND EARLY
# =====================================================================
def check_frame(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Assert the frame is usable before wasting an hour training on garbage."""
    missing = set(required) - set(df.columns)
    assert not missing, f"missing columns: {sorted(missing)}"
    assert len(df) > 0, "dataframe is empty"
    dupes = df.columns[df.columns.duplicated()].tolist()
    assert not dupes, f"duplicate column names: {dupes}"
    print(f"frame ok: {df.shape[0]} rows, {df.shape[1]} columns")

check_frame(pd.DataFrame({"a": [1], "b": [2]}), required=["a", "b"])

# =====================================================================
# 6. THE SCRIPT GUARD - lets a file be both runnable and importable
# =====================================================================
def main() -> None:
    print("running as a script")

if __name__ == "__main__":
    main()
~~~

### The notebook-to-module workflow

~~~text
1. Explore freely in a notebook.        <- messy is fine here
2. When a cell WORKS, move it into src/ as a named function.
3. Import it back into the notebook:
       %load_ext autoreload
       %autoreload 2
       from src.features import add_ratio
4. The notebook becomes a thin narrative; the logic lives in files
   that you can test, version and reuse.
~~~

:::tip The two magic lines
Put ~%load_ext autoreload~ and ~%autoreload 2~ at the top of every notebook. Then edits to
your ~.py~ files take effect immediately without restarting the kernel. This single habit
makes the notebook-plus-module workflow practical.
:::
`
}
],
quiz: [
{
q: 'What does ~[n**2 for n in nums if n % 2 == 0]~ produce?',
options: ['All squares', 'Squares of even numbers only', 'Even numbers only', 'An error'],
answer: 1,
why: 'The ~if~ filters first, then the expression transforms. It is a filter and a map in one line.'
},
{
q: 'Why is ~def f(items=[])~ dangerous?',
options: [
  'Lists cannot be default arguments',
  'The same list object is reused across every call, so it accumulates',
  'It is slower than using None',
  'It only works in Python 2'
],
answer: 1,
why: 'Default arguments are evaluated once, at function definition. Use ~items=None~ and create the list inside the function.'
},
{
q: 'In scikit-learn convention, what does a trailing underscore like ~coef_~ mean?',
options: [
  'It is a private attribute',
  'It was learned during fit and does not exist before',
  'It is deprecated',
  'It is a hyperparameter you set'
],
answer: 1,
why: 'No underscore = you set it (hyperparameter). Trailing underscore = the model learned it during fit.'
}
]
},

/* ============================================================ */
{
id: 'numpy',
title: 'NumPy: arrays and vectorisation',
summary: 'The array, why vectorised operations are 50x faster than loops, broadcasting, indexing, aggregation along axes, and the shape errors that will haunt you.',
tags: ['numpy', 'core', 'essential'],
intro: `
## Why NumPy exists

A Python list of a million numbers is a million separate Python objects, each with a type
tag and a reference count, scattered across memory. Adding two such lists means a million
interpreted iterations.

A NumPy array is **one contiguous block of raw numbers, all the same type**. Adding two
arrays runs a single compiled C loop over that block, using your CPU's vector instructions.

~~~text
PYTHON LIST                          NUMPY ARRAY
[ptr]->obj(1)  [ptr]->obj(2)  ...    [ 1 | 2 | 3 | 4 | 5 | 6 ]
scattered, boxed, any type           contiguous, raw, one dtype
loop in Python                        loop in C, vectorised
~~~

Typical speedup: **20x to 100x**. And the code is shorter.

## The mental model

An array has three properties you must always be able to state:

~~~text
a = np.zeros((3, 4))

a.shape  -> (3, 4)      how many elements along each axis
a.ndim   -> 2           how many axes
a.dtype  -> float64     what type every element is

    axis 0 (rows) --->  down
    axis 1 (cols) --->  across

        col0  col1  col2  col3
  row0 [  .     .     .     .  ]
  row1 [  .     .     .     .  ]
  row2 [  .     .     .     .  ]
~~~

**Every NumPy error you will ever get comes from not knowing the shape.** Print it.
`,
keyPoints: [
  'Vectorise: never write a Python for-loop over array elements if an array operation exists.',
  'axis=0 collapses rows (result is per-column); axis=1 collapses columns (result is per-row).',
  'Broadcasting aligns shapes from the right; dimensions must match or be 1.',
  'Basic slicing returns a VIEW that shares memory; fancy indexing returns a COPY.'
],
pitfalls: [
  'Assuming ~axis=0~ means "along rows". It means "collapse the row axis" - the result is one value per column.',
  'Modifying a slice and being surprised the original changed. Use ~.copy()~ when you mean a copy.',
  'Integer division surprises: ~np.array([1,2,3]) / 2~ gives floats, ~// 2~ gives ints.',
  'Comparing floats with ~==~. Use ~np.isclose~.'
],
levels: [
{
name: 'Arrays and why they are fast',
goal: 'Create arrays every way you will need, and measure the speedup vectorisation actually gives.',
md: `
~~~python numpy_basics.py
import numpy as np
import time

# =====================================================================
# CREATING ARRAYS
# =====================================================================
a = np.array([1, 2, 3, 4])                    # from a list
b = np.array([[1, 2, 3], [4, 5, 6]])          # 2-D from nested lists

print("a:", a, a.shape, a.dtype)
print("b:\\n", b, b.shape, b.dtype)

# Built-in constructors - use these, they are faster and clearer
print(np.zeros((2, 3)))                # 2x3 of 0.0
print(np.ones(4))                      # [1. 1. 1. 1.]
print(np.full((2, 2), 7))              # 2x2 of 7
print(np.eye(3))                       # 3x3 identity matrix
print(np.arange(0, 10, 2))             # [0 2 4 6 8]      like range()
print(np.linspace(0, 1, 5))            # [0. 0.25 0.5 0.75 1.]  N evenly spaced

# Random - always with an explicit generator for reproducibility
rng = np.random.default_rng(seed=42)
print(rng.random((2, 3)).round(3))            # uniform in [0,1)
print(rng.normal(loc=0, scale=1, size=5).round(3))    # gaussian
print(rng.integers(low=0, high=10, size=(2, 4)))      # random ints

# dtype matters for memory and precision
print(np.array([1, 2, 3]).dtype)              # int64
print(np.array([1, 2, 3], dtype=np.float32).dtype)
print("memory for 1M float64:", np.zeros(1_000_000).nbytes / 1e6, "MB")
print("memory for 1M float32:", np.zeros(1_000_000, dtype=np.float32).nbytes / 1e6, "MB")

# =====================================================================
# WHY IT IS FAST - measure it
# =====================================================================
N = 2_000_000
py_list = list(range(N))
np_arr = np.arange(N)

t0 = time.perf_counter()
result_py = [x * 2 + 1 for x in py_list]
t_py = time.perf_counter() - t0

t0 = time.perf_counter()
result_np = np_arr * 2 + 1
t_np = time.perf_counter() - t0

print(f"\\npython list comprehension: {t_py * 1000:8.1f} ms")
print(f"numpy vectorised         : {t_np * 1000:8.1f} ms")
print(f"speedup                  : {t_py / t_np:8.1f}x")

# Dot product: the operation at the heart of every neural network
x = rng.random(1_000_000)
y = rng.random(1_000_000)

t0 = time.perf_counter()
dot_py = sum(xi * yi for xi, yi in zip(x, y))
t_py = time.perf_counter() - t0

t0 = time.perf_counter()
dot_np = np.dot(x, y)
t_np = time.perf_counter() - t0

print(f"\\ndot product python: {t_py * 1000:8.1f} ms  -> {dot_py:.4f}")
print(f"dot product numpy : {t_np * 1000:8.1f} ms  -> {dot_np:.4f}")
print(f"speedup           : {t_py / t_np:8.1f}x")
~~~

~~~text
python list comprehension:    142.6 ms
numpy vectorised         :      4.1 ms
speedup                  :     34.8x

dot product python:    118.3 ms  -> 250071.7449
dot product numpy :      0.6 ms  -> 250071.7449
speedup           :    197.1x
~~~

### Element-wise maths, applied to everything at once

~~~python vectorised_math.py
import numpy as np
a = np.array([1.0, 4.0, 9.0, 16.0])
b = np.array([2.0, 2.0, 3.0, 4.0])

print(a + b)          # [ 3.  6. 12. 20.]
print(a * b)          # [ 2.  8. 27. 64.]   ELEMENT-WISE, not matrix multiply
print(a / b)          # [0.5 2.  3.  4. ]
print(a ** 0.5)       # [1. 2. 3. 4.]
print(np.sqrt(a))     # same
print(np.log(a).round(3))
print(np.exp([0, 1, 2]).round(3))

# comparisons give boolean arrays
print(a > 5)                  # [False False  True  True]
print((a > 5).sum())          # 2      <- True counts as 1
print(np.where(a > 5, "big", "small"))   # vectorised if/else

# matrix multiply is @ (or np.dot), NOT *
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print("element-wise A*B:\\n", A * B)
print("matrix product A@B:\\n", A @ B)
~~~

:::warn The number one NumPy confusion
~*~ is **element-wise multiplication**. ~@~ is **matrix multiplication**. In linear algebra
notation ~AB~ means the matrix product, so ~A * B~ in NumPy is *not* what a textbook means.
:::
`
},
{
name: 'Indexing, slicing and boolean masks',
goal: 'Select exactly the data you want - and understand when you get a view versus a copy.',
md: `
~~~python indexing.py
import numpy as np
rng = np.random.default_rng(0)

a = np.arange(10)
print(a)                       # [0 1 2 3 4 5 6 7 8 9]

# ---------------- 1-D: same as Python lists ------------------------
print(a[3])                    # 3
print(a[2:5])                  # [2 3 4]      stop is EXCLUSIVE
print(a[-2:])                  # [8 9]
print(a[::3])                  # [0 3 6 9]

# ---------------- 2-D: row, column --------------------------------
m = np.arange(24).reshape(4, 6)
print(m)
# [[ 0  1  2  3  4  5]
#  [ 6  7  8  9 10 11]
#  [12 13 14 15 16 17]
#  [18 19 20 21 22 23]]

print(m[1, 3])          # 9          single element (row 1, col 3)
print(m[1])             # row 1      [6 7 8 9 10 11]
print(m[:, 2])          # column 2   [ 2  8 14 20]
print(m[1:3, 2:5])      # sub-block
# [[ 8  9 10]
#  [14 15 16]]
print(m[::2, ::2])      # every other row and column
print(m[..., -1])       # last column, any number of dimensions

# ---------------- BOOLEAN MASKS - the workhorse --------------------
scores = np.array([55, 91, 78, 42, 88, 67, 95, 31])

mask = scores >= 70
print(mask)                      # [False  True  True False  True False  True False]
print(scores[mask])              # [91 78 88 95]     select where True
print(scores[scores >= 70])      # same thing, written inline

# combine with & (and), | (or), ~ (not) - NOT the words and/or/not
print(scores[(scores >= 50) & (scores < 90)])     # [55 78 67]
print(scores[(scores < 50) | (scores > 90)])      # [42 91 95 31]
print(scores[~(scores >= 70)])                    # [55 42 67 31]

# assign through a mask
clipped = scores.copy()
clipped[clipped < 50] = 50
print(clipped)                   # [55 91 78 50 88 67 95 50]

# count and locate
print("how many pass:", (scores >= 70).sum())
print("any failures :", (scores < 40).any())
print("all positive :", (scores > 0).all())
print("positions    :", np.where(scores >= 70)[0])     # [1 2 4 6]

# ---------------- FANCY INDEXING (a list of positions) -------------
print(scores[[0, 3, 7]])         # [55 42 31]
order = np.argsort(scores)       # positions that would sort the array
print("sorted        :", scores[order])
print("top 3 indices :", np.argsort(scores)[-3:][::-1])
print("top 3 values  :", scores[np.argsort(scores)[-3:][::-1]])

# ---------------- VIEW vs COPY - this bites everyone ---------------
original = np.arange(6)
view = original[1:4]             # basic slicing -> a VIEW, shares memory
view[0] = 999
print("after editing a slice:", original)     # [  0 999   2   3   4   5]  CHANGED

original = np.arange(6)
copy = original[1:4].copy()      # explicit copy
copy[0] = 999
print("after editing a copy :", original)     # [0 1 2 3 4 5]  unchanged

original = np.arange(6)
fancy = original[[1, 2, 3]]      # fancy indexing ALWAYS copies
fancy[0] = 999
print("after editing fancy  :", original)     # [0 1 2 3 4 5]  unchanged

print("\\nis it a view?", view.base is not None)
print("is it a copy?", copy.base is None)
~~~

:::danger The view trap in real code
~~~python
X_train = X[:800]          # a VIEW into X
X_train /= X_train.max()   # in-place - this also modifies X!
~~~
You just scaled part of your original data without meaning to. If you intend a copy,
write ~X[:800].copy()~ - or avoid in-place operators on slices entirely.
:::
`
},
{
name: 'Axes, aggregation and broadcasting',
goal: 'Master the two concepts that cause 90% of NumPy confusion: what axis means, and how shapes get aligned.',
md: `
## Axes: the rule that finally makes it click

**~axis=k~ means "collapse axis k".** The axis you name disappears from the result.

~~~text
m.shape = (4, 6)          axis 0 has length 4 (rows)
                          axis 1 has length 6 (cols)

m.sum(axis=0)  ->  collapses the 4 rows  ->  shape (6,)  one value PER COLUMN
m.sum(axis=1)  ->  collapses the 6 cols  ->  shape (4,)  one value PER ROW
m.sum()        ->  collapses everything  ->  scalar
~~~

~~~python axes.py
import numpy as np

m = np.array([[ 1,  2,  3],
              [ 4,  5,  6],
              [ 7,  8,  9],
              [10, 11, 12]])
print("shape:", m.shape)          # (4, 3)

print("sum(axis=0):", m.sum(axis=0), " shape", m.sum(axis=0).shape)  # [22 26 30] (3,)
print("sum(axis=1):", m.sum(axis=1), " shape", m.sum(axis=1).shape)  # [ 6 15 24 33] (4,)
print("sum()      :", m.sum())                                       # 78

# The memory hook:
#   axis=0 -> "down the columns"  -> one number per COLUMN
#   axis=1 -> "across the rows"   -> one number per ROW

print("\\ncolumn means :", m.mean(axis=0))
print("row maxima   :", m.max(axis=1))
print("column argmax:", m.argmax(axis=0))     # which ROW holds each column's max
print("std per col  :", m.std(axis=0).round(3))

# keepdims - keeps the collapsed axis as size 1, which makes broadcasting work
cm = m.mean(axis=0)
cm_keep = m.mean(axis=0, keepdims=True)
print("\\nwithout keepdims:", cm.shape)        # (3,)
print("with keepdims   :", cm_keep.shape)     # (1, 3)

# every aggregation supports axis
for fn in ["sum", "mean", "std", "min", "max", "prod"]:
    print(f"  {fn:5s} axis=0 -> {getattr(m, fn)(axis=0)}")

# cumulative versions do NOT collapse
print("\\ncumsum axis=0:\\n", m.cumsum(axis=0))
~~~

## Broadcasting: doing maths on different shapes

NumPy aligns shapes **from the right**. Two dimensions are compatible if they are equal,
or one of them is 1. A missing dimension counts as 1.

~~~text
EXAMPLE 1 - add a row vector to a matrix
    m       (4, 3)
    v       (   3)   -> treated as (1, 3)
    ------------------
    result  (4, 3)   v is repeated down every row

EXAMPLE 2 - add a column vector to a matrix
    m       (4, 3)
    c       (4, 1)
    ------------------
    result  (4, 3)   c is repeated across every column

EXAMPLE 3 - incompatible
    m       (4, 3)
    v       (   4)   -> (1, 4);  3 vs 4, neither is 1
    ------------------
    ERROR: operands could not be broadcast together
~~~

~~~python broadcasting.py
import numpy as np

m = np.arange(12).reshape(4, 3)
print("m:\\n", m)

# scalar - broadcast to everything
print("\\nm + 100:\\n", m + 100)

# row vector (3,) - added to every row
row = np.array([10, 20, 30])
print("\\nm + row:\\n", m + row)

# column vector (4,1) - added to every column
col = np.array([[1], [2], [3], [4]])
print("\\nm + col:\\n", m + col)

# THE MISTAKE: a 1-D array of length 4 is NOT a column vector
bad = np.array([1, 2, 3, 4])
try:
    m + bad
except ValueError as e:
    print("\\nerror:", e)
print("fix with reshape:", (m + bad.reshape(4, 1)).shape)
print("or with newaxis :", (m + bad[:, np.newaxis]).shape)

# ---------------------------------------------------------------
# THE APPLICATION YOU WILL USE CONSTANTLY: standardising features
# ---------------------------------------------------------------
rng = np.random.default_rng(0)
X = rng.normal(loc=[100, 5, 0.02], scale=[15, 2, 0.005], size=(1000, 3))
print("\\nbefore scaling:")
print("  means:", X.mean(axis=0).round(3))
print("  stds :", X.std(axis=0).round(3))

mu = X.mean(axis=0)          # shape (3,)  - one mean per feature
sigma = X.std(axis=0)        # shape (3,)
X_scaled = (X - mu) / sigma  # (1000,3) - (3,) -> broadcasts down every row

print("after scaling:")
print("  means:", X_scaled.mean(axis=0).round(6))
print("  stds :", X_scaled.std(axis=0).round(6))

# ---------------------------------------------------------------
# Broadcasting can also build things: an outer product / distance matrix
# ---------------------------------------------------------------
a = np.array([1, 2, 3])
b = np.array([10, 20, 30, 40])
outer = a[:, None] * b[None, :]        # (3,1) * (1,4) -> (3,4)
print("\\nouter product shape:", outer.shape)
print(outer)

# Pairwise squared distances between two sets of points, no loops at all
P = rng.random((5, 2))
Q = rng.random((7, 2))
diff = P[:, None, :] - Q[None, :, :]   # (5,1,2) - (1,7,2) -> (5,7,2)
dist = np.sqrt((diff ** 2).sum(axis=-1))
print("\\npairwise distance matrix shape:", dist.shape)   # (5, 7)
print(dist.round(3))
~~~

:::tip The debugging ritual
When NumPy complains, print shapes - all of them - before doing anything else:
~~~python
print("X", X.shape, "w", w.shape, "b", b.shape)
~~~
Nine times out of ten the fix is a ~.reshape(-1, 1)~ or a ~keepdims=True~.
:::
`
},
{
name: 'Reshaping, stacking and linear algebra',
goal: 'Change array shapes fluently and run the matrix operations that underlie every model.',
md: `
~~~python reshape_stack.py
import numpy as np
rng = np.random.default_rng(0)

a = np.arange(12)
print(a.reshape(3, 4))
print(a.reshape(4, 3))
print(a.reshape(2, 2, 3).shape)         # 3-D

# -1 means "work it out for me" - use it constantly
print(a.reshape(-1, 4).shape)           # (3, 4)
print(a.reshape(2, -1).shape)           # (2, 6)

# The two you will use in every sklearn script:
v = np.array([1, 2, 3, 4, 5])
print("column vector:", v.reshape(-1, 1).shape)   # (5, 1)  one feature, 5 samples
print("row vector   :", v.reshape(1, -1).shape)   # (1, 5)  one sample, 5 features

# flatten vs ravel
m = np.arange(6).reshape(2, 3)
print(m.ravel())          # view when possible - cheap
print(m.flatten())        # always a copy   - safe

# transpose
print("\\nm:\\n", m)
print("m.T:\\n", m.T)
print("shapes:", m.shape, "->", m.T.shape)

# add and remove axes
x = np.array([1, 2, 3])
print("\\nnewaxis:", x[np.newaxis, :].shape, x[:, np.newaxis].shape)
print("expand :", np.expand_dims(x, 0).shape)
print("squeeze:", np.ones((1, 3, 1)).squeeze().shape)     # (3,)

# =====================================================================
# STACKING AND SPLITTING
# =====================================================================
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print("\\nvstack (on top of each other):\\n", np.vstack([a, b]))       # (4,2)
print("hstack (side by side):\\n", np.hstack([a, b]))                 # (2,4)
print("concatenate axis=0:", np.concatenate([a, b], axis=0).shape)
print("concatenate axis=1:", np.concatenate([a, b], axis=1).shape)
print("stack (NEW axis)  :", np.stack([a, b]).shape)                  # (2,2,2)

# column_stack is the friendly way to build a feature matrix
age = np.array([34, 51, 22])
income = np.array([28000, 61000, 15000])
X = np.column_stack([age, income])
print("\\nfeature matrix:\\n", X, X.shape)

# splitting
big = np.arange(12).reshape(6, 2)
first, second = np.split(big, [4])          # split at row 4
print("\\nsplit:", first.shape, second.shape)

# =====================================================================
# LINEAR ALGEBRA - what every model runs on
# =====================================================================
A = np.array([[2.0, 1.0],
              [1.0, 3.0]])
b = np.array([5.0, 10.0])

print("\\nmatrix @ vector:", A @ b)
print("matrix @ matrix:\\n", A @ A)
print("transpose      :\\n", A.T)
print("determinant    :", round(np.linalg.det(A), 4))
print("inverse        :\\n", np.linalg.inv(A).round(4))
print("check A @ A_inv:\\n", (A @ np.linalg.inv(A)).round(10))

# Solving Ax = b  -- ALWAYS use solve, never inv(A) @ b
x = np.linalg.solve(A, b)
print("\\nsolution x     :", x.round(4))
print("verify A @ x   :", (A @ x).round(4), "should equal", b)

# eigenvalues - the machinery behind PCA
vals, vecs = np.linalg.eig(A)
print("\\neigenvalues :", vals.round(4))
print("eigenvectors:\\n", vecs.round(4))

# norms - how you measure the size of a vector
v = np.array([3.0, 4.0])
print("\\nL2 norm (length):", np.linalg.norm(v))          # 5.0
print("L1 norm (sum abs):", np.linalg.norm(v, ord=1))    # 7.0

# =====================================================================
# LEAST SQUARES: linear regression, from scratch, in one line
# =====================================================================
rng = np.random.default_rng(42)
n = 100
X_raw = rng.uniform(0, 10, (n, 2))
true_w = np.array([2.5, -1.3])
true_b = 4.0
y = X_raw @ true_w + true_b + rng.normal(0, 0.5, n)

# add a column of 1s so the intercept becomes just another weight
X_design = np.column_stack([np.ones(n), X_raw])       # (100, 3)

# the normal equation:  w = (X'X)^-1 X'y   -- but solved stably
w_hat, *_ = np.linalg.lstsq(X_design, y, rcond=None)

print(f"\\ntrue    : b={true_b:.3f}  w={true_w}")
print(f"recovered: b={w_hat[0]:.3f}  w={w_hat[1:].round(3)}")
~~~

~~~text
true    : b=4.000  w=[ 2.5 -1.3]
recovered: b=4.026  w=[ 2.494 -1.303]
~~~

**That is linear regression.** No library, no gradient descent - one call to ~lstsq~. You
will implement it three more ways in the Regression track, but the maths is already here.

:::warn Never invert a matrix to solve a system
~np.linalg.inv(A) @ b~ is slower and numerically far less stable than ~np.linalg.solve(A, b)~.
Same for regression: use ~lstsq~, not the explicit normal-equation inverse. This matters
enormously when features are correlated.
:::
`
}
],
quiz: [
{
q: 'For an array of shape (100, 5), what does ~X.mean(axis=0)~ return?',
options: [
  'A single number',
  'An array of 100 values, one mean per row',
  'An array of 5 values, one mean per column',
  'An array of shape (100, 5)'
],
answer: 2,
why: 'axis=0 collapses the row axis, so the 100 disappears and you are left with 5 values - the mean of each feature.'
},
{
q: 'What is the result of adding an array of shape (4, 3) to one of shape (3,)?',
options: ['Shape (4, 3), the vector added to each row', 'An error', 'Shape (3,)', 'Shape (4, 3, 3)'],
answer: 0,
why: 'Broadcasting aligns from the right: (3,) becomes (1,3), which stretches down all 4 rows. This is exactly how per-feature scaling works.'
},
{
q: '~view = arr[1:4]~ then ~view[0] = 999~. What happens to ~arr~?',
options: [
  'Nothing, view is a copy',
  'arr[1] becomes 999, because basic slicing returns a view sharing memory',
  'An error is raised',
  'arr[0] becomes 999'
],
answer: 1,
why: 'Basic slicing returns a view. Fancy indexing (a list of positions) and ~.copy()~ return copies.'
},
{
q: 'Which multiplies two matrices in the linear-algebra sense?',
options: ['~A * B~', '~A @ B~', '~A.multiply(B)~', '~np.multiply(A, B)~'],
answer: 1,
why: '~@~ (or np.dot / np.matmul) is the matrix product. ~*~ is element-wise, which is a very common source of silent bugs.'
}
]
},

/* ============================================================ */
{
id: 'pandas',
title: 'pandas: tables that think',
summary: 'Series and DataFrames, loading real files, selecting with loc and iloc, filtering, and the missing-data handling that every dataset demands.',
tags: ['pandas', 'core', 'essential'],
intro: `
## The two objects

~~~text
SERIES - one column, with an index
                                     DATAFRAME - a dict of Series sharing an index
  index   values
    0   |  34                          index |  age   city     churned
    1   |  51                         -------+---------------------------
    2   |  22                            0   |  34    Athens   False
    3   |  45                            1   |  51    Patras   True
                                         2   |  22    Athens   False
  s.values -> numpy array                3   |  45    Volos    True
  s.index  -> the labels
                                        df["age"]        -> a Series
                                        df[["age","city"]] -> a DataFrame
~~~

A DataFrame is a NumPy array with **labelled rows and columns and mixed dtypes**. That is
the whole difference - and it is why pandas is where real data lives.

## The three ways to select, and when to use which

| Syntax | Selects by | Use for |
|---|---|---|
| ~df["col"]~ | column name | grabbing a column |
| ~df.loc[rows, cols]~ | **labels** | almost everything |
| ~df.iloc[rows, cols]~ | **integer positions** | when you genuinely mean "the 3rd row" |

:::warn Use loc and iloc, not chained brackets
~df["a"]["b"] = 5~ may modify a copy and silently do nothing, producing the famous
~SettingWithCopyWarning~. ~df.loc[row, "col"] = 5~ always works. Make it a habit.
:::
`,
keyPoints: [
  '~.loc~ is label-based and its slices are INCLUSIVE of the endpoint; ~.iloc~ is positional and exclusive.',
  'Boolean masks are the primary filtering tool; combine them with & | ~ and wrap each condition in parentheses.',
  'Missing data is ~NaN~; ~==~ never matches it, so use ~.isna()~.',
  'Chained assignment can silently fail - always assign through ~.loc~.'
],
pitfalls: [
  'Writing ~df[df.a > 1 & df.b < 2]~ without parentheses - operator precedence makes it wrong.',
  'Using ~df.append~ in a loop (removed in pandas 2.0, and it was O(n squared) anyway). Build a list, then ~pd.concat~.',
  'Forgetting ~inplace=False~ is the default, so ~df.dropna()~ returns a new frame and changes nothing.',
  'Reading a CSV without checking dtypes, then wondering why a numeric column is text.'
],
levels: [
{
name: 'Loading, inspecting, selecting',
goal: 'Get any dataset into pandas and confidently pull out exactly the rows and columns you want.',
md: `
~~~python pandas_basics.py
import numpy as np
import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 30)

# =====================================================================
# CREATING
# =====================================================================
s = pd.Series([34, 51, 22, 45], name="age")
print(s)

df = pd.DataFrame({
    "name":    ["Anna", "Boris", "Chris", "Dora", "Elias", "Fay"],
    "age":     [34, 51, 22, 45, 29, 38],
    "city":    ["Athens", "Patras", "Athens", "Volos", "Patras", "Athens"],
    "income":  [28000, 61000, 15000, 42000, 33000, np.nan],
    "churned": [False, True, False, True, False, False],
})
print(df)

# =====================================================================
# LOADING REAL FILES
# =====================================================================
# df = pd.read_csv("data.csv")
# df = pd.read_csv("data.csv", sep=";", decimal=",")        # European CSVs
# df = pd.read_csv("data.csv", parse_dates=["signup_date"]) # dates as dates
# df = pd.read_csv("data.csv", na_values=["", "NA", "?", "-999"])
# df = pd.read_csv("big.csv", usecols=["a","b"], nrows=10000)   # sample a big file
# df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
# df = pd.read_json("data.json")
# df = pd.read_parquet("data.parquet")     # fast + keeps dtypes. Prefer it.
# df = pd.read_sql("SELECT * FROM users", con=engine)

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
tit = pd.read_csv(url)

# =====================================================================
# FIRST LOOK - do all five of these, every time
# =====================================================================
print("\\n1. shape :", tit.shape)
print("\\n2. head:")
print(tit.head(3))
print("\\n3. info:")
tit.info()
print("\\n4. describe (numeric):")
print(tit.describe().T.round(2))
print("\\n5. missing values:")
print(tit.isna().sum().sort_values(ascending=False).head())

# categorical summary
print("\\ndescribe (objects):")
print(tit.describe(include="object").T)

# =====================================================================
# SELECTING COLUMNS
# =====================================================================
print("\\none column (Series) :", type(tit["age"]).__name__)
print("two columns (DataFrame):", type(tit[["age", "fare"]]).__name__)
print(tit[["age", "fare", "survived"]].head(3))

# by dtype
print("\\nnumeric columns:", list(tit.select_dtypes("number").columns))
print("text columns   :", list(tit.select_dtypes("object").columns))

# =====================================================================
# SELECTING ROWS: loc (labels) vs iloc (positions)
# =====================================================================
print("\\n--- iloc: integer positions, stop EXCLUSIVE ---")
print(tit.iloc[0])                   # first row, as a Series
print(tit.iloc[0:3])                 # rows 0,1,2
print(tit.iloc[0:3, 0:4])            # rows 0-2, columns 0-3
print(tit.iloc[[0, 5, 10], [0, 2]])  # specific rows and columns

print("\\n--- loc: labels, stop INCLUSIVE ---")
print(tit.loc[0:2])                  # rows labelled 0,1,2  <- THREE rows
print(tit.loc[0:2, ["age", "fare"]])
print(tit.loc[:, "age":"fare"])      # column range by name

# The inclusive/exclusive difference, made explicit
print("\\niloc[0:2] gives", len(tit.iloc[0:2]), "rows")
print("loc[0:2]  gives", len(tit.loc[0:2]), "rows   <- surprising but correct")
~~~

### Filtering with boolean masks

~~~python filtering.py
import pandas as pd
tit = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

# one condition
adults = tit[tit["age"] >= 18]
print("adults:", len(adults))

# MULTIPLE conditions - every condition needs its own parentheses
young_survivors = tit[(tit["age"] < 18) & (tit["survived"] == 1)]
print("child survivors:", len(young_survivors))

first_or_second = tit[(tit["pclass"] == 1) | (tit["pclass"] == 2)]
print("1st or 2nd class:", len(first_or_second))

not_southampton = tit[~(tit["embark_town"] == "Southampton")]
print("not from Southampton:", len(not_southampton))

# isin - much cleaner than chained ORs
print("1st or 2nd, again:", len(tit[tit["pclass"].isin([1, 2])]))

# between
print("age 20-40:", len(tit[tit["age"].between(20, 40)]))

# string methods - all under .str
print("names containing 'Miss':", tit["who"].str.contains("woman").sum())
print("upper:", tit["sex"].str.upper().head(3).tolist())

# query() - readable for long conditions
print("query:", len(tit.query("age < 18 and survived == 1 and pclass == 3")))

# .loc with a mask AND a column selection, in one go
print(tit.loc[(tit["age"] > 60) & (tit["survived"] == 1), ["age", "sex", "pclass"]])

# sorting
print("\\noldest passengers:")
print(tit.sort_values("age", ascending=False)[["age", "sex", "survived"]].head(3))
print("\\nsort by two keys:")
print(tit.sort_values(["pclass", "fare"], ascending=[True, False])
         [["pclass", "fare"]].head(5))

# top-n without sorting the whole frame
print("\\nnlargest:", tit.nlargest(3, "fare")["fare"].tolist())
~~~

:::danger The parentheses rule
~df[df.age > 18 & df.survived == 1]~ is **wrong**. Python binds ~&~ tighter than ~>~, so it
evaluates ~18 & df.survived~ first and you get a confusing error or, worse, wrong rows.
Always: ~df[(df.age > 18) & (df.survived == 1)]~.
:::
`
},
{
name: 'Missing data, types and cleaning',
goal: 'Handle NaN properly, fix wrong dtypes, and run the standard cleaning pass on a messy real file.',
md: `
~~~python cleaning.py
import numpy as np
import pandas as pd

# A deliberately awful dataset
df = pd.DataFrame({
    "id":        [1, 2, 3, 4, 5, 6, 6],                     # duplicate row
    "Name ":     [" Anna", "BORIS", "chris ", None, "Elias", "Fay", "Fay"],
    "age":       [34, 51, 22, np.nan, 220, 38, 38],         # missing + impossible
    "income":    ["28,000", "61000", "15000", "42000", "n/a", "33000", "33000"],
    "signup":    ["2023-01-15", "2022-06-02", "not a date",
                  "2021-03-19", "2023-05-05", "2024-02-14", "2024-02-14"],
    "plan":      ["Small", "large", "SMALL", "Medium", "medium", "Large", "Large"],
    "churned":   [0, 1, 0, 1, 0, 0, 0],
})
print("BEFORE")
print(df)
print(df.dtypes)

# =====================================================================
# STEP 1 - tidy the column names
# =====================================================================
df.columns = (df.columns
                .str.strip()          # remove surrounding spaces
                .str.lower()          # lowercase
                .str.replace(" ", "_", regex=False))
print("\\ncolumns now:", list(df.columns))

# =====================================================================
# STEP 2 - drop exact duplicates
# =====================================================================
print("\\nduplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
# or de-duplicate on a key, keeping the most recent:
# df = df.sort_values("signup").drop_duplicates("id", keep="last")

# =====================================================================
# STEP 3 - fix dtypes
# =====================================================================
# text that should be numeric: strip separators first, then coerce
df["income"] = (df["income"].astype(str)
                            .str.replace(",", "", regex=False))
df["income"] = pd.to_numeric(df["income"], errors="coerce")   # bad -> NaN

# text that should be a date
df["signup"] = pd.to_datetime(df["signup"], errors="coerce")

# tidy string values
df["name"] = df["name"].str.strip().str.title()
df["plan"] = df["plan"].str.strip().str.lower()

# low-cardinality strings -> category (saves a lot of memory)
df["plan"] = df["plan"].astype("category")

print("\\ndtypes after fixing:")
print(df.dtypes)

# =====================================================================
# STEP 4 - impossible values become missing
# =====================================================================
df.loc[df["age"] > 120, "age"] = np.nan
df.loc[df["age"] < 0, "age"] = np.nan

# =====================================================================
# STEP 5 - understand the missingness BEFORE filling it
# =====================================================================
print("\\nmissing per column:")
print(pd.DataFrame({
    "missing": df.isna().sum(),
    "pct": (df.isna().mean() * 100).round(1),
}))

# NaN never equals anything, not even itself
print("\\nnp.nan == np.nan ->", np.nan == np.nan, " (always False!)")
print("use .isna() instead ->", df["age"].isna().sum(), "missing ages")

# =====================================================================
# STEP 6 - handle the missing values
# =====================================================================
work = df.copy()

# option A: drop
print("\\ndrop rows with ANY missing :", work.dropna().shape)
print("drop rows missing 'age'    :", work.dropna(subset=["age"]).shape)
print("drop columns >50% missing  :",
      work.dropna(axis=1, thresh=int(0.5 * len(work))).shape)

# option B: fill with a statistic
work["age"] = work["age"].fillna(work["age"].median())      # median resists outliers
work["income"] = work["income"].fillna(work["income"].median())

# option C: fill by group - usually much better
# work["age"] = work.groupby("plan", observed=True)["age"].transform(
#     lambda s: s.fillna(s.median()))

# option D: forward/backward fill - for time series ONLY
# ts = ts.ffill()      carry the last known value forward
# ts = ts.bfill()

# option E: mark that it WAS missing - sometimes the best feature you have
work["age_was_missing"] = df["age"].isna().astype(int)

print("\\nAFTER")
print(work)
print("\\nremaining missing:", work.isna().sum().sum())
~~~

### Choosing a missing-data strategy

| Situation | Do this |
|---|---|
| Under ~5% missing, at random | Fill with median (numeric) or mode (categorical) |
| A lot missing, but the column is important | Fill **and** add an ~was_missing~ indicator column |
| Over ~60% missing | Usually drop the column |
| Missing because of a real reason (no income = unemployed) | Fill with a sentinel and add the indicator - the missingness is signal |
| Time series | Forward-fill; never fill with the global mean |
| A whole row is empty | Drop the row |

:::tip Missingness is data
If income is missing more often for churned customers, then "income is missing" predicts
churn. Adding ~income_was_missing~ costs one column and sometimes beats every clever
imputation you could invent. Always check:
~~~python
df.groupby(df["income"].isna())["churned"].mean()
~~~
:::

:::warn Impute inside the pipeline
The medians above were computed from the whole frame. In a real workflow, imputation must
happen inside a ~Pipeline~ so each cross-validation fold computes its own median from its
own training rows. Otherwise you have leaked. See the Data Engineering track.
:::
`
},
{
name: 'Grouping, joining and reshaping',
goal: 'The three operations that turn raw rows into features: groupby aggregation, merges, and pivots.',
md: `
## groupby: split - apply - combine

~~~text
        SPLIT                 APPLY                COMBINE
  city    spend          mean per group        city    spend
  Athens    100    -->   Athens: (100+140)/2   Athens    120
  Patras     80          Patras: (80+90)/2     Patras     85
  Athens    140
  Patras     90
~~~

~~~python groupby.py
import numpy as np
import pandas as pd

tit = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")

# one column, one function
print(tit.groupby("pclass")["survived"].mean().round(3))

# several functions at once
print("\\n", tit.groupby("pclass")["fare"].agg(["count", "mean", "median", "std"]).round(2))

# several columns, different functions each - the professional form
summary = tit.groupby("pclass").agg(
    n=("survived", "size"),
    survival_rate=("survived", "mean"),
    avg_age=("age", "mean"),
    median_fare=("fare", "median"),
    pct_female=("sex", lambda s: (s == "female").mean()),
).round(3)
print("\\n", summary)

# group by MORE THAN ONE column
print("\\n", tit.groupby(["pclass", "sex"])["survived"].mean().round(3))

# ... and unstack it into a readable table
print("\\n", tit.groupby(["pclass", "sex"])["survived"].mean().unstack().round(3))

# =====================================================================
# transform: aggregate but keep the original number of rows
#            THE most useful groupby trick for feature engineering
# =====================================================================
tit["class_avg_fare"] = tit.groupby("pclass")["fare"].transform("mean")
tit["fare_vs_class"] = tit["fare"] / tit["class_avg_fare"]
print("\\n", tit[["pclass", "fare", "class_avg_fare", "fare_vs_class"]].head(5).round(2))

# group-wise imputation, the right way
tit["age_filled"] = tit.groupby(["pclass", "sex"])["age"].transform(
    lambda s: s.fillna(s.median()))
print("\\nages still missing after group-wise fill:", tit["age_filled"].isna().sum())

# filter whole groups
big_groups = tit.groupby("embark_town").filter(lambda g: len(g) > 100)
print("rows in ports with >100 passengers:", len(big_groups))

# apply a custom function per group
def group_report(g):
    return pd.Series({
        "n": len(g),
        "survived": g["survived"].sum(),
        "rate": g["survived"].mean(),
    })
print("\\n", tit.groupby("sex")[["survived"]].apply(
    lambda g: pd.Series({"n": len(g), "rate": g["survived"].mean()})).round(3))
~~~

## Joining tables

~~~python merging.py
import pandas as pd

customers = pd.DataFrame({
    "cust_id": [1, 2, 3, 4],
    "name":    ["Anna", "Boris", "Chris", "Dora"],
    "city":    ["Athens", "Patras", "Athens", "Volos"],
})
orders = pd.DataFrame({
    "order_id": [10, 11, 12, 13, 14],
    "cust_id":  [1, 1, 2, 2, 5],          # note: customer 5 does not exist
    "amount":   [50.0, 30.0, 120.0, 45.0, 99.0],
})

print("INNER (only matches):")
print(pd.merge(customers, orders, on="cust_id", how="inner"))

print("\\nLEFT (keep all customers):")
print(pd.merge(customers, orders, on="cust_id", how="left"))

print("\\nOUTER (keep everything):")
print(pd.merge(customers, orders, on="cust_id", how="outer"))

# Always check what happened - use the indicator
check = pd.merge(customers, orders, on="cust_id", how="outer", indicator=True)
print("\\nmerge audit:")
print(check["_merge"].value_counts())

# validate catches duplicate-key bugs before they silently multiply your rows
try:
    pd.merge(customers, orders, on="cust_id", validate="one_to_one")
except Exception as e:
    print("\\nvalidate caught it:", type(e).__name__, str(e)[:60])

# The realistic pattern: aggregate the child table, then left-join
per_customer = orders.groupby("cust_id").agg(
    n_orders=("order_id", "count"),
    total_spend=("amount", "sum"),
    avg_order=("amount", "mean"),
).reset_index()

features = customers.merge(per_customer, on="cust_id", how="left")
features[["n_orders", "total_spend", "avg_order"]] = \\
    features[["n_orders", "total_spend", "avg_order"]].fillna(0)
print("\\nFEATURE TABLE (this is what goes into the model):")
print(features)

# concat = stacking, not joining
print("\\nconcat rows:", pd.concat([customers, customers]).shape)
print("concat cols:", pd.concat([customers, customers], axis=1).shape)
~~~

~~~text
merge audit:
both          4
left_only     2
right_only    1
Name: _merge, dtype: int64
~~~

:::danger The silent row explosion
If both tables have duplicate keys, a merge produces the *cross product* of the matches.
A 1,000-row table joined to a 1,000-row table on a duplicated key can return 200,000 rows,
and your metrics will be quietly meaningless. **Always check ~len(df)~ before and after a
merge**, and use ~validate="one_to_many"~ when you know the expected relationship.
:::

## Reshaping: long and wide

~~~python reshape.py
import pandas as pd

long = pd.DataFrame({
    "city":  ["Athens", "Athens", "Patras", "Patras", "Volos", "Volos"],
    "year":  [2023, 2024, 2023, 2024, 2023, 2024],
    "sales": [100, 130, 80, 95, 60, 72],
})
print("LONG (tidy - one observation per row):")
print(long)

wide = long.pivot(index="city", columns="year", values="sales")
print("\\nWIDE (a report):")
print(wide)

print("\\nback to long:")
print(wide.reset_index().melt(id_vars="city", var_name="year", value_name="sales"))

# pivot_table aggregates, so it handles duplicate index/column pairs
tit = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
print("\\nsurvival rate by class and sex:")
print(pd.pivot_table(tit, values="survived", index="pclass",
                     columns="sex", aggfunc="mean", margins=True).round(3))

# crosstab - counts
print("\\ncounts:")
print(pd.crosstab(tit["pclass"], tit["sex"], margins=True))
~~~

:::note Long vs wide
**Long/tidy** (one observation per row) is what you model and plot with.
**Wide** is what humans read. Use ~melt~ to go long, ~pivot~ to go wide.
:::
`
},
{
name: 'Time series and performance',
goal: 'Handle dates properly, resample, compute rolling features, and make pandas fast on large frames.',
md: `
~~~python timeseries.py
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
dates = pd.date_range("2023-01-01", periods=730, freq="D")
trend = np.linspace(100, 160, 730)
weekly = 12 * np.sin(np.arange(730) * 2 * np.pi / 7)
yearly = 25 * np.sin(np.arange(730) * 2 * np.pi / 365)
noise = rng.normal(0, 6, 730)

ts = pd.DataFrame({"date": dates,
                   "sales": trend + weekly + yearly + noise}).set_index("date")
print(ts.head())

# =====================================================================
# DATETIME PARTS - your feature engineering for any dated dataset
# =====================================================================
feat = ts.copy()
feat["year"] = feat.index.year
feat["month"] = feat.index.month
feat["day_of_week"] = feat.index.dayofweek        # Monday = 0
feat["day_of_year"] = feat.index.dayofyear
feat["week"] = feat.index.isocalendar().week.astype(int)
feat["is_weekend"] = (feat.index.dayofweek >= 5).astype(int)
feat["is_month_end"] = feat.index.is_month_end.astype(int)
feat["quarter"] = feat.index.quarter

# CYCLICAL encoding - so December(12) is next to January(1), not 11 units away
feat["month_sin"] = np.sin(2 * np.pi * feat["month"] / 12)
feat["month_cos"] = np.cos(2 * np.pi * feat["month"] / 12)
feat["dow_sin"] = np.sin(2 * np.pi * feat["day_of_week"] / 7)
feat["dow_cos"] = np.cos(2 * np.pi * feat["day_of_week"] / 7)
print("\\n", feat.head(3).round(3))

# =====================================================================
# SELECTING BY TIME
# =====================================================================
print("\\nJune 2023      :", len(ts.loc["2023-06"]))
print("a date range   :", len(ts.loc["2023-06-01":"2023-08-31"]))
print("all of 2024    :", len(ts.loc["2024"]))

# =====================================================================
# RESAMPLING - change the frequency
# =====================================================================
print("\\nmonthly totals:")
print(ts.resample("ME").sum().head(4).round(1))
print("\\nweekly mean:")
print(ts.resample("W").mean().head(4).round(1))
print("\\nmonthly, several stats:")
print(ts.resample("ME")["sales"].agg(["min", "mean", "max"]).head(3).round(1))

# =====================================================================
# ROLLING WINDOWS - the classic time-series features
# =====================================================================
feat["ma_7"] = ts["sales"].rolling(7).mean()          # smooth out weekly noise
feat["ma_30"] = ts["sales"].rolling(30).mean()
feat["std_7"] = ts["sales"].rolling(7).std()
feat["max_30"] = ts["sales"].rolling(30).max()
feat["ewm_7"] = ts["sales"].ewm(span=7).mean()        # weights recent days more

# LAGS - what the value was N days ago
for lag in [1, 7, 14, 365]:
    feat[f"lag_{lag}"] = ts["sales"].shift(lag)

# change vs the past
feat["diff_1"] = ts["sales"].diff()
feat["pct_change_7"] = ts["sales"].pct_change(7)

print("\\n", feat[["sales", "ma_7", "lag_7", "diff_1"]].tail(4).round(2))

# expanding = all history up to now
feat["cumulative_mean"] = ts["sales"].expanding().mean()
~~~

:::danger The lag that ruins everything
~rolling(7).mean()~ in pandas is **trailing** by default - at row t it uses rows t-6..t,
including t itself. If ~sales~ at time t is your *target*, that feature contains the answer.

For forecasting, always shift first:
~~~python
feat["ma_7_safe"] = ts["sales"].shift(1).rolling(7).mean()
~~~
Now the feature at time t uses only days t-7..t-1. Rule: **any feature must be computable
using only data strictly before the prediction moment.**
:::

## Making pandas fast

~~~python performance.py
import numpy as np, pandas as pd, time

n = 500_000
rng = np.random.default_rng(0)
df = pd.DataFrame({
    "a": rng.random(n),
    "b": rng.random(n),
    "cat": rng.choice(["x", "y", "z"], n),
})

def timeit(fn, label):
    t0 = time.perf_counter(); out = fn(); dt = time.perf_counter() - t0
    print(f"{label:36s} {dt*1000:9.1f} ms")
    return out

# ---- 1. NEVER iterate rows -----------------------------------------
def with_iterrows():
    out = []
    for _, row in df.head(20000).iterrows():      # only 20k, it is that slow
        out.append(row["a"] * row["b"])
    return out

def with_apply():
    return df.head(20000).apply(lambda r: r["a"] * r["b"], axis=1)

def vectorised():
    return df.head(20000)["a"] * df.head(20000)["b"]

timeit(with_iterrows, "iterrows (20k rows)")
timeit(with_apply,    "apply axis=1 (20k rows)")
timeit(vectorised,    "vectorised (20k rows)")

# ---- 2. category dtype for repeated strings ------------------------
mem_obj = df["cat"].memory_usage(deep=True) / 1e6
df["cat_c"] = df["cat"].astype("category")
mem_cat = df["cat_c"].memory_usage(deep=True) / 1e6
print(f"\\nstring column : {mem_obj:6.2f} MB")
print(f"category column: {mem_cat:6.2f} MB   ({mem_obj/mem_cat:.0f}x smaller)")

# ---- 3. downcast numerics ------------------------------------------
before = df.memory_usage(deep=True).sum() / 1e6
df["a32"] = pd.to_numeric(df["a"], downcast="float")
print(f"\\nfloat64 -> float32 halves the memory of that column")

# ---- 4. build lists, then concat once ------------------------------
# BAD:   for ...: df = pd.concat([df, new_row])     O(n^2)
# GOOD:  rows = [...]; df = pd.DataFrame(rows)      O(n)

# ---- 5. use the right file format ----------------------------------
# CSV     : human readable, slow, loses dtypes
# Parquet : 5-10x smaller, 10x faster, keeps dtypes.  Use it.
# df.to_parquet("data.parquet")
# df = pd.read_parquet("data.parquet")
~~~

~~~text
iterrows (20k rows)                    1893.4 ms
apply axis=1 (20k rows)                 421.7 ms
vectorised (20k rows)                     0.9 ms

string column :  29.14 MB
category column:   0.50 MB   (58x smaller)
~~~

:::tip The performance hierarchy
**vectorised NumPy/pandas > ~.apply~ on a Series > ~.apply(axis=1)~ > ~itertuples~ > ~iterrows~**

If you find yourself writing a loop over rows, stop and ask what the vectorised expression
is. It exists roughly 95% of the time.
:::
`
}
],
quiz: [
{
q: 'How many rows does ~df.loc[0:2]~ return, versus ~df.iloc[0:2]~, on a default integer index?',
options: ['2 and 2', '3 and 2', '2 and 3', '3 and 3'],
answer: 1,
why: '.loc slices are label-based and INCLUSIVE of the endpoint (rows 0,1,2 = three rows). .iloc is positional and exclusive (rows 0,1 = two rows). This asymmetry catches everyone once.'
},
{
q: 'Which correctly filters for adults who survived?',
options: [
  '~df[df.age >= 18 & df.survived == 1]~',
  '~df[(df.age >= 18) & (df.survived == 1)]~',
  '~df[df.age >= 18 and df.survived == 1]~',
  '~df[df.age >= 18, df.survived == 1]~'
],
answer: 1,
why: 'Each condition needs parentheses because & binds tighter than the comparison operators, and you must use & rather than the Python keyword ~and~, which cannot operate on arrays.'
},
{
q: 'You want a per-row feature holding each passenger class average fare. Which is right?',
options: [
  '~df.groupby("pclass")["fare"].mean()~',
  '~df.groupby("pclass")["fare"].transform("mean")~',
  '~df.groupby("pclass")["fare"].agg("mean")~',
  '~df["fare"].mean()~'
],
answer: 1,
why: '~transform~ broadcasts the group result back to the original row count, so it can be assigned as a column. ~mean~ and ~agg~ collapse to one row per group.'
},
{
q: 'For a forecasting model, why is ~sales.rolling(7).mean()~ unsafe as a feature?',
options: [
  'It is too slow',
  'It includes the current row, which is the value you are trying to predict',
  'Rolling windows do not work on time series',
  'It produces NaN at the start'
],
answer: 1,
why: 'pandas rolling windows are trailing and inclusive of the current row, so the feature contains the target. Use ~.shift(1).rolling(7).mean()~ so only strictly past data is used.'
}
]
},

/* ============================================================ */
{
id: 'visualisation',
title: 'Visualisation with matplotlib and seaborn',
summary: 'Charts that answer questions - distributions, relationships, comparisons and model diagnostics - plus the plot types you will use in every EDA.',
tags: ['matplotlib', 'seaborn', 'eda'],
intro: `
## Plot to learn, not to decorate

Every chart in data science answers one of four questions:

| Question | Chart |
|---|---|
| How is this variable spread out? | histogram, KDE, box plot, violin |
| How do two variables relate? | scatter, line, hexbin, 2-D density |
| How do groups compare? | bar, grouped box, strip/swarm |
| How does the model behave? | residual plot, ROC curve, confusion matrix, learning curve |

## matplotlib's two interfaces

~~~python
# 1. pyplot / state machine - fine for one quick plot
plt.plot(x, y)
plt.title("quick")
plt.show()

# 2. object-oriented - use this for anything with more than one panel
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(x, y)
ax[0].set_title("left")
ax[1].hist(y)
fig.tight_layout()
~~~

Learn the second one. Every multi-panel figure and every reusable plotting function needs it.

:::tip The famous first question
Anscombe's quartet: four datasets with identical means, variances, correlations and
regression lines - and completely different shapes. **Always plot the data.** Summary
statistics can be identical for wildly different realities.
:::
`,
keyPoints: [
  'Use the object-oriented API (fig, ax) for anything beyond a single throwaway plot.',
  'Histogram for one variable, scatter for two, box/violin for group comparison.',
  'seaborn takes a tidy (long) DataFrame and does the grouping for you.',
  'Always label axes and state units. An unlabelled chart is not evidence.'
],
pitfalls: [
  'Bar charts that do not start at zero - they exaggerate differences.',
  'Scatter plots with 100,000 overlapping points; use alpha, hexbin or sampling.',
  'Pie charts with more than three slices. Use a bar chart.',
  'Forgetting ~plt.show()~ in a script, or ~%matplotlib inline~ in an old notebook.'
],
levels: [
{
name: 'The essential chart types',
goal: 'Produce every chart you need for exploratory analysis, with the code you will reuse forever.',
md: `
~~~python essential_charts.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

tips = sns.load_dataset("tips")        # ships with seaborn, no download needed
print(tips.head())

fig, axes = plt.subplots(2, 3, figsize=(16, 9))

# ---------- 1. HISTOGRAM: the shape of one variable -----------------
sns.histplot(data=tips, x="total_bill", bins=25, kde=True, ax=axes[0, 0])
axes[0, 0].set_title("1. Distribution of a single variable")
axes[0, 0].set_xlabel("total bill (USD)")

# ---------- 2. BOX PLOT: compare groups, see outliers ---------------
sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0, 1])
axes[0, 1].set_title("2. Group comparison + outliers")

# ---------- 3. SCATTER: relationship between two variables ----------
sns.scatterplot(data=tips, x="total_bill", y="tip",
                hue="time", size="size", alpha=0.75, ax=axes[0, 2])
axes[0, 2].set_title("3. Relationship, coloured by a third variable")

# ---------- 4. BAR: an aggregate per category -----------------------
sns.barplot(data=tips, x="day", y="tip", hue="sex",
            errorbar=("ci", 95), ax=axes[1, 0])
axes[1, 0].set_title("4. Mean tip per day, with 95% CI")

# ---------- 5. VIOLIN: full distribution per group ------------------
sns.violinplot(data=tips, x="day", y="total_bill", hue="smoker",
               split=True, ax=axes[1, 1])
axes[1, 1].set_title("5. Full shape, not just quartiles")

# ---------- 6. HEATMAP: a matrix, usually correlations --------------
corr = tips.select_dtypes("number").corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, ax=axes[1, 2])
axes[1, 2].set_title("6. Correlation matrix")

fig.suptitle("The six charts that cover most of exploratory analysis", fontsize=14)
fig.tight_layout()
plt.show()
~~~

### Reading each one

- **Histogram** - is it symmetric, skewed, bimodal? Are there impossible values at the edges?
- **Box plot** - the box is the middle 50%, the line is the median, the dots are outliers
  beyond 1.5 times the interquartile range.
- **Scatter** - linear? curved? heteroscedastic (fanning out)? clustered?
- **Bar with CI** - if the confidence intervals overlap heavily, the difference is not solid.
- **Violin** - a box plot that shows bimodality, which a box plot hides completely.
- **Heatmap** - dark red or dark blue pairs are highly correlated; watch for redundant features.

### More specialised, still essential

~~~python more_charts.py
import numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
flights = sns.load_dataset("flights")

fig, ax = plt.subplots(2, 3, figsize=(16, 9))

# pairplot-style: every numeric pair at once (run separately, it makes its own figure)
# sns.pairplot(tips, hue="time", diag_kind="kde")

# 1. count plot - frequency of a category
sns.countplot(data=tips, x="day", hue="smoker", ax=ax[0, 0])
ax[0, 0].set_title("category frequencies")

# 2. line plot - a time series
sns.lineplot(data=flights, x="year", y="passengers", ax=ax[0, 1])
ax[0, 1].set_title("trend over time (band = 95% CI across months)")

# 3. hexbin - a scatter with too many points
rng = np.random.default_rng(0)
big = pd.DataFrame({"x": rng.normal(size=50000),
                    "y": rng.normal(size=50000) * 0.6})
big["y"] += big["x"] * 0.8
hb = ax[0, 2].hexbin(big["x"], big["y"], gridsize=40, cmap="viridis")
ax[0, 2].set_title("50,000 points: hexbin, not scatter")
plt.colorbar(hb, ax=ax[0, 2])

# 4. ECDF - what fraction is below each value. Underused and excellent.
sns.ecdfplot(data=tips, x="total_bill", hue="time", ax=ax[1, 0])
ax[1, 0].set_title("ECDF: read percentiles straight off")

# 5. regression plot with a fitted line and CI
sns.regplot(data=tips, x="total_bill", y="tip",
            scatter_kws={"alpha": 0.4}, line_kws={"color": "crimson"}, ax=ax[1, 1])
ax[1, 1].set_title("scatter + fitted line + uncertainty")

# 6. strip/swarm - every individual point, for small data
sns.stripplot(data=tips, x="day", y="tip", hue="sex",
              dodge=True, alpha=0.7, ax=ax[1, 2])
ax[1, 2].set_title("show the raw points")

fig.tight_layout(); plt.show()

# FACETING - the same chart repeated per group. seaborn's superpower.
g = sns.FacetGrid(tips, col="time", row="smoker", height=3.2, aspect=1.3)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip", alpha=0.7)
g.add_legend()
g.figure.suptitle("Faceting: one panel per subgroup", y=1.02)
plt.show()
~~~

:::warn Two charts that mislead
1. **Bar charts with a truncated y-axis.** Starting at 90 instead of 0 turns a 2% difference
   into a visual mountain. For bars, always start at zero.
2. **Overplotted scatter.** With 50,000 points a scatter is a black blob that hides the
   density. Use ~alpha=0.1~, ~hexbin~, or a 2-D KDE.
:::
`
},
{
name: 'Model diagnostic plots',
goal: 'Build the four plots that tell you whether your model is trustworthy - not just what its score is.',
md: `
A score is one number. These plots tell you *how* the model is wrong, which is what you
need to improve it.

~~~python diagnostics.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc, precision_recall_curve)

sns.set_theme(style="whitegrid")

# =====================================================================
# REGRESSION DIAGNOSTICS
# =====================================================================
X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)
reg = RandomForestRegressor(n_estimators=120, n_jobs=-1, random_state=0).fit(X_tr, y_tr)
pred = reg.predict(X_te)
resid = y_te - pred

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

# --- 1. predicted vs actual: points should hug the diagonal ---------
ax[0].scatter(y_te, pred, alpha=0.15, s=8)
lims = [min(y_te.min(), pred.min()), max(y_te.max(), pred.max())]
ax[0].plot(lims, lims, "r--", lw=2)
ax[0].set_xlabel("actual"); ax[0].set_ylabel("predicted")
ax[0].set_title("1. Predicted vs actual\\n(tight around the line = good)")

# --- 2. residuals vs predicted: should be a shapeless cloud ---------
ax[1].scatter(pred, resid, alpha=0.15, s=8)
ax[1].axhline(0, color="r", ls="--", lw=2)
ax[1].set_xlabel("predicted"); ax[1].set_ylabel("residual (actual - predicted)")
ax[1].set_title("2. Residual plot\\n(any pattern = missing structure)")

# --- 3. residual distribution: should be centred and roughly normal -
sns.histplot(resid, bins=50, kde=True, ax=ax[2])
ax[2].axvline(0, color="r", ls="--")
ax[2].set_title(f"3. Residuals\\nmean={resid.mean():.3f}  std={resid.std():.3f}")
fig.tight_layout(); plt.show()

print("What to look for in the residual plot:")
print("  funnel shape  -> variance grows with the prediction (try log-transforming y)")
print("  curved band   -> the model is missing a non-linear relationship")
print("  a flat ceiling-> the target is capped (California housing caps at 5.0)")
~~~

~~~python classification_diagnostics.py
# =====================================================================
# CLASSIFICATION DIAGNOSTICS
# =====================================================================
Xc, yc = load_breast_cancer(return_X_y=True)
yc = 1 - yc                      # 1 = malignant, the class we care about
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, test_size=0.25,
                                              random_state=0, stratify=yc)
clf = RandomForestClassifier(n_estimators=300, random_state=0).fit(Xc_tr, yc_tr)
proba = clf.predict_proba(Xc_te)[:, 1]
pred = (proba >= 0.5).astype(int)

fig, ax = plt.subplots(1, 4, figsize=(20, 4.4))

# --- 1. confusion matrix --------------------------------------------
ConfusionMatrixDisplay(confusion_matrix(yc_te, pred),
                       display_labels=["benign", "malignant"]).plot(ax=ax[0], colorbar=False)
ax[0].set_title("1. Confusion matrix")

# --- 2. ROC curve ----------------------------------------------------
fpr, tpr, _ = roc_curve(yc_te, proba)
ax[1].plot(fpr, tpr, lw=2, label=f"AUC = {auc(fpr, tpr):.3f}")
ax[1].plot([0, 1], [0, 1], "k--", lw=1)
ax[1].set_xlabel("false positive rate"); ax[1].set_ylabel("true positive rate")
ax[1].set_title("2. ROC curve"); ax[1].legend()

# --- 3. precision-recall curve (better for imbalanced data) ---------
prec, rec, _ = precision_recall_curve(yc_te, proba)
ax[2].plot(rec, prec, lw=2)
ax[2].set_xlabel("recall"); ax[2].set_ylabel("precision")
ax[2].set_title("3. Precision-Recall curve")

# --- 4. predicted probability distribution --------------------------
sns.histplot(x=proba, hue=yc_te, bins=30, ax=ax[3],
             palette={0: "steelblue", 1: "crimson"}, alpha=0.6)
ax[3].axvline(0.5, color="k", ls="--")
ax[3].set_xlabel("predicted P(malignant)")
ax[3].set_title("4. Are the classes well separated?")
fig.tight_layout(); plt.show()

# --- 5. feature importance ------------------------------------------
data = load_breast_cancer()
imp = pd.Series(clf.feature_importances_, index=data.feature_names).nlargest(12)
plt.figure(figsize=(8, 5))
sns.barplot(x=imp.values, y=imp.index, hue=imp.index, palette="viridis", legend=False)
plt.xlabel("importance"); plt.title("5. Which features drive the model")
plt.tight_layout(); plt.show()
~~~

### What each plot tells you

| Plot | Healthy | Warning sign |
|---|---|---|
| Predicted vs actual | Tight cloud on the diagonal | Fanning, curvature, a ceiling |
| Residuals vs predicted | Shapeless horizontal band | Funnel (try log y), curve (add non-linearity) |
| Confusion matrix | Errors on the cheap side | Many false negatives on a costly class |
| ROC / PR curve | Rises fast to the top-left | Hugging the diagonal = near-random |
| Probability histogram | Two separated humps | One merged blob = the model cannot tell classes apart |
| Feature importance | A few clear drivers | One feature at 0.9 = suspect leakage |

:::danger The leakage tell
If one feature dominates importance and the score is suspiciously high, check whether that
feature is available at prediction time. A single feature carrying 90% of the importance is
almost always a leak, not a discovery.
:::
`
},
{
name: 'Publication-quality figures',
goal: 'Turn a working plot into one you can put in a report - consistent style, correct labels, saved at the right resolution.',
md: `
~~~python publication.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# =====================================================================
# 1. SET A HOUSE STYLE ONCE, AT THE TOP OF THE SCRIPT
# =====================================================================
mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,               # print quality when saving
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,         # remove chartjunk
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "legend.frameon": False,
    "figure.autolayout": True,
})
sns.set_palette("deep")

# =====================================================================
# 2. BUILD THE FIGURE WITH THE OBJECT-ORIENTED API
# =====================================================================
rng = np.random.default_rng(7)
months = pd.date_range("2023-01-01", periods=24, freq="ME")
data = pd.DataFrame({
    "month": months,
    "model_a": 0.72 + np.cumsum(rng.normal(0.004, 0.010, 24)),
    "model_b": 0.70 + np.cumsum(rng.normal(0.007, 0.012, 24)),
})

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(data["month"], data["model_a"], marker="o", ms=4, lw=2,
        label="Model A (logistic regression)")
ax.plot(data["month"], data["model_b"], marker="s", ms=4, lw=2,
        label="Model B (gradient boosting)")

# a confidence band
ax.fill_between(data["month"], data["model_b"] - 0.02, data["model_b"] + 0.02,
                alpha=0.15, color="C1")

# reference line with an explanatory label
ax.axhline(0.75, color="grey", ls="--", lw=1.2)
ax.text(data["month"].iloc[0], 0.753, "business requirement", fontsize=9, color="grey")

# annotate the single most important point
best = data.loc[data["model_b"].idxmax()]
ax.annotate(f"peak {best['model_b']:.3f}",
            xy=(best["month"], best["model_b"]),
            xytext=(-70, 24), textcoords="offset points",
            arrowprops=dict(arrowstyle="->", color="C1"),
            fontsize=9, color="C1")

# ---- labels: never optional ----------------------------------------
ax.set_title("Model accuracy on the monthly hold-out set")
ax.set_xlabel("Month")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.65, 0.90)
ax.legend(loc="lower right")

# a source note - makes a chart self-contained
fig.text(0.01, -0.02, "Source: monthly evaluation runs, n=4,200 per month",
         fontsize=8, color="grey")

# =====================================================================
# 3. SAVE PROPERLY
# =====================================================================
fig.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
fig.savefig("model_comparison.pdf", bbox_inches="tight")   # vector, for documents
plt.show()
~~~

### A reusable plotting function

Stop copying plot code between notebooks. Write it once:

~~~python plot_helpers.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_distributions(df, cols=None, ncols=4, figsize_per=(3.4, 2.6)):
    """Histogram every numeric column - the first thing to run on any dataset."""
    cols = cols or df.select_dtypes("number").columns.tolist()
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(figsize_per[0] * ncols, figsize_per[1] * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), bins=30, ax=ax, kde=True)
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
        skew = df[col].skew()
        if abs(skew) > 1:
            ax.text(0.97, 0.92, f"skew {skew:.1f}", transform=ax.transAxes,
                    ha="right", fontsize=8, color="crimson")
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_target_relationship(df, target, cols=None, ncols=4):
    """How does each feature relate to the target? Run this before modelling."""
    cols = cols or [c for c in df.select_dtypes("number").columns if c != target]
    nrows = int(np.ceil(len(cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, col in zip(axes, cols):
        ax.scatter(df[col], df[target], alpha=0.2, s=6)
        r = df[[col, target]].corr().iloc[0, 1]
        ax.set_title(f"{col}  (r={r:.2f})", fontsize=10)
    for ax in axes[len(cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig

# usage:
# import seaborn as sns
# tips = sns.load_dataset("tips")
# plot_distributions(tips); plt.show()
# plot_target_relationship(tips, target="tip"); plt.show()
~~~

:::tip The checklist before a chart leaves your machine
1. Does it have a title that states the **finding**, not the chart type?
2. Are both axes labelled, with units?
3. Does the y-axis start where it should (zero for bars)?
4. Is the legend readable and necessary?
5. Would it still make sense in greyscale, and to a colour-blind reader?
6. Is the source or sample size stated?
:::
`
}
],
quiz: [
{
q: 'You need to plot 80,000 points and the scatter is a solid blob. What do you do?',
options: [
  'Use a pie chart instead',
  'Use hexbin, a 2-D density plot, or heavy transparency',
  'Remove most of the data',
  'Increase the marker size'
],
answer: 1,
why: 'Overplotting hides density. Hexbin and 2-D KDE show where the mass actually is; alpha helps for moderate sizes. Sampling is a last resort because it discards information.'
},
{
q: 'A residual plot shows a clear funnel shape widening to the right. What does that mean?',
options: [
  'The model is perfect',
  'Error variance grows with the predicted value - try a log transform of the target',
  'There are too many features',
  'The learning rate is wrong'
],
answer: 1,
why: 'That is heteroscedasticity. Log-transforming a positive, right-skewed target usually stabilises it, and often improves the model as well.'
},
{
q: 'Which is the correct way to build a figure with four panels?',
options: [
  'Call plt.plot four times',
  '~fig, axes = plt.subplots(2, 2)~ then draw on each ~axes[i, j]~',
  'Create four separate scripts',
  'Use plt.figure() four times'
],
answer: 1,
why: 'The object-oriented API gives you explicit axes objects, which is the only manageable way to build, style and save multi-panel figures.'
}
]
}

]
});
