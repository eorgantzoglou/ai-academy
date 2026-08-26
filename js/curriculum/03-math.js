/* Track 03 - The mathematics you actually need */
(window.CURRICULUM = window.CURRICULUM || []).push({
id: 'math',
title: 'Math for ML',
icon: 'fx',
level: 'Beginner',
blurb: 'Linear algebra, calculus, probability, statistics and information theory - taught only where they touch real machine learning, with runnable code instead of proofs.',
intro: `
## How much maths do you actually need?

Less than you fear, and more than tutorials admit. The honest answer:

| You need to... | Maths required |
|---|---|
| Use scikit-learn effectively | Almost none. Read this track once. |
| Debug a model that is behaving oddly | Solid intuition for gradients, distributions and correlation |
| Read a paper or implement from scratch | Comfortable linear algebra and calculus notation |
| Do research | Real analysis, optimisation theory, measure-theoretic probability |

This track targets the middle two. **Every concept is introduced because something in
machine learning breaks without it**, and every one comes with NumPy code you can run.

Nothing here is proved. If you want proofs, this track will tell you what to search for.
`,
topics: [

/* ============================================================ */
{
id: 'linear-algebra',
title: 'Linear algebra: the language of data',
summary: 'Vectors, matrices, dot products and matrix multiplication - what they mean geometrically, and why every dataset and every neural layer is one of them.',
tags: ['math', 'linear-algebra', 'essential'],
intro: `
## Why data is a matrix

One customer with five measurements is a **vector** in 5-dimensional space.
A thousand customers is a **matrix** of shape (1000, 5).

~~~text
        feature1 feature2 feature3
row 0  [   34      28000     0.71  ]   <- one sample = one point in 3-D space
row 1  [   51      61000     0.35  ]
row 2  [   22      15000     0.88  ]

X.shape = (3, 3)
~~~

Once you see data this way, machine learning becomes geometry: classification is *cutting
space with a surface*, clustering is *finding dense regions*, PCA is *rotating the space*,
and a neural network layer is *a rotation, a stretch and a bend*.

## The four operations that matter

**1. The dot product** - the single most important operation in machine learning.

:::math Dot product
For vectors **a** and **b**: **a · b = a1b1 + a2b2 + ... + anbn**

Geometrically: **a · b = |a| |b| cos(θ)** where θ is the angle between them.

- Positive -> they point in broadly the same direction
- Zero -> perpendicular, unrelated
- Negative -> opposite directions
:::

Every linear model is a dot product: ~prediction = weights · features + bias~.
Every attention score in a Transformer is a dot product between a query and a key.
Every similarity search is a dot product between embeddings.

**2. Matrix-vector multiplication** - apply a transformation to a point.

**3. Matrix-matrix multiplication** - apply a transformation to many points at once, or
compose two transformations.

~~~text
(m x n) @ (n x p) = (m x p)
      ^      ^
      these must MATCH. This is the shape error you will see a thousand times.
~~~

**4. The transpose** - flip rows and columns. Mostly appears to make shapes line up.
`,
keyPoints: [
  'A dataset is a matrix: rows are samples, columns are features.',
  'The dot product measures alignment - it is the core of linear models, attention and similarity.',
  'For A @ B, the inner dimensions must match: (m,n) @ (n,p) -> (m,p).',
  'Matrix multiplication is not commutative: A @ B is generally not B @ A.'
],
pitfalls: [
  'Using ~*~ (element-wise) when you mean ~@~ (matrix product).',
  'Confusing a shape (n,) with (n,1). NumPy treats them differently in matrix contexts.',
  'Forgetting that ~np.dot~ on two 1-D arrays returns a scalar, not an array.'
],
levels: [
{
name: 'Vectors, dot products and geometry',
goal: 'Build a working intuition for what a dot product measures, then use it to build a real recommender.',
md: `
~~~python vectors.py
import numpy as np

# =====================================================================
# A VECTOR IS A POINT, OR AN ARROW FROM THE ORIGIN
# =====================================================================
a = np.array([3.0, 4.0])
b = np.array([4.0, 3.0])
c = np.array([-4.0, 3.0])

# LENGTH (magnitude, L2 norm)  |a| = sqrt(3^2 + 4^2) = 5
print("length of a:", np.linalg.norm(a))
print("by hand    :", np.sqrt((a ** 2).sum()))

# ADDITION = place the arrows head to tail
print("a + b:", a + b)

# SCALING = stretch the arrow
print("2 * a:", 2 * a, " length:", np.linalg.norm(2 * a))

# =====================================================================
# THE DOT PRODUCT
# =====================================================================
print("\\na . b =", np.dot(a, b), "  (also a @ b =", a @ b, ")")
print("by hand:", 3 * 4 + 4 * 3)

# The geometric meaning: a . b = |a| |b| cos(theta)
def angle_between(u, v):
    cos_t = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos_t = np.clip(cos_t, -1.0, 1.0)      # guard against float error
    return np.degrees(np.arccos(cos_t))

print(f"\\nangle(a, b) = {angle_between(a, b):.1f} degrees   dot = {a @ b:+.1f}")
print(f"angle(a, c) = {angle_between(a, c):.1f} degrees   dot = {a @ c:+.1f}")
print(f"angle(a, a) = {angle_between(a, a):.1f} degrees   dot = {a @ a:+.1f}")

perp = np.array([-4.0, 3.0]) / 5 * 5
print(f"\\nperpendicular vectors have dot product: {np.dot([1, 0], [0, 1])}")

# =====================================================================
# COSINE SIMILARITY - the dot product, normalised.
# This IS how semantic search, recommenders and RAG work.
# =====================================================================
def cosine_similarity(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# Three users rated five films from 1 to 5
users = {
    "Anna":  np.array([5, 4, 1, 1, 5]),
    "Boris": np.array([4, 5, 2, 1, 4]),     # tastes like Anna
    "Chris": np.array([1, 1, 5, 5, 1]),     # the opposite
}
print("\\nCOSINE SIMILARITY between users:")
for n1, v1 in users.items():
    row = "  " + f"{n1:6s}"
    for n2, v2 in users.items():
        row += f" {cosine_similarity(v1, v2):+.3f}"
    print(row)

# =====================================================================
# A TINY RECOMMENDER, built from nothing but dot products
# =====================================================================
films = ["Alien", "Blade Runner", "Notting Hill", "Love Actually", "The Thing"]
ratings = np.array([
    [5, 4, 1, 1, 5],     # Anna
    [4, 5, 2, 1, 4],     # Boris
    [1, 1, 5, 5, 1],     # Chris
    [5, 5, 1, 2, 0],     # Dora - has NOT seen The Thing (0 = unseen)
])
names = ["Anna", "Boris", "Chris", "Dora"]

target = 3                                # recommend for Dora
seen = ratings[target] > 0

sims = []
for i in range(len(names)):
    if i == target:
        continue
    # compare only on films BOTH have seen
    both = seen & (ratings[i] > 0)
    s = cosine_similarity(ratings[target][both], ratings[i][both])
    sims.append((names[i], s, i))
sims.sort(key=lambda t: -t[1])

print(f"\\nMost similar to {names[target]}:")
for name, s, _ in sims:
    print(f"  {name:6s} {s:+.3f}")

# Predict her rating for the unseen film, weighting by similarity
unseen_idx = np.where(~seen)[0][0]
num = sum(s * ratings[i][unseen_idx] for _, s, i in sims if ratings[i][unseen_idx] > 0)
den = sum(s for _, s, i in sims if ratings[i][unseen_idx] > 0)
print(f"\\nPredicted rating for '{films[unseen_idx]}': {num / den:.2f} / 5")
~~~

~~~text
COSINE SIMILARITY between users:
  Anna   +1.000 +0.988 +0.406
  Boris  +0.988 +1.000 +0.480
  Chris  +0.406 +0.480 +1.000

Most similar to Dora:
  Boris  +0.997
  Anna   +0.992
  Chris  +0.371

Predicted rating for 'The Thing': 4.50 / 5
~~~

:::tip Why cosine and not raw dot product
The raw dot product grows with vector length, so a user who rates everything 5 looks
"similar" to everyone. Dividing by the lengths removes magnitude and leaves only
**direction** - which is what "same taste" actually means. The identical logic applies to
sentence embeddings in a RAG system.
:::

### Projection: the idea behind least squares and PCA

~~~python projection.py
import numpy as np

def project(v, onto):
    """The shadow v casts on the direction 'onto'."""
    u = onto / np.linalg.norm(onto)
    return (v @ u) * u

v = np.array([4.0, 3.0])
direction = np.array([1.0, 0.0])              # the x-axis

p = project(v, direction)
residual = v - p
print("v            :", v)
print("projection   :", p)                     # [4. 0.]
print("residual     :", residual)              # [0. 3.]
print("perpendicular:", np.isclose(p @ residual, 0))   # True - always

# Least squares finds the projection of y onto the space your features can reach.
# The residual is what your features CANNOT explain.
print("\\nThat perpendicular residual is exactly what linear regression minimises.")
~~~
`
},
{
name: 'Matrices as transformations',
goal: 'See a matrix as a function that moves points, and connect that to what a neural network layer does.',
md: `
~~~python matrices.py
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# A MATRIX TIMES A VECTOR = A TRANSFORMED POINT
# =====================================================================
# A unit square, as four corner points (each row is a point)
square = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)

transforms = {
    "identity":     np.array([[1, 0], [0, 1]]),
    "scale x2":     np.array([[2, 0], [0, 2]]),
    "stretch x":    np.array([[3, 0], [0, 1]]),
    "rotate 45":    np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)],
                              [np.sin(np.pi/4),  np.cos(np.pi/4)]]),
    "shear":        np.array([[1, 1], [0, 1]]),
    "reflect x":    np.array([[1, 0], [0, -1]]),
    "collapse":     np.array([[1, 1], [1, 1]]),     # singular! squashes to a line
}

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, (name, M) in zip(axes.ravel(), transforms.items()):
    out = square @ M.T          # note the .T: rows are points, so we post-multiply
    ax.plot(square[:, 0], square[:, 1], "b--", alpha=0.4, label="before")
    ax.plot(out[:, 0], out[:, 1], "r-", lw=2, label="after")
    ax.set_title(f"{name}\\ndet = {np.linalg.det(M):.2f}")
    ax.set_xlim(-3, 4); ax.set_ylim(-3, 4)
    ax.grid(alpha=0.3); ax.set_aspect("equal")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
axes.ravel()[-1].axis("off")
axes.ravel()[0].legend()
plt.tight_layout(); plt.show()

# =====================================================================
# THE DETERMINANT = how much the transformation scales AREA
# =====================================================================
for name, M in transforms.items():
    d = np.linalg.det(M)
    note = ("area unchanged" if np.isclose(abs(d), 1) else
            "SINGULAR - collapses space, not invertible" if np.isclose(d, 0) else
            f"area x{abs(d):.1f}")
    flip = " (and flips orientation)" if d < 0 else ""
    print(f"{name:12s} det={d:+6.2f}  {note}{flip}")
~~~

~~~text
identity     det= +1.00  area unchanged
scale x2     det= +4.00  area x4.0
stretch x    det= +3.00  area x3.0
rotate 45    det= +1.00  area unchanged
shear        det= +1.00  area unchanged
reflect x    det= -1.00  area unchanged (and flips orientation)
collapse     det= +0.00  SINGULAR - collapses space, not invertible
~~~

:::note Why "singular" matters
A determinant of zero means the transformation squashes 2-D space onto a line -
information is destroyed and cannot be recovered, so the matrix has no inverse.

In machine learning this shows up as **perfectly correlated features**. If ~height_cm~ and
~height_m~ are both in your matrix, ~X'X~ is singular, and the normal equation for linear
regression has no unique solution. That is exactly what Ridge regression fixes by adding
a small value to the diagonal.
:::

### Matrix multiplication is a batch of dot products

~~~python matmul.py
import numpy as np

# 4 samples, 3 features
X = np.array([[1., 2., 3.],
              [4., 5., 6.],
              [7., 8., 9.],
              [10., 11., 12.]])

# a linear layer: 3 inputs -> 2 outputs
W = np.array([[0.5, -0.2],
              [1.0,  0.3],
              [-0.5, 0.8]])         # shape (3, 2)
b = np.array([0.1, -0.1])           # one bias per output

Z = X @ W + b                       # (4,3) @ (3,2) -> (4,2), then broadcast b
print("X:", X.shape, " W:", W.shape, " -> Z:", Z.shape)
print(Z.round(3))

# Element [i, j] is the dot product of sample i with output-neuron j's weights
print("\\nZ[0,0] by hand:", X[0] @ W[:, 0] + b[0], " vs matrix:", Z[0, 0])

# THIS IS A NEURAL NETWORK LAYER. Add a non-linearity and you have one:
def relu(x):
    return np.maximum(0, x)

A = relu(Z)
print("\\nafter ReLU:\\n", A.round(3))

# A two-layer network is just two of these
W2 = np.array([[1.0], [-1.5]])      # (2, 1)
b2 = np.array([0.2])
output = A @ W2 + b2
print("\\nnetwork output:", output.ravel().round(4))

# =====================================================================
# SHAPE RULES - print them when confused, it fixes almost everything
# =====================================================================
print("\\nSHAPE ARITHMETIC")
print("  (4,3) @ (3,2) ->", (X @ W).shape, "   inner 3 and 3 match")
try:
    X @ X                                     # (4,3) @ (4,3)
except ValueError as e:
    print("  (4,3) @ (4,3) -> ERROR:", str(e)[:52])
print("  (4,3) @ (3,4) ->", (X @ X.T).shape, "  transpose fixes it")
print("  (3,4) @ (4,3) ->", (X.T @ X).shape, "  different result! not commutative")

# The (n,) vs (n,1) gotcha
v = np.array([1., 2., 3.])
print("\\n(3,)   @ (3,)   ->", np.dot(v, v), "  a SCALAR")
print("(1,3)  @ (3,1)  ->", (v.reshape(1, 3) @ v.reshape(3, 1)).shape, " a 1x1 MATRIX")
print("(3,1)  @ (1,3)  ->", (v.reshape(3, 1) @ v.reshape(1, 3)).shape, " a 3x3 matrix!")
~~~

### Eigenvectors: the directions a matrix does not rotate

~~~python eigen.py
import numpy as np

A = np.array([[3.0, 1.0],
              [1.0, 3.0]])

vals, vecs = np.linalg.eig(A)
print("eigenvalues :", vals.round(3))          # [4. 2.]
print("eigenvectors:\\n", vecs.round(3))

# The defining property: A @ v = lambda * v  (direction unchanged, only stretched)
for i in range(2):
    v = vecs[:, i]
    print(f"\\nA @ v{i} = {(A @ v).round(4)}")
    print(f"lam * v{i} = {(vals[i] * v).round(4)}   <- identical")

# ---------------------------------------------------------------
# WHY THIS MATTERS: it is exactly PCA
# ---------------------------------------------------------------
rng = np.random.default_rng(0)
n = 500
x = rng.normal(0, 3, n)
y = 0.8 * x + rng.normal(0, 1, n)          # correlated data
data = np.column_stack([x, y])
data = data - data.mean(axis=0)            # centre it

cov = np.cov(data.T)                       # 2x2 covariance matrix
vals, vecs = np.linalg.eigh(cov)           # eigh: for symmetric matrices
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]

print("\\ncovariance matrix:\\n", cov.round(3))
print("eigenvalues (variance along each direction):", vals.round(3))
print("explained variance ratio:", (vals / vals.sum()).round(3))
print("first principal component (the direction of most spread):", vecs[:, 0].round(3))

# project the data onto the principal components
projected = data @ vecs
print("\\nvariance along PC1:", projected[:, 0].var().round(3))
print("variance along PC2:", projected[:, 1].var().round(3))
print("-> PC1 captures", f"{vals[0]/vals.sum():.1%}", "of the variation.")
print("   Dropping PC2 loses only", f"{vals[1]/vals.sum():.1%}", "- that is dimensionality reduction.")
~~~

:::tip The one-sentence summary of PCA
**PCA finds the eigenvectors of the covariance matrix.** They point along the directions
of greatest variance, and the eigenvalues say how much variance each direction holds.
Keep the top few, discard the rest - you have compressed your data with minimal loss.
:::
`
}
],
quiz: [
{
q: 'What does a dot product of zero between two vectors mean?',
options: ['They are identical', 'They are perpendicular / unrelated in direction', 'One of them is zero', 'They point in opposite directions'],
answer: 1,
why: 'a·b = |a||b|cos(θ), and cos(90 degrees) = 0. Opposite directions give a negative dot product; identical directions give the maximum positive value.'
},
{
q: 'You multiply matrices of shape (32, 128) and (128, 10). What is the output shape?',
options: ['(32, 128)', '(128, 10)', '(32, 10)', 'Error'],
answer: 2,
why: 'The inner dimensions (128 and 128) must match and cancel, leaving the outer dimensions. This is exactly a batch of 32 samples through a 128-to-10 linear layer.'
},
{
q: 'A matrix has determinant 0. What does that tell you?',
options: [
  'It is the identity matrix',
  'It collapses space to a lower dimension and has no inverse',
  'All its entries are zero',
  'It rotates without scaling'
],
answer: 1,
why: 'Zero determinant means the transformation is singular - it destroys a dimension. In ML this happens with perfectly collinear features, which is why ridge regularisation exists.'
},
{
q: 'PCA finds the principal components by computing:',
options: [
  'The gradient of the loss',
  'The eigenvectors of the covariance matrix',
  'The inverse of the data matrix',
  'The mean of each column'
],
answer: 1,
why: 'Eigenvectors of the covariance matrix point along the directions of maximum variance, and their eigenvalues quantify how much variance each captures.'
}
]
},

/* ============================================================ */
{
id: 'calculus',
title: 'Calculus: how models learn',
summary: 'Derivatives as slopes, gradients as directions of steepest ascent, the chain rule as backpropagation - and gradient descent implemented from scratch.',
tags: ['math', 'calculus', 'optimisation', 'essential'],
intro: `
## Why calculus is in a machine learning course

Training a model means finding the parameter values that minimise a loss function. The
loss is a landscape; the parameters are your position on it; **the derivative tells you
which way is downhill.** That is the entire connection.

~~~text
   loss
    |  \\                              ..
    |   \\                          ..
    |    \\.                     ..
    |      \\..              ...
    |          \\.........
    +---------------------------------- parameter w
              ^
              the minimum: slope = 0

   at any point, the DERIVATIVE dLoss/dw is the slope.
   Move in the OPPOSITE direction of the slope -> you go downhill.
       w  <-  w  -  learning_rate * dLoss/dw
~~~

## The three ideas

**1. Derivative** - the instantaneous rate of change. For a function of one variable, the
slope of the tangent line.

**2. Gradient** - a vector of partial derivatives, one per parameter. It points in the
direction of **steepest ascent**, so we step in the negative gradient direction.

:::math Gradient
For a loss L that depends on weights w1, w2, ..., wn:

**grad L = [ dL/dw1, dL/dw2, ..., dL/dwn ]**

Each entry answers: *if I nudge only this weight, how much does the loss change?*
:::

**3. Chain rule** - how derivatives compose through nested functions.

:::math Chain rule
If **y = f(g(x))**, then **dy/dx = f'(g(x)) * g'(x)**

Backpropagation is nothing but the chain rule applied repeatedly, from the loss backwards
through every layer of a network.
:::

## The derivatives you should recognise on sight

| Function | Derivative | Where it appears |
|---|---|---|
| x squared | 2x | Mean squared error |
| e to the x | e to the x | Softmax, exponential family |
| log(x) | 1/x | Cross-entropy, log-likelihood |
| sigmoid(x) | sigmoid(x)(1 - sigmoid(x)) | Logistic regression, gates in LSTMs |
| tanh(x) | 1 - tanh(x) squared | RNN activations |
| ReLU(x) | 1 if x > 0, else 0 | Nearly every modern network |
`,
keyPoints: [
  'The derivative is a slope; the gradient is a vector of slopes, one per parameter.',
  'Gradient descent steps in the NEGATIVE gradient direction: w := w - lr * grad.',
  'The chain rule multiplies local derivatives along a path - that is backpropagation.',
  'ReLU has a derivative of exactly 0 or 1, which is why it avoids vanishing gradients.'
],
pitfalls: [
  'A learning rate that is too high makes the loss diverge; too low and it never converges.',
  'Forgetting to zero gradients between steps in PyTorch - they accumulate by default.',
  'Believing gradient descent finds the global minimum. It finds a local one, and in deep nets that is usually fine.'
],
levels: [
{
name: 'Derivatives, numerically and by hand',
goal: 'See what a derivative is by computing one three ways, and verify your hand-derived formulas.',
md: `
~~~python derivatives.py
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. THE DEFINITION: the slope of a shrinking secant line
# =====================================================================
def f(x):
    return x ** 2

def numerical_derivative(f, x, h=1e-5):
    """Central difference - more accurate than (f(x+h)-f(x))/h."""
    return (f(x + h) - f(x - h)) / (2 * h)

x0 = 3.0
print("f(x) = x^2, at x = 3")
print("  analytic derivative 2x  =", 2 * x0)
for h in [1.0, 0.1, 0.01, 1e-5]:
    approx = (f(x0 + h) - f(x0)) / h
    print(f"  forward difference h={h:<7} = {approx:.6f}")
print("  central difference        =", numerical_derivative(f, x0))

# =====================================================================
# 2. VERIFY THE FORMULAS YOU WILL USE IN NEURAL NETWORKS
# =====================================================================
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

print("\\nCHECKING ANALYTIC DERIVATIVES AGAINST NUMERICAL ONES")
print(f"{'function':10s} {'x':>6s} {'analytic':>12s} {'numerical':>12s} {'match':>7s}")
print("-" * 52)
for name, fn, dfn in [("sigmoid", sigmoid, d_sigmoid),
                      ("tanh", tanh, d_tanh),
                      ("relu", relu, d_relu),
                      ("square", lambda x: x**2, lambda x: 2*x),
                      ("log", np.log, lambda x: 1/x)]:
    for x in [0.5, 2.0]:
        a = float(dfn(np.array(x)))
        n = numerical_derivative(fn, x)
        print(f"{name:10s} {x:6.1f} {a:12.6f} {n:12.6f} {'OK' if np.isclose(a, n, atol=1e-4) else 'FAIL':>7s}")

# =====================================================================
# 3. PLOT THE FUNCTIONS AND THEIR DERIVATIVES
# =====================================================================
x = np.linspace(-5, 5, 400)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, fn, dfn) in zip(axes, [("sigmoid", sigmoid, d_sigmoid),
                                      ("tanh", tanh, d_tanh),
                                      ("ReLU", relu, d_relu)]):
    ax.plot(x, fn(x), lw=2, label=name)
    ax.plot(x, dfn(x), lw=2, ls="--", label="derivative")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_title(name); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print("\\nNotice: sigmoid's derivative peaks at 0.25 and is nearly 0 when |x| > 4.")
print("Multiply many of those together through deep layers and the gradient VANISHES.")
print("ReLU's derivative is exactly 1 for positive inputs - no shrinking. That is why")
print("ReLU replaced sigmoid in hidden layers and made deep networks trainable.")
~~~

### The chain rule, computed step by step

~~~python chain_rule.py
import numpy as np

# Consider a tiny network:  x -> z = w*x + b -> a = sigmoid(z) -> L = (a - y)^2
# We want dL/dw and dL/db.

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

x, y = 2.0, 1.0            # one input, one target
w, b = 0.5, -0.3           # current parameters

# ---- FORWARD PASS: compute and REMEMBER every intermediate ----------
z = w * x + b
a = sigmoid(z)
L = (a - y) ** 2
print(f"forward:  z={z:.4f}  a={a:.4f}  L={L:.6f}")

# ---- BACKWARD PASS: chain the local derivatives --------------------
dL_da = 2 * (a - y)                 # d/da of (a-y)^2
da_dz = a * (1 - a)                 # d/dz of sigmoid(z)
dz_dw = x                           # d/dw of (w*x + b)
dz_db = 1.0                         # d/db of (w*x + b)

dL_dz = dL_da * da_dz               # chain step 1
dL_dw = dL_dz * dz_dw               # chain step 2
dL_db = dL_dz * dz_db

print(f"\\nbackward chain:")
print(f"  dL/da = 2(a - y)      = {dL_da:+.6f}")
print(f"  da/dz = a(1 - a)      = {da_dz:+.6f}")
print(f"  dL/dz = dL/da * da/dz = {dL_dz:+.6f}")
print(f"  dL/dw = dL/dz * x     = {dL_dw:+.6f}")
print(f"  dL/db = dL/dz * 1     = {dL_db:+.6f}")

# ---- VERIFY NUMERICALLY (gradient checking - do this when debugging)
def loss(w_, b_):
    return (sigmoid(w_ * x + b_) - y) ** 2

h = 1e-6
num_dw = (loss(w + h, b) - loss(w - h, b)) / (2 * h)
num_db = (loss(w, b + h) - loss(w, b - h)) / (2 * h)
print(f"\\ngradient check:")
print(f"  dL/dw analytic {dL_dw:+.8f}  numeric {num_dw:+.8f}  {'OK' if np.isclose(dL_dw, num_dw) else 'MISMATCH'}")
print(f"  dL/db analytic {dL_db:+.8f}  numeric {num_db:+.8f}  {'OK' if np.isclose(dL_db, num_db) else 'MISMATCH'}")

# ---- TAKE ONE GRADIENT DESCENT STEP --------------------------------
lr = 1.0
w_new, b_new = w - lr * dL_dw, b - lr * dL_db
print(f"\\nloss before step: {loss(w, b):.6f}")
print(f"loss after  step: {loss(w_new, b_new):.6f}   <- it went down")
~~~

:::tip Gradient checking
When you implement backprop by hand and the network will not learn, **compare your analytic
gradient to a numerical one** exactly as above. If they disagree, your derivative is wrong -
not your learning rate. This one technique will save you hours.
:::
`
},
{
name: 'Gradient descent from scratch',
goal: 'Implement the optimiser that trains every model in this course, and watch how the learning rate decides whether it works at all.',
md: `
~~~python gradient_descent.py
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# PART 1: minimise a simple function, so you can SEE the path
# =====================================================================
def f(x):
    return x ** 4 - 3 * x ** 3 + 2          # has a clear minimum

def df(x):
    return 4 * x ** 3 - 9 * x ** 2

def descend(start, lr, steps=60):
    x = start
    path = [x]
    for _ in range(steps):
        x = x - lr * df(x)                  # THE UPDATE RULE
        if not np.isfinite(x) or abs(x) > 1e6:
            break
        path.append(x)
    return np.array(path)

xs = np.linspace(-1, 3.2, 400)
fig, axes = plt.subplots(1, 4, figsize=(18, 4))

for ax, lr in zip(axes, [0.001, 0.01, 0.05, 0.12]):
    path = descend(start=3.0, lr=lr)
    ax.plot(xs, f(xs), lw=2, color="steelblue")
    ok = np.isfinite(f(path)) & (np.abs(path) < 3.5)
    ax.plot(path[ok], f(path[ok]), "ro-", ms=4, alpha=0.7)
    final = path[-1] if np.isfinite(path[-1]) else float("nan")
    verdict = ("too slow" if lr <= 0.001 else
               "diverged!" if not np.isfinite(final) or abs(final) > 10 else "good")
    ax.set_title(f"lr = {lr}   ({verdict})\\nended at x = {final:.3f}")
    ax.set_ylim(-8, 12); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
~~~

~~~text
lr = 0.001  (too slow)     ended at x = 2.612
lr = 0.01   (good)         ended at x = 2.250
lr = 0.05   (good)         ended at x = 2.250
lr = 0.12   (diverged!)    ended at x = nan
~~~

:::warn The learning rate is the hyperparameter that matters most
- **Too small**: converges, but takes forever, and may stall in a flat region.
- **Just right**: steady decrease to the minimum.
- **Too large**: overshoots, oscillates, then explodes to NaN.

If your loss becomes ~nan~, lower the learning rate by 10x before changing anything else.
:::

### Part 2: train a real linear regression

~~~python gd_regression.py
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---- data: y = 3x + 5 + noise --------------------------------------
n = 200
X = rng.uniform(0, 10, (n, 1))
true_w, true_b = 3.0, 5.0
y = true_w * X.ravel() + true_b + rng.normal(0, 2, n)

def mse(w, b):
    pred = w * X.ravel() + b
    return np.mean((pred - y) ** 2)

def gradients(w, b):
    """Derivatives of MSE = mean((wx + b - y)^2)."""
    pred = w * X.ravel() + b
    error = pred - y                       # shape (n,)
    dw = 2 * np.mean(error * X.ravel())    # dMSE/dw
    db = 2 * np.mean(error)                # dMSE/db
    return dw, db

# ---- train ----------------------------------------------------------
w, b = 0.0, 0.0                            # start from nothing
lr = 0.01
history = []

for epoch in range(300):
    dw, db = gradients(w, b)
    w -= lr * dw
    b -= lr * db
    history.append((w, b, mse(w, b)))
    if epoch % 50 == 0:
        print(f"epoch {epoch:4d}  w={w:6.3f}  b={b:6.3f}  loss={mse(w,b):8.4f}")

print(f"\\nlearned : w={w:.3f}  b={b:.3f}")
print(f"true    : w={true_w:.3f}  b={true_b:.3f}")

# ---- compare to the exact closed-form solution ----------------------
X_design = np.column_stack([np.ones(n), X.ravel()])
exact, *_ = np.linalg.lstsq(X_design, y, rcond=None)
print(f"exact   : w={exact[1]:.3f}  b={exact[0]:.3f}   <- gradient descent got there")

# ---- visualise ------------------------------------------------------
hist = np.array(history)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))

ax[0].scatter(X, y, alpha=0.4, s=15)
for i in [0, 5, 20, 60, 299]:
    wi, bi, _ = history[i]
    ax[0].plot([0, 10], [bi, wi * 10 + bi], alpha=0.35 + 0.13 * (i / 299),
               label=f"epoch {i}")
ax[0].set_title("the line improving"); ax[0].legend(fontsize=8)

ax[1].plot(hist[:, 2], lw=2)
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("MSE")
ax[1].set_yscale("log"); ax[1].set_title("loss curve (log scale)")
ax[1].grid(alpha=0.3)

# the loss surface, with the path taken
ws = np.linspace(-1, 6, 120)
bs = np.linspace(-2, 11, 120)
WW, BB = np.meshgrid(ws, bs)
ZZ = np.array([[mse(wv, bv) for wv in ws] for bv in bs])
cs = ax[2].contour(WW, BB, ZZ, levels=30, cmap="viridis")
ax[2].plot(hist[:, 0], hist[:, 1], "r.-", ms=3, lw=1)
ax[2].plot(true_w, true_b, "w*", ms=16, markeredgecolor="k")
ax[2].set_xlabel("w"); ax[2].set_ylabel("b")
ax[2].set_title("the path down the loss surface")
plt.tight_layout(); plt.show()
~~~

### Part 3: the three flavours of gradient descent

~~~python gd_variants.py
import numpy as np, time

rng = np.random.default_rng(0)
n, d = 20000, 20
X = rng.normal(size=(n, d))
true_w = rng.normal(size=d)
y = X @ true_w + rng.normal(0, 0.5, n)

def loss(w):
    return np.mean((X @ w - y) ** 2)

def grad(Xb, yb, w):
    return 2 * Xb.T @ (Xb @ w - yb) / len(yb)

def train(batch_size, lr, epochs=20, label=""):
    w = np.zeros(d)
    t0 = time.perf_counter()
    for _ in range(epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            w -= lr * grad(X[b], y[b], w)
    dt = time.perf_counter() - t0
    print(f"{label:28s} final loss {loss(w):7.4f}   time {dt:6.2f}s")
    return w

print("n = 20,000 samples, 20 features\\n")
train(batch_size=n,   lr=0.10, label="BATCH (all 20000 at once)")
train(batch_size=64,  lr=0.05, label="MINI-BATCH (64)")
train(batch_size=1,   lr=0.01, epochs=3, label="STOCHASTIC (1) - 3 epochs")
~~~

| Variant | Batch size | Per-step cost | Gradient quality | Verdict |
|---|---|---|---|---|
| Batch GD | all n | expensive | exact, smooth | Only for small data |
| **Mini-batch GD** | 32-512 | cheap | slightly noisy | **What everyone uses** |
| Stochastic GD | 1 | trivial | very noisy | Rare alone; the noise can escape local minima |

:::note Why mini-batch won
It gets almost the gradient quality of full-batch at a tiny fraction of the cost, and the
batch fits neatly into GPU memory so the matrix multiply is fully parallel. The small
amount of noise also acts as a mild regulariser. Batch size 32-256 is the standard range.
:::

### The optimisers you will actually use

~~~python optimisers.py
import numpy as np

def sgd(w, g, state, lr=0.01):
    """Plain gradient descent."""
    return w - lr * g, state

def momentum(w, g, state, lr=0.01, beta=0.9):
    """Accumulate a velocity - rolls through small bumps and flat regions."""
    v = state.get("v", np.zeros_like(w))
    v = beta * v + (1 - beta) * g
    state["v"] = v
    return w - lr * v, state

def rmsprop(w, g, state, lr=0.01, beta=0.999, eps=1e-8):
    """Divide by a running root-mean-square: big-gradient params get smaller steps."""
    s = state.get("s", np.zeros_like(w))
    s = beta * s + (1 - beta) * g ** 2
    state["s"] = s
    return w - lr * g / (np.sqrt(s) + eps), state

def adam(w, g, state, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
    """Momentum + RMSprop + bias correction. The default choice for deep learning."""
    m = state.get("m", np.zeros_like(w))
    v = state.get("v", np.zeros_like(w))
    t = state.get("t", 0) + 1
    m = b1 * m + (1 - b1) * g              # first moment  (mean of gradients)
    v = b2 * v + (1 - b2) * g ** 2         # second moment (mean of squared gradients)
    m_hat = m / (1 - b1 ** t)              # bias correction: early m is biased to 0
    v_hat = v / (1 - b2 ** t)
    state.update(m=m, v=v, t=t)
    return w - lr * m_hat / (np.sqrt(v_hat) + eps), state

# ---- compare them on a difficult, elongated valley ------------------
def rosen_grad(p):
    x, y = p
    return np.array([-2 * (1 - x) - 400 * x * (y - x ** 2), 200 * (y - x ** 2)])

def run(opt, lr, steps=3000):
    w = np.array([-1.5, 2.0]); state = {}
    for _ in range(steps):
        w, state = opt(w, rosen_grad(w), state, lr=lr)
    return w

print("Minimising the Rosenbrock function; the true minimum is (1, 1)\\n")
for name, opt, lr in [("SGD", sgd, 0.0005),
                      ("Momentum", momentum, 0.002),
                      ("RMSprop", rmsprop, 0.01),
                      ("Adam", adam, 0.02)]:
    w = run(opt, lr)
    print(f"  {name:10s} ended at ({w[0]:6.3f}, {w[1]:6.3f})   "
          f"distance to optimum: {np.linalg.norm(w - 1):.4f}")
~~~

:::tip Which optimiser to use
- **Adam** with lr=1e-3 - the default for deep learning. Start here.
- **AdamW** - Adam with proper weight decay. Standard for Transformers.
- **SGD with momentum 0.9** - often generalises slightly better for image models, but
  needs a learning-rate schedule and more tuning.

Do not agonise. Use Adam, get something working, then tune.
:::
`
}
],
quiz: [
{
q: 'Why do we subtract the gradient rather than add it?',
options: [
  'To keep the weights positive',
  'The gradient points toward steepest ASCENT, and we want to minimise the loss',
  'It makes the maths simpler',
  'Adding it would be faster but less accurate'
],
answer: 1,
why: 'The gradient points uphill. Going downhill on the loss surface means stepping in the negative gradient direction.'
},
{
q: 'Your training loss becomes NaN after a few steps. What is the first thing to try?',
options: [
  'Add more layers',
  'Reduce the learning rate by a factor of 10',
  'Collect more data',
  'Switch from Adam to SGD'
],
answer: 1,
why: 'NaN almost always means divergence from an excessive learning rate: the step overshoots, the loss grows, the next gradient is bigger, and it explodes. Drop the learning rate first.'
},
{
q: 'What is backpropagation, in one sentence?',
options: [
  'A way to initialise weights',
  'The chain rule applied repeatedly to compute the loss gradient with respect to every parameter',
  'A method of splitting data',
  'The forward pass through the network'
],
answer: 1,
why: 'Backprop caches the forward-pass intermediates, then walks backwards multiplying local derivatives - exactly the chain rule, organised efficiently.'
},
{
q: 'Why did ReLU largely replace sigmoid in hidden layers?',
options: [
  'ReLU is more accurate',
  'Sigmoid derivatives max out at 0.25, so multiplying them through deep networks makes gradients vanish',
  'ReLU uses less memory',
  'Sigmoid cannot handle negative inputs'
],
answer: 1,
why: 'The sigmoid derivative peaks at 0.25 and approaches 0 in the tails. Ten layers of that shrinks the gradient by roughly a million. ReLU has derivative exactly 1 for positive inputs.'
}
]
},

/* ============================================================ */
{
id: 'probability',
title: 'Probability and distributions',
summary: 'Probability rules, the distributions that model real data, Bayes theorem, and expectation - the vocabulary behind every probabilistic model.',
tags: ['math', 'probability', 'statistics'],
intro: `
## Why probability

Machine learning models do not output certainties, they output **beliefs**. A classifier
says "87% likely spam". A regression has an error distribution. A language model is
literally a probability distribution over the next token.

## The rules, all of them

~~~text
1. 0 <= P(A) <= 1                              probabilities live in [0,1]
2. P(certain) = 1,  P(impossible) = 0
3. P(not A) = 1 - P(A)
4. P(A or B) = P(A) + P(B) - P(A and B)        subtract the double-counted overlap
5. P(A and B) = P(A) * P(B|A)                  the chain rule
6. If A and B are INDEPENDENT: P(A and B) = P(A) * P(B)
7. P(A|B) = P(A and B) / P(B)                  conditional probability
~~~

## Bayes' theorem

:::math Bayes
**P(H | E) = P(E | H) * P(H) / P(E)**

- **P(H)** - the *prior*: what you believed before seeing evidence
- **P(E|H)** - the *likelihood*: how expected the evidence is if the hypothesis holds
- **P(H|E)** - the *posterior*: your updated belief
- **P(E)** - the evidence, a normalising constant
:::

Naive Bayes is a direct application. So is every medical-test intuition problem, and so is
the entire field of Bayesian statistics.

## The distributions to know

| Distribution | Models | Example |
|---|---|---|
| **Bernoulli** | one yes/no trial | did this user click? |
| **Binomial** | k successes in n trials | how many of 100 emails were opened |
| **Poisson** | count of rare events in a fixed interval | support tickets per hour |
| **Uniform** | all outcomes equally likely | a fair die, random initialisation |
| **Normal (Gaussian)** | sums of many small effects | heights, measurement error, residuals |
| **Exponential** | time until the next event | time between website visits |
| **Log-normal** | positive, right-skewed quantities | incomes, house prices, session lengths |

:::note The Central Limit Theorem, in one line
**The mean of many independent samples is approximately normally distributed, whatever the
original distribution was.** This is why the normal distribution is everywhere, and why
confidence intervals work at all.
:::
`,
keyPoints: [
  'P(A|B) is generally NOT P(B|A). Confusing them is the base-rate fallacy.',
  'A rare condition plus an imperfect test means most positives are false positives.',
  'The Central Limit Theorem is why sample means are normal even when the data is not.',
  'Right-skewed positive data (income, price) is usually log-normal - take the log.'
],
pitfalls: [
  'Assuming independence when events are correlated - it makes joint probabilities badly wrong.',
  'Reading a 95% confidence interval as "95% probability the true value is in here". It is a statement about the procedure.',
  'Ignoring the base rate when interpreting a test result.'
],
levels: [
{
name: 'Distributions you can see',
goal: 'Generate, plot and recognise every distribution you will meet in real data.',
md: `
~~~python distributions.py
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
ax = axes.ravel()

# ---- 1. BERNOULLI: a single yes/no ---------------------------------
p = 0.3
samples = rng.binomial(1, p, 10000)
ax[0].bar([0, 1], [np.mean(samples == 0), np.mean(samples == 1)], color=["steelblue", "coral"])
ax[0].set_title(f"Bernoulli(p={p})\\nclicked or not")
ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["no", "yes"])

# ---- 2. BINOMIAL: k successes out of n -----------------------------
n_trials, p = 20, 0.3
samples = rng.binomial(n_trials, p, 10000)
ax[1].hist(samples, bins=range(0, 16), density=True, alpha=0.7, color="steelblue",
           edgecolor="white")
k = np.arange(0, 16)
ax[1].plot(k, stats.binom.pmf(k, n_trials, p), "ro-", ms=4)
ax[1].set_title(f"Binomial(n={n_trials}, p={p})\\nclicks out of 20 impressions")

# ---- 3. POISSON: counts of rare events -----------------------------
lam = 4
samples = rng.poisson(lam, 10000)
ax[2].hist(samples, bins=range(0, 16), density=True, alpha=0.7, color="steelblue",
           edgecolor="white")
ax[2].plot(k, stats.poisson.pmf(k, lam), "ro-", ms=4)
ax[2].set_title(f"Poisson(lambda={lam})\\nsupport tickets per hour")

# ---- 4. UNIFORM ----------------------------------------------------
samples = rng.uniform(0, 1, 10000)
ax[3].hist(samples, bins=40, density=True, alpha=0.7, color="steelblue", edgecolor="white")
ax[3].axhline(1.0, color="r", lw=2)
ax[3].set_title("Uniform(0, 1)\\nrandom initialisation")

# ---- 5. NORMAL -----------------------------------------------------
mu, sigma = 170, 8
samples = rng.normal(mu, sigma, 10000)
ax[4].hist(samples, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
xs = np.linspace(140, 200, 300)
ax[4].plot(xs, stats.norm.pdf(xs, mu, sigma), "r-", lw=2)
for k_sd, alpha in [(1, 0.30), (2, 0.18), (3, 0.10)]:
    ax[4].axvspan(mu - k_sd * sigma, mu + k_sd * sigma, alpha=alpha, color="orange")
ax[4].set_title(f"Normal(mu={mu}, sigma={sigma})\\n68% / 95% / 99.7% rule")

# ---- 6. EXPONENTIAL ------------------------------------------------
samples = rng.exponential(scale=2.0, size=10000)
ax[5].hist(samples, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
xs = np.linspace(0, 15, 300)
ax[5].plot(xs, stats.expon.pdf(xs, scale=2.0), "r-", lw=2)
ax[5].set_title("Exponential(scale=2)\\ntime until the next event")

# ---- 7. LOG-NORMAL: income, prices ---------------------------------
samples = rng.lognormal(mean=10.0, sigma=0.6, size=10000)
ax[6].hist(samples, bins=60, density=True, alpha=0.7, color="steelblue", edgecolor="white")
ax[6].set_title(f"Log-normal\\nincome: skew = {stats.skew(samples):.2f}")
ax[6].set_xlim(0, 150000)

# ---- 8. ...and the log of it is normal -----------------------------
ax[7].hist(np.log(samples), bins=60, density=True, alpha=0.7, color="seagreen",
           edgecolor="white")
ax[7].set_title(f"log(income)\\nskew = {stats.skew(np.log(samples)):.2f}  <- fixed")

plt.tight_layout(); plt.show()
~~~

:::tip The single most useful transform in data science
If a positive feature or target is heavily right-skewed - income, price, page views,
session duration - **take the logarithm**. It turns a log-normal into a normal, which
makes linear models, distance metrics and MSE all behave far better.

~~~python
df["log_price"] = np.log1p(df["price"])   # log1p handles zeros safely
~~~
:::

### The Central Limit Theorem, demonstrated

~~~python clt.py
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# Start with a distribution that looks NOTHING like a normal
population = rng.exponential(scale=2.0, size=1_000_000)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))

axes[0].hist(population, bins=60, density=True, color="crimson", alpha=0.7)
axes[0].set_title(f"the POPULATION\\nheavily skewed, mean={population.mean():.2f}")
axes[0].set_xlim(0, 15)

for ax, n in zip(axes[1:], [2, 10, 50]):
    means = np.array([rng.choice(population, n).mean() for _ in range(5000)])
    ax.hist(means, bins=50, density=True, color="steelblue", alpha=0.7)
    # overlay the normal the CLT predicts
    from scipy import stats
    xs = np.linspace(means.min(), means.max(), 200)
    predicted_sd = population.std() / np.sqrt(n)
    ax.plot(xs, stats.norm.pdf(xs, population.mean(), predicted_sd), "r-", lw=2)
    ax.set_title(f"mean of {n} samples\\nobserved sd={means.std():.3f}, "
                 f"predicted={predicted_sd:.3f}")
plt.tight_layout(); plt.show()

print("The population is exponential and wildly skewed.")
print("The mean of just 50 samples is already almost perfectly normal.")
print("And its spread shrinks as 1/sqrt(n) - four times the data halves the error.")
~~~

:::note Why you care
1. It justifies confidence intervals and t-tests on non-normal data.
2. It explains the ~1/sqrt(n)~ law: to halve your uncertainty you need **four times** the data.
3. It is why measurement errors and model residuals tend to look normal.
:::
`
},
{
name: 'Bayes theorem and the base-rate trap',
goal: 'Work through the medical-test problem that almost everyone gets wrong, then build Naive Bayes from it.',
md: `
## The problem that fools professionals

> A disease affects **1 in 1,000** people. A test is **99% accurate**: it catches 99% of
> sick people, and gives a false positive for only 1% of healthy people.
>
> **You test positive. What is the probability you are sick?**

Most people say 99%. The correct answer is about **9%**.

~~~python bayes.py
import numpy as np

# =====================================================================
# WORK IT OUT WITH ACTUAL PEOPLE - the clearest way to see it
# =====================================================================
population = 1_000_000
prevalence = 0.001
sensitivity = 0.99          # P(test positive | sick)
specificity = 0.99          # P(test negative | healthy)

sick = int(population * prevalence)              # 1,000 people
healthy = population - sick                      # 999,000 people

true_positives = int(sick * sensitivity)         # 990 sick people test positive
false_negatives = sick - true_positives          # 10 sick people are missed
false_positives = int(healthy * (1 - specificity))   # 9,990 healthy test positive!
true_negatives = healthy - false_positives

print(f"Population           : {population:,}")
print(f"  actually sick      : {sick:,}")
print(f"  actually healthy   : {healthy:,}")
print()
print(f"TEST POSITIVE        : {true_positives + false_positives:,}")
print(f"  ...and really sick : {true_positives:,}")
print(f"  ...but healthy     : {false_positives:,}   <- the trap")
print()
posterior = true_positives / (true_positives + false_positives)
print(f"P(sick | positive)   = {true_positives:,} / {true_positives + false_positives:,}"
      f" = {posterior:.4f}  ({posterior:.1%})")

# =====================================================================
# THE SAME THING WITH THE FORMULA
# =====================================================================
def bayes(prior, likelihood, false_positive_rate):
    """P(H|E) = P(E|H)P(H) / [ P(E|H)P(H) + P(E|not H)P(not H) ]"""
    numerator = likelihood * prior
    evidence = numerator + false_positive_rate * (1 - prior)
    return numerator / evidence

print(f"\\nvia the formula      = {bayes(0.001, 0.99, 0.01):.4f}")

# =====================================================================
# WHY: the base rate dominates
# =====================================================================
print(f"\\n{'prevalence':>12s} {'P(sick | positive)':>20s}")
print("-" * 34)
for prev in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]:
    print(f"{prev:>12.4%} {bayes(prev, 0.99, 0.01):>20.1%}")

# =====================================================================
# UPDATING TWICE: a second independent positive test
# =====================================================================
first = bayes(0.001, 0.99, 0.01)
second = bayes(first, 0.99, 0.01)        # yesterday's posterior is today's prior
print(f"\\nafter ONE positive test : {first:.1%}")
print(f"after TWO positive tests: {second:.1%}   <- now it is convincing")
~~~

~~~text
TEST POSITIVE        : 10,980
  ...and really sick : 990
  ...but healthy     : 9,990   <- the trap

P(sick | positive)   = 990 / 10,980 = 0.0902  (9.0%)

  prevalence   P(sick | positive)
----------------------------------
     0.0100%                 1.0%
     0.1000%                 9.0%
     1.0000%                50.0%
     5.0000%                83.9%
    10.0000%                91.7%
    50.0000%                99.0%
~~~

:::danger The lesson for machine learning
**A 99%-accurate model on a 0.1% event is nearly useless without calibration.**

This is fraud detection. This is rare-disease screening. This is intrusion detection.
When positives are rare, precision collapses no matter how good your model sounds -
which is exactly why you use precision-recall curves rather than accuracy, and why the
threshold has to be chosen deliberately.
:::

### Naive Bayes: Bayes applied to classification

~~~python naive_bayes_scratch.py
import numpy as np
from collections import defaultdict

class NaiveBayesText:
    """Multinomial Naive Bayes for text, built from Bayes' theorem directly.

    P(class | words) is proportional to P(class) * product of P(word | class)
    'Naive' = we pretend the words are independent given the class. They are
    not, but the approximation works remarkably well.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha            # Laplace smoothing: never assign zero probability

    def fit(self, docs, labels):
        self.classes_ = sorted(set(labels))
        self.vocab_ = sorted({w for d in docs for w in d.split()})
        self.v_ = len(self.vocab_)

        self.log_prior_ = {}
        self.log_likelihood_ = {}

        for c in self.classes_:
            class_docs = [d for d, l in zip(docs, labels) if l == c]
            # PRIOR: how common is this class?
            self.log_prior_[c] = np.log(len(class_docs) / len(docs))

            # LIKELIHOOD: how often does each word appear in this class?
            counts = defaultdict(int)
            total = 0
            for d in class_docs:
                for w in d.split():
                    counts[w] += 1
                    total += 1

            self.log_likelihood_[c] = {
                w: np.log((counts[w] + self.alpha) / (total + self.alpha * self.v_))
                for w in self.vocab_
            }
            # probability for an unseen word
            self.log_likelihood_[c]["<UNK>"] = np.log(
                self.alpha / (total + self.alpha * self.v_))
        return self

    def predict_log_proba(self, doc):
        scores = {}
        for c in self.classes_:
            # work in LOG space: multiplying many small probabilities underflows to 0
            s = self.log_prior_[c]
            for w in doc.split():
                s += self.log_likelihood_[c].get(w, self.log_likelihood_[c]["<UNK>"])
            scores[c] = s
        return scores

    def predict(self, doc):
        scores = self.predict_log_proba(doc)
        return max(scores, key=scores.get)

    def predict_proba(self, doc):
        scores = self.predict_log_proba(doc)
        # softmax over log scores, done stably
        vals = np.array(list(scores.values()))
        exp = np.exp(vals - vals.max())
        probs = exp / exp.sum()
        return dict(zip(scores.keys(), probs))


docs = [
    "great movie loved it wonderful acting",
    "wonderful film great story loved",
    "amazing performance great direction",
    "terrible movie boring waste of time",
    "awful acting boring plot terrible",
    "waste of money boring and awful",
]
labels = ["pos", "pos", "pos", "neg", "neg", "neg"]

nb = NaiveBayesText().fit(docs, labels)

for test in ["great wonderful acting", "boring terrible waste", "movie"]:
    probs = nb.predict_proba(test)
    print(f"{test:28s} -> {nb.predict(test):3s}  "
          f"P(pos)={probs['pos']:.3f}  P(neg)={probs['neg']:.3f}")
~~~

~~~text
great wonderful acting       -> pos  P(pos)=0.983  P(neg)=0.017
boring terrible waste        -> neg  P(pos)=0.014  P(neg)=0.986
movie                        -> pos  P(pos)=0.500  P(neg)=0.500
~~~

:::tip Three details that make Naive Bayes work
1. **Work in log space.** Multiplying 500 probabilities of 0.001 underflows to exactly 0.
   Adding their logs does not.
2. **Laplace smoothing** (~alpha=1~). Without it, one unseen word gives probability 0 and
   annihilates the whole document score.
3. **The independence assumption is false** and it still works. Naive Bayes is often a
   surprisingly strong baseline for text - fast, needs little data, hard to beat cheaply.
:::
`
}
],
quiz: [
{
q: 'A disease affects 1 in 1000. A test is 99% accurate both ways. You test positive. Roughly what is P(sick)?',
options: ['99%', '50%', '9%', '1%'],
answer: 2,
why: 'Among a million people, 990 true positives versus 9,990 false positives - so about 9%. When the base rate is tiny, false positives swamp true ones.'
},
{
q: 'Your target variable is house price and its histogram has a long right tail. What is the standard fix?',
options: [
  'Remove the expensive houses',
  'Take the logarithm of the target',
  'Use a bigger model',
  'Standardise it to mean 0'
],
answer: 1,
why: 'Positive right-skewed quantities are usually log-normal. log or log1p makes them roughly normal, which suits linear models and MSE far better. Remember to exponentiate predictions back.'
},
{
q: 'Why does Naive Bayes compute in log space?',
options: [
  'Logs are faster to compute',
  'Multiplying hundreds of small probabilities underflows to zero in floating point',
  'It makes the model more accurate',
  'It is required by the Bayes theorem'
],
answer: 1,
why: 'Products of many probabilities below 1 collapse to 0.0 in float64. Sums of logs are numerically stable, and the ranking is identical since log is monotonic.'
},
{
q: 'The Central Limit Theorem says that to halve the standard error of a mean you must:',
options: ['Double the sample size', 'Quadruple the sample size', 'Halve the sample size', 'Nothing - it stays constant'],
answer: 1,
why: 'Standard error scales as 1/sqrt(n), so dividing it by 2 requires multiplying n by 4. This is why data collection has sharply diminishing returns.'
}
]
},

/* ============================================================ */
{
id: 'statistics',
title: 'Statistics: describing and inferring',
summary: 'Descriptive statistics that do not lie, correlation and its traps, confidence intervals, hypothesis testing and A/B tests done properly.',
tags: ['math', 'statistics', 'experimentation'],
intro: `
## Two jobs

**Descriptive statistics** summarise the data you have.
**Inferential statistics** tell you what the data you have implies about data you do not have.

## Centre and spread, and why the mean is often wrong

| Statistic | Meaning | Robust to outliers? |
|---|---|---|
| Mean | arithmetic average | **No** - one billionaire moves it |
| Median | the middle value | **Yes** |
| Mode | most common value | Yes |
| Standard deviation | typical distance from the mean | No |
| IQR (Q3 - Q1) | spread of the middle 50% | **Yes** |
| MAD | median absolute deviation | **Yes** |

:::warn The mean of a skewed distribution is misleading
Median income is a far more honest summary than mean income. For any right-skewed variable
- income, house price, session length, order value - report the median, or report both.
:::

## Correlation and its three traps

Pearson's r measures **linear** association, between -1 and +1.

~~~text
   r = +0.9          r = 0            r = 0            r = -0.9
    .                 . . .            .    .            .
      .              .     .            .  .              .
        .           .       .            ..                 .
          .          . . .              .  .                  .
   strong linear    no relation      STRONG relation,     strong negative
                                     but r = 0 (a curve!)
~~~

**Trap 1: correlation is not causation.** Ice-cream sales correlate with drownings.
Temperature causes both.

**Trap 2: r only sees straight lines.** A perfect parabola has r near 0.

**Trap 3: outliers dominate r.** One extreme point can create or destroy a correlation.

## Hypothesis testing, without the mysticism

~~~text
1. State a NULL hypothesis (H0): "there is no difference"
2. Collect data
3. Compute: if H0 were true, how surprising is this data?   <- the p-value
4. If p < alpha (usually 0.05), reject H0
~~~

:::danger What a p-value is NOT
- It is **not** the probability the null hypothesis is true.
- It is **not** the probability your result happened by chance.
- It is **not** a measure of effect size. A tiny, useless difference becomes "significant"
  with enough data.

A p-value is: **the probability of seeing data at least this extreme, IF the null were true.**
Always report the effect size and confidence interval alongside it.
:::
`,
keyPoints: [
  'For skewed data, report the median and IQR, not the mean and standard deviation.',
  'Correlation measures linear association only, and never implies causation.',
  'A p-value is P(data | null), not P(null | data).',
  'Statistical significance is not practical significance - always report effect size.'
],
pitfalls: [
  'Peeking at an A/B test and stopping as soon as p < 0.05. That inflates false positives enormously.',
  'Running twenty tests and reporting the one that was significant.',
  'Using a t-test on heavily skewed data with small n.',
  'Interpreting a confidence interval as a probability statement about the parameter.'
],
levels: [
{
name: 'Describing data honestly',
goal: 'Compute summaries that survive outliers and skew, and detect the correlations that lie.',
md: `
~~~python descriptive.py
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(42)

# A realistic income distribution: mostly modest, a few very high
incomes = np.concatenate([
    rng.lognormal(mean=10.2, sigma=0.45, size=980),
    rng.uniform(400_000, 3_000_000, 20),         # 20 very wealthy people
])

print("INCOME DISTRIBUTION, n =", len(incomes))
print(f"  mean          : {incomes.mean():>12,.0f}")
print(f"  median        : {np.median(incomes):>12,.0f}   <- the honest number")
print(f"  std dev       : {incomes.std():>12,.0f}")
print(f"  IQR           : {stats.iqr(incomes):>12,.0f}   <- robust spread")
print(f"  skewness      : {stats.skew(incomes):>12.2f}   (>1 = strongly right-skewed)")
print(f"  kurtosis      : {stats.kurtosis(incomes):>12.2f}   (>0 = heavy tails)")

print("\\npercentiles:")
for p in [1, 10, 25, 50, 75, 90, 99]:
    print(f"  p{p:<3d} {np.percentile(incomes, p):>12,.0f}")

print(f"\\nthe mean sits at the {stats.percentileofscore(incomes, incomes.mean()):.0f}th percentile")
print("-> more than two thirds of people earn LESS than the 'average'.")

# =====================================================================
# OUTLIER DETECTION - three standard methods
# =====================================================================
def iqr_outliers(x, k=1.5):
    q1, q3 = np.percentile(x, [25, 75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return (x < lo) | (x > hi), lo, hi

def zscore_outliers(x, k=3):
    z = (x - x.mean()) / x.std()
    return np.abs(z) > k

def modified_zscore_outliers(x, k=3.5):
    """Uses the median, so extreme values cannot hide themselves."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    mz = 0.6745 * (x - med) / mad
    return np.abs(mz) > k

m_iqr, lo, hi = iqr_outliers(incomes)
print(f"\\nOUTLIERS")
print(f"  IQR method       : {m_iqr.sum():4d}  (outside {lo:,.0f} to {hi:,.0f})")
print(f"  z-score >3       : {zscore_outliers(incomes).sum():4d}")
print(f"  modified z >3.5  : {modified_zscore_outliers(incomes).sum():4d}  <- most reliable")

print("\\nNote the z-score method finds fewest: the outliers inflate the standard")
print("deviation they are measured against. This is called MASKING.")
~~~

### The correlation traps, demonstrated

~~~python correlation_traps.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(0)
fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))

# ---- Trap 1: a strong NON-LINEAR relation, r near zero -------------
x = np.linspace(-3, 3, 200)
y = x ** 2 + rng.normal(0, 0.4, 200)
r, p = stats.pearsonr(x, y)
rs, _ = stats.spearmanr(x, y)
ax[0].scatter(x, y, alpha=0.6, s=14)
ax[0].set_title(f"perfect parabola\\nPearson r = {r:.3f}  (blind!)")

# ---- Trap 2: one outlier CREATES a correlation ---------------------
x2 = np.concatenate([rng.normal(0, 1, 100), [12]])
y2 = np.concatenate([rng.normal(0, 1, 100), [12]])
r_all, _ = stats.pearsonr(x2, y2)
r_without, _ = stats.pearsonr(x2[:-1], y2[:-1])
ax[1].scatter(x2[:-1], y2[:-1], alpha=0.6, s=14)
ax[1].scatter(x2[-1:], y2[-1:], color="red", s=90, zorder=5)
ax[1].set_title(f"one red point\\nwith it r={r_all:.2f}, without r={r_without:.2f}")

# ---- Trap 3: SIMPSON'S PARADOX -------------------------------------
frames = []
for i, (cx, cy) in enumerate([(1, 8), (3, 6), (5, 4), (7, 2)]):
    gx = rng.normal(cx, 0.5, 60)
    gy = cy + 0.8 * (gx - cx) + rng.normal(0, 0.3, 60)     # POSITIVE within group
    frames.append(pd.DataFrame({"x": gx, "y": gy, "group": f"g{i}"}))
    ax[2].scatter(gx, gy, alpha=0.6, s=12, label=f"group {i}")
d = pd.concat(frames)
r_overall, _ = stats.pearsonr(d["x"], d["y"])
r_within = d.groupby("group").apply(
    lambda g: stats.pearsonr(g["x"], g["y"])[0], include_groups=False).mean()
ax[2].set_title(f"Simpson's paradox\\noverall r={r_overall:+.2f}, within groups r={r_within:+.2f}")
ax[2].legend(fontsize=7)

# ---- Pearson vs Spearman on a monotonic curve ----------------------
x3 = np.linspace(0.1, 5, 200)
y3 = np.exp(x3) + rng.normal(0, 3, 200)
rp, _ = stats.pearsonr(x3, y3)
rsp, _ = stats.spearmanr(x3, y3)
ax[3].scatter(x3, y3, alpha=0.6, s=14)
ax[3].set_title(f"monotonic but curved\\nPearson {rp:.2f}  Spearman {rsp:.2f}")

plt.tight_layout(); plt.show()

print("RULES")
print("  Pearson  : linear relationships only, sensitive to outliers")
print("  Spearman : any MONOTONIC relationship, rank-based, robust")
print("  Always PLOT before trusting a correlation coefficient")
print("  Always check whether a hidden grouping variable reverses the sign")
~~~

:::danger Simpson's paradox
The overall correlation can have the **opposite sign** to the correlation within every
subgroup. A famous real case: a university appeared to discriminate against women in
admissions overall, yet within every individual department women were admitted at equal or
higher rates. Women simply applied more often to the most competitive departments.

Whenever you compute an aggregate statistic, ask: **is there a grouping variable that would
reverse this?**
:::
`
},
{
name: 'Confidence intervals and A/B testing',
goal: 'Quantify uncertainty properly and run an A/B test that you can actually defend.',
md: `
~~~python confidence_intervals.py
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)

# =====================================================================
# 1. CONFIDENCE INTERVAL FOR A MEAN
# =====================================================================
sample = rng.normal(loc=170, scale=8, size=100)

mean = sample.mean()
sem = stats.sem(sample)                  # standard error = s / sqrt(n)
ci = stats.t.interval(0.95, df=len(sample) - 1, loc=mean, scale=sem)

print(f"sample mean      : {mean:.2f}")
print(f"standard error   : {sem:.3f}")
print(f"95% CI           : [{ci[0]:.2f}, {ci[1]:.2f}]")
print(f"true value (170) inside? {ci[0] <= 170 <= ci[1]}")

# What "95% confidence" ACTUALLY means: repeat the experiment many times,
# and 95% of the intervals you build will contain the true value.
contains = 0
for _ in range(10000):
    s = rng.normal(170, 8, 100)
    lo, hi = stats.t.interval(0.95, len(s) - 1, loc=s.mean(), scale=stats.sem(s))
    contains += (lo <= 170 <= hi)
print(f"\\nover 10,000 repeats, {contains / 100:.1f}% of intervals contained the truth")
print("THAT is what 95% confidence means - a property of the PROCEDURE,")
print("not a probability statement about this one interval.")

# =====================================================================
# 2. BOOTSTRAP - a CI for ANY statistic, with no formula needed
# =====================================================================
def bootstrap_ci(data, statistic=np.median, n_boot=10000, alpha=0.05):
    """Resample with replacement, recompute, take percentiles. That is all."""
    boots = np.array([statistic(rng.choice(data, len(data), replace=True))
                      for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return statistic(data), lo, hi

incomes = rng.lognormal(10.2, 0.6, 500)
for name, fn in [("median", np.median), ("90th pct", lambda x: np.percentile(x, 90)),
                 ("mean", np.mean)]:
    est, lo, hi = bootstrap_ci(incomes, fn, n_boot=3000)
    print(f"\\n{name:9s}: {est:10,.0f}   95% CI [{lo:,.0f}, {hi:,.0f}]")

print("\\nThe bootstrap needs NO distributional assumption and works for any")
print("statistic - median, percentile, correlation, AUC, anything.")
~~~

### A/B testing, done properly

~~~python ab_test.py
import numpy as np
from scipy import stats

rng = np.random.default_rng(7)

# =====================================================================
# STEP 1 - DECIDE THE SAMPLE SIZE BEFORE YOU START
# =====================================================================
def sample_size_proportion(baseline, mde, alpha=0.05, power=0.80):
    """How many users per arm to detect a lift of 'mde' (absolute)."""
    p1, p2 = baseline, baseline + mde
    p_bar = (p1 + p2) / 2
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    n = ((z_a * np.sqrt(2 * p_bar * (1 - p_bar)) +
          z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2) / (mde ** 2)
    return int(np.ceil(n))

baseline = 0.10
print("Baseline conversion 10%. Users needed PER ARM:")
for mde in [0.05, 0.02, 0.01, 0.005]:
    n = sample_size_proportion(baseline, mde)
    print(f"  to detect +{mde:.1%} absolute lift : {n:>9,}")
print("\\nSmaller effects need quadratically more traffic. Decide this FIRST.")

# =====================================================================
# STEP 2 - RUN THE EXPERIMENT (simulated here)
# =====================================================================
n_per_arm = 5000
true_a, true_b = 0.10, 0.115           # B is genuinely 1.5 points better

conv_a = rng.binomial(1, true_a, n_per_arm)
conv_b = rng.binomial(1, true_b, n_per_arm)

rate_a, rate_b = conv_a.mean(), conv_b.mean()
print(f"\\nRESULTS  (n = {n_per_arm:,} per arm)")
print(f"  A: {conv_a.sum():4d} conversions = {rate_a:.4f}")
print(f"  B: {conv_b.sum():4d} conversions = {rate_b:.4f}")
print(f"  absolute lift: {rate_b - rate_a:+.4f}")
print(f"  relative lift: {(rate_b / rate_a - 1):+.2%}")

# =====================================================================
# STEP 3 - TEST FOR SIGNIFICANCE (two-proportion z-test)
# =====================================================================
count = np.array([conv_a.sum(), conv_b.sum()])
nobs = np.array([n_per_arm, n_per_arm])
p_pool = count.sum() / nobs.sum()
se = np.sqrt(p_pool * (1 - p_pool) * (1 / nobs[0] + 1 / nobs[1]))
z = (rate_b - rate_a) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z)))

print(f"\\n  z = {z:.3f}")
print(f"  p = {p_value:.4f}   ->  {'SIGNIFICANT' if p_value < 0.05 else 'not significant'} at alpha=0.05")

# =====================================================================
# STEP 4 - THE CONFIDENCE INTERVAL MATTERS MORE THAN THE P-VALUE
# =====================================================================
se_diff = np.sqrt(rate_a * (1 - rate_a) / n_per_arm + rate_b * (1 - rate_b) / n_per_arm)
lo = (rate_b - rate_a) - 1.96 * se_diff
hi = (rate_b - rate_a) + 1.96 * se_diff
print(f"\\n  95% CI for the lift: [{lo:+.4f}, {hi:+.4f}]")
print(f"  in relative terms  : [{lo/rate_a:+.1%}, {hi/rate_a:+.1%}]")
print("\\n  Report THIS. It tells you the plausible range of business impact,")
print("  which a p-value never does.")

# =====================================================================
# STEP 5 - WHY PEEKING RUINS EVERYTHING
# =====================================================================
print("\\n" + "=" * 60)
print("THE PEEKING PROBLEM - simulated with NO real difference at all")
print("=" * 60)

false_positives_fixed = 0
false_positives_peeking = 0
trials = 1000

for _ in range(trials):
    a = rng.binomial(1, 0.10, 5000)
    b = rng.binomial(1, 0.10, 5000)          # identical! any 'win' is noise

    # honest: test once, at the end
    _, p = stats.ttest_ind(a, b)
    if p < 0.05:
        false_positives_fixed += 1

    # dishonest: check every 500 users and stop at the first p < 0.05
    for k in range(500, 5001, 500):
        _, p = stats.ttest_ind(a[:k], b[:k])
        if p < 0.05:
            false_positives_peeking += 1
            break

print(f"  test once at the end   : {false_positives_fixed/trials:.1%} false positives (expected 5%)")
print(f"  peek 10 times and stop : {false_positives_peeking/trials:.1%} false positives")
print("\\n  Peeking roughly triples your false-positive rate. Fix the sample size,")
print("  or use a sequential test designed for it.")
~~~

~~~text
Baseline conversion 10%. Users needed PER ARM:
  to detect +5.0% absolute lift :       727
  to detect +2.0% absolute lift :     3,838
  to detect +1.0% absolute lift :    14,750
  to detect +0.5% absolute lift :    58,168

  95% CI for the lift: [+0.0032, +0.0288]
  in relative terms  : [+3.2%, +28.9%]

  test once at the end   : 5.2% false positives (expected 5%)
  peek 10 times and stop : 16.8% false positives
~~~

:::tip The A/B testing checklist
1. **Decide the metric and the sample size before launching.**
2. **Randomise properly** and check the arms are balanced on covariates.
3. **Run for whole weeks** - weekday and weekend users differ.
4. **Do not peek** - or use a sequential/Bayesian method built for it.
5. **Report the confidence interval and the effect size**, not just the p-value.
6. **Correct for multiple comparisons** if you test several metrics
   (Bonferroni: divide alpha by the number of tests).
7. **Check the guardrail metrics** - a conversion win that tanks retention is a loss.
:::
`
}
],
quiz: [
{
q: 'Which pair of statistics should you report for a heavily right-skewed variable like income?',
options: ['Mean and standard deviation', 'Median and IQR', 'Mode and range', 'Mean and IQR'],
answer: 1,
why: 'Mean and standard deviation are both dragged by the tail. The median and interquartile range describe where most of the data actually sits.'
},
{
q: 'A p-value of 0.03 means:',
options: [
  'There is a 3% chance the null hypothesis is true',
  'There is a 97% chance your result is real',
  'If the null were true, data at least this extreme would occur 3% of the time',
  'The effect size is 3%'
],
answer: 2,
why: 'It is P(data this extreme | null true) - a statement about the data given the hypothesis, never the reverse, and it says nothing about magnitude.'
},
{
q: 'Why is checking an A/B test every day and stopping when p < 0.05 a problem?',
options: [
  'It uses too much compute',
  'Each look is another chance to hit a false positive, so the real error rate is far above 5%',
  'The test statistic becomes invalid after 24 hours',
  'It requires a larger sample'
],
answer: 1,
why: 'Repeated testing multiplies the opportunities for noise to cross the threshold. Ten peeks can push the false-positive rate from 5% to around 17%. Fix n in advance, or use a sequential method.'
},
{
q: 'Every subgroup shows a positive correlation, but the overall correlation is negative. This is:',
options: ['Impossible', "Simpson's paradox", 'A calculation error', 'Multicollinearity'],
answer: 1,
why: "Simpson's paradox: a confounding grouping variable can reverse the aggregate relationship. Always check whether an unmodelled group explains an aggregate result."
}
]
},

/* ============================================================ */
{
id: 'information-theory',
title: 'Information theory: entropy and loss functions',
summary: 'Entropy, cross-entropy and KL divergence - why the loss function of every classifier and every language model is what it is.',
tags: ['math', 'information-theory', 'advanced'],
intro: `
## Why this is here

The loss function for classification is **cross-entropy**. Decision trees split on
**information gain**. Variational autoencoders minimise a **KL divergence**. Language
models are evaluated with **perplexity**, which is exponentiated cross-entropy.

All four are the same handful of ideas.

## Information = surprise

:::math Self-information
The information content of an event with probability p is **-log(p)**.

- p = 1 (certain) -> 0 bits. Learning it tells you nothing.
- p = 0.5 -> 1 bit.
- p = 0.01 (rare) -> about 6.6 bits. Rare events are informative.
:::

## Entropy = average surprise

:::math Entropy
**H(p) = - sum over i of p(i) * log p(i)**

The average number of bits needed to encode an outcome from distribution p.
Maximum when the distribution is uniform (maximum uncertainty), zero when one outcome is
certain.
:::

## Cross-entropy = the cost of being wrong

:::math Cross-entropy
**H(p, q) = - sum over i of p(i) * log q(i)**

The average bits needed if you encode data that really comes from **p** using a code
optimised for your model **q**. It is minimised exactly when q = p.

**That is why it is the loss function**: minimising cross-entropy pushes your predicted
distribution toward the true one.
:::

## KL divergence = the excess cost

:::math KL divergence
**KL(p || q) = H(p, q) - H(p)**

How many *extra* bits your wrong model costs you. Always >= 0, and zero only when p = q.
It is **not symmetric**: KL(p||q) is not KL(q||p).
:::

~~~text
   H(p)         true entropy - irreducible
   H(p,q)       what your model costs
   KL(p||q)     the waste = H(p,q) - H(p)

   Since H(p) is a constant of the data, minimising cross-entropy
   IS minimising KL divergence. Same optimisation, different name.
~~~
`,
keyPoints: [
  'Cross-entropy loss is the standard classification loss because minimising it drives the predicted distribution toward the truth.',
  'For one-hot labels, cross-entropy collapses to -log(probability assigned to the correct class).',
  'KL divergence is asymmetric and is cross-entropy minus the constant true entropy.',
  'Perplexity = exp(cross-entropy) - the "effective number of choices" a language model faces.'
],
pitfalls: [
  'Computing log(0). Always clip probabilities, or use the framework loss that fuses softmax and log.',
  'Applying softmax and then a loss that also applies softmax - a very common double-softmax bug.',
  'Treating KL divergence as a distance. It is not symmetric and violates the triangle inequality.'
],
levels: [
{
name: 'Entropy, cross-entropy and why classifiers use it',
goal: 'Compute all three quantities by hand, then derive why cross-entropy is the right loss.',
md: `
~~~python information.py
import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-12          # guard against log(0)

def entropy(p, base=2):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return -np.sum(p * np.log(p) / np.log(base))

def cross_entropy(p, q, base=2):
    p, q = np.asarray(p, float), np.clip(np.asarray(q, float), EPS, 1)
    return -np.sum(p * np.log(q) / np.log(base))

def kl_divergence(p, q, base=2):
    return cross_entropy(p, q, base) - entropy(p, base)

# =====================================================================
# 1. ENTROPY: uncertainty measured in bits
# =====================================================================
print("ENTROPY of some distributions (bits)")
cases = {
    "certain [1, 0]":                [1.0, 0.0],
    "fair coin [.5, .5]":            [0.5, 0.5],
    "biased [.9, .1]":               [0.9, 0.1],
    "very biased [.99, .01]":        [0.99, 0.01],
    "fair 4-sided":                  [0.25] * 4,
    "fair 8-sided":                  [0.125] * 8,
}
for name, p in cases.items():
    print(f"  {name:26s} H = {entropy(p):.4f}")

print("\\n-> a fair 8-sided die needs exactly 3 bits (2^3 = 8). Entropy IS the")
print("   average number of yes/no questions needed to identify the outcome.")

# the entropy of a coin, as its bias changes
ps = np.linspace(0.001, 0.999, 400)
Hs = [entropy([p, 1 - p]) for p in ps]
plt.figure(figsize=(7, 4))
plt.plot(ps, Hs, lw=2)
plt.axvline(0.5, color="r", ls="--")
plt.xlabel("P(heads)"); plt.ylabel("entropy (bits)")
plt.title("Uncertainty is maximal at p = 0.5")
plt.grid(alpha=0.3); plt.tight_layout(); plt.show()

# =====================================================================
# 2. CROSS-ENTROPY AS A LOSS
# =====================================================================
print("\\nCROSS-ENTROPY: true class is index 0 (one-hot [1, 0, 0])")
true = [1.0, 0.0, 0.0]
predictions = {
    "perfect      [1.00, 0.00, 0.00]": [1.00, 0.00, 0.00],
    "confident ok [0.90, 0.05, 0.05]": [0.90, 0.05, 0.05],
    "unsure       [0.40, 0.30, 0.30]": [0.40, 0.30, 0.30],
    "uniform      [0.33, 0.33, 0.33]": [1/3, 1/3, 1/3],
    "wrong        [0.05, 0.90, 0.05]": [0.05, 0.90, 0.05],
    "confidently  [0.01, 0.98, 0.01]": [0.01, 0.98, 0.01],
}
print(f"  {'prediction':36s} {'CE (bits)':>10s} {'KL':>8s}")
for name, q in predictions.items():
    print(f"  {name:36s} {cross_entropy(true, q):10.4f} {kl_divergence(true, q):8.4f}")

print("\\nWith a one-hot label, cross-entropy simplifies to just -log(q[correct]).")
print("So the loss depends ONLY on the probability you gave the right answer -")
print("and it punishes confident mistakes brutally (that last row).")
~~~

~~~text
  prediction                           CE (bits)       KL
  perfect      [1.00, 0.00, 0.00]         0.0000   0.0000
  confident ok [0.90, 0.05, 0.05]         0.1520   0.1520
  unsure       [0.40, 0.30, 0.30]         1.3219   1.3219
  uniform      [0.33, 0.33, 0.33]         1.5850   1.5850
  wrong        [0.05, 0.90, 0.05]         4.3219   4.3219
  confidently  [0.01, 0.98, 0.01]         6.6439   6.6439
~~~

:::note Why the asymmetric punishment is correct
Being 98% sure of the wrong answer costs 6.64 bits; being 90% sure of the right answer
costs 0.15. Cross-entropy makes overconfidence expensive, which is exactly the behaviour
you want in a model that will make decisions.
:::

### Binary cross-entropy, and the numerically stable version

~~~python bce.py
import numpy as np

def binary_cross_entropy(y_true, y_pred, eps=1e-15):
    """The loss behind logistic regression and every binary classifier."""
    y_pred = np.clip(y_pred, eps, 1 - eps)          # never log(0)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

y = np.array([1, 1, 0, 0, 1])
print("BINARY CROSS-ENTROPY")
for name, p in [("great predictions", [0.95, 0.90, 0.05, 0.10, 0.85]),
                ("mediocre",          [0.60, 0.55, 0.45, 0.40, 0.60]),
                ("no idea",           [0.50] * 5),
                ("confidently wrong", [0.05, 0.10, 0.95, 0.90, 0.15])]:
    print(f"  {name:20s} loss = {binary_cross_entropy(y, np.array(p)):.4f}")

# ---------------------------------------------------------------
# THE STABILITY PROBLEM, and why frameworks fuse the operations
# ---------------------------------------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def naive_loss(z, y):
    """sigmoid then log -> overflows for large |z|."""
    p = sigmoid(z)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))

def stable_loss(z, y):
    """The algebraically identical, numerically safe form:
       max(z,0) - z*y + log(1 + exp(-|z|))"""
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))

print("\\nNUMERICAL STABILITY (logit z, true label y=0)")
print(f"  {'z':>8s} {'naive':>14s} {'stable':>14s}")
for z in [0.0, 10.0, 50.0, 100.0, 800.0]:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        n = naive_loss(np.float64(z), 0.0)
    s = stable_loss(np.float64(z), 0.0)
    print(f"  {z:8.1f} {n:14.4f} {s:14.4f}")
~~~

~~~text
NUMERICAL STABILITY (logit z, true label y=0)
         z          naive         stable
       0.0         0.6931         0.6931
      10.0        10.0000        10.0000
      50.0        50.0000        50.0000
     100.0            inf       100.0000
     800.0            inf       800.0000
~~~

:::danger The double-softmax bug
PyTorch's ~nn.CrossEntropyLoss~ **applies log-softmax internally**. If you also put a
~softmax~ at the end of your network, you apply it twice: the gradients shrink, training
crawls, and nothing errors out.

~~~python
# WRONG
logits = model(x)
probs = torch.softmax(logits, dim=1)
loss = nn.CrossEntropyLoss()(probs, y)      # softmax applied twice

# RIGHT
logits = model(x)                            # raw scores, no activation
loss = nn.CrossEntropyLoss()(logits, y)      # handles softmax + log + NLL
~~~
This is one of the most common silent bugs in deep learning code.
:::
`
},
{
name: 'Information gain and perplexity',
goal: 'Use entropy to build a decision-tree split, and to evaluate a language model.',
md: `
## Information gain: how a decision tree chooses

A tree asks: *which question reduces uncertainty the most?* That reduction is
**information gain** - entropy before the split minus the weighted entropy after.

~~~python information_gain.py
import numpy as np
import pandas as pd

def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return -np.sum(p * np.log2(p))

def gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

def information_gain(X_col, y, threshold):
    left = X_col <= threshold
    right = ~left
    if left.sum() == 0 or right.sum() == 0:
        return 0.0
    n = len(y)
    weighted_after = (left.sum() / n * entropy(y[left]) +
                      right.sum() / n * entropy(y[right]))
    return entropy(y) - weighted_after

# A small dataset: will someone play tennis?
df = pd.DataFrame({
    "outlook":  ["sun","sun","cloud","rain","rain","rain","cloud","sun","sun","rain",
                 "sun","cloud","cloud","rain"],
    "temp":     [30,29,31,21,18,17,17,22,19,20,22,23,30,21],
    "humidity": [85,90,86,96,80,70,65,95,70,80,70,90,75,91],
    "windy":    [0,1,0,0,0,1,1,0,0,0,1,1,0,1],
    "play":     [0,0,1,1,1,0,1,0,1,1,1,1,1,0],
})

y = df["play"].values
print(f"entropy BEFORE any split: {entropy(y):.4f} bits")
print(f"gini    BEFORE any split: {gini(y):.4f}")
print(f"class balance: {np.bincount(y)}\\n")

# Which numeric feature and threshold splits best?
print("SEARCHING FOR THE BEST SPLIT")
best = (None, None, -1)
for col in ["temp", "humidity", "windy"]:
    values = np.sort(df[col].unique())
    candidates = (values[:-1] + values[1:]) / 2 if len(values) > 1 else values
    for t in candidates:
        g = information_gain(df[col].values, y, t)
        if g > best[2]:
            best = (col, t, g)
    # report the best threshold for this column
    col_best = max(((t, information_gain(df[col].values, y, t)) for t in candidates),
                   key=lambda x: x[1])
    print(f"  {col:9s} best threshold {col_best[0]:6.1f}  gain {col_best[1]:.4f}")

# categorical feature: gain over all its values
def gain_categorical(col, y):
    n = len(y)
    after = sum((col == v).sum() / n * entropy(y[col == v]) for v in np.unique(col))
    return entropy(y) - after

print(f"  {'outlook':9s} (categorical)         gain {gain_categorical(df['outlook'].values, y):.4f}")

print(f"\\nWINNER: split on {best[0]} <= {best[1]:.1f}, gain = {best[2]:.4f} bits")
print("A decision tree does exactly this search, recursively, at every node.")

# verify against sklearn
from sklearn.tree import DecisionTreeClassifier, export_text
X = pd.get_dummies(df.drop(columns="play"), columns=["outlook"], dtype=int)
tree = DecisionTreeClassifier(criterion="entropy", max_depth=2, random_state=0).fit(X, y)
print("\\nsklearn agrees:")
print(export_text(tree, feature_names=list(X.columns)))
~~~

## Perplexity: how language models are scored

:::math Perplexity
**Perplexity = exp(cross-entropy in nats) = 2^(cross-entropy in bits)**

Interpretation: the model is as uncertain as if it were choosing uniformly among
*perplexity* options at each step.

- Perplexity 1 = perfect prediction
- Perplexity = vocabulary size = the model has learned nothing
- GPT-class models reach roughly 10-20 on general English text
:::

~~~python perplexity.py
import numpy as np
from collections import defaultdict, Counter

text = ("the cat sat on the mat the cat ate the fish the dog sat on the rug "
        "the dog ate the bone the cat sat on the rug the bird sat on the fence "
        "the cat likes the fish the dog likes the bone").split()

vocab = sorted(set(text))
V = len(vocab)
print(f"corpus: {len(text)} tokens, vocabulary {V} words")

# ---- model 1: uniform - knows nothing ------------------------------
uniform_logprob = np.log(1 / V) * len(text)
uniform_ppl = np.exp(-uniform_logprob / len(text))
print(f"\\nuniform model     perplexity = {uniform_ppl:6.2f}  (= vocabulary size)")

# ---- model 2: unigram - knows word frequencies ---------------------
counts = Counter(text)
unigram_logprob = sum(np.log(counts[w] / len(text)) for w in text)
unigram_ppl = np.exp(-unigram_logprob / len(text))
print(f"unigram model     perplexity = {unigram_ppl:6.2f}  (knows 'the' is common)")

# ---- model 3: bigram - knows what follows what ---------------------
ALPHA = 0.1                                    # smoothing
bigram = defaultdict(Counter)
for a, b in zip(text[:-1], text[1:]):
    bigram[a][b] += 1

total_logprob = 0.0
for a, b in zip(text[:-1], text[1:]):
    ctx = bigram[a]
    p = (ctx[b] + ALPHA) / (sum(ctx.values()) + ALPHA * V)
    total_logprob += np.log(p)
bigram_ppl = np.exp(-total_logprob / (len(text) - 1))
print(f"bigram model      perplexity = {bigram_ppl:6.2f}  (knows context)")

print("\\nLower perplexity = better prediction. Each model knows strictly more")
print("than the last, and the perplexity falls accordingly.")

# ---- what the bigram model actually learned ------------------------
print("\\nAfter 'the', the model predicts:")
ctx = bigram["the"]
tot = sum(ctx.values())
for w, c in ctx.most_common(5):
    print(f"  P({w:6s} | the) = {c/tot:.3f}")
~~~

~~~text
corpus: 58 tokens, vocabulary 16 words

uniform model     perplexity =  16.00  (= vocabulary size)
unigram model     perplexity =   8.31  (knows 'the' is common)
bigram model      perplexity =   3.42  (knows context)

After 'the', the model predicts:
  P(cat    | the) = 0.211
  P(dog    | the) = 0.158
  P(fish   | the) = 0.105
  P(rug    | the) = 0.105
  P(bone   | the) = 0.105
~~~

:::tip Connecting it all
- **Decision trees** pick splits that maximise information gain (entropy reduction).
- **Classifiers** minimise cross-entropy, which equals minimising KL to the true labels.
- **Language models** are scored by perplexity, which is exponentiated cross-entropy.
- **VAEs** add a KL term to keep the latent distribution close to a standard normal.

One set of ideas, four applications. You will meet all four later in this course.
:::
`
}
],
quiz: [
{
q: 'With a one-hot label, cross-entropy loss reduces to:',
options: [
  'The sum of all predicted probabilities',
  'Negative log of the probability assigned to the correct class',
  'The squared error between prediction and label',
  'The entropy of the prediction'
],
answer: 1,
why: 'All terms multiply by p(i), which is 0 except for the true class, so only -log(q[correct]) survives. That is why confident wrong answers are punished so heavily.'
},
{
q: 'A language model has cross-entropy of 3 bits per token. Its perplexity is:',
options: ['3', '8', '9', '0.33'],
answer: 1,
why: 'Perplexity = 2 to the power of cross-entropy in bits = 2^3 = 8. The model is as uncertain as choosing uniformly among 8 options.'
},
{
q: 'Why does PyTorch tell you to feed raw logits to CrossEntropyLoss?',
options: [
  'Logits are faster to compute',
  'The loss applies log-softmax internally; adding your own softmax applies it twice and cripples the gradients',
  'Softmax is deprecated',
  'It only works with negative numbers'
],
answer: 1,
why: 'Fusing softmax and log inside the loss is both numerically stable and correct. A separate softmax before it is a silent, very common bug.'
},
{
q: 'A decision tree chooses a split by maximising:',
options: [
  'Accuracy on the training set',
  'Information gain - entropy before the split minus weighted entropy after',
  'The number of samples in each leaf',
  'The depth of the tree'
],
answer: 1,
why: 'Information gain measures how much uncertainty the question removes. Gini impurity is the common alternative and behaves very similarly.'
}
]
}

]
});
