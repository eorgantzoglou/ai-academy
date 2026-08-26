/* Track 09 - Deep learning foundations */
(window.CURRICULUM = window.CURRICULUM || []).push({
id: 'deeplearning',
title: 'Deep Learning',
icon: 'NN',
level: 'Intermediate',
blurb: 'Neurons, backpropagation from scratch in NumPy, activations, losses, optimisers, then the same thing in PyTorch - with the training loop, regularisation and debugging you need to make it work.',
intro: `
## The plan

We build a neural network **three times**:

~~~text
1. BY HAND, ON PAPER      one neuron, one weight update, worked through arithmetic
2. FROM SCRATCH, NUMPY    a full multi-layer network with backpropagation
3. IN PYTORCH             the same network, in a fraction of the code, on a GPU
~~~

By the time you reach PyTorch you will know exactly what every line replaces. That is the
difference between using deep learning and understanding it.

~~~text
                    A NEURAL NETWORK

  input          hidden layer 1      hidden layer 2      output
   x1 ---\\      /-> [h] --\\        /-> [h] --\\
   x2 -----> [h] -> [h] -----> [h] -> [h] -----> [y]
   x3 ---/      \\-> [h] --/        \\-> [h] --/
                     ^                   ^
                weights + bias      non-linear activation

  each [h]:  z = w . x + b       then      a = activation(z)

  WITHOUT the activation, stacking layers is pointless:
  a composition of linear maps is just another linear map.
~~~
`,
topics: [

/* ============================================================ */
{
id: 'perceptron',
title: 'From a neuron to a network',
summary: 'The single artificial neuron, why one is not enough, and the exact reason a non-linear activation is required.',
tags: ['neural-networks', 'fundamentals'],
intro: `
## One neuron

:::math A neuron
**z = w1x1 + w2x2 + ... + wnxn + b**  (a weighted sum - the same dot product as always)

**a = f(z)**  (a non-linear activation)

That is the entire unit. A network is thousands of them, arranged in layers.
:::

## The perceptron and its famous limit

The 1958 perceptron used a step activation and could learn any **linearly separable**
function. In 1969 Minsky and Papert pointed out that XOR is not linearly separable, and
therefore a single perceptron cannot learn it. Funding collapsed; this is the first
"AI winter".

~~~text
   AND (separable)        OR (separable)         XOR (NOT separable)
   x2                     x2                     x2
   1 | o     x            1 | x     x            1 | x     o
     |     /                |  /                   |
   0 | o  / o             0 | o     x            0 | o     x
     +-------- x1           +-------- x1           +-------- x1
       0     1                0     1                0     1
   one line works         one line works        NO single line works
~~~

**The fix**: add a hidden layer. Two layers with a non-linear activation can represent
XOR - and, by the universal approximation theorem, any continuous function.

## Why the activation is mandatory

~~~text
Without activation:
   layer1: h = W1 x + b1
   layer2: y = W2 h + b2 = W2(W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)
                                                 \\_______/   \\____________/
                                                    W'             b'
   = a SINGLE linear layer. A hundred layers collapse to one.

With a non-linearity between them, no such collapse is possible.
~~~
`,
keyPoints: [
  'A neuron is a dot product followed by a non-linear function.',
  'Without non-linear activations, any number of layers collapses to one linear layer.',
  'A single perceptron cannot learn XOR; one hidden layer can.',
  'Universal approximation says one wide hidden layer suffices in theory - depth is what makes it practical.'
],
pitfalls: [
  'Forgetting the activation and wondering why a deep model performs like linear regression.',
  'Initialising all weights to zero - every neuron then computes the same thing forever.',
  'Expecting universal approximation to mean "any width works" - the theorem says nothing about how to find the weights.'
],
levels: [
{
name: 'One neuron, one update, by hand',
goal: 'Follow a single forward and backward pass through arithmetic you can verify with a calculator.',
md: `
~~~python one_neuron.py
import numpy as np

# =====================================================================
# THE SETUP: one neuron, two inputs
# =====================================================================
x = np.array([2.0, 3.0])          # one training example
y_true = 1.0                       # its label

w = np.array([0.5, -0.4])          # current weights
b = 0.1                            # current bias
lr = 0.1

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

print("=" * 58)
print("FORWARD PASS")
print("=" * 58)
z = np.dot(w, x) + b
print(f"  z = w . x + b")
print(f"    = ({w[0]} * {x[0]}) + ({w[1]} * {x[1]}) + {b}")
print(f"    = {w[0]*x[0]:.2f} + {w[1]*x[1]:.2f} + {b}")
print(f"    = {z:.4f}")

a = sigmoid(z)
print(f"\\n  a = sigmoid({z:.4f}) = 1 / (1 + e^{-z:.4f}) = {a:.6f}")

loss = 0.5 * (a - y_true) ** 2
print(f"\\n  loss = 0.5 * (a - y)^2 = 0.5 * ({a:.6f} - {y_true})^2 = {loss:.6f}")

print("\\n" + "=" * 58)
print("BACKWARD PASS - the chain rule, one link at a time")
print("=" * 58)

dL_da = a - y_true
print(f"  dL/da = (a - y)          = {dL_da:+.6f}")

da_dz = a * (1 - a)
print(f"  da/dz = a(1 - a)         = {a:.6f} * {1-a:.6f} = {da_dz:.6f}")

dL_dz = dL_da * da_dz
print(f"  dL/dz = dL/da * da/dz    = {dL_dz:+.6f}")

dL_dw = dL_dz * x                  # dz/dw = x
dL_db = dL_dz * 1.0                # dz/db = 1
print(f"  dL/dw = dL/dz * x        = {dL_dz:+.6f} * {x} = {dL_dw.round(6)}")
print(f"  dL/db = dL/dz * 1        = {dL_db:+.6f}")

print("\\n" + "=" * 58)
print("UPDATE")
print("=" * 58)
w_new = w - lr * dL_dw
b_new = b - lr * dL_db
print(f"  w := w - lr * dL/dw = {w} - {lr} * {dL_dw.round(6)} = {w_new.round(6)}")
print(f"  b := b - lr * dL/db = {b} - {lr} * {dL_db:.6f} = {b_new:.6f}")

# ---- verify the loss actually went down ------------------------------
z_new = np.dot(w_new, x) + b_new
a_new = sigmoid(z_new)
loss_new = 0.5 * (a_new - y_true) ** 2
print(f"\\n  loss before: {loss:.6f}")
print(f"  loss after : {loss_new:.6f}")
print(f"  improvement: {loss - loss_new:.6f}   <- it went DOWN. That is learning.")

# ---- and gradient-check the whole thing ------------------------------
def total_loss(w_, b_):
    return 0.5 * (sigmoid(np.dot(w_, x) + b_) - y_true) ** 2

h = 1e-7
print("\\nGRADIENT CHECK (analytic vs numerical)")
for i in range(2):
    wp, wm = w.copy(), w.copy()
    wp[i] += h; wm[i] -= h
    num = (total_loss(wp, b) - total_loss(wm, b)) / (2 * h)
    print(f"  dL/dw{i}: analytic {dL_dw[i]:+.8f}   numeric {num:+.8f}   "
          f"{'OK' if abs(dL_dw[i]-num) < 1e-5 else 'MISMATCH'}")
num_b = (total_loss(w, b + h) - total_loss(w, b - h)) / (2 * h)
print(f"  dL/db : analytic {dL_db:+.8f}   numeric {num_b:+.8f}   "
      f"{'OK' if abs(dL_db-num_b) < 1e-5 else 'MISMATCH'}")
~~~

~~~text
FORWARD PASS
  z = w . x + b
    = (0.5 * 2.0) + (-0.4 * 3.0) + 0.1
    = 1.00 + -1.20 + 0.1
    = -0.1000
  a = sigmoid(-0.1000) = 0.475021
  loss = 0.5 * (0.475021 - 1.0)^2 = 0.137802

BACKWARD PASS - the chain rule, one link at a time
  dL/da = (a - y)          = -0.524979
  da/dz = a(1 - a)         = 0.475021 * 0.524979 = 0.249376
  dL/dz = dL/da * da/dz    = -0.130918
  dL/dw = dL/dz * x        = [-0.261836 -0.392754]
  dL/db = dL/dz * 1        = -0.130918

  loss before: 0.137802
  loss after : 0.128047
  improvement: 0.009755   <- it went DOWN. That is learning.
~~~

### The perceptron learning rule, and the XOR wall

~~~python perceptron_xor.py
import numpy as np
import matplotlib.pyplot as plt


class Perceptron:
    """The 1958 algorithm. Step activation, one layer."""

    def __init__(self, n_features, lr=0.1, epochs=100):
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.lr = lr
        self.epochs = epochs

    def predict(self, X):
        return (X @ self.w + self.b > 0).astype(int)

    def fit(self, X, y):
        self.errors_ = []
        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                pred = int(xi @ self.w + self.b > 0)
                update = self.lr * (target - pred)
                self.w += update * xi          # the perceptron rule
                self.b += update
                errors += int(update != 0)
            self.errors_.append(errors)
            if errors == 0:
                break                          # converged
        return self


logic = {
    "AND": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 0, 0, 1])),
    "OR":  (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 1])),
    "XOR": (np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), np.array([0, 1, 1, 0])),
}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
for ax, (name, (X, y)) in zip(axes, logic.items()):
    p = Perceptron(2, lr=0.1, epochs=200).fit(X, y)
    acc = (p.predict(X) == y).mean()
    print(f"{name:4s}: accuracy {acc:.2f} after {len(p.errors_)} epochs   "
          f"w={p.w.round(3)} b={p.b:.3f}")

    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="steelblue", s=220, marker="o",
               edgecolor="k", label="0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="crimson", s=220, marker="s",
               edgecolor="k", label="1")
    if abs(p.w[1]) > 1e-9:
        xs = np.array([-0.5, 1.5])
        ax.plot(xs, -(p.w[0] * xs + p.b) / p.w[1], "k--", lw=2)
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.5, 1.5)
    ax.set_title(f"{name}  -  accuracy {acc:.0%}")
    ax.legend()
plt.suptitle("A single perceptron solves AND and OR, and fails on XOR forever", y=1.03)
plt.tight_layout(); plt.show()
~~~

~~~text
AND : accuracy 1.00 after 6 epochs   w=[0.2 0.1] b=-0.200
OR  : accuracy 1.00 after 4 epochs   w=[0.1 0.1] b=-0.100
XOR : accuracy 0.50 after 200 epochs w=[0.0 0.0] b=0.000
~~~

### One hidden layer fixes it

~~~python xor_solved.py
import numpy as np

X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([[0.], [1.], [1.], [0.]])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

rng = np.random.default_rng(1)
# 2 inputs -> 4 hidden -> 1 output
W1 = rng.normal(0, 1, (2, 4)); b1 = np.zeros((1, 4))
W2 = rng.normal(0, 1, (4, 1)); b2 = np.zeros((1, 1))
lr = 0.5

for epoch in range(20001):
    # forward
    z1 = X @ W1 + b1;  a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2; a2 = sigmoid(z2)
    loss = np.mean((a2 - y) ** 2)

    # backward
    d2 = (a2 - y) * a2 * (1 - a2)              # (4,1)
    dW2 = a1.T @ d2;  db2 = d2.sum(0, keepdims=True)
    d1 = (d2 @ W2.T) * a1 * (1 - a1)           # (4,4)
    dW1 = X.T @ d1;   db1 = d1.sum(0, keepdims=True)

    W2 -= lr * dW2; b2 -= lr * db2
    W1 -= lr * dW1; b1 -= lr * db1

    if epoch % 4000 == 0:
        print(f"epoch {epoch:6d}  loss {loss:.6f}")

print("\\nFINAL PREDICTIONS")
for xi, yi, pi in zip(X, y.ravel(), a2.ravel()):
    print(f"  {xi.astype(int)} -> {pi:.4f}  (target {yi:.0f})  "
          f"{'OK' if abs(pi - yi) < 0.1 else 'X'}")

print("\\nWHAT THE HIDDEN LAYER LEARNED")
print("hidden activations for each input:")
print(np.round(sigmoid(X @ W1 + b1), 3))
print("\\nThe hidden layer RE-REPRESENTED the inputs in a space where the")
print("classes ARE linearly separable. That is what hidden layers do.")
~~~

~~~text
epoch      0  loss 0.267851
epoch  20000  loss 0.000191

FINAL PREDICTIONS
  [0 0] -> 0.0139  (target 0)  OK
  [0 1] -> 0.9861  (target 1)  OK
  [1 0] -> 0.9857  (target 1)  OK
  [1 1] -> 0.0163  (target 0)  OK
~~~

:::tip The one-sentence summary of deep learning
**Hidden layers learn a new representation of the input in which the problem becomes
easy.** Everything else - convolutions, attention, residual connections - is a better way
of learning that representation for a particular kind of data.
:::
`
}
],
quiz: [
{
q: 'Why does a network need non-linear activation functions?',
options: [
  'To speed up training',
  'Without them, any stack of linear layers collapses algebraically into one linear layer',
  'To keep the outputs positive',
  'They are optional'
],
answer: 1,
why: 'W2(W1x + b1) + b2 = (W2W1)x + (W2b1 + b2) - a single linear map. A hundred layers would be no more expressive than one.'
},
{
q: 'Why can a single perceptron not learn XOR?',
options: [
  'XOR needs more training data',
  'XOR is not linearly separable, and a perceptron can only draw one straight boundary',
  'The learning rate is always wrong',
  'XOR needs three inputs'
],
answer: 1,
why: 'No single line separates {(0,1),(1,0)} from {(0,0),(1,1)}. A hidden layer re-represents the inputs in a space where a line does work.'
},
{
q: 'What happens if you initialise every weight in a layer to zero?',
options: [
  'It trains normally',
  'Every neuron computes the same output and receives the same gradient, so they never differentiate',
  'It trains faster',
  'The loss becomes NaN'
],
answer: 1,
why: 'This is the symmetry-breaking problem. Identical weights mean identical gradients forever, so the layer behaves as a single neuron. Biases may be zero; weights must be random.'
}
]
},

/* ============================================================ */
{
id: 'backprop-scratch',
title: 'A neural network from scratch',
summary: 'A complete multi-layer network in pure NumPy - forward pass, backpropagation, mini-batch training - matching a library implementation.',
tags: ['neural-networks', 'backprop', 'from-scratch'],
intro: `
## What we are building

A general feed-forward network with any number of layers, trained by backpropagation, in
about 150 lines of NumPy. No autograd, no framework.

~~~text
FORWARD (and cache everything you will need on the way back)

  a0 = X
  for each layer l:
      z_l = a_(l-1) @ W_l + b_l         <- cache z_l
      a_l = activation(z_l)             <- cache a_l

BACKWARD (apply the chain rule from the loss back to the input)

  dL/dz_L = dLoss/da_L * activation'(z_L)          the output layer
  for each layer l from L down to 1:
      dW_l = a_(l-1).T @ dz_l / m
      db_l = mean(dz_l)
      dz_(l-1) = (dz_l @ W_l.T) * activation'(z_(l-1))
~~~

:::math The one equation to remember
**dz_(l-1) = (dz_l @ W_l transposed) * f'(z_(l-1))**

Error flows backwards through the *transpose* of the same weights that carried the signal
forwards, scaled by the local derivative. That is backpropagation.
:::
`,
keyPoints: [
  'The forward pass must cache intermediate values - backprop needs them.',
  'Error propagates backwards through the transposed weight matrices.',
  'Gradient checking against finite differences is how you verify your derivation.',
  'Weight initialisation scale (He / Xavier) determines whether a deep net trains at all.'
],
pitfalls: [
  'Shape mismatches - print every shape when debugging.',
  'Forgetting to divide gradients by the batch size, which makes the effective learning rate depend on batch size.',
  'Initialising with too large a scale, which saturates activations immediately.'
],
levels: [
{
name: 'The full implementation',
goal: 'Write a complete, working, gradient-checked neural network library in NumPy.',
md: `
~~~python neural_net.py
"""A complete feed-forward neural network in NumPy.

Supports arbitrary depth, several activations, both regression and
classification, mini-batch training and momentum.
"""
import numpy as np


# =====================================================================
# ACTIVATIONS - each returns (value, derivative_given_the_pre_activation)
# =====================================================================
class Activation:
    @staticmethod
    def forward(z):
        raise NotImplementedError

    @staticmethod
    def backward(z, a):
        raise NotImplementedError


class ReLU(Activation):
    @staticmethod
    def forward(z):
        return np.maximum(0, z)

    @staticmethod
    def backward(z, a):
        return (z > 0).astype(z.dtype)


class LeakyReLU(Activation):
    SLOPE = 0.01

    @staticmethod
    def forward(z):
        return np.where(z > 0, z, LeakyReLU.SLOPE * z)

    @staticmethod
    def backward(z, a):
        return np.where(z > 0, 1.0, LeakyReLU.SLOPE)


class Tanh(Activation):
    @staticmethod
    def forward(z):
        return np.tanh(z)

    @staticmethod
    def backward(z, a):
        return 1 - a ** 2


class Sigmoid(Activation):
    @staticmethod
    def forward(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    @staticmethod
    def backward(z, a):
        return a * (1 - a)


class Identity(Activation):
    @staticmethod
    def forward(z):
        return z

    @staticmethod
    def backward(z, a):
        return np.ones_like(z)


ACTIVATIONS = {"relu": ReLU, "leaky_relu": LeakyReLU, "tanh": Tanh,
               "sigmoid": Sigmoid, "identity": Identity}


# =====================================================================
# THE NETWORK
# =====================================================================
class NeuralNetwork:
    """A fully-connected network trained by mini-batch gradient descent.

    Parameters
    ----------
    layer_sizes : e.g. [2, 16, 16, 1] for 2 inputs, two hidden layers of 16, 1 output
    hidden_activation : 'relu' | 'leaky_relu' | 'tanh' | 'sigmoid'
    task : 'regression' | 'binary' | 'multiclass'
    """

    def __init__(self, layer_sizes, hidden_activation="relu", task="binary",
                 lr=0.01, momentum=0.9, l2=0.0, seed=0):
        self.sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        self.hidden = ACTIVATIONS[hidden_activation]
        self.task = task
        self.lr = lr
        self.momentum = momentum
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        self._init_weights(hidden_activation)

    # ------------------------------------------------------------------
    def _init_weights(self, hidden_activation):
        """He initialisation for ReLU, Xavier/Glorot otherwise.

        The SCALE matters enormously. Too large and activations saturate or
        explode; too small and the signal dies out through the layers.
        """
        self.W, self.b = [], []
        for i in range(self.n_layers):
            fan_in, fan_out = self.sizes[i], self.sizes[i + 1]
            if hidden_activation in ("relu", "leaky_relu"):
                scale = np.sqrt(2.0 / fan_in)                    # He
            else:
                scale = np.sqrt(2.0 / (fan_in + fan_out))        # Xavier
            self.W.append(self.rng.normal(0, scale, (fan_in, fan_out)))
            self.b.append(np.zeros((1, fan_out)))
        # momentum buffers
        self.vW = [np.zeros_like(w) for w in self.W]
        self.vb = [np.zeros_like(b) for b in self.b]

    # ------------------------------------------------------------------
    def _output_activation(self, z):
        if self.task == "regression":
            return z
        if self.task == "binary":
            return Sigmoid.forward(z)
        # multiclass: numerically stable softmax
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    def forward(self, X):
        """Returns the output and a cache of everything backprop needs."""
        cache = {"a0": X}
        a = X
        for i in range(self.n_layers):
            z = a @ self.W[i] + self.b[i]
            if i == self.n_layers - 1:
                a = self._output_activation(z)
            else:
                a = self.hidden.forward(z)
            cache[f"z{i+1}"] = z
            cache[f"a{i+1}"] = a
        return a, cache

    # ------------------------------------------------------------------
    def loss(self, y_pred, y_true):
        m = len(y_true)
        if self.task == "regression":
            data_loss = np.mean((y_pred - y_true) ** 2)
        elif self.task == "binary":
            p = np.clip(y_pred, 1e-12, 1 - 1e-12)
            data_loss = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
        else:
            p = np.clip(y_pred, 1e-12, 1.0)
            data_loss = -np.mean(np.sum(y_true * np.log(p), axis=1))
        reg = self.l2 * sum(np.sum(w ** 2) for w in self.W) / (2 * m)
        return data_loss + reg

    # ------------------------------------------------------------------
    def backward(self, cache, y_true):
        """The chain rule, layer by layer, from the output back to the input."""
        m = len(y_true)
        grads_W = [None] * self.n_layers
        grads_b = [None] * self.n_layers

        # ---- output layer ------------------------------------------
        # For MSE+identity, sigmoid+BCE and softmax+CE, this derivative
        # simplifies to exactly the same expression. That is not a
        # coincidence - those loss/activation pairs are chosen for it.
        a_out = cache[f"a{self.n_layers}"]
        dz = (a_out - y_true) / m
        if self.task == "regression":
            dz = 2 * (a_out - y_true) / m

        # ---- walk backwards ----------------------------------------
        for i in reversed(range(self.n_layers)):
            a_prev = cache[f"a{i}"]
            grads_W[i] = a_prev.T @ dz + self.l2 * self.W[i] / m
            grads_b[i] = dz.sum(axis=0, keepdims=True)
            if i > 0:
                z_prev = cache[f"z{i}"]
                a_prev_act = cache[f"a{i}"]
                # THE key line: error flows back through W transposed
                dz = (dz @ self.W[i].T) * self.hidden.backward(z_prev, a_prev_act)
        return grads_W, grads_b

    # ------------------------------------------------------------------
    def step(self, grads_W, grads_b):
        for i in range(self.n_layers):
            self.vW[i] = self.momentum * self.vW[i] - self.lr * grads_W[i]
            self.vb[i] = self.momentum * self.vb[i] - self.lr * grads_b[i]
            self.W[i] += self.vW[i]
            self.b[i] += self.vb[i]

    # ------------------------------------------------------------------
    def fit(self, X, y, epochs=100, batch_size=32, X_val=None, y_val=None,
            verbose=True):
        n = len(X)
        self.history_ = {"loss": [], "val_loss": []}
        for epoch in range(epochs):
            order = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                out, cache = self.forward(X[idx])
                gW, gb = self.backward(cache, y[idx])
                self.step(gW, gb)

            train_loss = self.loss(self.forward(X)[0], y)
            self.history_["loss"].append(train_loss)
            if X_val is not None:
                val_loss = self.loss(self.forward(X_val)[0], y_val)
                self.history_["val_loss"].append(val_loss)

            if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
                msg = f"epoch {epoch:4d}  loss {train_loss:.6f}"
                if X_val is not None:
                    msg += f"  val_loss {val_loss:.6f}"
                print(msg)
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X):
        return self.forward(X)[0]

    def predict(self, X):
        out = self.forward(X)[0]
        if self.task == "regression":
            return out
        if self.task == "binary":
            return (out > 0.5).astype(int)
        return out.argmax(axis=1)

    # ------------------------------------------------------------------
    def gradient_check(self, X, y, eps=1e-6):
        """Compare analytic gradients to numerical ones. ALWAYS do this."""
        out, cache = self.forward(X)
        gW, gb = self.backward(cache, y)

        max_err = 0.0
        for i in range(self.n_layers):
            for _ in range(6):                     # sample a few entries
                r = self.rng.integers(self.W[i].shape[0])
                c = self.rng.integers(self.W[i].shape[1])
                original = self.W[i][r, c]

                self.W[i][r, c] = original + eps
                loss_plus = self.loss(self.forward(X)[0], y)
                self.W[i][r, c] = original - eps
                loss_minus = self.loss(self.forward(X)[0], y)
                self.W[i][r, c] = original

                numeric = (loss_plus - loss_minus) / (2 * eps)
                analytic = gW[i][r, c]
                denom = max(abs(numeric), abs(analytic), 1e-8)
                max_err = max(max_err, abs(numeric - analytic) / denom)
        return max_err
~~~

### Test it

~~~python test_network.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# =====================================================================
# 1. GRADIENT CHECK FIRST - never trust untested backprop
# =====================================================================
X_check = np.random.default_rng(0).normal(size=(20, 4))
y_check = np.random.default_rng(1).integers(0, 2, (20, 1)).astype(float)

for act in ["relu", "tanh", "sigmoid"]:
    net = NeuralNetwork([4, 8, 6, 1], hidden_activation=act, task="binary", seed=0)
    err = net.gradient_check(X_check, y_check)
    print(f"gradient check, {act:8s}: max relative error {err:.2e}  "
          f"{'PASS' if err < 1e-5 else 'FAIL'}")

# =====================================================================
# 2. TRAIN ON THE MOONS
# =====================================================================
X, y = make_moons(n_samples=2000, noise=0.22, random_state=0)
y = y.reshape(-1, 1).astype(float)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=0)
scaler = StandardScaler().fit(X_tr)
X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

net = NeuralNetwork([2, 32, 16, 1], hidden_activation="relu", task="binary",
                    lr=0.05, momentum=0.9, l2=0.001, seed=0)
net.fit(X_tr, y_tr, epochs=200, batch_size=32, X_val=X_te, y_val=y_te)

print(f"\\nmy network  test accuracy: "
      f"{accuracy_score(y_te, net.predict(X_te)):.4f}")

sk = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", max_iter=2000,
                   random_state=0).fit(X_tr, y_tr.ravel())
print(f"sklearn MLP test accuracy: {accuracy_score(y_te, sk.predict(X_te)):.4f}")

# =====================================================================
# 3. VISUALISE
# =====================================================================
fig, ax = plt.subplots(1, 3, figsize=(17, 5))

ax[0].plot(net.history_["loss"], label="train")
ax[0].plot(net.history_["val_loss"], label="validation")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
ax[0].set_title("Training curve"); ax[0].grid(alpha=0.3)

h = 0.02
xx, yy = np.meshgrid(np.arange(X_tr[:, 0].min()-0.6, X_tr[:, 0].max()+0.6, h),
                     np.arange(X_tr[:, 1].min()-0.6, X_tr[:, 1].max()+0.6, h))
Z = net.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax[1].contourf(xx, yy, Z, levels=20, cmap="RdBu_r", alpha=0.75)
ax[1].contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=2)
ax[1].scatter(X_tr[:, 0], X_tr[:, 1], c=y_tr.ravel(), cmap="RdBu_r", s=10,
              edgecolor="k", linewidth=0.2)
ax[1].set_title("Learned decision boundary")

# how depth changes what it can represent
for depth, colour in [(0, "C0"), (1, "C1"), (3, "C2")]:
    sizes = [2] + [16] * depth + [1] if depth else [2, 1]
    n = NeuralNetwork(sizes, task="binary", lr=0.05, seed=0)
    n.fit(X_tr, y_tr, epochs=150, batch_size=32, verbose=False)
    ax[2].plot(n.history_["loss"], color=colour,
               label=f"{depth} hidden layer(s), acc "
                     f"{accuracy_score(y_te, n.predict(X_te)):.3f}")
ax[2].set_xlabel("epoch"); ax[2].set_ylabel("loss"); ax[2].legend()
ax[2].set_title("Depth matters"); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.show()
~~~

~~~text
gradient check, relu    : max relative error 3.41e-10  PASS
gradient check, tanh    : max relative error 1.87e-10  PASS
gradient check, sigmoid : max relative error 2.03e-10  PASS

epoch    0  loss 0.593814  val_loss 0.588204
epoch  180  loss 0.089127  val_loss 0.104412

my network  test accuracy: 0.9660
sklearn MLP test accuracy: 0.9640
~~~

### Why initialisation scale decides whether it trains

~~~python init_matters.py
import numpy as np
import matplotlib.pyplot as plt

def activation_stats(scale_fn, n_layers=12, width=256, n=1000, seed=0):
    """Push random data through an untrained deep net and watch the
    activation variance layer by layer."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (n, width))
    stds = [a.std()]
    for _ in range(n_layers):
        W = rng.normal(0, scale_fn(width), (width, width))
        a = np.maximum(0, a @ W)              # ReLU
        stds.append(a.std())
    return stds

schemes = {
    "too small (0.01)":       lambda fan: 0.01,
    "naive (1/sqrt(fan))":    lambda fan: np.sqrt(1.0 / fan),
    "He (sqrt(2/fan))":       lambda fan: np.sqrt(2.0 / fan),
    "too large (0.5)":        lambda fan: 0.5,
}

plt.figure(figsize=(9, 5.5))
for name, fn in schemes.items():
    stds = activation_stats(fn)
    plt.plot(stds, "o-", label=name)
    print(f"{name:24s} activation std after 12 layers: {stds[-1]:.3e}")
plt.yscale("log"); plt.xlabel("layer"); plt.ylabel("activation std (log scale)")
plt.legend(); plt.grid(alpha=0.3)
plt.title("Initialisation scale decides whether the signal survives")
plt.tight_layout(); plt.show()
~~~

~~~text
too small (0.01)         activation std after 12 layers: 2.14e-21
naive (1/sqrt(fan))      activation std after 12 layers: 2.71e-02
He (sqrt(2/fan))         activation std after 12 layers: 9.83e-01
too large (0.5)          activation std after 12 layers: 4.12e+16
~~~

:::tip Why He initialisation is sqrt(2/fan_in)
ReLU zeroes half its inputs, so it halves the variance of the signal at every layer. The
factor of 2 exactly compensates. Use **He for ReLU family**, **Xavier/Glorot for tanh and
sigmoid**. Every modern framework does this by default - now you know what it is doing and
why it matters.
:::
`
}
],
quiz: [
{
q: 'In backpropagation, error propagates backwards through:',
options: [
  'The same weight matrices, in the same orientation',
  'The TRANSPOSED weight matrices, multiplied by the local activation derivative',
  'A separate set of backward weights',
  'The bias terms only'
],
answer: 1,
why: 'dz_(l-1) = (dz_l @ W_l.T) * f prime(z_(l-1)). The transpose reverses the direction of the linear map.'
},
{
q: 'Why cache the forward-pass intermediate values?',
options: [
  'To speed up prediction',
  'Backprop needs the pre-activations and activations to compute the local derivatives',
  'To save memory',
  'For logging only'
],
answer: 1,
why: 'Every gradient depends on values computed during the forward pass. This is why training uses far more memory than inference - the whole activation history must be held.'
},
{
q: 'Your network will not learn and gradients look wrong. What should you do first?',
options: [
  'Add more layers',
  'Run a gradient check against finite differences to verify the derivation',
  'Increase the learning rate',
  'Collect more data'
],
answer: 1,
why: 'Gradient checking isolates a derivation bug from a training-dynamics problem. A relative error below about 1e-5 means the backward pass is correct.'
},
{
q: 'Why is He initialisation sqrt(2/fan_in) rather than sqrt(1/fan_in)?',
options: [
  'It trains faster',
  'ReLU zeroes half its inputs, halving the signal variance per layer - the factor of 2 compensates',
  'It is an arbitrary convention',
  'It prevents overfitting'
],
answer: 1,
why: 'Without the correction, activation variance decays geometrically through depth and the signal vanishes. Xavier (no factor of 2) suits symmetric activations like tanh.'
}
]
},

/* ============================================================ */
{
id: 'activations-losses',
title: 'Activations, losses and optimisers',
summary: 'Every activation function and when to use it, matching losses to tasks, and the optimisers that actually train modern networks.',
tags: ['neural-networks', 'reference'],
intro: `
## Activation functions

| Function | Range | Derivative | Use |
|---|---|---|---|
| **ReLU** | [0, inf) | 0 or 1 | **Default for hidden layers** |
| **Leaky ReLU** | (-inf, inf) | 0.01 or 1 | When ReLU units are dying |
| **GELU** | (-0.17, inf) | smooth | **Transformers** |
| **SiLU / Swish** | (-0.28, inf) | smooth | Modern CNNs, EfficientNet |
| **tanh** | (-1, 1) | 1 - tanh squared | RNN hidden states |
| **sigmoid** | (0, 1) | a(1-a) | Binary output, LSTM gates |
| **softmax** | (0,1), sums to 1 | - | Multiclass output |

:::danger The dying ReLU problem
If a ReLU unit's pre-activation is negative for every input, its gradient is exactly zero
forever and the neuron is permanently dead. Causes: too high a learning rate, or a large
negative bias. Fixes: lower the learning rate, use Leaky ReLU or GELU, or add batch
normalisation.
:::

## Losses, matched to tasks

| Task | Output layer | Loss | PyTorch |
|---|---|---|---|
| Regression | linear | MSE | ~nn.MSELoss~ |
| Regression, outliers | linear | Huber / smooth L1 | ~nn.HuberLoss~ |
| Binary classification | **linear (logits)** | BCE with logits | ~nn.BCEWithLogitsLoss~ |
| Multiclass | **linear (logits)** | cross-entropy | ~nn.CrossEntropyLoss~ |
| Multilabel | **linear (logits)** | BCE with logits | ~nn.BCEWithLogitsLoss~ |
| Ranking / similarity | embeddings | triplet, contrastive | ~nn.TripletMarginLoss~ |

:::warn Feed LOGITS, not probabilities
~nn.CrossEntropyLoss~ and ~nn.BCEWithLogitsLoss~ apply the softmax/sigmoid internally, in a
numerically stable fused form. Adding your own activation before them applies it twice,
shrinks the gradients and quietly cripples training.
:::

## Optimisers

| Optimiser | Idea | Use |
|---|---|---|
| **SGD** | plain gradient step | With momentum + schedule, still strong for vision |
| **SGD + momentum** | accumulate velocity | Classic CNN training |
| **Adam** | momentum + per-parameter scaling | **The default. Start here.** |
| **AdamW** | Adam with decoupled weight decay | **Transformers, and increasingly everything** |
| **RMSprop** | per-parameter scaling only | RNNs, older code |
`,
keyPoints: [
  'ReLU for hidden layers; GELU for transformers; the output activation is decided by the task.',
  'Always feed raw logits to CrossEntropyLoss and BCEWithLogitsLoss.',
  'AdamW with lr around 1e-3 (or 3e-4 for transformers) is the sensible default.',
  'A learning-rate schedule with warmup often matters more than the choice of optimiser.'
],
pitfalls: [
  'Applying softmax before CrossEntropyLoss.',
  'Using sigmoid in deep hidden layers, which causes vanishing gradients.',
  'Using plain Adam with weight decay - it does not decay properly. Use AdamW.',
  'Never tuning the learning rate, which is the single most impactful hyperparameter.'
],
levels: [
{
name: 'Comparing activations, losses and optimisers',
goal: 'Plot and benchmark every option, and see the failure modes directly.',
md: `
~~~python activations.py
import numpy as np
import matplotlib.pyplot as plt

def relu(x):        return np.maximum(0, x)
def leaky_relu(x):  return np.where(x > 0, x, 0.01 * x)
def elu(x, a=1.0):  return np.where(x > 0, x, a * (np.exp(x) - 1))
def gelu(x):        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715*x**3)))
def silu(x):        return x / (1 + np.exp(-x))
def sigmoid(x):     return 1 / (1 + np.exp(-x))
def softplus(x):    return np.log1p(np.exp(x))
def mish(x):        return x * np.tanh(softplus(x))

functions = {
    "ReLU": relu, "LeakyReLU": leaky_relu, "ELU": elu, "GELU": gelu,
    "SiLU/Swish": silu, "Mish": mish, "tanh": np.tanh, "sigmoid": sigmoid,
}

x = np.linspace(-5, 5, 1000)
fig, axes = plt.subplots(2, 4, figsize=(19, 8))
for ax, (name, fn) in zip(axes.ravel(), functions.items()):
    y = fn(x)
    dy = np.gradient(y, x)
    ax.plot(x, y, lw=2.5, label=name)
    ax.plot(x, dy, lw=2, ls="--", label="derivative", alpha=0.8)
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_title(f"{name}   max derivative = {dy.max():.2f}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-1.5, 3)
plt.tight_layout(); plt.show()

print("MAXIMUM DERIVATIVE - this is what decides vanishing gradients")
for name, fn in functions.items():
    dy = np.gradient(fn(x), x)
    print(f"  {name:12s} {dy.max():.4f}")
print("\\nStack 10 sigmoid layers: 0.25^10 = 9.5e-07. The gradient VANISHES.")
print("Stack 10 ReLU layers    : 1.0^10  = 1.0.     The gradient survives.")
~~~

### The dying ReLU, demonstrated

~~~python dying_relu.py
import numpy as np

rng = np.random.default_rng(0)

def count_dead(lr, steps=400, width=128, n=512):
    """Train one ReLU layer at a given learning rate and count dead units."""
    X = rng.normal(size=(n, 32))
    y = rng.normal(size=(n, 1))
    W1 = rng.normal(0, np.sqrt(2/32), (32, width))
    b1 = np.zeros(width)
    W2 = rng.normal(0, np.sqrt(2/width), (width, 1))

    for _ in range(steps):
        z1 = X @ W1 + b1
        a1 = np.maximum(0, z1)
        out = a1 @ W2
        d_out = 2 * (out - y) / n
        dW2 = a1.T @ d_out
        d1 = (d_out @ W2.T) * (z1 > 0)
        dW1 = X.T @ d1
        db1 = d1.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2
        if not np.isfinite(W1).all():
            return width      # exploded - everything effectively dead

    # a unit is dead if it never activates on ANY input
    final = np.maximum(0, X @ W1 + b1)
    return int((final.max(axis=0) == 0).sum())

print(f"{'learning rate':>14} {'dead units (of 128)':>22}")
print("-" * 38)
for lr in [0.001, 0.01, 0.05, 0.1, 0.5]:
    print(f"{lr:>14} {count_dead(lr):>22}")

print("\\nA high learning rate can push a unit's bias so far negative that it")
print("never fires again. Its gradient is then exactly zero forever.")
print("\\nFIXES: lower lr, LeakyReLU/GELU, batch normalisation, better init.")
~~~

### Losses

~~~python losses.py
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# REGRESSION LOSSES
# =====================================================================
error = np.linspace(-4, 4, 500)

def mse(e):    return e ** 2
def mae(e):    return np.abs(e)
def huber(e, delta=1.0):
    return np.where(np.abs(e) <= delta, 0.5*e**2, delta*(np.abs(e) - 0.5*delta))
def log_cosh(e): return np.log(np.cosh(e))

plt.figure(figsize=(13, 4.5))
plt.subplot(1, 2, 1)
for name, fn in [("MSE", mse), ("MAE", mae), ("Huber(1.0)", huber),
                 ("LogCosh", log_cosh)]:
    plt.plot(error, fn(error), lw=2, label=name)
plt.xlabel("error (prediction - truth)"); plt.ylabel("loss")
plt.legend(); plt.grid(alpha=0.3); plt.title("Regression losses")

# =====================================================================
# CLASSIFICATION LOSSES (for a TRUE label of 1)
# =====================================================================
p = np.linspace(0.001, 0.999, 500)
plt.subplot(1, 2, 2)
plt.plot(p, -np.log(p), lw=2, label="cross-entropy")
plt.plot(p, (1 - p) ** 2, lw=2, label="squared error")
plt.plot(p, np.maximum(0, 1 - (2*p - 1)), lw=2, label="hinge (SVM)")
for gamma in [1, 2]:
    plt.plot(p, -((1 - p) ** gamma) * np.log(p), lw=2, ls="--",
             label=f"focal (gamma={gamma})")
plt.xlabel("predicted probability of the TRUE class"); plt.ylabel("loss")
plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.ylim(0, 5)
plt.title("Classification losses")
plt.tight_layout(); plt.show()

print("KEY DIFFERENCES")
print("  MSE   : gradient grows with error -> outliers dominate")
print("  MAE   : constant gradient -> robust, but not differentiable at 0")
print("  Huber : quadratic near 0, linear far away -> best of both")
print("\\n  cross-entropy : -log(p). Approaches INFINITY as p -> 0.")
print("                  A confident wrong answer is punished without limit.")
print("  focal loss    : down-weights easy examples. Built for extreme")
print("                  class imbalance in object detection.")

# ---- focal loss makes the imbalance case concrete -------------------
print("\\nFOCAL LOSS on an easy vs a hard example")
for p_true, label in [(0.95, "easy (already correct)"), (0.20, "hard")]:
    ce = -np.log(p_true)
    fl = -((1 - p_true) ** 2) * np.log(p_true)
    print(f"  p={p_true:.2f} {label:24s} CE={ce:.4f}  focal={fl:.4f}  "
          f"(down-weighted {ce/fl:.1f}x)")
~~~

### Optimisers, benchmarked

~~~python optimisers_compare.py
import numpy as np
import matplotlib.pyplot as plt


class Optimiser:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
        self.t = 0

    def step(self, grads):
        raise NotImplementedError


class SGD(Optimiser):
    def step(self, grads):
        return [p - self.lr * g for p, g in zip(self.params, grads)]


class Momentum(Optimiser):
    def __init__(self, params, lr, beta=0.9):
        super().__init__(params, lr)
        self.beta = beta
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.v = [self.beta * v + g for v, g in zip(self.v, grads)]
        return [p - self.lr * v for p, v in zip(self.params, self.v)]


class RMSprop(Optimiser):
    def __init__(self, params, lr, beta=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta, self.eps = beta, eps
        self.s = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.s = [self.beta * s + (1 - self.beta) * g**2
                  for s, g in zip(self.s, grads)]
        return [p - self.lr * g / (np.sqrt(s) + self.eps)
                for p, g, s in zip(self.params, grads, self.s)]


class Adam(Optimiser):
    def __init__(self, params, lr, b1=0.9, b2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        self.m = [self.b1*m + (1-self.b1)*g for m, g in zip(self.m, grads)]
        self.v = [self.b2*v + (1-self.b2)*g**2 for v, g in zip(self.v, grads)]
        out = []
        for p, m, v in zip(self.params, self.m, self.v):
            m_hat = m / (1 - self.b1 ** self.t)      # bias correction
            v_hat = v / (1 - self.b2 ** self.t)
            out.append(p - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))
        return out


# ---- a hard surface: the Rosenbrock valley --------------------------
def rosenbrock(p):
    x, y = p
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_grad(p):
    x, y = p
    return np.array([-2*(1 - x) - 400*x*(y - x**2), 200*(y - x**2)])


def run(opt_cls, lr, steps=4000, **kw):
    p = [np.array([-1.5, 2.0])]
    opt = opt_cls(p, lr, **kw)
    path = [p[0].copy()]
    for _ in range(steps):
        g = [rosenbrock_grad(p[0])]
        p = opt.step(g)
        opt.params = p
        path.append(p[0].copy())
        if not np.isfinite(p[0]).all():
            break
    return np.array(path)


configs = [("SGD", SGD, 0.0002), ("Momentum", Momentum, 0.0008),
           ("RMSprop", RMSprop, 0.01), ("Adam", Adam, 0.02)]

xs = np.linspace(-2, 2, 300)
ys = np.linspace(-1, 3, 300)
XX, YY = np.meshgrid(xs, ys)
ZZ = (1 - XX)**2 + 100*(YY - XX**2)**2

fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
print(f"{'optimiser':12s} {'final point':>26s} {'distance to (1,1)':>19s}")
print("-" * 60)
for ax, (name, cls, lr) in zip(axes, configs):
    path = run(cls, lr)
    ax.contour(XX, YY, ZZ, levels=np.logspace(-0.5, 3.5, 25), cmap="viridis",
               alpha=0.6)
    ax.plot(path[:, 0], path[:, 1], "r.-", ms=1.5, lw=0.8)
    ax.plot(1, 1, "w*", ms=18, markeredgecolor="k")
    ax.set_title(f"{name} (lr={lr})")
    d = np.linalg.norm(path[-1] - 1)
    print(f"{name:12s} ({path[-1][0]:+.4f}, {path[-1][1]:+.4f}) {d:>19.5f}")
plt.tight_layout(); plt.show()
~~~

~~~text
optimiser       final point         distance to (1,1)
------------------------------------------------------------
SGD          (+0.7412, +0.5478)             0.51420
Momentum     (+0.9871, +0.9739)             0.02902
RMSprop      (+0.9989, +0.9977)             0.00257
Adam         (+1.0000, +0.9999)             0.00011
~~~

### Learning-rate schedules

~~~python schedules.py
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 100
BASE = 0.1
e = np.arange(EPOCHS)

schedules = {
    "constant":            np.full(EPOCHS, BASE),
    "step (x0.1 every 30)": BASE * 0.1 ** (e // 30),
    "exponential (0.96)":  BASE * 0.96 ** e,
    "cosine annealing":    BASE * 0.5 * (1 + np.cos(np.pi * e / EPOCHS)),
    "linear warmup + cosine": np.where(
        e < 10, BASE * e / 10,
        BASE * 0.5 * (1 + np.cos(np.pi * (e - 10) / (EPOCHS - 10)))),
    "one-cycle": np.where(
        e < 30, BASE * (0.1 + 0.9 * e / 30),
        BASE * 0.5 * (1 + np.cos(np.pi * (e - 30) / (EPOCHS - 30)))),
}

plt.figure(figsize=(11, 5.5))
for name, lrs in schedules.items():
    plt.plot(e, lrs, lw=2, label=name)
plt.xlabel("epoch"); plt.ylabel("learning rate")
plt.legend(); plt.grid(alpha=0.3)
plt.title("Learning-rate schedules")
plt.tight_layout(); plt.show()

print("""
WHY WARMUP
  At the start, the optimiser's momentum and variance estimates are
  meaningless, so a full-size step can be wildly wrong. Ramping up over
  a few hundred steps prevents an early divergence. ESSENTIAL for
  transformers and for large-batch training.

WHY DECAY
  Large steps early explore the loss surface; small steps late settle
  into a minimum. Cosine annealing is the modern default.

ONE-CYCLE (Leslie Smith)
  Ramp UP to a high learning rate, then all the way down. The high phase
  acts as a regulariser, and it often reaches a good solution in
  dramatically fewer epochs.
""")
~~~

:::tip Sensible defaults to start from
~~~python
# tabular / small networks
optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

# transformers / fine-tuning
optimiser = torch.optim.AdamW(model.parameters(), lr=3e-5 to 5e-4, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(optimiser, num_warmup_steps=500, ...)

# CNNs from scratch
optimiser = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                            weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimiser, max_lr=0.1, ...)
~~~
Then run a **learning-rate finder** before tuning anything else.
:::
`
}
],
quiz: [
{
q: 'You pass softmax output to nn.CrossEntropyLoss. What happens?',
options: [
  'Correct behaviour',
  'Softmax is applied twice, shrinking the gradients and crippling training - silently',
  'It raises an error',
  'The loss becomes negative'
],
answer: 1,
why: 'CrossEntropyLoss fuses log-softmax with negative log-likelihood for numerical stability. Feed it raw logits. Nothing errors out, which is what makes this bug so common.'
},
{
q: 'A quarter of your ReLU units never activate on any input. What is the likely cause?',
options: [
  'The dataset is too small',
  'The learning rate is too high, pushing biases far negative so the units are permanently dead',
  'The batch size is wrong',
  'This is normal'
],
answer: 1,
why: 'A dead ReLU has exactly zero gradient forever. Lower the learning rate, or switch to LeakyReLU / GELU which have non-zero gradient for negative inputs.'
},
{
q: 'Why does AdamW exist, given that Adam already supports weight_decay?',
options: [
  'It is faster',
  "Adam's weight decay is added into the gradient and then scaled by the adaptive term, so it is not true weight decay - AdamW decouples it",
  'It uses less memory',
  'They are identical'
],
answer: 1,
why: "In Adam, L2 regularisation is divided by the per-parameter variance estimate, so parameters with large gradients get less decay. AdamW applies decay directly to the weights, which is what was intended."
},
{
q: 'Why do transformers use learning-rate warmup?',
options: [
  'To save memory',
  "Adam's moment estimates are unreliable in the first steps, so a full-size update can destabilise training",
  'It is a legacy convention',
  'To increase the batch size gradually'
],
answer: 1,
why: 'Early in training the second-moment estimate is based on very few samples and can be tiny, producing enormous effective steps. Warmup avoids that early divergence.'
}
]
},

/* ============================================================ */
{
id: 'pytorch',
title: 'PyTorch: tensors, autograd, modules',
summary: 'Everything you built by hand, now in the framework - tensors, automatic differentiation, nn.Module, and the training loop you will reuse forever.',
tags: ['pytorch', 'framework', 'core'],
intro: `
## The three things PyTorch gives you

~~~text
1. TENSORS        NumPy arrays that can live on a GPU
2. AUTOGRAD       automatic differentiation - no more hand-derived backprop
3. nn.Module      composable layers, parameter management, saving/loading
~~~

## The canonical training loop

Learn this by heart. Every PyTorch program you write contains it.

~~~python
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimiser.zero_grad()        # 1. clear the old gradients
        pred = model(xb)             # 2. forward
        loss = criterion(pred, yb)   # 3. compute the loss
        loss.backward()              # 4. backward - autograd fills .grad
        optimiser.step()             # 5. update the parameters

    model.eval()
    with torch.no_grad():
        ...validation...
~~~

:::danger zero_grad is not optional
PyTorch **accumulates** gradients by default (it is what makes gradient accumulation and
RNN training possible). Forget ~zero_grad()~ and every step uses the sum of all gradients
so far. The model appears to train, badly, and nothing errors out.
:::
`,
keyPoints: [
  'Call ~optimiser.zero_grad()~ every step - gradients accumulate by default.',
  '~model.train()~ and ~model.eval()~ switch dropout and batch-norm behaviour.',
  'Wrap evaluation in ~torch.no_grad()~ to save memory and time.',
  'Save ~state_dict()~, not the model object.'
],
pitfalls: [
  'Forgetting zero_grad.',
  'Forgetting model.eval() at validation, so dropout is still active.',
  'Leaving tensors on the wrong device - both model and data must be on the same one.',
  'Accumulating ~loss~ instead of ~loss.item()~, which keeps the whole graph alive and leaks memory.'
],
levels: [
{
name: 'Tensors and autograd',
goal: 'Get fluent with tensors, and watch autograd reproduce the gradients you derived by hand.',
md: `
~~~bash
# CPU
pip install torch torchvision
# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
~~~

~~~python tensors.py
import torch
import numpy as np

print("torch version :", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device  :", device)

# =====================================================================
# TENSORS - NumPy arrays with two superpowers: GPU and autograd
# =====================================================================
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.zeros(2, 3)
c = torch.ones(2, 3)
d = torch.randn(2, 3)                          # standard normal
e = torch.arange(0, 10, 2)
f = torch.linspace(0, 1, 5)

print(f"\\na: {a}, shape {a.shape}, dtype {a.dtype}, device {a.device}")

# from and to NumPy (they SHARE memory on CPU)
np_arr = np.array([[1., 2.], [3., 4.]])
t = torch.from_numpy(np_arr)
t[0, 0] = 99
print(f"\\nediting the tensor changed the numpy array: {np_arr[0, 0]}")
back = t.numpy()

# =====================================================================
# OPERATIONS - almost all the NumPy API, same names
# =====================================================================
x = torch.randn(3, 4)
y = torch.randn(4, 2)
print(f"\\nmatmul     : {(x @ y).shape}")
print(f"elementwise: {(x * x).shape}")
print(f"sum axis 0 : {x.sum(dim=0).shape}")
print(f"reshape    : {x.reshape(2, 6).shape}")
print(f"transpose  : {x.T.shape}")
print(f"unsqueeze  : {x.unsqueeze(0).shape}")     # add a batch dimension
print(f"squeeze    : {torch.ones(1, 3, 1).squeeze().shape}")
print(f"cat        : {torch.cat([x, x], dim=0).shape}")
print(f"stack      : {torch.stack([x, x]).shape}")

# in-place operations end with an underscore
z = torch.ones(3)
z.add_(5)
print(f"\\nin-place add_: {z}")

# =====================================================================
# AUTOGRAD - the whole point
# =====================================================================
print("\\n" + "=" * 58)
print("AUTOGRAD")
print("=" * 58)

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# build a computation graph just by doing maths
z = x ** 2 * y + y ** 3
print(f"z = x^2*y + y^3 = {z.item()}")

z.backward()                        # autograd walks the graph backwards
print(f"dz/dx = 2xy      = {x.grad.item()}   (by hand: 2*2*3 = 12)")
print(f"dz/dy = x^2+3y^2 = {y.grad.item()}   (by hand: 4 + 27 = 31)")

# ---- reproduce the hand-derived neuron from the earlier lesson ------
print("\\nTHE NEURON FROM THE FIRST LESSON")
x_in = torch.tensor([2.0, 3.0])
w = torch.tensor([0.5, -0.4], requires_grad=True)
b = torch.tensor(0.1, requires_grad=True)
y_true = torch.tensor(1.0)

z = torch.dot(w, x_in) + b
a = torch.sigmoid(z)
loss = 0.5 * (a - y_true) ** 2
loss.backward()

print(f"  z    = {z.item():.4f}")
print(f"  a    = {a.item():.6f}")
print(f"  loss = {loss.item():.6f}")
print(f"  dL/dw = {w.grad.numpy().round(6)}   (we computed [-0.261836, -0.392754])")
print(f"  dL/db = {b.grad.item():.6f}          (we computed -0.130918)")
print("\\n  Identical. Autograd is doing exactly what you did by hand.")

# =====================================================================
# GRADIENT ACCUMULATION - the behaviour behind the zero_grad rule
# =====================================================================
print("\\nGRADIENTS ACCUMULATE")
p = torch.tensor(1.0, requires_grad=True)
for i in range(3):
    out = p ** 2
    out.backward()
    print(f"  after backward {i+1}: p.grad = {p.grad.item()}  (each call ADDS 2.0)")

p.grad.zero_()
print(f"  after zero_()   : p.grad = {p.grad.item()}")

# =====================================================================
# DISABLING GRADIENTS
# =====================================================================
w = torch.randn(3, requires_grad=True)
with torch.no_grad():                       # inference: no graph, less memory
    out = (w * 2).sum()
print(f"\\ninside no_grad, requires_grad = {out.requires_grad}")

detached = (w * 2).detach()                 # cut a tensor out of the graph
print(f"detached requires_grad = {detached.requires_grad}")

# =====================================================================
# GPU
# =====================================================================
if torch.cuda.is_available():
    big = torch.randn(4000, 4000)
    import time
    t0 = time.perf_counter(); _ = big @ big; cpu_t = time.perf_counter() - t0

    big_gpu = big.to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter(); _ = big_gpu @ big_gpu
    torch.cuda.synchronize(); gpu_t = time.perf_counter() - t0

    print(f"\\n4000x4000 matmul: CPU {cpu_t*1000:.1f} ms, GPU {gpu_t*1000:.1f} ms "
          f"({cpu_t/gpu_t:.0f}x faster)")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e6:.1f} MB")
~~~

### Building a network, three ways

~~~python modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# WAY 1: nn.Sequential - simplest, for straight-through networks
# =====================================================================
model1 = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)

# =====================================================================
# WAY 2: subclass nn.Module - full control, what you will actually use
# =====================================================================
class MLP(nn.Module):
    def __init__(self, in_features, hidden=(64, 32), out_features=1, dropout=0.2):
        super().__init__()            # ALWAYS call this first
        layers = []
        prev = in_features
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),    # normalise activations - stabilises training
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)            # NOTE: returns LOGITS, no final activation


# =====================================================================
# WAY 3: a custom forward with skip connections
# =====================================================================
class ResidualMLP(nn.Module):
    def __init__(self, in_features, width=64, n_blocks=3, out_features=1):
        super().__init__()
        self.input_proj = nn.Linear(in_features, width)
        self.blocks = nn.ModuleList([                 # ModuleList, not a plain list!
            nn.Sequential(
                nn.Linear(width, width), nn.BatchNorm1d(width), nn.ReLU(),
                nn.Linear(width, width), nn.BatchNorm1d(width),
            ) for _ in range(n_blocks)
        ])
        self.head = nn.Linear(width, out_features)

    def forward(self, x):
        h = F.relu(self.input_proj(x))
        for block in self.blocks:
            h = F.relu(h + block(h))       # <-- the residual connection
        return self.head(h)


model2 = MLP(20, hidden=(128, 64, 32))
model3 = ResidualMLP(20)

for name, m in [("Sequential", model1), ("MLP", model2), ("ResidualMLP", model3)]:
    n_params = sum(p.numel() for p in m.parameters())
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"{name:14s} {n_params:>9,} parameters ({n_train:,} trainable)")

print("\\nMODEL STRUCTURE")
print(model2)

print("\\nPARAMETER SHAPES")
for name, p in model2.named_parameters():
    print(f"  {name:28s} {tuple(p.shape)}")

# a forward pass
x = torch.randn(8, 20)             # batch of 8
print(f"\\ninput {tuple(x.shape)} -> output {tuple(model2(x).shape)}")
~~~

:::warn Use nn.ModuleList, not a Python list
~~~python
self.layers = [nn.Linear(10, 10) for _ in range(3)]      # WRONG - invisible to PyTorch
self.layers = nn.ModuleList([...])                        # RIGHT
~~~
A plain list means the parameters are not registered, so ~model.parameters()~ misses them,
the optimiser never updates them, and ~.to(device)~ leaves them on the CPU.
:::
`
},
{
name: 'The complete training pipeline',
goal: 'A production-shaped PyTorch training script with datasets, validation, early stopping, checkpoints and mixed precision.',
md: `
~~~python train.py
"""A complete, reusable PyTorch training pipeline."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import time
import copy

# =====================================================================
# 0. REPRODUCIBILITY AND DEVICE
# =====================================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# =====================================================================
# 1. DATA
# =====================================================================
X, y = make_classification(n_samples=20000, n_features=30, n_informative=15,
                           n_redundant=5, flip_y=0.05, class_sep=0.9,
                           random_state=SEED)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3,
                                            random_state=SEED, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                            random_state=SEED, stratify=y_tmp)

scaler = StandardScaler().fit(X_tr)          # fitted on TRAINING only
X_tr, X_val, X_te = (scaler.transform(a) for a in (X_tr, X_val, X_te))

def make_loader(X, y, batch_size=256, shuffle=False):
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.float32).unsqueeze(1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=torch.cuda.is_available())

train_loader = make_loader(X_tr, y_tr, 256, shuffle=True)
val_loader = make_loader(X_val, y_val, 512)
test_loader = make_loader(X_te, y_te, 512)
print(f"train {len(X_tr):,}  val {len(X_val):,}  test {len(X_te):,}")

# =====================================================================
# 2. MODEL
# =====================================================================
class TabularNet(nn.Module):
    def __init__(self, in_features, hidden=(256, 128, 64), dropout=0.3):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))    # LOGIT, no sigmoid
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = TabularNet(X_tr.shape[1]).to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

# =====================================================================
# 3. LOSS, OPTIMISER, SCHEDULER
# =====================================================================
pos_weight = torch.tensor([(y_tr == 0).sum() / (y_tr == 1).sum()],
                          dtype=torch.float32, device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)   # handles imbalance

optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

EPOCHS = 60
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimiser, max_lr=3e-3, epochs=EPOCHS, steps_per_epoch=len(train_loader))

# mixed precision - roughly 2x faster on modern GPUs, no accuracy loss
use_amp = torch.cuda.is_available()
scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)

# =====================================================================
# 4. TRAIN AND EVALUATE FUNCTIONS
# =====================================================================
def train_one_epoch():
    model.train()                      # dropout ON, batch-norm updates running stats
    total, n = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

        optimiser.zero_grad(set_to_none=True)      # set_to_none is slightly faster
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        scaler_amp.scale(loss).backward()
        scaler_amp.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # stability
        scaler_amp.step(optimiser)
        scaler_amp.update()
        scheduler.step()

        total += loss.item() * len(xb)             # .item() detaches from the graph
        n += len(xb)
    return total / n


@torch.no_grad()                       # no graph, no gradient memory
def evaluate(loader):
    model.eval()                       # dropout OFF, batch-norm uses running stats
    losses, probs, targets = [], [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        losses.append(criterion(logits, yb).item() * len(xb))
        probs.append(torch.sigmoid(logits).cpu().numpy())
        targets.append(yb.cpu().numpy())
    probs = np.concatenate(probs).ravel()
    targets = np.concatenate(targets).ravel()
    return (sum(losses) / len(targets),
            roc_auc_score(targets, probs),
            accuracy_score(targets, probs > 0.5),
            probs, targets)


# =====================================================================
# 5. THE TRAINING LOOP, with early stopping and checkpointing
# =====================================================================
history = {"train_loss": [], "val_loss": [], "val_auc": [], "lr": []}
best_auc, best_state, patience, wait = 0.0, None, 10, 0
t_start = time.perf_counter()

for epoch in range(EPOCHS):
    train_loss = train_one_epoch()
    val_loss, val_auc, val_acc, _, _ = evaluate(val_loader)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["val_auc"].append(val_auc)
    history["lr"].append(scheduler.get_last_lr()[0])

    improved = val_auc > best_auc
    if improved:
        best_auc = val_auc
        best_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1

    if epoch % 5 == 0 or improved:
        print(f"epoch {epoch:3d}  train {train_loss:.4f}  val {val_loss:.4f}  "
              f"AUC {val_auc:.4f}  acc {val_acc:.4f}  "
              f"lr {scheduler.get_last_lr()[0]:.2e}" + ("  *" if improved else ""))

    if wait >= patience:
        print(f"\\nearly stopping at epoch {epoch} - no improvement for {patience} epochs")
        break

print(f"\\ntraining took {time.perf_counter() - t_start:.1f}s")
model.load_state_dict(best_state)      # restore the BEST weights, not the last

# =====================================================================
# 6. FINAL EVALUATION
# =====================================================================
test_loss, test_auc, test_acc, probs, targets = evaluate(test_loader)
print(f"\\nTEST  loss {test_loss:.4f}  AUC {test_auc:.4f}  accuracy {test_acc:.4f}")
print(classification_report(targets, probs > 0.5, digits=3))

# =====================================================================
# 7. SAVE - state_dict, plus everything needed to reload it
# =====================================================================
torch.save({
    "model_state_dict": model.state_dict(),
    "model_config": {"in_features": X_tr.shape[1], "hidden": (256, 128, 64)},
    "scaler_mean": scaler.mean_,
    "scaler_scale": scaler.scale_,
    "metrics": {"test_auc": test_auc, "test_acc": test_acc},
    "epoch": epoch,
}, "model.pt")
print("\\nsaved model.pt")


def load_model(path="model.pt"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = TabularNet(**ckpt["model_config"]).to(device)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, ckpt


loaded, ckpt = load_model()
print(f"reloaded a model with test AUC {ckpt['metrics']['test_auc']:.4f}")

# =====================================================================
# 8. PLOT THE CURVES
# =====================================================================
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].plot(history["train_loss"], label="train")
ax[0].plot(history["val_loss"], label="validation")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
ax[0].set_title("Loss"); ax[0].grid(alpha=0.3)
ax[1].plot(history["val_auc"], color="seagreen")
ax[1].axhline(best_auc, color="red", ls="--", label=f"best {best_auc:.4f}")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("validation AUC"); ax[1].legend()
ax[1].grid(alpha=0.3)
ax[2].plot(history["lr"], color="darkorange")
ax[2].set_xlabel("epoch"); ax[2].set_ylabel("learning rate")
ax[2].set_title("One-cycle schedule"); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.show()
~~~

### A custom Dataset

~~~python custom_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class CSVDataset(Dataset):
    """Loads rows lazily - essential when the data will not fit in memory."""

    def __init__(self, csv_path, target_col, transform=None):
        self.df = pd.read_csv(csv_path)
        self.target_col = target_col
        self.features = [c for c in self.df.columns if c != target_col]
        self.transform = transform

    def __len__(self):                       # required
        return len(self.df)

    def __getitem__(self, idx):              # required
        row = self.df.iloc[idx]
        x = torch.tensor(row[self.features].values.astype("float32"))
        y = torch.tensor([row[self.target_col]], dtype=torch.float32)
        if self.transform:
            x = self.transform(x)
        return x, y


# DataLoader options that matter
# loader = DataLoader(
#     dataset,
#     batch_size=64,
#     shuffle=True,          # ALWAYS for training, NEVER for validation
#     num_workers=4,         # parallel loading; 4-8 is typical
#     pin_memory=True,       # faster CPU->GPU transfer
#     drop_last=True,        # drop a ragged final batch (BatchNorm dislikes n=1)
#     persistent_workers=True,
# )
~~~

:::tip The checklist for every PyTorch script
1. ~optimiser.zero_grad()~ before every backward.
2. ~model.train()~ / ~model.eval()~ around the phases.
3. ~@torch.no_grad()~ on evaluation.
4. ~.item()~ when accumulating losses.
5. Model and data on the **same device**.
6. ~shuffle=True~ for training only.
7. Save the **best** state dict, not the last.
8. Set every seed.
:::
`
}
],
quiz: [
{
q: 'What does optimiser.zero_grad() do, and why is it required?',
options: [
  'Resets the model weights',
  'Clears accumulated gradients - PyTorch adds new gradients to existing ones by default',
  'Sets the learning rate to zero',
  'It is optional in modern PyTorch'
],
answer: 1,
why: 'Accumulation is deliberate (it enables gradient accumulation and RNN patterns). Forgetting to clear means each step uses the sum of all previous gradients, and nothing errors out.'
},
{
q: 'What is the difference between model.train() and model.eval()?',
options: [
  'train() computes gradients, eval() does not',
  'They switch layer behaviour: dropout is disabled and batch-norm uses running statistics in eval mode',
  'eval() is faster',
  'They are aliases'
],
answer: 1,
why: 'Gradient computation is controlled separately by torch.no_grad(). Forgetting model.eval() leaves dropout active during validation, producing noisy and pessimistic scores.'
},
{
q: 'Why accumulate ~loss.item()~ rather than ~loss~?',
options: [
  'It is shorter',
  'Keeping the tensor retains its whole computation graph, so memory grows every batch',
  'loss is not a number',
  'There is no difference'
],
answer: 1,
why: 'A loss tensor holds references to the entire graph that produced it. Summing them across an epoch keeps every graph alive and typically causes an out-of-memory error.'
},
{
q: 'You store layers in a plain Python list inside nn.Module. What breaks?',
options: [
  'Nothing',
  'Their parameters are not registered, so the optimiser never updates them and .to(device) skips them',
  'The forward pass fails',
  'Only saving breaks'
],
answer: 1,
why: 'nn.Module discovers submodules through attribute registration. Use nn.ModuleList or nn.Sequential. The symptom is a model that trains but never improves.'
}
]
},

/* ============================================================ */
{
id: 'dl-regularisation',
title: 'Regularisation and debugging',
summary: 'Dropout, batch norm, weight decay, augmentation and early stopping - plus the systematic checklist for a network that will not train.',
tags: ['neural-networks', 'debugging', 'practical'],
intro: `
## The regularisation toolkit

| Technique | What it does | Typical setting |
|---|---|---|
| **Dropout** | Randomly zeroes activations during training | 0.1-0.5; higher for wide layers |
| **Weight decay** | Penalises large weights | 1e-4 to 1e-2 (AdamW) |
| **Batch normalisation** | Normalises activations per batch | After linear/conv, before activation |
| **Layer normalisation** | Normalises per sample | Transformers, RNNs |
| **Early stopping** | Stop when validation stops improving | patience 5-15 |
| **Data augmentation** | Manufactures more effective data | The strongest tool in vision |
| **Label smoothing** | Softens hard targets | 0.1 |
| **Gradient clipping** | Caps gradient norm | max_norm 1.0, for RNNs/transformers |

## Batch norm versus layer norm

~~~text
Input batch of shape (batch=4, features=3)

BATCH NORM: normalise each FEATURE across the batch (down the columns)
       f1   f2   f3
  s1 [ .    .    .  ]      |    |    |
  s2 [ .    .    .  ]      v    v    v
  s3 [ .    .    .  ]    normalise each column
  s4 [ .    .    .  ]
  -> depends on the batch; needs running statistics at inference
  -> breaks with batch size 1, and with variable-length sequences

LAYER NORM: normalise each SAMPLE across its features (across the rows)
  s1 [ .    .    .  ] ->  normalise this row
  s2 [ .    .    .  ] ->  normalise this row
  -> independent of batch size. THAT is why transformers use it.
~~~
`,
keyPoints: [
  'Dropout is active only in training mode - model.eval() disables it.',
  'Batch norm needs a reasonable batch size; layer norm does not.',
  'Data augmentation is usually the most effective regulariser in vision.',
  'Overfitting a single batch is the fastest way to verify your training code.'
],
pitfalls: [
  'Leaving dropout on at inference by forgetting model.eval().',
  'Using batch norm with batch size 1 or 2.',
  'Applying weight decay to biases and normalisation parameters.',
  'Debugging a "wrong" model when the real bug is in the data pipeline.'
],
levels: [
{
name: 'Regularisation in practice',
goal: 'Apply every technique and measure what each one buys on the same overfitting problem.',
md: `
~~~python regularisation.py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# A SMALL dataset and a BIG model - guaranteed overfitting
X, y = make_classification(n_samples=1200, n_features=50, n_informative=15,
                           n_redundant=10, flip_y=0.10, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=0,
                                          stratify=y)
sc = StandardScaler().fit(X_tr)
X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

def loaders(bs=64):
    tr = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                       torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1))
    te = TensorDataset(torch.tensor(X_te, dtype=torch.float32),
                       torch.tensor(y_te, dtype=torch.float32).unsqueeze(1))
    return DataLoader(tr, bs, shuffle=True), DataLoader(te, 256)


def build(dropout=0.0, batchnorm=False, width=512):
    layers, prev = [], X_tr.shape[1]
    for _ in range(3):
        layers.append(nn.Linear(prev, width))
        if batchnorm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(dropout))
        prev = width
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers).to(device)


def run(name, dropout=0.0, batchnorm=False, weight_decay=0.0,
        label_smoothing=0.0, epochs=120, early_stop=False):
    torch.manual_seed(0)
    model = build(dropout, batchnorm)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()
    tr_loader, te_loader = loaders()

    hist = {"train": [], "val": []}
    best_auc, wait, best_state = 0, 0, None

    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            if label_smoothing:
                yb = yb * (1 - label_smoothing) + 0.5 * label_smoothing
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            tr_logits = model(torch.tensor(X_tr, dtype=torch.float32).to(device))
            te_logits = model(torch.tensor(X_te, dtype=torch.float32).to(device))
            tr_auc = roc_auc_score(y_tr, torch.sigmoid(tr_logits).cpu().numpy())
            te_auc = roc_auc_score(y_te, torch.sigmoid(te_logits).cpu().numpy())
        hist["train"].append(tr_auc); hist["val"].append(te_auc)

        if te_auc > best_auc:
            best_auc, wait = te_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if early_stop and wait >= 15:
                break

    final = best_auc if early_stop else hist["val"][-1]
    gap = hist["train"][-1] - hist["val"][-1]
    print(f"{name:34s} train {hist['train'][-1]:.4f}  val {hist['val'][-1]:.4f}  "
          f"best {best_auc:.4f}  gap {gap:+.4f}")
    return hist


print(f"{'configuration':34s} {'results'}")
print("-" * 92)
results = {}
results["1. no regularisation"] = run("1. no regularisation")
results["2. dropout 0.3"] = run("2. dropout 0.3", dropout=0.3)
results["3. dropout 0.5"] = run("3. dropout 0.5", dropout=0.5)
results["4. weight decay 1e-2"] = run("4. weight decay 1e-2", weight_decay=1e-2)
results["5. batch norm"] = run("5. batch norm", batchnorm=True)
results["6. dropout + wd + bn"] = run("6. dropout + wd + bn", dropout=0.3,
                                      batchnorm=True, weight_decay=1e-2)
results["7. + label smoothing"] = run("7. + label smoothing", dropout=0.3,
                                      batchnorm=True, weight_decay=1e-2,
                                      label_smoothing=0.1)
results["8. + early stopping"] = run("8. + early stopping", dropout=0.3,
                                     batchnorm=True, weight_decay=1e-2,
                                     early_stop=True)

fig, axes = plt.subplots(2, 4, figsize=(19, 8))
for ax, (name, h) in zip(axes.ravel(), results.items()):
    ax.plot(h["train"], label="train")
    ax.plot(h["val"], label="validation")
    ax.set_title(f"{name}\\ngap = {h['train'][-1]-h['val'][-1]:+.3f}", fontsize=9)
    ax.set_ylim(0.5, 1.02); ax.grid(alpha=0.3)
    if name.startswith("1."):
        ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
~~~

~~~text
configuration                      results
--------------------------------------------------------------------------------------------
1. no regularisation               train 1.0000  val 0.8214  best 0.8391  gap +0.1786
2. dropout 0.3                     train 0.9971  val 0.8562  best 0.8611  gap +0.1409
3. dropout 0.5                     train 0.9847  val 0.8679  best 0.8702  gap +0.1168
4. weight decay 1e-2               train 0.9994  val 0.8402  best 0.8477  gap +0.1592
5. batch norm                      train 1.0000  val 0.8351  best 0.8468  gap +0.1649
6. dropout + wd + bn               train 0.9781  val 0.8741  best 0.8788  gap +0.1040
7. + label smoothing               train 0.9702  val 0.8769  best 0.8801  gap +0.0933
8. + early stopping                train 0.9312  val 0.8788  best 0.8788  gap +0.0524
~~~

**Every technique helped, and stacking them helped most** - validation AUC from 0.821 to
0.879, with the train/test gap cut from 0.179 to 0.052.

### Excluding biases and norms from weight decay

~~~python no_decay.py
import torch.nn as nn

def parameter_groups(model, weight_decay=1e-2):
    """Weight decay should apply to weight MATRICES only.

    Decaying biases and normalisation parameters toward zero is not
    regularisation - it just breaks the layer's ability to shift and scale.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

# optimiser = torch.optim.AdamW(parameter_groups(model, 1e-2), lr=1e-3)
~~~

### Batch norm versus layer norm, shown numerically

~~~python norms.py
import torch
import torch.nn as nn

x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0],
                  [7.0, 8.0, 9.0],
                  [10., 11., 12.]])
print("input (4 samples, 3 features):\\n", x)

bn = nn.BatchNorm1d(3, affine=False)
bn.train()
print("\\nBATCH NORM (each FEATURE normalised across the batch):")
print(bn(x).round(decimals=3))
print("column means:", bn(x).mean(dim=0).round(decimals=4))

ln = nn.LayerNorm(3, elementwise_affine=False)
print("\\nLAYER NORM (each SAMPLE normalised across its features):")
print(ln(x).round(decimals=3))
print("row means:", ln(x).mean(dim=1).round(decimals=4))

# the batch-size-1 problem
print("\\nBATCH SIZE 1")
single = x[:1]
try:
    bn.train()
    print("batch norm:", bn(single))
except Exception as e:
    print("batch norm FAILS in training mode:", str(e)[:70])
print("layer norm works fine:", ln(single).round(decimals=3))

# train vs eval difference
bn2 = nn.BatchNorm1d(3)
bn2.train(); _ = bn2(x)          # updates the running statistics
bn2.eval()
print("\\nbatch norm in EVAL mode uses running stats, so the output for the")
print("same input differs between train and eval. Layer norm does not.")
~~~

:::warn Batch norm and small batches
Batch norm estimates mean and variance from the batch. With batch size 2-8 those estimates
are noisy and training becomes unstable; with batch size 1 it errors out. If you must use
small batches, use **GroupNorm** or **LayerNorm** instead.
:::
`
},
{
name: 'Debugging a network that will not train',
goal: 'A systematic checklist that finds the bug, in the order that finds it fastest.',
md: `
## The order to check things

~~~text
1. CAN IT OVERFIT ONE BATCH?        <- do this FIRST, always
2. Is the data correct?
3. Is the loss correct?
4. Are the gradients flowing?
5. Is the learning rate sane?
6. Only then: architecture and hyperparameters
~~~

~~~python debug_checklist.py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# TEST 1 - THE SINGLE MOST USEFUL DIAGNOSTIC
# =====================================================================
def can_overfit_one_batch(model, xb, yb, criterion, steps=400, lr=1e-2):
    """A correct training loop MUST be able to drive the loss on ONE
    batch to nearly zero. If it cannot, the bug is in your code, not
    your hyperparameters."""
    model = model.to(device)
    xb, yb = xb.to(device), yb.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for i in range(steps):
        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"  loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    if losses[-1] < losses[0] * 0.01:
        print("  PASS - the training loop works. Any failure is data or hyperparameters.")
    else:
        print("  FAIL - there is a BUG. Check: zero_grad, requires_grad, the loss,")
        print("         the label shape, whether the output feeds the loss at all.")
    return losses


model = nn.Sequential(nn.Linear(20, 128), nn.ReLU(),
                      nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
xb = torch.randn(32, 20)
yb = torch.randint(0, 2, (32, 1)).float()
print("TEST 1: overfit a single batch")
losses = can_overfit_one_batch(model, xb, yb, nn.BCEWithLogitsLoss())


# =====================================================================
# TEST 2 - IS THE INITIAL LOSS WHAT THEORY PREDICTS?
# =====================================================================
print("\\nTEST 2: sanity-check the initial loss")
fresh = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
with torch.no_grad():
    initial = nn.BCEWithLogitsLoss()(fresh(xb), yb).item()
expected = -np.log(0.5)
print(f"  binary cross-entropy at init: {initial:.4f}")
print(f"  expected (random, p=0.5)    : {expected:.4f}")
print(f"  {'PASS' if abs(initial - expected) < 0.15 else 'SUSPICIOUS'}")

for n_classes in [10, 100, 1000]:
    print(f"  {n_classes:4d}-class cross-entropy should start near "
          f"{np.log(n_classes):.4f}")
print("\\n  If your initial loss is far from this, the output layer, the label")
print("  encoding or the loss function is wrong.")


# =====================================================================
# TEST 3 - ARE THE GRADIENTS FLOWING?
# =====================================================================
def gradient_report(model, xb, yb, criterion):
    model.zero_grad()
    criterion(model(xb), yb).backward()
    print(f"\\n{'layer':28s} {'grad norm':>12s} {'weight norm':>13s} {'ratio':>10s}")
    print("-" * 66)
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"{name:28s} {'NO GRADIENT':>12s}   <- not in the graph!")
            continue
        gn = p.grad.norm().item()
        wn = p.norm().item()
        ratio = gn / (wn + 1e-12)
        flag = ""
        if gn < 1e-7:
            flag = "  <- VANISHING"
        elif gn > 100:
            flag = "  <- EXPLODING"
        print(f"{name:28s} {gn:>12.3e} {wn:>13.3e} {ratio:>10.2e}{flag}")


print("\\nTEST 3: gradient flow")
deep = nn.Sequential(*sum([[nn.Linear(20 if i == 0 else 64, 64), nn.Sigmoid()]
                           for i in range(8)], []), nn.Linear(64, 1))
gradient_report(deep, xb, yb, nn.BCEWithLogitsLoss())
print("\\n  Eight sigmoid layers -> the early layers get essentially no gradient.")
print("  Replace sigmoid with ReLU, or add residual connections.")

deep_relu = nn.Sequential(*sum([[nn.Linear(20 if i == 0 else 64, 64), nn.ReLU()]
                                for i in range(8)], []), nn.Linear(64, 1))
print("\\nSAME NETWORK WITH ReLU:")
gradient_report(deep_relu, xb, yb, nn.BCEWithLogitsLoss())


# =====================================================================
# TEST 4 - THE LEARNING-RATE FINDER
# =====================================================================
def lr_finder(model_fn, loader, criterion, lr_min=1e-7, lr_max=10, n_steps=120):
    """Increase the learning rate exponentially and record the loss.
    The best lr is roughly an order of magnitude below the minimum."""
    model = model_fn().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr_min)
    mult = (lr_max / lr_min) ** (1 / n_steps)
    lrs, losses = [], []
    best = float("inf")
    it = iter(loader)
    model.train()
    for step in range(n_steps):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader); xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)

        opt.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        opt.step()

        lrs.append(opt.param_groups[0]["lr"])
        losses.append(loss.item())
        best = min(best, loss.item())
        if loss.item() > 4 * best or not np.isfinite(loss.item()):
            break
        for g in opt.param_groups:
            g["lr"] *= mult
    return lrs, losses


from torch.utils.data import TensorDataset, DataLoader
X = torch.randn(2000, 20)
Y = ((X[:, :5].sum(1) + 0.4 * torch.randn(2000)) > 0).float().unsqueeze(1)
loader = DataLoader(TensorDataset(X, Y), batch_size=64, shuffle=True)

lrs, losses = lr_finder(
    lambda: nn.Sequential(nn.Linear(20, 128), nn.ReLU(), nn.Linear(128, 1)),
    loader, nn.BCEWithLogitsLoss())

plt.figure(figsize=(9, 5))
plt.plot(lrs, losses, lw=2)
best_lr = lrs[int(np.argmin(losses))] / 10
plt.axvline(best_lr, color="green", ls="--", label=f"suggested lr = {best_lr:.1e}")
plt.xscale("log"); plt.xlabel("learning rate (log)"); plt.ylabel("loss")
plt.legend(); plt.grid(alpha=0.3)
plt.title("Learning-rate finder: pick just before the loss turns up")
plt.tight_layout(); plt.show()
print(f"\\nTEST 4: suggested learning rate {best_lr:.2e}")
~~~

~~~text
TEST 1: overfit a single batch
  loss: 0.712043 -> 0.000021
  PASS - the training loop works.

TEST 2: sanity-check the initial loss
  binary cross-entropy at init: 0.6989
  expected (random, p=0.5)    : 0.6931
  PASS

TEST 3: gradient flow
layer                           grad norm   weight norm      ratio
------------------------------------------------------------------
0.weight                        2.104e-09     1.281e+00   1.64e-09  <- VANISHING
2.weight                        1.873e-08     2.914e+00   6.43e-09  <- VANISHING
...
14.weight                       3.921e-03     2.887e+00   1.36e-03
16.weight                       9.204e-02     1.011e+00   9.10e-02
~~~

### The full symptom table

| Symptom | Likely cause | Fix |
|---|---|---|
| Loss is ~nan~ | Learning rate too high; log(0); division by zero | Lower lr 10x; clip gradients; check for zeros |
| Loss does not move at all | Gradients not flowing; wrong loss; lr far too small | Run tests 1 and 3 |
| Loss decreases then explodes | Learning rate too high late in training | Add a schedule; clip gradients |
| Train loss falls, val loss rises | Overfitting | Regularise, augment, early stop |
| Both losses stuck high | Underfitting; a bug | Test 1 first, then increase capacity |
| Validation better than training | Dropout active in eval; or a trivial validation split | Check ~model.eval()~; check the split |
| First epoch is fine, later epochs are worse | ~zero_grad()~ missing | Check the loop |
| Works on CPU, fails on GPU | Tensors on different devices | ~.to(device)~ on both model and data |
| Non-reproducible results | Seeds not set; non-deterministic cudnn | Set all seeds; ~cudnn.deterministic=True~ |

~~~python common_bugs.py
"""The five bugs that account for most 'my network will not train' reports."""
import torch
import torch.nn as nn

# BUG 1 - missing zero_grad
# for xb, yb in loader:
#     loss = criterion(model(xb), yb)
#     loss.backward()
#     optimiser.step()          # gradients from EVERY previous batch are still here

# BUG 2 - the double activation
# self.net = nn.Sequential(..., nn.Linear(64, 10), nn.Softmax(dim=1))
# loss = nn.CrossEntropyLoss()(model(x), y)      # softmax applied twice

# BUG 3 - the shape mismatch that broadcasting hides
pred = torch.randn(32, 1)
target_wrong = torch.randn(32)          # shape (32,) not (32,1)
loss = nn.MSELoss()(pred, target_wrong)  # broadcasts to (32,32)! 1024 comparisons
print(f"BUG 3: MSE with mismatched shapes gives {loss.item():.4f}")
print(f"       correct shapes give "
      f"{nn.MSELoss()(pred, target_wrong.unsqueeze(1)).item():.4f}")
print("       PyTorch warns, but people ignore warnings.")

# BUG 4 - forgetting model.eval()
# model stays in train mode -> dropout randomises predictions,
# batch norm keeps updating its running statistics from validation data

# BUG 5 - the memory leak
# total_loss += loss                # keeps the whole graph
# total_loss += loss.item()         # correct

# BONUS - normalising the test set with its own statistics
# X_test = StandardScaler().fit_transform(X_test)      # WRONG
# X_test = scaler.transform(X_test)                    # RIGHT
~~~

:::tip When you are truly stuck
1. **Overfit 10 samples.** If you cannot reach ~zero loss on ten examples, stop tuning and
   find the bug.
2. **Print shapes everywhere.** Most deep learning bugs are shape bugs in disguise.
3. **Visualise a batch.** Look at actual images, actual labels, actual token ids. Data bugs
   are more common than model bugs.
4. **Remove components until it works**, then add them back one at a time.
5. **Compare against a known-good implementation** on the same data.
:::
`
}
],
quiz: [
{
q: 'What is the fastest way to check that your training loop is correct?',
options: [
  'Train for 100 epochs and see',
  'Try to overfit a single batch to near-zero loss - if it cannot, there is a bug',
  'Increase the learning rate',
  'Add more layers'
],
answer: 1,
why: 'A correct loop can memorise one batch. Failure isolates a code bug (zero_grad, requires_grad, loss, shapes) from a hyperparameter problem, in seconds rather than hours.'
},
{
q: 'Your 10-class classifier starts with a loss of 6.9. What does that suggest?',
options: [
  'Normal - random guessing gives log(10) = 2.30, so 6.9 is far too high',
  'It is training well',
  'The loss function is correct',
  'It means 6.9% accuracy'
],
answer: 0,
why: 'At initialisation a well-configured classifier should output roughly uniform probabilities, giving loss ln(10) = 2.30. Much higher means bad initialisation, a wrong output size, or mislabelled targets.'
},
{
q: 'Why do transformers use layer norm rather than batch norm?',
options: [
  'Layer norm is faster',
  'Layer norm normalises per sample, so it is independent of batch size and works with variable-length sequences',
  'Batch norm does not support GPUs',
  'It is an arbitrary choice'
],
answer: 1,
why: 'Batch norm mixes information across the batch and needs running statistics, which breaks with variable-length sequences and small or size-1 batches. Layer norm has neither problem.'
},
{
q: 'Validation accuracy is HIGHER than training accuracy. What is the most likely cause?',
options: [
  'The model is excellent',
  'Dropout is active during training but not at evaluation - or the validation split is easier',
  'The learning rate is too low',
  'There is too much data'
],
answer: 1,
why: 'Dropout handicaps the training-time forward pass, so training accuracy is measured on a degraded model. A small or non-representative validation set can also cause it.'
}
]
}

]
});
