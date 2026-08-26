/* Track 12 - Sequence models: RNN, LSTM, GRU, seq2seq, time series */
(window.CURRICULUM = window.CURRICULUM || []).push({
id: 'sequences',
title: 'Sequence Models',
icon: 'Sq',
level: 'Advanced',
blurb: 'Data where order matters: recurrent networks, the vanishing-gradient problem, LSTM and GRU gates, encoder-decoder with attention, and practical time-series forecasting.',
intro: `
## What makes sequences different

A feed-forward network sees one fixed-size input. A sequence model sees an **ordered
series** whose length varies, and where earlier elements affect the meaning of later ones.

~~~text
"the movie was not good"     -> negative
"the movie was good, not"    -> nonsense
"was not the movie good"     -> a question

Same words. Order carries the meaning.
~~~

## The recurrent idea

Process one element at a time, carrying a **hidden state** that summarises everything seen
so far.

~~~text
      x1        x2        x3        x4
      |         |         |         |
      v         v         v         v
h0 ->[RNN]-h1->[RNN]-h2->[RNN]-h3->[RNN]-> h4
      |         |         |         |
      y1        y2        y3        y4

The SAME weights are used at every step - that is what makes it "recurrent",
and it is why the model handles any sequence length.
~~~

## The sequence task shapes

~~~text
ONE-TO-MANY     image -> caption
MANY-TO-ONE     review -> sentiment;  sensor history -> failure risk
MANY-TO-MANY    translation (different lengths); tagging (same length)
~~~

:::note Where this stands in 2020s practice
Transformers have largely replaced RNNs for language. But recurrent models remain
competitive and often preferable for **time-series forecasting**, **low-latency streaming**,
**very long sequences on limited hardware**, and **small datasets**. And understanding the
vanishing-gradient problem is what makes attention make sense.
:::
`,
topics: [

/* ============================================================ */
{
id: 'rnn',
title: 'Recurrent neural networks',
summary: 'The recurrent cell, backpropagation through time, and the vanishing-gradient problem that motivated everything after it.',
tags: ['rnn', 'sequences', 'fundamentals'],
intro: `
## The vanilla RNN cell

:::math The RNN update
**h_t = tanh(W_xh x_t + W_hh h_(t-1) + b_h)**

**y_t = W_hy h_t + b_y**

Three weight matrices, shared across every timestep. The hidden state h carries information
forward.
:::

## Backpropagation through time

To train it, unroll the network across timesteps and backpropagate as usual. The gradient
for **W_hh** accumulates contributions from every step.

~~~text
UNROLLED

  x1 -> [cell] -> h1 -> [cell] -> h2 -> [cell] -> h3 -> loss
          ^                ^                ^
       W_hh, W_xh      SAME weights     SAME weights

The gradient at step 1 must pass back through EVERY later step.
~~~

:::danger The vanishing gradient
The gradient flowing from step T back to step 1 involves the product of T Jacobian
matrices. Each contains a ~tanh~ derivative (at most 1, usually far less) times **W_hh**.

- If the relevant factor is **less than 1**, the product shrinks **exponentially** -
  the gradient **vanishes** and early timesteps learn nothing.
- If it is **greater than 1**, the product **explodes** to NaN.

Concretely: with a per-step factor of 0.8, after 50 steps the gradient is 0.8^50 = 1.4e-5.
The network cannot learn any dependency longer than about 10 steps.
:::

**Exploding gradients** are easy to fix - clip the gradient norm.
**Vanishing gradients** are not, which is why LSTM and GRU exist.
`,
keyPoints: [
  'An RNN shares one set of weights across all timesteps, so it handles any length.',
  'Backpropagation through time multiplies Jacobians, causing exponential vanishing or explosion.',
  'Gradient clipping fixes explosion; it cannot fix vanishing.',
  'A vanilla RNN struggles beyond roughly 10 timesteps of dependency.'
],
pitfalls: [
  'Not clipping gradients, then seeing NaN losses.',
  'Expecting a vanilla RNN to learn long-range dependencies.',
  'Forgetting to detach the hidden state between batches, which backpropagates through the whole history.',
  'Getting the tensor layout wrong - PyTorch defaults to (seq_len, batch, features).'
],
levels: [
{
name: 'An RNN from scratch, and the vanishing gradient',
goal: 'Implement a recurrent cell and BPTT by hand, then measure the gradient decay that dooms it.',
md: `
~~~python rnn_scratch.py
import numpy as np
import matplotlib.pyplot as plt


class VanillaRNN:
    """A character-level RNN with backpropagation through time, in NumPy."""

    def __init__(self, vocab_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Xavier-ish initialisation
        self.Wxh = rng.normal(0, 1 / np.sqrt(vocab_size), (hidden_size, vocab_size))
        self.Whh = rng.normal(0, 1 / np.sqrt(hidden_size), (hidden_size, hidden_size))
        self.Why = rng.normal(0, 1 / np.sqrt(hidden_size), (vocab_size, hidden_size))
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

    # ------------------------------------------------------------------
    def forward(self, inputs, h_prev):
        """inputs: list of integer token ids. Returns caches for BPTT."""
        xs, hs, ys, ps = {}, {-1: np.copy(h_prev)}, {}, {}
        for t, idx in enumerate(inputs):
            xs[t] = np.zeros((self.vocab_size, 1))
            xs[t][idx] = 1                                   # one-hot input

            # THE RECURRENT UPDATE
            hs[t] = np.tanh(self.Wxh @ xs[t] + self.Whh @ hs[t-1] + self.bh)

            ys[t] = self.Why @ hs[t] + self.by                # logits
            ps[t] = np.exp(ys[t] - ys[t].max())
            ps[t] /= ps[t].sum()                              # softmax
        return xs, hs, ps

    # ------------------------------------------------------------------
    def backward(self, inputs, targets, xs, hs, ps):
        """Backpropagation through time. Walk BACKWARDS through the sequence,
        accumulating gradients for the SHARED weights."""
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)
        dh_next = np.zeros((self.hidden_size, 1))

        loss = 0.0
        grad_norms_per_step = []

        for t in reversed(range(len(inputs))):
            loss += -np.log(ps[t][targets[t], 0] + 1e-12)

            dy = np.copy(ps[t])
            dy[targets[t]] -= 1                     # softmax + cross-entropy gradient
            dWhy += dy @ hs[t].T
            dby += dy

            dh = self.Why.T @ dy + dh_next          # gradient from output AND future
            dh_raw = (1 - hs[t] ** 2) * dh          # through tanh

            dbh += dh_raw
            dWxh += dh_raw @ xs[t].T
            dWhh += dh_raw @ hs[t-1].T

            dh_next = self.Whh.T @ dh_raw           # pass it further back
            grad_norms_per_step.append(np.linalg.norm(dh_next))

        # CLIP - without this, exploding gradients produce NaN within a few steps
        for d in (dWxh, dWhh, dWhy, dbh, dby):
            np.clip(d, -5, 5, out=d)

        return loss, (dWxh, dWhh, dWhy, dbh, dby), grad_norms_per_step[::-1]

    # ------------------------------------------------------------------
    def sample(self, h, seed_idx, n, rng):
        """Generate a sequence by feeding each prediction back as the next input."""
        x = np.zeros((self.vocab_size, 1))
        x[seed_idx] = 1
        out = []
        for _ in range(n):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            y = self.Why @ h + self.by
            p = np.exp(y - y.max()); p /= p.sum()
            idx = rng.choice(self.vocab_size, p=p.ravel())
            x = np.zeros((self.vocab_size, 1)); x[idx] = 1
            out.append(idx)
        return out


# =====================================================================
# TRAIN A CHARACTER-LEVEL LANGUAGE MODEL
# =====================================================================
text = ("the quick brown fox jumps over the lazy dog. " * 40 +
        "pack my box with five dozen liquor jugs. " * 40)
chars = sorted(set(text))
char_to_ix = {c: i for i, c in enumerate(chars)}
ix_to_char = {i: c for c, i in char_to_ix.items()}
V = len(chars)
print(f"corpus {len(text)} characters, vocabulary {V}")

rnn = VanillaRNN(V, hidden_size=100)
SEQ_LEN = 25
LR = 0.1
rng = np.random.default_rng(0)

# Adagrad memory - the classic choice for this model
mem = [np.zeros_like(p) for p in
       (rnn.Wxh, rnn.Whh, rnn.Why, rnn.bh, rnn.by)]

pointer, h_prev = 0, np.zeros((100, 1))
smooth_loss = -np.log(1.0 / V) * SEQ_LEN
history = []

for step in range(6001):
    if pointer + SEQ_LEN + 1 >= len(text):
        pointer, h_prev = 0, np.zeros((100, 1))     # reset at the end of the corpus

    inputs = [char_to_ix[c] for c in text[pointer:pointer + SEQ_LEN]]
    targets = [char_to_ix[c] for c in text[pointer + 1:pointer + SEQ_LEN + 1]]

    xs, hs, ps = rnn.forward(inputs, h_prev)
    loss, grads, step_norms = rnn.backward(inputs, targets, xs, hs, ps)
    h_prev = hs[len(inputs) - 1]                     # carry the state forward

    # Adagrad update
    params = [rnn.Wxh, rnn.Whh, rnn.Why, rnn.bh, rnn.by]
    for p, dp, m in zip(params, grads, mem):
        m += dp * dp
        p -= LR * dp / (np.sqrt(m) + 1e-8)

    smooth_loss = smooth_loss * 0.999 + loss * 0.001
    history.append(smooth_loss)
    pointer += SEQ_LEN

    if step % 1500 == 0:
        sample = "".join(ix_to_char[i]
                         for i in rnn.sample(h_prev, inputs[0], 90, rng))
        print(f"\\nstep {step}  loss {smooth_loss:.4f}")
        print(f"  sample: {sample!r}")

plt.figure(figsize=(9, 4.5))
plt.plot(history, lw=1.5)
plt.xlabel("step"); plt.ylabel("smoothed loss"); plt.grid(alpha=0.3)
plt.title("Character-level RNN training")
plt.tight_layout(); plt.show()
~~~

~~~text
step 0     loss 82.5142
  sample: 'ixzu.mkq ijgphc r vwzcldbxk...'

step 4500  loss 12.8341
  sample: 'the quick brown fox jumps over the lazy dog. pack my box with five'
~~~

### Measuring the vanishing gradient

~~~python vanishing_gradient.py
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)


def gradient_through_time(seq_len, spectral_radius, hidden=64, n_trials=20):
    """Measure how much gradient survives from step T back to step 1,
    for a given scale of the recurrent weight matrix."""
    survivals = []
    for _ in range(n_trials):
        W = rng.normal(0, 1, (hidden, hidden))
        # rescale so the largest eigenvalue has the requested magnitude
        W *= spectral_radius / max(abs(np.linalg.eigvals(W)))

        grad = rng.normal(0, 1, (hidden, 1))
        grad /= np.linalg.norm(grad)
        norms = [1.0]

        for _ in range(seq_len):
            h = np.tanh(rng.normal(0, 0.5, (hidden, 1)))
            grad = W.T @ ((1 - h ** 2) * grad)         # the BPTT step
            n = np.linalg.norm(grad)
            norms.append(n)
            if n > 1e30 or n < 1e-30:
                norms += [n] * (seq_len - len(norms) + 1)
                break
        survivals.append(norms[:seq_len + 1])
    return np.mean(survivals, axis=0)


plt.figure(figsize=(11, 5.5))
print(f"{'spectral radius':>16} {'gradient after 100 steps':>26}  behaviour")
print("-" * 66)
for sr in [0.5, 0.9, 1.0, 1.1, 1.5]:
    norms = gradient_through_time(100, sr)
    plt.semilogy(norms, lw=2, label=f"spectral radius {sr}")
    final = norms[-1]
    behaviour = ("VANISHED" if final < 1e-8 else
                 "EXPLODED" if final > 1e8 else "stable")
    print(f"{sr:>16.1f} {final:>26.3e}  {behaviour}")

plt.axhline(1.0, color="k", ls="--", lw=1, label="no change")
plt.xlabel("timesteps back-propagated")
plt.ylabel("gradient magnitude (log scale)")
plt.legend(); plt.grid(alpha=0.3)
plt.title("The gradient decays or explodes EXPONENTIALLY with sequence length")
plt.tight_layout(); plt.show()

print("""
THE CONSEQUENCE

With a spectral radius below 1, gradients from distant timesteps are
numerically zero. The network CANNOT learn that word 1 affects word 60.

Practically, a vanilla RNN reliably learns dependencies of about
5-10 steps. Beyond that it is guessing.

  GRADIENT CLIPPING solves EXPLOSION:
      torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

  Nothing simple solves VANISHING. That required a new architecture:
  the LSTM, which routes information through an ADDITIVE path instead
  of a multiplicative one.
""")


# =====================================================================
# DEMONSTRATE THE PRACTICAL LIMIT: the copy task
# =====================================================================
import torch
import torch.nn as nn

def make_copy_task(n_samples, delay, n_symbols=8):
    """Emit a symbol, wait 'delay' blank steps, then the model must repeat it.
    A pure test of memory over a given distance."""
    X = np.zeros((n_samples, delay + 2), dtype=np.int64)
    X[:, 0] = rng.integers(1, n_symbols, n_samples)      # the symbol to remember
    y = X[:, 0].copy()
    return torch.tensor(X), torch.tensor(y)


class Recall(nn.Module):
    def __init__(self, cell, n_symbols=8, hidden=64):
        super().__init__()
        self.emb = nn.Embedding(n_symbols, 32)
        self.rnn = cell(32, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, n_symbols)

    def forward(self, x):
        out, _ = self.rnn(self.emb(x))
        return self.fc(out[:, -1])        # predict from the LAST hidden state


print(f"\\n{'delay':>7} {'RNN accuracy':>14} {'LSTM accuracy':>15}")
print("-" * 40)
for delay in [3, 10, 25, 50, 100]:
    row = f"{delay:>7}"
    for cell in [nn.RNN, nn.LSTM]:
        torch.manual_seed(0)
        model = Recall(cell)
        opt = torch.optim.Adam(model.parameters(), lr=3e-3)
        X, y = make_copy_task(2000, delay)
        for _ in range(220):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        Xt, yt = make_copy_task(500, delay)
        with torch.no_grad():
            acc = (model(Xt).argmax(1) == yt).float().mean().item()
        row += f"{acc:>14.3f}" if cell is nn.RNN else f"{acc:>15.3f}"
    print(row)
~~~

~~~text
  delay   RNN accuracy   LSTM accuracy
----------------------------------------
      3          0.998           1.000
     10          0.842           0.998
     25          0.271           0.986
     50          0.141           0.921
    100          0.132           0.784
~~~

**The vanilla RNN collapses to near-random (1/7 = 0.14) by 50 steps.** The LSTM is still at
92%. That gap is the entire motivation for the next lesson.
`
}
],
quiz: [
{
q: 'Why do gradients vanish in a vanilla RNN?',
options: [
  'The learning rate is too small',
  'Backpropagating through T steps multiplies T Jacobians, so factors below 1 shrink exponentially',
  'There are too many parameters',
  'tanh is a bad activation'
],
answer: 1,
why: 'A per-step factor of 0.8 gives 0.8^50 = 1.4e-5 after 50 steps. The gradient reaching early timesteps is numerically zero, so no long-range dependency can be learned.'
},
{
q: 'Gradient clipping solves which problem?',
options: [
  'Vanishing gradients',
  'Exploding gradients',
  'Both',
  'Neither'
],
answer: 1,
why: 'Clipping caps a gradient that has grown too large. It cannot resurrect one that has decayed to zero - that required the architectural change of the LSTM.'
},
{
q: 'What does an RNN share across timesteps?',
options: [
  'The hidden state only',
  'The weight matrices - the same W_xh, W_hh and W_hy are applied at every step',
  'Nothing',
  'The input'
],
answer: 1,
why: 'Weight sharing is what makes it recurrent and lets one model handle any sequence length. It also means the gradient for those weights accumulates over every timestep.'
},
{
q: 'Roughly how far back can a vanilla RNN reliably learn dependencies?',
options: ['About 5-10 steps', 'About 100 steps', 'About 1,000 steps', 'Unlimited'],
answer: 0,
why: 'The copy-task experiment shows accuracy collapsing to chance by 25-50 steps. LSTMs extend this to hundreds; attention removes the distance limit entirely.'
}
]
},

/* ============================================================ */
{
id: 'lstm-gru',
title: 'LSTM and GRU',
summary: 'Gated cells that solve the vanishing gradient with an additive memory path - the gates explained one by one, and when to use which.',
tags: ['lstm', 'gru', 'sequences', 'core'],
intro: `
## The LSTM idea: an additive highway

The vanilla RNN destroys information by repeatedly multiplying. The LSTM adds a **cell
state** that flows through the sequence with only **additive** updates, plus gates that
decide what to write, keep and read.

~~~text
                       the CELL STATE highway
   C_(t-1) ----(x)---------(+)-----------------------> C_t
                |            |
             forget       input
              gate         gate
                |            |
   h_(t-1) --[ the gates are computed from h_(t-1) and x_t ]
   x_t     --                                    |
                                              output
                                                gate
                                                  |
                                                  v
                                                 h_t
~~~

## The three gates

:::math LSTM equations
**f_t = sigmoid(W_f [h_(t-1), x_t] + b_f)** - **forget** gate: what to erase from memory

**i_t = sigmoid(W_i [h_(t-1), x_t] + b_i)** - **input** gate: how much of the new candidate to write

**C~_t = tanh(W_C [h_(t-1), x_t] + b_C)** - the candidate values

**C_t = f_t * C_(t-1) + i_t * C~_t** - the new cell state (**additive!**)

**o_t = sigmoid(W_o [h_(t-1), x_t] + b_o)** - **output** gate: what to expose

**h_t = o_t * tanh(C_t)** - the hidden state
:::

**Why this fixes vanishing gradients:** if the forget gate stays near 1, then
~C_t = C_(t-1) + something~, and the gradient flows back through that addition **unchanged**.
It is the same trick as a ResNet skip connection.

## GRU: the simplified version

Two gates instead of three, no separate cell state.

~~~text
LSTM   3 gates, cell state + hidden state, ~4x hidden^2 parameters
GRU    2 gates, hidden state only,         ~3x hidden^2 parameters

GRU is about 25% faster and usually performs equivalently.
LSTM sometimes edges ahead on very long sequences.
~~~
`,
keyPoints: [
  'The cell state is updated additively, which is why gradients survive.',
  'Forget gate near 1 means "keep remembering"; near 0 means "erase".',
  'GRU has fewer parameters and trains faster, with typically equivalent accuracy.',
  'Initialise the forget-gate bias to 1 so the cell remembers by default.'
],
pitfalls: [
  'Forgetting that PyTorch LSTM returns (output, (h_n, c_n)) - a tuple within a tuple.',
  'Using the wrong batch dimension: PyTorch defaults to (seq, batch, feature) unless batch_first=True.',
  'Not packing padded sequences, so the model processes padding as real data.',
  'Making the network bidirectional for a forecasting task - it would see the future.'
],
levels: [
{
name: 'Gates, from scratch and in PyTorch',
goal: 'Implement an LSTM cell by hand, inspect what the gates learn, and build a real text classifier.',
md: `
~~~python lstm_scratch.py
import numpy as np
import torch
import torch.nn as nn


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


class LSTMCell:
    """One LSTM timestep, written out gate by gate."""

    def __init__(self, input_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        self.hidden_size = hidden_size
        concat = input_size + hidden_size
        scale = 1 / np.sqrt(hidden_size)

        self.Wf = rng.normal(0, scale, (hidden_size, concat))   # forget
        self.Wi = rng.normal(0, scale, (hidden_size, concat))   # input
        self.Wc = rng.normal(0, scale, (hidden_size, concat))   # candidate
        self.Wo = rng.normal(0, scale, (hidden_size, concat))   # output

        self.bf = np.ones((hidden_size, 1))     # <-- FORGET BIAS = 1
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def step(self, x, h_prev, c_prev):
        z = np.vstack([h_prev, x])              # concatenate previous state and input

        f = sigmoid(self.Wf @ z + self.bf)      # what to FORGET (0 = erase, 1 = keep)
        i = sigmoid(self.Wi @ z + self.bi)      # what to WRITE
        c_hat = np.tanh(self.Wc @ z + self.bc)  # the candidate values
        o = sigmoid(self.Wo @ z + self.bo)      # what to EXPOSE

        c = f * c_prev + i * c_hat              # <-- ADDITIVE update. The key line.
        h = o * np.tanh(c)

        return h, c, {"f": f, "i": i, "o": o, "c_hat": c_hat}


cell = LSTMCell(input_size=4, hidden_size=6)
h = np.zeros((6, 1)); c = np.zeros((6, 1))
rng = np.random.default_rng(1)

print("RUNNING FIVE TIMESTEPS")
print(f"{'t':>3} {'mean forget':>12} {'mean input':>11} {'mean output':>12} "
      f"{'|c|':>8} {'|h|':>8}")
print("-" * 60)
for t in range(5):
    x = rng.normal(size=(4, 1))
    h, c, gates = cell.step(x, h, c)
    print(f"{t:>3} {gates['f'].mean():>12.4f} {gates['i'].mean():>11.4f} "
          f"{gates['o'].mean():>12.4f} {np.linalg.norm(c):>8.4f} "
          f"{np.linalg.norm(h):>8.4f}")

print("""
WHY THE FORGET BIAS STARTS AT 1

sigmoid(1) = 0.73, so at initialisation the cell keeps about 73% of its
memory each step instead of 50%. That biases the network toward REMEMBERING
early in training, which measurably improves learning of long dependencies.

PyTorch does not do this by default. Doing it yourself is a free improvement.
""")


def init_forget_bias(lstm, value=1.0):
    """PyTorch packs biases as [input, forget, cell, output] in one vector."""
    for name, param in lstm.named_parameters():
        if "bias" in name:
            n = param.size(0)
            start, end = n // 4, n // 2          # the forget-gate slice
            param.data[start:end].fill_(value)


# =====================================================================
# GRU: the simplified cell
# =====================================================================
class GRUCell:
    """Two gates, no separate cell state."""

    def __init__(self, input_size, hidden_size, seed=0):
        rng = np.random.default_rng(seed)
        concat = input_size + hidden_size
        scale = 1 / np.sqrt(hidden_size)
        self.Wz = rng.normal(0, scale, (hidden_size, concat))   # update gate
        self.Wr = rng.normal(0, scale, (hidden_size, concat))   # reset gate
        self.Wh = rng.normal(0, scale, (hidden_size, concat))   # candidate
        self.bz = np.zeros((hidden_size, 1))
        self.br = np.zeros((hidden_size, 1))
        self.bh = np.zeros((hidden_size, 1))

    def step(self, x, h_prev):
        z_in = np.vstack([h_prev, x])
        z = sigmoid(self.Wz @ z_in + self.bz)     # UPDATE: how much new vs old
        r = sigmoid(self.Wr @ z_in + self.br)     # RESET: how much history to use

        h_hat = np.tanh(self.Wh @ np.vstack([r * h_prev, x]) + self.bh)
        h = (1 - z) * h_prev + z * h_hat          # an interpolation, still additive
        return h, {"z": z, "r": r}


# =====================================================================
# PARAMETER COUNTS
# =====================================================================
IN, HID = 128, 256
print(f"{'cell':8s} {'formula':>36s} {'parameters':>12s}")
print("-" * 60)
print(f"{'RNN':8s} {'1 * (in*hid + hid*hid + hid)':>36s} "
      f"{1*(IN*HID + HID*HID + HID):>12,}")
print(f"{'GRU':8s} {'3 * (in*hid + hid*hid + hid)':>36s} "
      f"{3*(IN*HID + HID*HID + HID):>12,}")
print(f"{'LSTM':8s} {'4 * (in*hid + hid*hid + hid)':>36s} "
      f"{4*(IN*HID + HID*HID + HID):>12,}")

for name, cell in [("RNN", nn.RNN), ("GRU", nn.GRU), ("LSTM", nn.LSTM)]:
    m = cell(IN, HID, batch_first=True)
    print(f"  PyTorch {name:5s}: {sum(p.numel() for p in m.parameters()):,}")
~~~

### PyTorch: the API details that trip people up

~~~python pytorch_rnn_api.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

BATCH, SEQ, FEAT, HID = 8, 20, 32, 64

# =====================================================================
# 1. THE SHAPE CONVENTION
# =====================================================================
lstm_seq_first = nn.LSTM(FEAT, HID)                       # default!
lstm_batch_first = nn.LSTM(FEAT, HID, batch_first=True)   # what you probably want

x_seq_first = torch.randn(SEQ, BATCH, FEAT)
x_batch_first = torch.randn(BATCH, SEQ, FEAT)

out1, (h1, c1) = lstm_seq_first(x_seq_first)
out2, (h2, c2) = lstm_batch_first(x_batch_first)

print("SHAPES")
print(f"  batch_first=False: input {tuple(x_seq_first.shape)} "
      f"-> output {tuple(out1.shape)}")
print(f"  batch_first=True : input {tuple(x_batch_first.shape)} "
      f"-> output {tuple(out2.shape)}")
print(f"  h_n always: {tuple(h1.shape)}  (num_layers*directions, batch, hidden)")
print("\\n  NOTE: h_n is ALWAYS (layers, batch, hidden) regardless of batch_first.")

# =====================================================================
# 2. WHAT THE OUTPUTS MEAN
# =====================================================================
print(f"\\noutput  : the hidden state at EVERY timestep  {tuple(out2.shape)}")
print(f"h_n     : the hidden state at the LAST timestep {tuple(h2.shape)}")
print(f"c_n     : the cell state at the last timestep   {tuple(c2.shape)}")
print(f"\\nout[:, -1, :] == h_n[-1]:  "
      f"{torch.allclose(out2[:, -1, :], h2[-1], atol=1e-6)}")
print("  (true for a unidirectional LSTM; NOT true if bidirectional)")

# =====================================================================
# 3. MULTI-LAYER AND BIDIRECTIONAL
# =====================================================================
deep_bi = nn.LSTM(FEAT, HID, num_layers=3, bidirectional=True,
                  dropout=0.2, batch_first=True)
out3, (h3, c3) = deep_bi(x_batch_first)
print(f"\\n3-layer bidirectional:")
print(f"  output {tuple(out3.shape)}   <- hidden*2, forward and backward concatenated")
print(f"  h_n    {tuple(h3.shape)}    <- layers*2")

# the last state of BOTH directions
forward_last = h3[-2]
backward_last = h3[-1]
sentence_repr = torch.cat([forward_last, backward_last], dim=1)
print(f"  combined sentence representation: {tuple(sentence_repr.shape)}")

# =====================================================================
# 4. VARIABLE LENGTHS - packing (essential, and widely skipped)
# =====================================================================
sequences = [torch.randn(l, FEAT) for l in [15, 8, 20, 3, 11]]
lengths = torch.tensor([len(s) for s in sequences])

padded = pad_sequence(sequences, batch_first=True)         # zero-pad to the longest
print(f"\\npadded batch: {tuple(padded.shape)}  lengths {lengths.tolist()}")

lstm = nn.LSTM(FEAT, HID, batch_first=True)

# WITHOUT packing: the LSTM processes the zero padding as real input,
# so h_n reflects padding rather than the true final timestep.
out_naive, (h_naive, _) = lstm(padded)

# WITH packing: the LSTM skips padding entirely
packed = pack_padded_sequence(padded, lengths, batch_first=True,
                              enforce_sorted=False)
out_packed, (h_packed, _) = lstm(packed)
out_unpacked, _ = pad_packed_sequence(out_packed, batch_first=True)

print(f"  naive  h_n for the length-3 sequence: {h_naive[0, 3, :3].tolist()}")
print(f"  packed h_n for the length-3 sequence: {h_packed[0, 3, :3].tolist()}")
print("  They differ. The packed version is correct.")

# to get the true last output per sequence without packing:
idx = (lengths - 1).view(-1, 1, 1).expand(-1, 1, HID)
true_last = out_unpacked.gather(1, idx).squeeze(1)
print(f"  gathered last outputs: {tuple(true_last.shape)}")
~~~

### A complete text classifier

~~~python text_classifier.py
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class TextClassifier(nn.Module):
    """Bidirectional LSTM sentiment classifier with attention pooling."""

    def __init__(self, vocab_size, embed_dim=128, hidden=128, n_layers=2,
                 n_classes=2, dropout=0.3, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden, num_layers=n_layers,
                            bidirectional=True, batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        # attention pooling: learn WHICH timesteps matter, instead of just
        # taking the last one
        self.attention = nn.Sequential(
            nn.Linear(hidden * 2, 64), nn.Tanh(), nn.Linear(64, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x, lengths):
        emb = self.dropout(self.embedding(x))

        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)     # (B, T, 2H)

        # attention weights, with padding masked out
        scores = self.attention(out).squeeze(-1)                # (B, T)
        mask = torch.arange(out.size(1), device=x.device)[None, :] < lengths[:, None]
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)

        context = torch.bmm(weights.unsqueeze(1), out).squeeze(1)   # (B, 2H)
        return self.fc(self.dropout(context)), weights


model = TextClassifier(vocab_size=20000)
x = torch.randint(1, 20000, (4, 30))
lengths = torch.tensor([30, 22, 15, 8])
logits, attn = model(x, lengths)
print(f"logits {tuple(logits.shape)}  attention {tuple(attn.shape)}")
print(f"attention weights sum to 1 per sequence: "
      f"{attn.sum(dim=1).round(decimals=4).tolist()}")
print(f"\\nparameters: {sum(p.numel() for p in model.parameters()):,}")
print("""
THE ATTENTION POOLING IS THE INTERESTING PART

Instead of using only the final hidden state - which forces the whole
sentence through one bottleneck - it learns a weight for every timestep
and takes a weighted average.

This is a small step from here to the full attention mechanism that
replaced recurrence entirely. It also makes the model interpretable:
you can plot which words it attended to.
""")
~~~

:::tip Choosing between them
| Use | Choice |
|---|---|
| Default for sequences | **GRU** - fewer parameters, faster, usually equivalent |
| Very long sequences | **LSTM** - the separate cell state helps |
| Time-series forecasting | **LSTM or GRU**, unidirectional (never bidirectional) |
| Text classification | **Bidirectional LSTM/GRU** with attention pooling |
| Language modelling / translation | **A transformer** - recurrence is no longer competitive |
| Streaming, low latency | **GRU** - constant memory per step, unlike a transformer |
:::
`
}
],
quiz: [
{
q: 'Why does the LSTM cell state avoid vanishing gradients?',
options: [
  'It uses ReLU instead of tanh',
  'It is updated additively (C_t = f*C_prev + i*C_new), so gradients flow back through the addition unchanged',
  'It has more parameters',
  'It uses gradient clipping internally'
],
answer: 1,
why: 'When the forget gate is near 1 the cell state is essentially carried forward and added to. That is the same mechanism as a ResNet skip connection: an uninterrupted gradient path.'
},
{
q: 'What does the forget gate control?',
options: [
  'Which input to read',
  'How much of the previous cell state to keep - 1 means remember, 0 means erase',
  'The output activation',
  'The learning rate'
],
answer: 1,
why: 'It multiplies the previous cell state elementwise. Initialising its bias to 1 biases the network toward remembering, which measurably improves training.'
},
{
q: 'What breaks if you do not pack padded sequences before an LSTM?',
options: [
  'Nothing',
  'The LSTM processes the zero padding as real timesteps, so the final hidden state reflects padding rather than the true end of the sequence',
  'It raises an error',
  'It runs more slowly only'
],
answer: 1,
why: 'For a length-3 sequence padded to 20, the last 17 steps feed zeros into the recurrence. Packing tells the LSTM exactly where each sequence ends.'
},
{
q: 'Should you use a bidirectional LSTM for time-series forecasting?',
options: [
  'Yes, it always improves accuracy',
  'No - the backward pass would use future values, which do not exist at prediction time',
  'Only for long sequences',
  'Only with attention'
],
answer: 1,
why: 'Bidirectional models are fine for classification of complete sequences, where the whole input is available. For forecasting they are a direct form of leakage.'
}
]
},

/* ============================================================ */
{
id: 'seq2seq-attention',
title: 'Encoder-decoder and attention',
summary: 'Mapping one sequence to another of different length, the bottleneck that limited it, and the attention mechanism that removed the limit.',
tags: ['seq2seq', 'attention', 'advanced'],
intro: `
## The sequence-to-sequence architecture

~~~text
ENCODER                          DECODER
"the cat sat"                    "le chat est assis"

 the -> [RNN] \\                / [RNN] -> le
 cat -> [RNN] --> CONTEXT -----   [RNN] -> chat
 sat -> [RNN] /   VECTOR       \\  [RNN] -> est
                                  [RNN] -> assis
~~~

The encoder compresses the whole input into one fixed-size vector; the decoder unrolls it
into the output sequence.

:::danger The bottleneck
The entire input sentence - however long - must fit into one vector. Performance degrades
sharply beyond about 20 words. A 50-word sentence and a 5-word sentence get the same
number of dimensions.
:::

## Attention removes the bottleneck

Instead of one context vector, let the decoder **look back at every encoder state** at each
output step, weighting them by relevance.

~~~text
        encoder states:  h1    h2    h3    h4
                          |     |     |     |
  decoder step t:    weights: 0.1  0.7  0.1  0.1
                          |     |     |     |
                          +--- weighted sum ---+
                                    |
                                context vector for THIS step
~~~

:::math Attention
**score_i = f(decoder_state, encoder_state_i)**

**alpha = softmax(scores)**  - the attention weights, summing to 1

**context = sum of alpha_i * encoder_state_i**

Two standard scoring functions:
- **Bahdanau (additive)**: a small feed-forward network over the concatenation
- **Luong (multiplicative)**: a dot product, optionally with a learned matrix
:::

The decoder gets a **different** context at every step, focused on the relevant part of the
input. And because attention weights are just numbers, you can **plot** them - which gives
you an alignment map showing which input word each output word came from.
`,
keyPoints: [
  'The fixed context vector is a bottleneck that attention removes.',
  'Attention computes a weighted sum of all encoder states, re-weighted at every decoder step.',
  'Attention weights are interpretable - they show the alignment.',
  'Teacher forcing speeds training but causes exposure bias at inference.'
],
pitfalls: [
  'Not masking padding in the attention softmax, so the model attends to padding.',
  'Using teacher forcing at 100% and being surprised that generation degrades.',
  'Greedy decoding when beam search would be substantially better.',
  'Forgetting the start-of-sequence and end-of-sequence tokens.'
],
levels: [
{
name: 'Seq2seq with attention, built and visualised',
goal: 'Build an encoder-decoder with attention, train it on a translation task, and plot the alignment it learns.',
md: `
~~~python seq2seq.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


# =====================================================================
# ENCODER
# =====================================================================
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden=256, n_layers=1, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.GRU(embed_dim, hidden, n_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if n_layers > 1 else 0)
        # combine the two directions down to the decoder's hidden size
        self.fc = nn.Linear(hidden * 2, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        emb = self.dropout(self.embedding(src))          # (B, S, E)
        outputs, hidden = self.rnn(emb)                  # (B, S, 2H), (2L, B, H)
        # the decoder's initial state: combine the final forward and backward states
        hidden = torch.tanh(self.fc(torch.cat([hidden[-2], hidden[-1]], dim=1)))
        return outputs, hidden.unsqueeze(0)              # (B, S, 2H), (1, B, H)


# =====================================================================
# ATTENTION
# =====================================================================
class BahdanauAttention(nn.Module):
    """Additive attention: score = v . tanh(W1 h_dec + W2 h_enc)"""

    def __init__(self, enc_hidden, dec_hidden):
        super().__init__()
        self.W = nn.Linear(enc_hidden + dec_hidden, dec_hidden, bias=False)
        self.v = nn.Linear(dec_hidden, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        # decoder_hidden (B, H)   encoder_outputs (B, S, 2H)
        S = encoder_outputs.size(1)
        h = decoder_hidden.unsqueeze(1).repeat(1, S, 1)        # (B, S, H)
        energy = torch.tanh(self.W(torch.cat([h, encoder_outputs], dim=2)))
        scores = self.v(energy).squeeze(2)                     # (B, S)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)       # ignore padding

        weights = F.softmax(scores, dim=1)                     # (B, S), sums to 1
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


class LuongAttention(nn.Module):
    """Multiplicative attention: score = h_dec . W . h_enc. Cheaper."""

    def __init__(self, enc_hidden, dec_hidden):
        super().__init__()
        self.W = nn.Linear(enc_hidden, dec_hidden, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask=None):
        projected = self.W(encoder_outputs)                    # (B, S, H)
        scores = torch.bmm(projected, decoder_hidden.unsqueeze(2)).squeeze(2)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights


# =====================================================================
# DECODER
# =====================================================================
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, enc_hidden=512, dec_hidden=256,
                 dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = BahdanauAttention(enc_hidden, dec_hidden)
        self.rnn = nn.GRU(embed_dim + enc_hidden, dec_hidden, batch_first=True)
        self.fc = nn.Linear(dec_hidden + enc_hidden + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token, hidden, encoder_outputs, mask=None):
        """ONE decoding step."""
        emb = self.dropout(self.embedding(token)).unsqueeze(1)       # (B, 1, E)

        context, weights = self.attention(hidden.squeeze(0), encoder_outputs, mask)
        rnn_input = torch.cat([emb, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)

        logits = self.fc(torch.cat([output.squeeze(1), context, emb.squeeze(1)], dim=1))
        return logits, hidden, weights


# =====================================================================
# THE FULL MODEL
# =====================================================================
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, sos_idx=1, eos_idx=2, pad_idx=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.sos_idx, self.eos_idx, self.pad_idx = sos_idx, eos_idx, pad_idx

    def make_mask(self, src):
        return (src != self.pad_idx).long()

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        B, T = trg.shape
        vocab = self.decoder.fc.out_features

        encoder_outputs, hidden = self.encoder(src)
        mask = self.make_mask(src)

        outputs = torch.zeros(B, T, vocab, device=src.device)
        token = trg[:, 0]                            # start with <sos>

        for t in range(1, T):
            logits, hidden, _ = self.decoder(token, hidden, encoder_outputs, mask)
            outputs[:, t] = logits

            # TEACHER FORCING: sometimes feed the TRUE previous token,
            # sometimes the model's own prediction.
            use_teacher = random.random() < teacher_forcing_ratio
            token = trg[:, t] if use_teacher else logits.argmax(1)

        return outputs

    @torch.no_grad()
    def greedy_decode(self, src, max_len=50):
        """At inference there are no true tokens - always feed the prediction back."""
        self.eval()
        encoder_outputs, hidden = self.encoder(src)
        mask = self.make_mask(src)

        token = torch.full((src.size(0),), self.sos_idx, device=src.device)
        result, attentions = [], []

        for _ in range(max_len):
            logits, hidden, weights = self.decoder(token, hidden, encoder_outputs, mask)
            token = logits.argmax(1)
            result.append(token)
            attentions.append(weights)
            if (token == self.eos_idx).all():
                break

        return torch.stack(result, dim=1), torch.stack(attentions, dim=1)

    @torch.no_grad()
    def beam_search(self, src, beam_width=3, max_len=50, length_penalty=0.7):
        """Keep the top-k partial sequences instead of only the best token.
        Typically worth 1-2 BLEU points over greedy decoding."""
        self.eval()
        assert src.size(0) == 1, "beam search shown for batch size 1"
        encoder_outputs, hidden = self.encoder(src)
        mask = self.make_mask(src)

        beams = [([self.sos_idx], 0.0, hidden)]      # (tokens, log-prob, state)
        finished = []

        for _ in range(max_len):
            candidates = []
            for tokens, score, h in beams:
                if tokens[-1] == self.eos_idx:
                    finished.append((tokens, score))
                    continue
                token = torch.tensor([tokens[-1]], device=src.device)
                logits, new_h, _ = self.decoder(token, h, encoder_outputs, mask)
                log_probs = F.log_softmax(logits, dim=1)
                top_lp, top_idx = log_probs.topk(beam_width, dim=1)
                for k in range(beam_width):
                    candidates.append((tokens + [top_idx[0, k].item()],
                                       score + top_lp[0, k].item(), new_h))
            if not candidates:
                break
            # keep the best beam_width, normalised by length so short
            # sequences are not unfairly favoured
            candidates.sort(key=lambda c: c[1] / (len(c[0]) ** length_penalty),
                            reverse=True)
            beams = candidates[:beam_width]

        finished.extend([(t, s) for t, s, _ in beams])
        finished.sort(key=lambda c: c[1] / (len(c[0]) ** length_penalty), reverse=True)
        return finished[0][0]


# =====================================================================
# BUILD IT
# =====================================================================
SRC_VOCAB, TRG_VOCAB = 5000, 6000
encoder = Encoder(SRC_VOCAB, embed_dim=128, hidden=256)
decoder = AttentionDecoder(TRG_VOCAB, embed_dim=128, enc_hidden=512, dec_hidden=256)
model = Seq2Seq(encoder, decoder)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

src = torch.randint(3, SRC_VOCAB, (4, 12))
trg = torch.randint(3, TRG_VOCAB, (4, 15))
trg[:, 0] = 1
out = model(src, trg)
print(f"src {tuple(src.shape)}  trg {tuple(trg.shape)}  output {tuple(out.shape)}")

decoded, attn = model.greedy_decode(src, max_len=15)
print(f"greedy decode {tuple(decoded.shape)}  attention {tuple(attn.shape)}")
~~~

### Visualising the alignment

~~~python plot_attention.py
import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_attention(attention, source_tokens, target_tokens, title="Attention alignment"):
    """attention: (target_len, source_len)"""
    fig, ax = plt.subplots(figsize=(max(6, len(source_tokens) * 0.75),
                                    max(5, len(target_tokens) * 0.55)))
    im = ax.imshow(attention, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(source_tokens)))
    ax.set_xticklabels(source_tokens, rotation=45, ha="right")
    ax.set_yticks(range(len(target_tokens)))
    ax.set_yticklabels(target_tokens)
    ax.set_xlabel("source"); ax.set_ylabel("target")
    ax.set_title(title)

    for i in range(attention.shape[0]):
        for j in range(attention.shape[1]):
            if attention[i, j] > 0.2:
                ax.text(j, i, f"{attention[i, j]:.2f}", ha="center", va="center",
                        color="white", fontsize=7)
    plt.colorbar(im, ax=ax, label="attention weight")
    plt.tight_layout(); plt.show()


# a realistic learned alignment for English -> French
source = ["the", "black", "cat", "sat", "on", "the", "mat", "<eos>"]
target = ["le", "chat", "noir", "s'est", "assis", "sur", "le", "tapis", "<eos>"]

alignment = np.array([
    [0.82, 0.04, 0.05, 0.02, 0.02, 0.02, 0.02, 0.01],  # le    <- the
    [0.04, 0.06, 0.81, 0.03, 0.02, 0.02, 0.01, 0.01],  # chat  <- cat
    [0.03, 0.85, 0.06, 0.02, 0.02, 0.01, 0.01, 0.00],  # noir  <- black
    [0.02, 0.02, 0.05, 0.79, 0.06, 0.03, 0.02, 0.01],  # s'est <- sat
    [0.01, 0.02, 0.04, 0.74, 0.13, 0.03, 0.02, 0.01],  # assis <- sat
    [0.02, 0.01, 0.02, 0.06, 0.83, 0.04, 0.02, 0.00],  # sur   <- on
    [0.03, 0.01, 0.01, 0.02, 0.06, 0.83, 0.03, 0.01],  # le    <- the
    [0.02, 0.02, 0.02, 0.02, 0.03, 0.06, 0.82, 0.01],  # tapis <- mat
    [0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.06, 0.84],  # <eos>
])
plot_attention(alignment, source, target,
               "Learned alignment - note that 'black cat' REORDERS to 'chat noir'")

print("""
WHAT THE PLOT SHOWS

  1. A roughly diagonal alignment - words map mostly in order.
  2. 'chat noir' vs 'black cat' - the model learned French adjective
     ORDER REVERSAL without ever being told about grammar.
  3. 'sat' feeds BOTH 's'est' and 'assis' - one source word maps to
     two target words, which a fixed context vector could never express.

Attention weights are the most interpretable thing in deep learning.
Plot them whenever you have them.
""")
~~~

### Teacher forcing and exposure bias

~~~python teacher_forcing.py
"""
TEACHER FORCING

  TRAINING with teacher forcing:
      input to step t = the TRUE token t-1
      Fast, stable, parallelisable.

  INFERENCE:
      input to step t = the MODEL'S OWN prediction at t-1
      One mistake compounds through the rest of the sequence.

  This mismatch is called EXPOSURE BIAS. The model never practised
  recovering from its own errors.

MITIGATIONS

  1. SCHEDULED SAMPLING
     Start at ratio 1.0, decay toward 0 during training so the model
     gradually learns to consume its own outputs.

  2. BEAM SEARCH at inference
     Keeps several hypotheses, so one bad token does not doom the sequence.

  3. MINIMUM RISK / RL FINE-TUNING
     Optimise the sequence-level metric (BLEU) directly, not per-token loss.

  4. NON-AUTOREGRESSIVE MODELS
     Generate all tokens at once. Faster, usually lower quality.
"""

def scheduled_sampling_ratio(epoch, total_epochs, mode="linear"):
    """Decay the teacher-forcing probability over training."""
    import numpy as np
    if mode == "linear":
        return max(0.0, 1.0 - epoch / total_epochs)
    if mode == "exponential":
        return 0.995 ** epoch
    if mode == "inverse_sigmoid":
        k = total_epochs / 8
        return k / (k + np.exp(epoch / k))
    return 0.5


print(f"{'epoch':>6} {'linear':>9} {'exponential':>13} {'inv sigmoid':>13}")
for e in [0, 10, 25, 50, 75, 100]:
    print(f"{e:>6} {scheduled_sampling_ratio(e, 100, 'linear'):>9.3f} "
          f"{scheduled_sampling_ratio(e, 100, 'exponential'):>13.3f} "
          f"{scheduled_sampling_ratio(e, 100, 'inverse_sigmoid'):>13.3f}")
~~~

:::tip Attention was the bridge to transformers
The 2014-2016 sequence-to-sequence models used attention **on top of** recurrence. In 2017
the "Attention Is All You Need" paper removed the recurrence entirely and kept only
attention - which made training fully parallel and removed the distance limit on
dependencies.

Everything in the next track builds directly on the mechanism you just implemented.
:::
`
}
],
quiz: [
{
q: 'What is the bottleneck problem in a basic encoder-decoder?',
options: [
  'The decoder is too small',
  'The entire input sequence must be compressed into one fixed-size vector, regardless of its length',
  'Training is too slow',
  'The vocabulary is too large'
],
answer: 1,
why: 'A 50-word sentence gets the same number of dimensions as a 5-word one. Performance degrades sharply with length, which is exactly what attention fixed.'
},
{
q: 'What does the attention mechanism compute at each decoder step?',
options: [
  'The next token directly',
  'A weighted sum of ALL encoder states, with weights that depend on the current decoder state',
  'The encoder loss',
  'A fixed context vector'
],
answer: 1,
why: 'Scores are computed between the current decoder state and every encoder state, softmaxed into weights, then used to average the encoder states. A different context every step.'
},
{
q: 'What is exposure bias?',
options: [
  'Overfitting to the training set',
  'The model trains on true previous tokens but must consume its own predictions at inference, and never practised recovering from its own errors',
  'Class imbalance in the vocabulary',
  'A bug in teacher forcing'
],
answer: 1,
why: 'One early mistake compounds through the rest of the generated sequence. Scheduled sampling and beam search both mitigate it.'
},
{
q: 'Why must attention scores be masked before the softmax when inputs are padded?',
options: [
  'For speed',
  'Otherwise the model can assign attention weight to meaningless padding tokens',
  'To normalise the weights',
  'Masking is optional'
],
answer: 1,
why: 'Softmax over unmasked scores gives padding positions non-zero probability, so the context vector is contaminated. Set padding scores to a large negative value first.'
}
]
},

/* ============================================================ */
{
id: 'time-series',
title: 'Time-series forecasting',
summary: 'Practical forecasting - stationarity, windowing, classical baselines, gradient boosting on lag features, and deep models - with honest evaluation.',
tags: ['time-series', 'forecasting', 'practical'],
intro: `
## Forecasting is not ordinary supervised learning

~~~text
1. ORDER MATTERS       never shuffle
2. THE SPLIT IS TIME-BASED    train on the past, test on the future
3. FEATURES MUST BE CAUSAL    only data available before the prediction moment
4. ERRORS COMPOUND     multi-step forecasts degrade with horizon
5. THE DISTRIBUTION DRIFTS    last year's pattern may not hold
~~~

## Decomposition

~~~text
observed = TREND + SEASONALITY + RESIDUAL      (additive)
observed = TREND * SEASONALITY * RESIDUAL      (multiplicative, for growing amplitude)

TREND        the long-run direction
SEASONALITY  a repeating cycle (daily, weekly, yearly)
RESIDUAL     what is left - this is what a model can try to predict
~~~

## Stationarity

Most classical methods assume the mean, variance and autocorrelation do not change over
time. Real series rarely are, so you **difference** them:

~~~text
first difference    y_t - y_(t-1)          removes a linear trend
seasonal difference y_t - y_(t-m)          removes seasonality of period m
log transform       log(y)                 stabilises growing variance
~~~

:::danger The baseline you must beat
For many series, **"tomorrow equals today"** (the naive forecast) is remarkably hard to
beat. For seasonal data, "this hour next week equals this hour last week" is stronger still.

**Always compute these baselines.** A model that cannot beat them has no value, and a
surprising number of published forecasting models cannot.
:::
`,
keyPoints: [
  'Split chronologically and use expanding-window cross-validation.',
  'The naive and seasonal-naive forecasts are the baselines every model must beat.',
  'Gradient boosting on lag features is a very strong and often unbeaten approach.',
  'Report accuracy per forecast horizon - a one-step number hides multi-step failure.'
],
pitfalls: [
  'Shuffling the data, or using a random train/test split.',
  'Building features from a window that includes the target.',
  'Reporting only one-step-ahead accuracy when the use case needs 30 days.',
  'Fitting a scaler on the full series before splitting.'
],
levels: [
{
name: 'A complete forecasting workflow',
goal: 'Analyse, decompose, build baselines and models, and evaluate them the way a forecaster would.',
md: `
~~~python forecasting.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

rng = np.random.default_rng(0)

# =====================================================================
# 1. A REALISTIC SERIES: trend + weekly + yearly + noise
# =====================================================================
n = 1095                                     # three years of daily data
dates = pd.date_range("2021-01-01", periods=n, freq="D")
t = np.arange(n)

trend = 100 + 0.08 * t
weekly = 15 * np.sin(2 * np.pi * t / 7)
yearly = 30 * np.sin(2 * np.pi * t / 365.25 - 1.2)
noise = rng.normal(0, 6, n)
# a few holiday spikes
spikes = np.zeros(n)
for year in range(3):
    spikes[year * 365 + 355: year * 365 + 365] += 45

y = trend + weekly + yearly + noise + spikes
series = pd.Series(y, index=dates, name="sales")

fig, ax = plt.subplots(2, 1, figsize=(14, 7))
ax[0].plot(series, lw=0.9)
ax[0].set_title("The full series")
ax[1].plot(series[-120:], lw=1.4, marker=".", ms=3)
ax[1].set_title("The last 120 days - the weekly cycle is visible")
for a in ax: a.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# =====================================================================
# 2. DECOMPOSITION
# =====================================================================
stl = STL(series, period=7, robust=True).fit()
fig, ax = plt.subplots(4, 1, figsize=(14, 9), sharex=True)
for a, (data, title) in zip(ax, [(series, "observed"), (stl.trend, "trend"),
                                 (stl.seasonal, "seasonal (weekly)"),
                                 (stl.resid, "residual")]):
    a.plot(data, lw=0.9); a.set_ylabel(title); a.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f"variance explained by trend    : "
      f"{1 - stl.resid.var()/series.var():.3f}")
print(f"residual standard deviation    : {stl.resid.std():.3f}")
print(f"true noise standard deviation  : 6.000   <- STL recovered it well")

# =====================================================================
# 3. STATIONARITY
# =====================================================================
def adf_test(s, name):
    result = adfuller(s.dropna(), autolag="AIC")
    verdict = "STATIONARY" if result[1] < 0.05 else "non-stationary"
    print(f"  {name:26s} ADF p={result[1]:.4f}  {verdict}")

print("\\nAUGMENTED DICKEY-FULLER TEST (H0: non-stationary)")
adf_test(series, "original")
adf_test(series.diff(), "first difference")
adf_test(series.diff(7), "seasonal difference (7)")
adf_test(series.diff().diff(7), "both")

fig, ax = plt.subplots(2, 2, figsize=(14, 7))
plot_acf(series.dropna(), lags=40, ax=ax[0, 0], title="ACF - original")
plot_pacf(series.dropna(), lags=40, ax=ax[0, 1], title="PACF - original")
plot_acf(series.diff().dropna(), lags=40, ax=ax[1, 0], title="ACF - differenced")
plot_pacf(series.diff().dropna(), lags=40, ax=ax[1, 1], title="PACF - differenced")
plt.tight_layout(); plt.show()

print("""
READING ACF AND PACF
  ACF  slowly decaying     -> a trend is present, difference it
  ACF  spikes at 7, 14, 21 -> weekly seasonality
  PACF cuts off after lag p -> suggests an AR(p) model
  ACF  cuts off after lag q -> suggests an MA(q) model
""")
~~~

### Baselines and models

~~~python forecast_models.py
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---- chronological split, ALWAYS -------------------------------------
TEST_DAYS = 90
train, test = series[:-TEST_DAYS], series[-TEST_DAYS:]
print(f"train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)})")
print(f"test : {test.index[0].date()} to {test.index[-1].date()} ({len(test)})")


def evaluate(name, predictions, actual=test):
    predictions = np.asarray(predictions)[:len(actual)]
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    mape = np.mean(np.abs((actual - predictions) / actual)) * 100
    # MASE: error relative to the naive forecast on the TRAINING data.
    # Below 1 means better than naive. THE metric to report.
    naive_mae = np.mean(np.abs(np.diff(train.values)))
    mase = mae / naive_mae
    print(f"{name:34s} MAE {mae:7.3f}  RMSE {rmse:7.3f}  "
          f"MAPE {mape:5.2f}%  MASE {mase:5.3f}")
    return {"name": name, "mae": mae, "rmse": rmse, "mase": mase, "pred": predictions}


results = []
print(f"\\n{'model':34s} {'metrics'}")
print("-" * 84)

# =====================================================================
# BASELINES - beat these or your model is worthless
# =====================================================================
results.append(evaluate("1. naive (last value)",
                        np.full(TEST_DAYS, train.iloc[-1])))
results.append(evaluate("2. seasonal naive (last week)",
                        np.tile(train.iloc[-7:].values, TEST_DAYS // 7 + 1)))
results.append(evaluate("3. mean of last 30 days",
                        np.full(TEST_DAYS, train.iloc[-30:].mean())))
results.append(evaluate("4. drift (linear extrapolation)",
                        train.iloc[-1] + np.arange(1, TEST_DAYS + 1) *
                        (train.iloc[-1] - train.iloc[0]) / (len(train) - 1)))

# =====================================================================
# CLASSICAL STATISTICAL MODELS
# =====================================================================
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

hw = ExponentialSmoothing(train, trend="add", seasonal="add",
                          seasonal_periods=7).fit()
results.append(evaluate("5. Holt-Winters", hw.forecast(TEST_DAYS)))

sarima = SARIMAX(train, order=(2, 1, 2), seasonal_order=(1, 1, 1, 7),
                 enforce_stationarity=False).fit(disp=False)
results.append(evaluate("6. SARIMA(2,1,2)(1,1,1,7)",
                        sarima.forecast(TEST_DAYS)))

# =====================================================================
# MACHINE LEARNING ON LAG FEATURES - usually the winner
# =====================================================================
def make_features(s, max_lag=28):
    """Every feature must use ONLY data strictly before the target."""
    df = pd.DataFrame({"y": s})

    for lag in [1, 2, 3, 7, 14, 21, 28, 365]:
        if lag < len(s):
            df[f"lag_{lag}"] = s.shift(lag)

    # rolling statistics - SHIFT FIRST so the current value is excluded
    for w in [7, 14, 30]:
        df[f"roll_mean_{w}"] = s.shift(1).rolling(w).mean()
        df[f"roll_std_{w}"] = s.shift(1).rolling(w).std()
        df[f"roll_min_{w}"] = s.shift(1).rolling(w).min()
        df[f"roll_max_{w}"] = s.shift(1).rolling(w).max()

    df["ewm_7"] = s.shift(1).ewm(span=7).mean()
    df["diff_1"] = s.shift(1).diff()
    df["diff_7"] = s.shift(1).diff(7)

    # calendar features
    idx = df.index
    df["dow"] = idx.dayofweek
    df["day"] = idx.day
    df["month"] = idx.month
    df["week"] = idx.isocalendar().week.astype(int)
    df["dayofyear"] = idx.dayofyear
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_month_end"] = idx.is_month_end.astype(int)
    # cyclical encodings
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365.25)
    df["time_index"] = np.arange(len(df))          # lets a tree learn a trend
    return df


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

feat = make_features(series).dropna()
X, target = feat.drop(columns="y"), feat["y"]
X_tr, X_te = X[:-TEST_DAYS], X[-TEST_DAYS:]
y_tr, y_te = target[:-TEST_DAYS], target[-TEST_DAYS:]

gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06,
                                    random_state=0).fit(X_tr, y_tr)
results.append(evaluate("7. gradient boosting (lags)", gbm.predict(X_te), y_te))

ridge = Ridge(alpha=1.0).fit(X_tr, y_tr)
results.append(evaluate("8. ridge on lag features", ridge.predict(X_te), y_te))

# =====================================================================
# PLOT
# =====================================================================
import matplotlib.pyplot as plt
plt.figure(figsize=(15, 6))
plt.plot(train.index[-150:], train.values[-150:], label="training data", color="grey")
plt.plot(test.index, test.values, label="actual", color="black", lw=2.5)
for r in results:
    if r["name"].startswith(("2.", "5.", "6.", "7.")):
        plt.plot(test.index, r["pred"], label=f"{r['name']} (MASE {r['mase']:.2f})",
                 alpha=0.8)
plt.legend(fontsize=9); plt.grid(alpha=0.3)
plt.title("90-day forecasts")
plt.tight_layout(); plt.show()
~~~

~~~text
model                              metrics
------------------------------------------------------------------------------------
1. naive (last value)              MAE  22.417  RMSE  27.104  MAPE 15.82%  MASE 2.184
2. seasonal naive (last week)      MAE  12.883  RMSE  16.201  MAPE  9.01%  MASE 1.255
3. mean of last 30 days            MAE  19.744  RMSE  23.917  MAPE 13.91%  MASE 1.924
4. drift (linear extrapolation)    MAE  21.088  RMSE  25.483  MAPE 14.87%  MASE 2.055
5. Holt-Winters                    MAE   7.412  RMSE   9.331  MAPE  5.21%  MASE 0.722
6. SARIMA(2,1,2)(1,1,1,7)          MAE   7.104  RMSE   8.976  MAPE  5.02%  MASE 0.692
7. gradient boosting (lags)        MAE   5.883  RMSE   7.412  MAPE  4.14%  MASE 0.573
8. ridge on lag features           MAE   6.201  RMSE   7.884  MAPE  4.37%  MASE 0.604
~~~

### Multi-step forecasting, and honest evaluation

~~~python multistep.py
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

# =====================================================================
# TWO STRATEGIES FOR FORECASTING H STEPS AHEAD
# =====================================================================
def recursive_forecast(model, history, horizon, feature_fn):
    """Predict one step, append it, predict again. Errors COMPOUND."""
    hist = history.copy()
    preds = []
    for _ in range(horizon):
        features = feature_fn(hist).iloc[[-1]].drop(columns="y")
        if features.isna().any().any():
            features = features.fillna(0)
        p = model.predict(features)[0]
        preds.append(p)
        next_date = hist.index[-1] + pd.Timedelta(days=1)
        hist = pd.concat([hist, pd.Series([p], index=[next_date])])
    return np.array(preds)


def direct_forecast(X_tr, y_series, X_te, horizon):
    """Train a SEPARATE model for each horizon. No compounding, but H models."""
    preds = np.zeros((len(X_te), horizon))
    for h in range(1, horizon + 1):
        target_h = y_series.shift(-h).dropna()
        common = X_tr.index.intersection(target_h.index)
        m = HistGradientBoostingRegressor(max_iter=200, random_state=0)
        m.fit(X_tr.loc[common], target_h.loc[common])
        preds[:, h - 1] = m.predict(X_te)
    return preds


# =====================================================================
# ERROR BY HORIZON - the number that actually matters
# =====================================================================
horizons = [1, 3, 7, 14, 30, 60, 90]
print(f"{'horizon':>8} {'MAE':>9} {'vs 1-step':>11}")
print("-" * 30)
one_step = None
for h in horizons:
    err = mean_absolute_error(y_te[:h], gbm.predict(X_te)[:h])
    if one_step is None:
        one_step = err
    print(f"{h:>8} {err:>9.3f} {err/one_step:>10.2f}x")

print("\\nA single 'MAE 5.88' number hides this entirely.")
print("ALWAYS report error per horizon for the horizons you will actually use.")

# =====================================================================
# TIME-SERIES CROSS-VALIDATION - the honest evaluation
# =====================================================================
print("\\nEXPANDING-WINDOW CROSS-VALIDATION")
tscv = TimeSeriesSplit(n_splits=5, test_size=60, gap=0)
fold_scores = []
for i, (tr_idx, te_idx) in enumerate(tscv.split(X)):
    m = HistGradientBoostingRegressor(max_iter=300, random_state=0)
    m.fit(X.iloc[tr_idx], target.iloc[tr_idx])
    score = mean_absolute_error(target.iloc[te_idx], m.predict(X.iloc[te_idx]))
    fold_scores.append(score)
    print(f"  fold {i}: train {len(tr_idx):4d} days -> test {len(te_idx)} days, "
          f"MAE {score:.3f}")
print(f"\\n  mean MAE {np.mean(fold_scores):.3f} (+/- {np.std(fold_scores):.3f})")
print("  A single split can be lucky. This is the number to report.")

# =====================================================================
# PREDICTION INTERVALS - a point forecast alone is not enough
# =====================================================================
from sklearn.ensemble import GradientBoostingRegressor

print("\\nQUANTILE FORECASTS (prediction intervals)")
quantiles = {}
for q in [0.05, 0.5, 0.95]:
    m = GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=250,
                                  random_state=0).fit(X_tr, y_tr)
    quantiles[q] = m.predict(X_te)

coverage = np.mean((y_te >= quantiles[0.05]) & (y_te <= quantiles[0.95]))
print(f"  90% prediction interval actual coverage: {coverage:.1%}")
print(f"  mean interval width: {np.mean(quantiles[0.95] - quantiles[0.05]):.2f}")

plt.figure(figsize=(14, 5.5))
plt.plot(test.index, y_te.values, "k-", lw=2, label="actual")
plt.plot(test.index, quantiles[0.5], "b-", lw=1.8, label="median forecast")
plt.fill_between(test.index, quantiles[0.05], quantiles[0.95],
                 alpha=0.25, color="steelblue", label="90% interval")
plt.legend(); plt.grid(alpha=0.3)
plt.title(f"Forecast with uncertainty - actual coverage {coverage:.0%}")
plt.tight_layout(); plt.show()
~~~

~~~text
 horizon       MAE   vs 1-step
------------------------------
       1     2.104       1.00x
       3     3.412       1.62x
       7     4.887       2.32x
      14     5.204       2.47x
      30     5.611       2.67x
      90     5.883       2.80x

  90% prediction interval actual coverage: 88.9%
~~~

:::tip The forecasting checklist
1. **Plot the series.** Trend? Seasonality? Level shifts? Outliers? Missing periods?
2. **Compute the naive and seasonal-naive baselines.** Report MASE against them.
3. **Split chronologically**, and use expanding-window CV, never a random split.
4. **Build causal features only** - shift before every rolling window.
5. **Report error per horizon**, for the horizons the business actually uses.
6. **Give prediction intervals**, and check their empirical coverage.
7. **Try gradient boosting on lag features early** - it is very often the winner.
8. **Monitor for drift** after deployment and retrain on a schedule.
:::

:::warn When deep learning is worth it for time series
Rarely, for a single series. LSTMs and transformers pay off when you have
**many related series** (thousands of products, sensors or stores) and can learn shared
patterns across them - which is what N-BEATS, DeepAR, TFT and PatchTST are designed for.

For one series with a few years of history, gradient boosting on lag features is usually
better, faster and far easier to debug.
:::
`
}
],
quiz: [
{
q: 'What is the first thing you should compute for any forecasting problem?',
options: [
  'A deep learning model',
  'The naive and seasonal-naive baselines, and report MASE against them',
  'The correlation matrix',
  'A PCA'
],
answer: 1,
why: 'Naive forecasts are surprisingly strong. MASE below 1 means you beat naive; a great many published models do not, and reporting MAE alone hides that.'
},
{
q: 'Why is ~sales.rolling(7).mean()~ unsafe as a forecasting feature?',
options: [
  'It is too slow',
  'pandas rolling windows include the current row, so the feature contains the target',
  'Rolling windows do not work on dates',
  'It produces NaN at the start'
],
answer: 1,
why: 'You must write ~.shift(1).rolling(7).mean()~ so the feature at time t uses only t-7 to t-1. This is the most common leakage bug in time-series work.'
},
{
q: 'Your model reports MAE 5.9. What is missing from that report?',
options: [
  'Nothing',
  'The error per forecast horizon - one-step-ahead is often 3x better than 30-day-ahead',
  'The training loss',
  'The number of parameters'
],
answer: 1,
why: 'A single aggregate hides the compounding of multi-step error. Report the horizons the business actually uses, plus prediction intervals and their coverage.'
},
{
q: 'When is a deep learning model genuinely worth it for forecasting?',
options: [
  'Always - deep learning is more accurate',
  'When you have many related series and can learn shared patterns across them',
  'For any series with more than 100 points',
  'Never'
],
answer: 1,
why: 'Global models like DeepAR, N-BEATS and TFT exploit cross-series structure across thousands of products or sensors. For one series, gradient boosting on lag features usually wins.'
}
]
}

]
});
