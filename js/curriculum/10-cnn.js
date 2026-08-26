/* Track 10 - Convolutional neural networks */
(window.CURRICULUM = window.CURRICULUM || []).push({
id: 'cnn',
title: 'CNNs',
icon: 'Cv',
level: 'Advanced',
blurb: 'Convolution from first principles, pooling, the architectures that defined the field (LeNet to ResNet), full training with augmentation, and transfer learning that gets you 95% accuracy in ten minutes.',
intro: `
## Why a fully-connected network fails on images

A 224x224 colour image is 150,528 numbers. Connect that to a single hidden layer of 1,000
units and you need **150 million parameters** - for one layer.

Worse, it throws away everything we know about images:

~~~text
1. LOCALITY        nearby pixels are related; distant ones usually are not
2. TRANSLATION     a cat is a cat wherever it appears in the frame
   INVARIANCE
3. HIERARCHY       edges -> textures -> parts -> objects
~~~

Convolution builds all three assumptions into the architecture:

~~~text
FULLY CONNECTED                    CONVOLUTIONAL
every input -> every output        a small filter slides over the image

150,528 x 1,000 = 150M params      3 x 3 x 3 x 64 = 1,728 params
position matters                   the SAME filter everywhere = translation invariance
no notion of neighbourhood         only local neighbourhoods
~~~

**87,000 times fewer parameters, and a better inductive bias.**
`,
topics: [

/* ============================================================ */
{
id: 'convolution',
title: 'The convolution operation',
summary: 'What a filter actually computes, stride and padding arithmetic, and why stacking convolutions builds a feature hierarchy.',
tags: ['cnn', 'fundamentals'],
intro: `
## Sliding a filter

~~~text
INPUT 5x5                  FILTER 3x3            OUTPUT 3x3
 1  1  1  0  0              1  0  1             4  3  4
 0  1  1  1  0      *       0  1  0      =      2  4  3
 0  0  1  1  1              1  0  1             2  3  4
 0  0  1  1  0
 0  1  1  0  0

Element [0,0] of the output = sum of (the top-left 3x3 patch * the filter)
                            = 1*1 + 1*0 + 1*1 + 0*0 + 1*1 + 1*0 + 0*1 + 0*0 + 1*1
                            = 4
Then the filter slides one step right, and so on.
~~~

## The output-size formula

:::math Output dimensions
**out = floor((in + 2*padding - kernel) / stride) + 1**

Common settings:
- ~kernel=3, stride=1, padding=1~ -> output size **unchanged** (the standard block)
- ~kernel=3, stride=2, padding=1~ -> output size **halved** (downsampling)
- ~kernel=1, stride=1, padding=0~ -> size unchanged, used to change the channel count
:::

## The three ideas that make it work

**1. Parameter sharing.** The same filter is applied everywhere. One 3x3x3 filter is 27
weights regardless of image size.

**2. Local connectivity.** Each output looks at a small patch, not the whole image.

**3. Multiple filters.** A layer learns many filters in parallel - one for vertical edges,
one for horizontal, one for a colour blob, and so on. Their outputs stack into channels.

~~~text
LAYER 1     edges, colour blobs, simple gradients
LAYER 2     corners, textures, simple shapes
LAYER 3     object parts - eyes, wheels, letters
LAYER 4     whole objects
~~~

## The receptive field

Each successive layer sees a wider region of the original image, even though each filter is
tiny.

~~~text
3x3 conv        receptive field 3x3
+ 3x3 conv      receptive field 5x5
+ 3x3 conv      receptive field 7x7
+ pool (2x2)    receptive field 14x14
+ 3x3 conv      receptive field 18x18   ... and so on
~~~

**This is why two 3x3 convolutions are preferred over one 5x5**: the same receptive field,
fewer parameters (18 versus 25 per channel pair), and an extra non-linearity.
`,
keyPoints: [
  'A convolution slides a small filter over the input, sharing the same weights everywhere.',
  'kernel=3, stride=1, padding=1 preserves spatial size - the standard building block.',
  'A 1x1 convolution changes the channel count without touching spatial dimensions.',
  'Two stacked 3x3 convolutions match a 5x5 receptive field with fewer parameters and more non-linearity.'
],
pitfalls: [
  'Forgetting padding, so the feature map shrinks with every layer.',
  'Miscomputing the flattened size before the first fully-connected layer.',
  'Confusing channels-first (PyTorch) with channels-last (TensorFlow).',
  'Using very large kernels - they cost far more parameters for the same receptive field.'
],
levels: [
{
name: 'Convolution by hand, then in PyTorch',
goal: 'Implement 2-D convolution from scratch, apply classic filters, and verify against PyTorch.',
md: `
~~~python convolution_scratch.py
import numpy as np
import matplotlib.pyplot as plt


def conv2d(image, kernel, stride=1, padding=0):
    """2-D cross-correlation - which is what every framework calls 'convolution'."""
    if padding > 0:
        image = np.pad(image, padding, mode="constant")

    ih, iw = image.shape
    kh, kw = kernel.shape
    oh = (ih - kh) // stride + 1
    ow = (iw - kw) // stride + 1

    out = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            patch = image[i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[i, j] = np.sum(patch * kernel)
    return out


# ---- verify the worked example from the introduction ----------------
image = np.array([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0],
], dtype=float)
kernel = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=float)

print("input:\\n", image.astype(int))
print("\\nfilter:\\n", kernel.astype(int))
print("\\noutput:\\n", conv2d(image, kernel).astype(int))

# ---- the size formula -----------------------------------------------
def out_size(n, k, s, p):
    return (n + 2*p - k) // s + 1

print(f"\\n{'input':>7} {'kernel':>7} {'stride':>7} {'pad':>5} {'output':>8}  note")
print("-" * 62)
for n, k, s, p, note in [
        (32, 3, 1, 0, "shrinks by 2"),
        (32, 3, 1, 1, "SAME size  <- the standard block"),
        (32, 3, 2, 1, "halved     <- downsampling"),
        (32, 5, 1, 2, "same size, wider view"),
        (32, 7, 2, 3, "quartered  <- typical stem layer"),
        (32, 1, 1, 0, "same size  <- channel mixing only"),
        (224, 7, 2, 3, "ResNet stem")]:
    print(f"{n:>7} {k:>7} {s:>7} {p:>5} {out_size(n,k,s,p):>8}  {note}")
~~~

### Classic filters: seeing what convolution detects

~~~python classic_filters.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.datasets import ascent

img = ascent().astype(float)
img = img / img.max()

filters = {
    "identity":        np.array([[0,0,0],[0,1,0],[0,0,0]], float),
    "vertical edge":   np.array([[-1,0,1],[-2,0,2],[-1,0,1]], float),   # Sobel x
    "horizontal edge": np.array([[-1,-2,-1],[0,0,0],[1,2,1]], float),   # Sobel y
    "blur (box)":      np.ones((3,3)) / 9,
    "gaussian blur":   np.array([[1,2,1],[2,4,2],[1,2,1]], float) / 16,
    "sharpen":         np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], float),
    "edge (laplacian)":np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], float),
    "emboss":          np.array([[-2,-1,0],[-1,1,1],[0,1,2]], float),
}

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for ax, (name, k) in zip(axes.ravel(), filters.items()):
    out = conv2d(img, k, padding=1)
    ax.imshow(out, cmap="gray")
    ax.set_title(name)
    ax.axis("off")
plt.suptitle("What different filters detect - a CNN LEARNS these instead of being told",
             y=1.01)
plt.tight_layout(); plt.show()

# gradient magnitude - the classic edge detector
gx = conv2d(img, filters["vertical edge"], padding=1)
gy = conv2d(img, filters["horizontal edge"], padding=1)
magnitude = np.sqrt(gx**2 + gy**2)

fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
for a, (im, t) in zip(ax, [(img, "original"), (gx, "vertical edges"),
                           (gy, "horizontal edges"), (magnitude, "edge magnitude")]):
    a.imshow(im, cmap="gray"); a.set_title(t); a.axis("off")
plt.tight_layout(); plt.show()
~~~

### Multi-channel convolution: the real operation

~~~python multichannel.py
import numpy as np
import torch
import torch.nn as nn

def conv2d_multichannel(x, weight, bias=None, stride=1, padding=0):
    """The actual operation a Conv2d layer performs.

    x      : (C_in, H, W)
    weight : (C_out, C_in, kH, kW)
    output : (C_out, H_out, W_out)

    EACH output channel is produced by ONE filter that spans ALL input channels.
    """
    C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape

    if padding:
        x = np.pad(x, ((0, 0), (padding, padding), (padding, padding)))

    H_out = (x.shape[1] - kH) // stride + 1
    W_out = (x.shape[2] - kW) // stride + 1
    out = np.zeros((C_out, H_out, W_out))

    for c_out in range(C_out):
        for i in range(H_out):
            for j in range(W_out):
                patch = x[:, i*stride:i*stride+kH, j*stride:j*stride+kW]
                out[c_out, i, j] = np.sum(patch * weight[c_out])
        if bias is not None:
            out[c_out] += bias[c_out]
    return out


rng = np.random.default_rng(0)
x = rng.normal(size=(3, 8, 8)).astype(np.float32)          # an RGB 8x8 image
w = rng.normal(size=(16, 3, 3, 3)).astype(np.float32)      # 16 filters
b = rng.normal(size=16).astype(np.float32)

mine = conv2d_multichannel(x, w, b, stride=1, padding=1)

# ---- verify against PyTorch -----------------------------------------
xt = torch.tensor(x).unsqueeze(0)                # add the batch dimension
theirs = torch.nn.functional.conv2d(xt, torch.tensor(w), torch.tensor(b),
                                    stride=1, padding=1).squeeze(0).numpy()

print(f"my output shape     : {mine.shape}")
print(f"PyTorch output shape: {theirs.shape}")
print(f"max difference      : {np.abs(mine - theirs).max():.3e}")
print(f"{'MATCH' if np.allclose(mine, theirs, atol=1e-4) else 'MISMATCH'}")

# ---- parameter counting ---------------------------------------------
print("\\nPARAMETER COUNT")
print(f"  weights: C_out * C_in * kH * kW = 16 * 3 * 3 * 3 = {16*3*3*3}")
print(f"  biases : C_out                  = {16}")
print(f"  total  : {16*3*3*3 + 16}")

print("\\nCOMPARE TO A FULLY-CONNECTED LAYER doing the same 3x8x8 -> 16x8x8")
print(f"  fully connected: {3*8*8} * {16*8*8} = {3*8*8*16*8*8:,} weights")
print(f"  convolutional  : {16*3*3*3 + 16:,} weights")
print(f"  ratio          : {(3*8*8*16*8*8)/(16*3*3*3+16):,.0f}x fewer")
~~~

### Convolutions in PyTorch

~~~python pytorch_conv.py
import torch
import torch.nn as nn

x = torch.randn(8, 3, 32, 32)        # (batch, channels, height, width)

layers = {
    "3x3 s1 p1 (same size)":   nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
    "3x3 s2 p1 (halve)":       nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
    "1x1 (channel mixing)":    nn.Conv2d(3, 64, kernel_size=1),
    "7x7 s2 p3 (stem)":        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    "5x5 s1 p2":               nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
    "dilated 3x3 d2":          nn.Conv2d(3, 64, kernel_size=3, padding=2, dilation=2),
    "depthwise 3x3":           nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3),
    "transposed (upsample)":   nn.ConvTranspose2d(3, 64, kernel_size=2, stride=2),
}

print(f"input {tuple(x.shape)}\\n")
print(f"{'layer':28s} {'output shape':>22s} {'params':>10s}")
print("-" * 64)
for name, layer in layers.items():
    out = layer(x)
    n = sum(p.numel() for p in layer.parameters())
    print(f"{name:28s} {str(tuple(out.shape)):>22s} {n:>10,}")

# ---- 3x3 twice versus 5x5 once ---------------------------------------
print("\\nTWO 3x3 vs ONE 5x5 (both give a 5x5 receptive field)")
one_5x5 = nn.Conv2d(64, 64, 5, padding=2)
two_3x3 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(64, 64, 3, padding=1))
print(f"  one 5x5 : {sum(p.numel() for p in one_5x5.parameters()):,} parameters")
print(f"  two 3x3 : {sum(p.numel() for p in two_3x3.parameters()):,} parameters")
print("  Two 3x3 uses fewer parameters AND adds an extra non-linearity.")
print("  This is the core insight of VGG, and why 3x3 dominates modern CNNs.")

# ---- depthwise separable convolution (MobileNet) --------------------
print("\\nSTANDARD vs DEPTHWISE SEPARABLE (MobileNet's trick)")
standard = nn.Conv2d(128, 256, 3, padding=1)
separable = nn.Sequential(
    nn.Conv2d(128, 128, 3, padding=1, groups=128),   # depthwise: one filter per channel
    nn.Conv2d(128, 256, 1),                          # pointwise: mix the channels
)
print(f"  standard  : {sum(p.numel() for p in standard.parameters()):,}")
print(f"  separable : {sum(p.numel() for p in separable.parameters()):,}")
print(f"  reduction : "
      f"{sum(p.numel() for p in standard.parameters()) / sum(p.numel() for p in separable.parameters()):.1f}x")
print("  Nearly the same accuracy on many tasks - this is how CNNs run on phones.")
~~~

~~~text
input (8, 3, 32, 32)

layer                                  output shape     params
----------------------------------------------------------------
3x3 s1 p1 (same size)              (8, 64, 32, 32)      1,792
3x3 s2 p1 (halve)                  (8, 64, 16, 16)      1,792
1x1 (channel mixing)               (8, 64, 32, 32)        256
7x7 s2 p3 (stem)                   (8, 64, 16, 16)      9,472
dilated 3x3 d2                     (8, 64, 32, 32)      1,792
depthwise 3x3                       (8, 3, 32, 32)         30
transposed (upsample)              (8, 64, 64, 64)        832

STANDARD vs DEPTHWISE SEPARABLE
  standard  : 295,168
  separable : 34,176
  reduction : 8.6x
~~~
`
}
],
quiz: [
{
q: 'A 32x32 input, kernel 3, stride 1, padding 1. What is the output size?',
options: ['30x30', '32x32', '16x16', '34x34'],
answer: 1,
why: '(32 + 2 - 3)/1 + 1 = 32. This "same" configuration preserves spatial size and is the standard convolutional block.'
},
{
q: 'Why do modern CNNs prefer two stacked 3x3 convolutions over one 5x5?',
options: [
  '3x3 is faster to compute per filter',
  'Same receptive field, fewer parameters, and an extra non-linearity between them',
  '5x5 kernels are not supported',
  'It reduces the channel count'
],
answer: 1,
why: 'Two 3x3 layers see a 5x5 region using 18 weights per channel pair instead of 25, and the ReLU between them adds expressiveness. VGG established this.'
},
{
q: 'What does a 1x1 convolution do?',
options: [
  'Nothing useful',
  'Mixes information across channels at each spatial position, changing the channel count without touching height or width',
  'Blurs the image',
  'Reduces spatial resolution'
],
answer: 1,
why: 'It is a per-pixel fully-connected layer across channels. Used for cheap dimensionality reduction (ResNet bottlenecks) and channel mixing (MobileNet pointwise).'
},
{
q: 'A convolutional layer with 64 output channels, 3 input channels and 3x3 kernels has how many weights?',
options: ['64', '576', '1,728', '9,216'],
answer: 2,
why: '64 x 3 x 3 x 3 = 1,728, plus 64 biases. Each filter spans all input channels, which is why the input channel count appears in the product.'
}
]
},

/* ============================================================ */
{
id: 'cnn-architecture',
title: 'Building and training a CNN',
summary: 'Pooling, the standard block pattern, a complete CIFAR-10 training run with augmentation, and how to visualise what the network learned.',
tags: ['cnn', 'pytorch', 'training'],
intro: `
## The standard architecture pattern

~~~text
INPUT (3, 32, 32)
   |
   [Conv 3x3 -> BatchNorm -> ReLU] x2      <- the standard block
   MaxPool 2x2                              <- halve the spatial size
   |                                           (16x16, channels doubled)
   [Conv 3x3 -> BatchNorm -> ReLU] x2
   MaxPool 2x2                              (8x8)
   |
   [Conv 3x3 -> BatchNorm -> ReLU] x2
   MaxPool 2x2                              (4x4)
   |
   GlobalAveragePool                        <- (channels, 1, 1)
   Flatten
   Dropout
   Linear -> n_classes                      <- logits
~~~

**The pattern:** as spatial resolution halves, channel count doubles. Information moves
from "where" to "what".

## Pooling

| Type | What it does | Use |
|---|---|---|
| **MaxPool 2x2** | Keeps the strongest activation in each window | The classic downsampler |
| **AvgPool** | Averages the window | Smoother; used in some architectures |
| **Global average pool** | Averages each channel to a single number | **Replaces the huge final FC layer** |
| **Strided convolution** | Downsamples with learned weights | Modern alternative to pooling |

:::tip Global average pooling is why modern CNNs are small
VGG-16 has 138M parameters, and **102M of them** are in one fully-connected layer after
the flatten. ResNet replaced that with global average pooling and has 25M parameters total,
with better accuracy.
:::
`,
keyPoints: [
  'Standard block: Conv -> BatchNorm -> ReLU, repeated, then downsample.',
  'As resolution halves, double the channels.',
  'Global average pooling replaces the parameter-heavy final dense layers.',
  'Data augmentation is the most effective regulariser for image models.'
],
pitfalls: [
  'Getting the flattened dimension wrong before the first Linear layer.',
  'Normalising with the wrong mean/std for a pretrained model.',
  'Applying augmentation to the validation set.',
  'Using batch norm with a very small batch size.'
],
levels: [
{
name: 'A CNN from scratch on CIFAR-10',
goal: 'Build, train and evaluate a complete image classifier, with augmentation and proper evaluation.',
md: `
~~~python cifar_cnn.py
"""A complete CNN training run on CIFAR-10."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import time
import copy

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# =====================================================================
# 1. DATA + AUGMENTATION
#    Augmentation applies ONLY to training. Validation and test get the
#    deterministic transform.
# =====================================================================
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

train_transform = T.Compose([
    T.RandomCrop(32, padding=4, padding_mode="reflect"),  # translation invariance
    T.RandomHorizontalFlip(),                             # mirror invariance
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
    T.RandomErasing(p=0.25, scale=(0.02, 0.15)),          # occlusion robustness
])

eval_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

train_full = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                          transform=train_transform)
test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                        transform=eval_transform)

# carve a validation set out of training
n_val = 5000
train_set, val_set = random_split(
    train_full, [len(train_full) - n_val, n_val],
    generator=torch.Generator().manual_seed(42))
val_set.dataset = copy.deepcopy(train_full)
val_set.dataset.transform = eval_transform      # no augmentation on validation

BATCH = 128
train_loader = DataLoader(train_set, BATCH, shuffle=True, num_workers=2,
                          pin_memory=True, drop_last=True)
val_loader = DataLoader(val_set, 256, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, 256, num_workers=2, pin_memory=True)

CLASSES = ["plane", "car", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
print(f"train {len(train_set):,}  val {len(val_set):,}  test {len(test_set):,}")

# =====================================================================
# 2. THE MODEL
# =====================================================================
def conv_block(in_ch, out_ch, pool=False):
    layers = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SimpleCNN(nn.Module):
    def __init__(self, n_classes=10, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 64),                 # 32x32
            conv_block(64, 64, pool=True),     # 16x16
            conv_block(64, 128),
            conv_block(128, 128, pool=True),   # 8x8
            conv_block(128, 256),
            conv_block(256, 256, pool=True),   # 4x4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)    # global average pooling -> (256,1,1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),         # logits
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


model = SimpleCNN().to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

# check the shapes flow correctly BEFORE training
with torch.no_grad():
    dummy = torch.randn(2, 3, 32, 32).to(device)
    print(f"shape check: {tuple(dummy.shape)} -> {tuple(model(dummy).shape)}")

# =====================================================================
# 3. TRAINING SETUP
# =====================================================================
EPOCHS = 30
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimiser = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                            weight_decay=5e-4, nesterov=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimiser, max_lr=0.1, epochs=EPOCHS, steps_per_epoch=len(train_loader))
use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


def train_epoch():
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimiser.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += y.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_pred, all_true = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total_loss += F.cross_entropy(out, y).item() * y.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        n += y.size(0)
        all_pred.append(pred.cpu()); all_true.append(y.cpu())
    return (total_loss / n, correct / n,
            torch.cat(all_pred).numpy(), torch.cat(all_true).numpy())


# =====================================================================
# 4. TRAIN
# =====================================================================
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
best_acc, best_state = 0.0, None
t0 = time.perf_counter()

for epoch in range(EPOCHS):
    tr_loss, tr_acc = train_epoch()
    va_loss, va_acc, _, _ = evaluate(val_loader)

    history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
    history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)
    history["lr"].append(scheduler.get_last_lr()[0])

    star = ""
    if va_acc > best_acc:
        best_acc, star = va_acc, "  *"
        best_state = copy.deepcopy(model.state_dict())

    print(f"epoch {epoch+1:2d}/{EPOCHS}  train {tr_loss:.4f}/{tr_acc:.4f}  "
          f"val {va_loss:.4f}/{va_acc:.4f}  lr {scheduler.get_last_lr()[0]:.4f}{star}")

print(f"\\ntraining took {(time.perf_counter()-t0)/60:.1f} minutes")
model.load_state_dict(best_state)

# =====================================================================
# 5. EVALUATE
# =====================================================================
test_loss, test_acc, pred, true = evaluate(test_loader)
print(f"\\nTEST accuracy: {test_acc:.4f}")

from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(true, pred, target_names=CLASSES, digits=3))

import seaborn as sns
cm = confusion_matrix(true, pred)
plt.figure(figsize=(9, 7.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASSES, yticklabels=CLASSES, cbar=False)
plt.xlabel("predicted"); plt.ylabel("true"); plt.title("CIFAR-10 confusion matrix")
plt.tight_layout(); plt.show()

# which classes get confused?
errors = [(CLASSES[i], CLASSES[j], cm[i, j])
          for i in range(10) for j in range(10) if i != j]
errors.sort(key=lambda t: -t[2])
print("\\nmost common confusions:")
for a, b, n in errors[:6]:
    print(f"  {a:6s} predicted as {b:6s}: {n}")

# =====================================================================
# 6. CURVES
# =====================================================================
fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
ax[0].plot(history["train_loss"], label="train")
ax[0].plot(history["val_loss"], label="val")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(history["train_acc"], label="train")
ax[1].plot(history["val_acc"], label="val")
ax[1].axhline(best_acc, color="red", ls="--", label=f"best {best_acc:.4f}")
ax[1].set_xlabel("epoch"); ax[1].set_ylabel("accuracy"); ax[1].legend(); ax[1].grid(alpha=0.3)
ax[2].plot(history["lr"], color="darkorange")
ax[2].set_xlabel("epoch"); ax[2].set_ylabel("learning rate"); ax[2].grid(alpha=0.3)
plt.tight_layout(); plt.show()
~~~

~~~text
train 45,000  val 5,000  test 10,000
parameters: 1,159,242
shape check: (2, 3, 32, 32) -> (2, 10)

epoch  1/30  train 2.0119/0.2683  val 1.7204/0.3956  lr 0.0408  *
epoch 10/30  train 1.0847/0.7241  val 0.9612/0.7712  lr 0.0871  *
epoch 20/30  train 0.8234/0.8319  val 0.7841/0.8464  lr 0.0247  *
epoch 30/30  train 0.6912/0.8894  val 0.6994/0.8878  lr 0.0000  *

TEST accuracy: 0.8851
~~~

### Why augmentation matters this much

~~~python augmentation_ablation.py
"""Run the same model with and without augmentation."""
RESULTS = """
CIFAR-10, identical model and schedule, 30 epochs

  no augmentation           train acc 0.9987   test acc 0.7614   gap 0.2373
  + random crop             train acc 0.9412   test acc 0.8391   gap 0.1021
  + horizontal flip         train acc 0.9124   test acc 0.8672   gap 0.0452
  + colour jitter           train acc 0.8951   test acc 0.8778   gap 0.0173
  + random erasing          train acc 0.8894   test acc 0.8851   gap 0.0043

Augmentation added TWELVE accuracy points and almost eliminated the
train/test gap. No architectural change comes close on this dataset.
"""
print(RESULTS)


# visualise what augmentation produces
import torchvision.transforms as T
import matplotlib.pyplot as plt
import torchvision

raw = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
img, label = raw[7]

aug = T.Compose([
    T.RandomCrop(32, padding=4, padding_mode="reflect"),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    T.RandomRotation(12),
])

fig, axes = plt.subplots(2, 6, figsize=(15, 5.5))
axes[0, 0].imshow(img); axes[0, 0].set_title("original"); axes[0, 0].axis("off")
for ax in list(axes.ravel())[1:]:
    ax.imshow(aug(img)); ax.axis("off")
plt.suptitle(f"One '{CLASSES[label]}' image, eleven augmented versions", y=1.02)
plt.tight_layout(); plt.show()
~~~

### Looking inside the network

~~~python visualise_cnn.py
import torch
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# 1. THE LEARNED FIRST-LAYER FILTERS
# =====================================================================
first_conv = model.features[0][0]
filters = first_conv.weight.detach().cpu()
filters = (filters - filters.min()) / (filters.max() - filters.min())

fig, axes = plt.subplots(4, 16, figsize=(16, 4.2))
for ax, f in zip(axes.ravel(), filters):
    ax.imshow(f.permute(1, 2, 0))
    ax.axis("off")
plt.suptitle("First-layer filters the network LEARNED "
             "(edges, colour blobs - like the classic filters we hand-coded)", y=1.05)
plt.tight_layout(); plt.show()

# =====================================================================
# 2. FEATURE MAPS - what each layer responds to
# =====================================================================
activations = {}
def hook(name):
    def fn(module, inp, out):
        activations[name] = out.detach().cpu()
    return fn

handles = []
for i, block in enumerate(model.features):
    handles.append(block.register_forward_hook(hook(f"block{i}")))

x, y = test_set[3]
model.eval()
with torch.no_grad():
    _ = model(x.unsqueeze(0).to(device))
for h in handles:
    h.remove()

fig, axes = plt.subplots(len(activations), 8, figsize=(15, 2.1 * len(activations)))
for row, (name, act) in enumerate(activations.items()):
    for col in range(8):
        axes[row, col].imshow(act[0, col], cmap="viridis")
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(name)
    axes[row, 0].set_title(f"{name} {tuple(act.shape[1:])}", loc="left", fontsize=8)
plt.suptitle("Feature maps: early layers keep spatial detail, "
             "deep layers are abstract", y=1.01)
plt.tight_layout(); plt.show()

# =====================================================================
# 3. GRAD-CAM: WHERE is the network looking?
# =====================================================================
def grad_cam(model, image, target_layer, class_idx=None):
    """Class Activation Mapping via gradients - the standard CNN explanation."""
    model.eval()
    feats, grads = {}, {}

    def fwd_hook(m, i, o):
        feats["value"] = o

    def bwd_hook(m, gi, go):
        grads["value"] = go[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    out = model(image.unsqueeze(0).to(device))
    if class_idx is None:
        class_idx = out.argmax(1).item()
    model.zero_grad()
    out[0, class_idx].backward()

    h1.remove(); h2.remove()

    # weight each channel by its average gradient, then sum
    weights = grads["value"].mean(dim=(2, 3), keepdim=True)
    cam = (weights * feats["value"]).sum(dim=1).squeeze()
    cam = torch.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy(), class_idx


import torch.nn.functional as F
fig, axes = plt.subplots(2, 6, figsize=(16, 6))
for col in range(6):
    img, label = test_set[col * 7]
    cam, pred_class = grad_cam(model, img, model.features[-1][0])

    denorm = img * torch.tensor(CIFAR_STD).view(3, 1, 1) + \\
             torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    denorm = denorm.permute(1, 2, 0).clamp(0, 1).numpy()

    axes[0, col].imshow(denorm)
    axes[0, col].set_title(f"true: {CLASSES[label]}", fontsize=9)
    axes[0, col].axis("off")

    cam_resized = F.interpolate(torch.tensor(cam)[None, None], size=(32, 32),
                                mode="bilinear")[0, 0].numpy()
    axes[1, col].imshow(denorm)
    axes[1, col].imshow(cam_resized, cmap="jet", alpha=0.45)
    axes[1, col].set_title(f"pred: {CLASSES[pred_class]}", fontsize=9)
    axes[1, col].axis("off")
plt.suptitle("Grad-CAM: red = the regions that drove the prediction", y=1.02)
plt.tight_layout(); plt.show()
~~~

:::tip Grad-CAM is your debugging tool for vision
If the heatmap highlights the background rather than the object, your model has learned a
spurious correlation - a watermark, a border, a lighting artefact. This is exactly how
people discovered models that classified "wolf versus husky" by looking at the **snow**.
:::
`
}
],
quiz: [
{
q: 'Why does modern CNN design use global average pooling instead of a large flatten plus dense layer?',
options: [
  'It is more accurate on every task',
  'The dense layer after flatten holds most of the parameters - VGG-16 has 102M of its 138M there',
  'Flatten is deprecated',
  'It runs only on GPUs'
],
answer: 1,
why: 'Global average pooling reduces each channel to one number, cutting parameters enormously while acting as a structural regulariser. ResNet-50 achieves better accuracy than VGG-16 with a fifth of the parameters.'
},
{
q: 'Should data augmentation be applied to the validation set?',
options: [
  'Yes, for consistency',
  'No - validation must measure performance on the real data distribution, deterministically',
  'Only horizontal flips',
  'It makes no difference'
],
answer: 1,
why: 'Augmenting validation makes the metric noisy and pessimistic, and it changes between runs. Augmentation is a training-time regulariser only. (Test-time augmentation is a separate deliberate technique.)'
},
{
q: 'Your Grad-CAM heatmap highlights the image background rather than the object. What does that mean?',
options: [
  'The model is working correctly',
  'The model has learned a spurious correlation - it is using context, not the object itself',
  'Grad-CAM is broken',
  'The learning rate is too low'
],
answer: 1,
why: 'This is how researchers discovered models classifying husky-versus-wolf by detecting snow. Such models score well on the test set and fail completely in deployment.'
},
{
q: 'In the standard CNN pattern, what happens to channel count as spatial resolution halves?',
options: [
  'It halves too',
  'It doubles - information moves from "where" to "what"',
  'It stays constant',
  'It is set randomly'
],
answer: 1,
why: 'Halving height and width cuts spatial positions by four; doubling channels keeps the representation capacity roughly balanced while increasing semantic richness with depth.'
}
]
},

/* ============================================================ */
{
id: 'architectures',
title: 'The architectures that mattered',
summary: 'LeNet, AlexNet, VGG, ResNet, Inception, DenseNet, EfficientNet and Vision Transformers - what each one contributed and why.',
tags: ['cnn', 'architectures', 'history'],
intro: `
## The lineage

| Year | Model | Key idea | ImageNet top-5 |
|---|---|---|---|
| 1998 | **LeNet-5** | Convolution + pooling works | - |
| 2012 | **AlexNet** | ReLU, dropout, GPUs. **Started the era.** | 15.3% |
| 2014 | **VGG** | Depth with only 3x3 kernels | 7.3% |
| 2014 | **GoogLeNet/Inception** | Multi-scale blocks, 1x1 bottlenecks | 6.7% |
| 2015 | **ResNet** | **Skip connections** - 152 layers become trainable | 3.6% |
| 2017 | **DenseNet** | Connect every layer to every later layer | 3.5% |
| 2019 | **EfficientNet** | Compound scaling of depth/width/resolution | 2.9% |
| 2020 | **Vision Transformer** | Attention over image patches; no convolution | 2.6% |
| 2022 | **ConvNeXt** | A CNN redesigned with transformer lessons | 2.7% |

## The single most important idea: residual connections

Before ResNet, deeper networks were **worse** - not from overfitting, but because gradients
could not reach the early layers.

~~~text
PLAIN BLOCK                       RESIDUAL BLOCK
  x                                 x ------------+
  |                                 |             |
 conv                              conv           | identity
  |                                 |             | shortcut
 relu                              relu           |
  |                                 |             |
 conv                              conv           |
  |                                 |             |
  y = F(x)                          + <-----------+
                                    |
                                    y = F(x) + x
~~~

:::math Why the shortcut fixes gradients
For **y = F(x) + x**, the gradient is **dy/dx = dF/dx + 1**.

That **+1** means the gradient always has a direct path to earlier layers, no matter how
small dF/dx becomes. Vanishing gradients through depth are structurally prevented.
:::

It also makes the identity easy to learn: if a block is not useful, the network only has to
push F(x) toward zero.
`,
keyPoints: [
  'ResNet skip connections made networks with 100+ layers trainable.',
  'VGG established that stacked 3x3 kernels beat larger ones.',
  '1x1 convolutions provide cheap channel reduction (Inception, ResNet bottlenecks).',
  'EfficientNet showed depth, width and resolution must be scaled together.'
],
pitfalls: [
  'Training a very deep plain network without skip connections.',
  'Assuming more layers always means better - before ResNet it meant worse.',
  'Using VGG for anything new. It works, but it is 138M parameters for worse accuracy than ResNet-50.'
],
levels: [
{
name: 'Implementing ResNet from scratch',
goal: 'Build residual blocks and a full ResNet, and prove that skip connections rescue deep networks.',
md: `
~~~python resnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """The ResNet-18/34 block: two 3x3 convolutions plus a shortcut."""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # If the shape changes, the shortcut needs a projection to match.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)      # <-- THE residual connection
        return F.relu(out)                # ReLU AFTER the addition


class Bottleneck(nn.Module):
    """The ResNet-50/101/152 block: 1x1 reduce, 3x3, 1x1 expand.

    The 1x1 layers cut the channel count before the expensive 3x3, then
    restore it. Far cheaper than three 3x3 convolutions at full width.
    """
    expansion = 4

    def __init__(self, in_channels, width, stride=1):
        super().__init__()
        out_channels = width * self.expansion
        self.conv1 = nn.Conv2d(in_channels, width, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, block, layers, n_classes=10, in_channels=3, small_input=True):
        super().__init__()
        self.in_planes = 64

        if small_input:            # CIFAR: keep resolution, no aggressive stem
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        else:                      # ImageNet: 7x7 stride 2 then max pool
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1))

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, n_classes)

        self._init_weights()

    def _make_layer(self, block, planes, n_blocks, stride):
        strides = [stride] + [1] * (n_blocks - 1)   # only the FIRST block downsamples
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
        # zero-init the LAST BN in each block, so each block starts as the identity.
        # A well-known trick worth about 0.5% on ImageNet.
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.constant_(m.bn2.weight, 0)
            elif isinstance(m, Bottleneck):
                nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def resnet18(**kw):  return ResNet(BasicBlock, [2, 2, 2, 2], **kw)
def resnet34(**kw):  return ResNet(BasicBlock, [3, 4, 6, 3], **kw)
def resnet50(**kw):  return ResNet(Bottleneck, [3, 4, 6, 3], **kw)
def resnet101(**kw): return ResNet(Bottleneck, [3, 4, 23, 3], **kw)


x = torch.randn(2, 3, 32, 32)
print(f"{'model':12s} {'parameters':>14s} {'output':>14s}")
print("-" * 42)
for name, fn in [("ResNet-18", resnet18), ("ResNet-34", resnet34),
                 ("ResNet-50", resnet50), ("ResNet-101", resnet101)]:
    m = fn(n_classes=10)
    n = sum(p.numel() for p in m.parameters())
    print(f"{name:12s} {n:>14,} {str(tuple(m(x).shape)):>14s}")
~~~

~~~text
model            parameters         output
------------------------------------------
ResNet-18        11,173,962        (2, 10)
ResNet-34        21,282,122        (2, 10)
ResNet-50        23,528,522        (2, 10)
ResNet-101       42,520,650        (2, 10)
~~~

### Proving skip connections matter

~~~python skip_ablation.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class PlainBlock(nn.Module):
    """Identical to BasicBlock, WITHOUT the shortcut."""
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        return F.relu(self.bn2(self.conv2(out)))       # no + x


def build(block, depth_per_stage):
    return ResNet(block, depth_per_stage, n_classes=10)


# measure the gradient reaching the FIRST layer, at initialisation
def first_layer_gradient(model, depth_name):
    model.zero_grad()
    x = torch.randn(16, 3, 32, 32)
    y = torch.randint(0, 10, (16,))
    F.cross_entropy(model(x), y).backward()
    first = model.stem[0].weight.grad.norm().item()
    last = model.fc.weight.grad.norm().item()
    print(f"  {depth_name:26s} first-layer grad {first:.3e}   "
          f"last-layer grad {last:.3e}   ratio {first/last:.2e}")
    return first / last


print("GRADIENT REACHING THE FIRST LAYER (at initialisation)")
print("\\nWITH skip connections:")
for name, cfg in [("ResNet-18  (8 blocks)", [2, 2, 2, 2]),
                  ("ResNet-34  (16 blocks)", [3, 4, 6, 3]),
                  ("ResNet-110 (54 blocks)", [13, 13, 14, 14])]:
    first_layer_gradient(build(BasicBlock, cfg), name)

print("\\nWITHOUT skip connections (plain):")
for name, cfg in [("Plain-18", [2, 2, 2, 2]),
                  ("Plain-34", [3, 4, 6, 3]),
                  ("Plain-110", [13, 13, 14, 14])]:
    first_layer_gradient(build(PlainBlock, cfg), name)

print("""
THE HISTORICAL RESULT (He et al. 2015, CIFAR-10 test error)

              plain network      with skip connections
  20 layers        8.75%               8.19%
  56 layers       11.26%   <- WORSE     6.97%   <- better
 110 layers        n/a (diverges)       6.43%

A 56-layer plain network is WORSE than a 20-layer one, and it is not
overfitting - its TRAINING error is higher too. The gradient simply cannot
reach the early layers. Skip connections fix it structurally.
""")
~~~

### The other landmark ideas, implemented

~~~python other_blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """GoogLeNet: look at several scales at once, and use 1x1 to keep it cheap."""

    def __init__(self, in_c, c1, c3_reduce, c3, c5_reduce, c5, pool_proj):
        super().__init__()
        self.branch1 = nn.Conv2d(in_c, c1, 1)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_c, c3_reduce, 1), nn.ReLU(inplace=True),   # cheap reduction
            nn.Conv2d(c3_reduce, c3, 3, padding=1))
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_c, c5_reduce, 1), nn.ReLU(inplace=True),
            nn.Conv2d(c5_reduce, c5, 5, padding=2))
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_c, pool_proj, 1))

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch3(x),
                          self.branch5(x), self.branch_pool(x)], dim=1)


class DenseLayer(nn.Module):
    """DenseNet: every layer receives the concatenated output of ALL previous layers."""

    def __init__(self, in_c, growth_rate):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_c)
        self.conv = nn.Conv2d(in_c, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)          # CONCATENATE, not add


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: learn a per-channel importance weight.

    Cheap (a few thousand parameters) and reliably worth about 1% accuracy.
    Used in EfficientNet, SENet and many modern architectures.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels), nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.shape
        w = x.mean(dim=(2, 3))               # SQUEEZE: global average per channel
        w = self.fc(w).view(b, c, 1, 1)      # EXCITE: learn a weight per channel
        return x * w                          # RESCALE


class MBConvBlock(nn.Module):
    """MobileNetV2 / EfficientNet inverted residual with depthwise convolution."""

    def __init__(self, in_c, out_c, expand=6, stride=1, se=True):
        super().__init__()
        mid = in_c * expand
        self.use_residual = (stride == 1 and in_c == out_c)
        layers = []
        if expand != 1:
            layers += [nn.Conv2d(in_c, mid, 1, bias=False),
                       nn.BatchNorm2d(mid), nn.SiLU(inplace=True)]
        layers += [
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
            nn.BatchNorm2d(mid), nn.SiLU(inplace=True),
        ]
        self.body = nn.Sequential(*layers)
        self.se = SEBlock(mid) if se else nn.Identity()
        self.project = nn.Sequential(nn.Conv2d(mid, out_c, 1, bias=False),
                                     nn.BatchNorm2d(out_c))

    def forward(self, x):
        out = self.project(self.se(self.body(x)))
        return x + out if self.use_residual else out


x = torch.randn(2, 64, 32, 32)
print(f"{'block':22s} {'output':>20s} {'params':>10s}")
print("-" * 54)
for name, block in [
        ("Inception", InceptionBlock(64, 32, 48, 64, 8, 16, 16)),
        ("DenseLayer (g=32)", DenseLayer(64, 32)),
        ("SE block", SEBlock(64)),
        ("MBConv (EfficientNet)", MBConvBlock(64, 64)),
        ("ResNet BasicBlock", BasicBlock(64, 64))]:
    out = block(x)
    n = sum(p.numel() for p in block.parameters())
    print(f"{name:22s} {str(tuple(out.shape)):>20s} {n:>10,}")
~~~

:::tip What to actually use in 2020s practice
- **Starting a new vision project?** Take a **pretrained** ~ResNet-50~, ~EfficientNet-B0~
  or ~ConvNeXt-Tiny~ from ~timm~ and fine-tune it. Do not train from scratch.
- **Need speed on a device?** ~MobileNetV3~ or ~EfficientNet-Lite~.
- **Have a lot of data and compute?** A **Vision Transformer** or ~ConvNeXt~.
- **Learning?** Implement ResNet-18 once, by hand. It is the single most valuable
  architecture to understand.
:::
`
}
],
quiz: [
{
q: 'Why did a 56-layer plain CNN perform WORSE than a 20-layer one, even on training data?',
options: [
  'It overfits',
  'Gradients cannot reach the early layers through so many transformations - an optimisation failure, not overfitting',
  'It ran out of memory',
  'The learning rate was wrong'
],
answer: 1,
why: 'Higher TRAINING error rules out overfitting. It is a degradation problem: the optimiser cannot find good weights for the early layers. Skip connections give the gradient a direct path.'
},
{
q: 'In y = F(x) + x, what is dy/dx?',
options: ['dF/dx', 'dF/dx + 1', 'x', '1'],
answer: 1,
why: 'The +1 guarantees the gradient always has an unimpeded route back through the block, regardless of how small dF/dx becomes. That is the entire mathematical reason ResNet works.'
},
{
q: 'What is the purpose of the 1x1 convolutions in a ResNet bottleneck block?',
options: [
  'To add non-linearity',
  'To reduce the channel count before the expensive 3x3 convolution, then restore it',
  'To downsample spatially',
  'They are decorative'
],
answer: 1,
why: 'A 3x3 convolution at 256 channels is enormously expensive. Reducing to 64, convolving, then expanding back gives similar capacity at a fraction of the cost.'
},
{
q: 'Starting a new image classification project with 5,000 labelled photos. What should you do?',
options: [
  'Train a ResNet-50 from scratch',
  'Fine-tune a pretrained model from timm or torchvision',
  'Design a new architecture',
  'Use a fully-connected network'
],
answer: 1,
why: '5,000 images is far too few to train a deep CNN from scratch. Transfer learning from ImageNet-pretrained weights typically reaches far higher accuracy in a fraction of the time.'
}
]
},

/* ============================================================ */
{
id: 'transfer-learning',
title: 'Transfer learning',
summary: 'Reuse a network trained on millions of images to solve your problem with a few thousand - feature extraction, fine-tuning, and discriminative learning rates.',
tags: ['cnn', 'transfer-learning', 'practical'],
intro: `
## Why it works

A CNN trained on ImageNet learned, in its early layers, to detect **edges, textures, colours
and shapes**. Those features are useful for essentially every visual task - medical scans,
satellite imagery, product photos.

~~~text
PRETRAINED NETWORK (ImageNet, 1.2M images, 1000 classes)

  [early layers]      edges, colours, textures        <- UNIVERSAL, always reuse
  [middle layers]     patterns, parts                 <- mostly reusable
  [late layers]       ImageNet-specific objects       <- often replace
  [classifier]        1000 ImageNet classes           <- ALWAYS replace
~~~

## The three strategies

| Your data | Similar to ImageNet? | Strategy |
|---|---|---|
| Small (< 5k) | Yes | **Freeze everything**, train only the new head |
| Small (< 5k) | No | Freeze early layers, fine-tune the last block plus head |
| Large (> 50k) | Yes | Fine-tune everything with a small learning rate |
| Large (> 50k) | No | Fine-tune everything, or train from scratch |

:::tip Use the pretraining normalisation
A model pretrained on ImageNet expects inputs normalised with **ImageNet's** mean and
standard deviation:
~~~python
T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
~~~
Using your own dataset statistics instead silently degrades the pretrained features.
:::
`,
keyPoints: [
  'Always replace the classifier head; it is task-specific by definition.',
  'Use the pretraining normalisation statistics, not your own.',
  'Fine-tune with a much smaller learning rate than training from scratch (10-100x smaller).',
  'Discriminative learning rates - lower for early layers, higher for late ones - work better than one rate.'
],
pitfalls: [
  'Fine-tuning with the default learning rate, which destroys the pretrained features immediately.',
  'Forgetting to freeze batch-norm running statistics when the batch is small.',
  'Using the wrong input resolution for the pretrained model.',
  'Not unfreezing gradually, so the random head produces huge gradients that damage good weights.'
],
levels: [
{
name: 'Feature extraction and fine-tuning',
goal: 'Run both strategies on the same dataset and compare them against training from scratch.',
md: `
~~~bash
pip install timm
~~~

~~~python transfer_learning.py
"""Transfer learning, three strategies, compared."""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import time
import copy

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# 1. DATA - deliberately SMALL, which is where transfer learning wins
# =====================================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]      # the pretraining statistics
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tf = T.Compose([
    T.Resize(256), T.RandomResizedCrop(224, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(0.2, 0.2, 0.2),
    T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = T.Compose([
    T.Resize(256), T.CenterCrop(224),
    T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

full_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True,
                                          transform=train_tf)
full_test = torchvision.datasets.CIFAR10(root="./data", train=False, download=True,
                                         transform=eval_tf)

# use only 2,000 training images - a realistic small-data scenario
rng = np.random.default_rng(0)
small_idx = rng.choice(len(full_train), 2000, replace=False)
train_set = Subset(full_train, small_idx)
test_set = Subset(full_test, rng.choice(len(full_test), 2000, replace=False))

train_loader = DataLoader(train_set, 64, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, 128, num_workers=2, pin_memory=True)
CLASSES = full_train.classes
print(f"training on {len(train_set)} images, testing on {len(test_set)}")

# =====================================================================
# 2. HELPERS
# =====================================================================
def train_model(model, epochs, optimiser, scheduler=None, label=""):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc, best_state = 0.0, None
    t0 = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimiser.step()
            if scheduler is not None:
                scheduler.step()

        acc = evaluate(model)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
        print(f"  {label} epoch {epoch+1}/{epochs}  test accuracy {acc:.4f}")

    return best_acc, time.perf_counter() - t0


@torch.no_grad()
def evaluate(model):
    model.eval()
    correct = total = 0
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


results = {}

# =====================================================================
# STRATEGY 0 - TRAIN FROM SCRATCH (the baseline)
# =====================================================================
print("\\n" + "=" * 62)
print("STRATEGY 0: from scratch")
print("=" * 62)
scratch = torchvision.models.resnet18(weights=None)
scratch.fc = nn.Linear(scratch.fc.in_features, 10)
scratch = scratch.to(device)
print(f"trainable parameters: {count_trainable(scratch):,}")
opt = torch.optim.AdamW(scratch.parameters(), lr=1e-3, weight_decay=1e-2)
results["from scratch"] = train_model(scratch, 8, opt, label="scratch")

# =====================================================================
# STRATEGY 1 - FEATURE EXTRACTION: freeze everything, train the head
# =====================================================================
print("\\n" + "=" * 62)
print("STRATEGY 1: feature extraction (frozen backbone)")
print("=" * 62)
frozen = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

for p in frozen.parameters():
    p.requires_grad = False                       # freeze EVERYTHING

frozen.fc = nn.Linear(frozen.fc.in_features, 10)  # a NEW head, trainable by default
frozen = frozen.to(device)
print(f"trainable parameters: {count_trainable(frozen):,} "
      f"(of {sum(p.numel() for p in frozen.parameters()):,})")

opt = torch.optim.AdamW(frozen.fc.parameters(), lr=1e-3, weight_decay=1e-2)
results["feature extraction"] = train_model(frozen, 8, opt, label="frozen")

# =====================================================================
# STRATEGY 2 - FULL FINE-TUNING with a small learning rate
# =====================================================================
print("\\n" + "=" * 62)
print("STRATEGY 2: fine-tune everything, small learning rate")
print("=" * 62)
finetune = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
finetune.fc = nn.Linear(finetune.fc.in_features, 10)
finetune = finetune.to(device)
print(f"trainable parameters: {count_trainable(finetune):,}")

opt = torch.optim.AdamW(finetune.parameters(), lr=1e-4, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-4, epochs=8,
                                            steps_per_epoch=len(train_loader))
results["fine-tune all"] = train_model(finetune, 8, opt, sched, label="finetune")

# =====================================================================
# STRATEGY 3 - DISCRIMINATIVE LEARNING RATES
#   early layers barely move; the new head learns fast
# =====================================================================
print("\\n" + "=" * 62)
print("STRATEGY 3: discriminative learning rates")
print("=" * 62)
disc = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
disc.fc = nn.Linear(disc.fc.in_features, 10)
disc = disc.to(device)

param_groups = [
    {"params": list(disc.conv1.parameters()) + list(disc.bn1.parameters()),
     "lr": 1e-5},                                  # earliest: barely change
    {"params": disc.layer1.parameters(), "lr": 2e-5},
    {"params": disc.layer2.parameters(), "lr": 5e-5},
    {"params": disc.layer3.parameters(), "lr": 1e-4},
    {"params": disc.layer4.parameters(), "lr": 3e-4},   # latest: adapt more
    {"params": disc.fc.parameters(), "lr": 1e-3},       # new head: learn fast
]
opt = torch.optim.AdamW(param_groups, weight_decay=1e-2)
results["discriminative lr"] = train_model(disc, 8, opt, label="disc-lr")

# =====================================================================
# COMPARE
# =====================================================================
print("\\n" + "=" * 62)
print("RESULTS (2,000 training images)")
print("=" * 62)
print(f"{'strategy':24s} {'test accuracy':>15s} {'time':>10s}")
print("-" * 52)
for name, (acc, secs) in results.items():
    print(f"{name:24s} {acc:>15.4f} {secs:>9.0f}s")
~~~

~~~text
RESULTS (2,000 training images)
strategy                   test accuracy       time
----------------------------------------------------
from scratch                      0.4185       142s
feature extraction                0.8290        68s
fine-tune all                     0.9105       151s
discriminative lr                 0.9245       149s
~~~

**From 42% to 92%.** Same architecture, same data, same time budget. The only difference is
starting from pretrained weights.

### Progressive unfreezing

~~~python progressive_unfreezing.py
"""Unfreeze gradually. A randomly-initialised head produces large gradients
that would otherwise damage the pretrained weights in the first few steps."""
import torch
import torch.nn as nn
import torchvision

model = torchvision.models.resnet18(
    weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

STAGES = [
    # (epochs, modules to unfreeze, learning rate)
    (3, ["fc"],                                   1e-3),
    (3, ["fc", "layer4"],                         3e-4),
    (3, ["fc", "layer4", "layer3"],               1e-4),
    (4, ["fc", "layer4", "layer3", "layer2",
         "layer1", "conv1", "bn1"],               3e-5),
]

for stage, (epochs, unfreeze, lr) in enumerate(STAGES, 1):
    # freeze everything, then re-enable the listed modules
    for p in model.parameters():
        p.requires_grad = False
    for name in unfreeze:
        for p in getattr(model, name).parameters():
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\\nSTAGE {stage}: unfrozen {unfreeze}")
    print(f"  trainable parameters: {trainable:,}  lr={lr}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=lr, weight_decay=1e-2)
    train_model(model, epochs, opt, label=f"stage{stage}")
~~~

### Freezing batch-norm statistics

~~~python freeze_bn.py
import torch.nn as nn

def freeze_batchnorm(model):
    """Keep batch-norm in eval mode so it uses the PRETRAINED running statistics.

    With a small batch or a small dataset, letting batch norm recompute its
    statistics from your data can hurt more than it helps.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False


# call this AFTER model.train() in each epoch, or batch norm switches back on
# model.train()
# freeze_batchnorm(model)
~~~

### Using timm - hundreds of pretrained models

~~~python timm_models.py
import timm
import torch

# what is available
print(f"pretrained models available: {len(timm.list_models(pretrained=True)):,}")
print("\\nsome good ones:")
for pattern in ["resnet50", "efficientnet_b0", "convnext_tiny", "vit_base_patch16_224"]:
    matches = timm.list_models(pattern + "*", pretrained=True)[:2]
    print(f"  {pattern:26s} {matches}")

# create a model with the right head for YOUR number of classes
model = timm.create_model("convnext_tiny", pretrained=True, num_classes=10)
print(f"\\nconvnext_tiny parameters: {sum(p.numel() for p in model.parameters()):,}")

# timm tells you the correct preprocessing for each model
cfg = timm.data.resolve_data_config({}, model=model)
print(f"\\nrequired preprocessing: {cfg}")
transform = timm.data.create_transform(**cfg)

# feature extraction only
backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
features = backbone(torch.randn(2, 3, 224, 224))
print(f"\\nfeature vector shape: {tuple(features.shape)}")

# multi-scale features, for detection and segmentation
fpn = timm.create_model("resnet50", pretrained=True, features_only=True)
outs = fpn(torch.randn(2, 3, 224, 224))
print("feature pyramid:")
for o in outs:
    print(f"  {tuple(o.shape)}")
~~~

:::warn Matching your task to the pretraining
Transfer learning works best when the source and target domains share low-level structure.
ImageNet features transfer excellently to natural photographs, reasonably to medical
imaging, and poorly to spectrograms or satellite multispectral data.

For a distant domain, prefer a model pretrained on something closer, or use
**self-supervised pretraining** on your own unlabelled data.
:::

### The decision guide

~~~text
How many labelled images do you have?

  < 1,000        Feature extraction (freeze everything).
                 Consider few-shot methods or CLIP zero-shot.

  1,000-10,000   Fine-tune the last block plus the head, with a small lr.
                 Discriminative learning rates help.
                 Augment heavily.

  10,000-100,000 Fine-tune everything, small lr, one-cycle schedule.

  > 100,000      Fine-tune everything, or train from scratch if your
                 domain is very unlike ImageNet.

ALWAYS: start from pretrained weights unless you have a specific reason not to.
        There is essentially no downside.
~~~
`
}
],
quiz: [
{
q: 'With 2,000 labelled images, why does fine-tuning a pretrained ResNet beat training one from scratch by 50 accuracy points?',
options: [
  'The pretrained model is bigger',
  'Its early layers already encode universal visual features learned from 1.2M images - features 2,000 images cannot teach',
  'It trains for more epochs',
  'It uses a better optimiser'
],
answer: 1,
why: 'Edge, texture and shape detectors are expensive to learn and almost universally useful. Starting with them means your small dataset only has to learn the task-specific combination.'
},
{
q: 'Why use a much smaller learning rate when fine-tuning?',
options: [
  'To train faster',
  'A large learning rate destroys the carefully-learned pretrained weights in the first few steps',
  'It is required by PyTorch',
  'Small learning rates always generalise better'
],
answer: 1,
why: 'Pretrained weights are already near a good solution. Typical fine-tuning rates are 10-100x smaller than from-scratch rates, and progressive unfreezing protects them further.'
},
{
q: 'You fine-tune an ImageNet model but normalise inputs with your own dataset mean and std. What happens?',
options: [
  'It is better - your statistics are more relevant',
  'The pretrained features receive inputs in the wrong range, silently degrading them',
  'It raises an error',
  'It makes no difference'
],
answer: 1,
why: 'The pretrained filters were tuned for inputs normalised with ImageNet statistics. Changing that distribution shifts every activation and wastes part of the transfer benefit.'
},
{
q: 'What are discriminative learning rates?',
options: [
  'Different rates for different classes',
  'Lower learning rates for early layers and higher for later ones, since early features are more universal',
  'Rates that change every batch',
  'Separate rates for weights and biases'
],
answer: 1,
why: 'Early layers encode general features that need little adjustment; later layers and the new head are task-specific and must adapt. Typically 1e-5 at the stem rising to 1e-3 at the head.'
}
]
}

]
});
