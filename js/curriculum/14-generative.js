/* Track 14 - Generative models */
(window.CURRICULUM = window.CURRICULUM || []).push({
id: 'generative',
title: 'Generative Models',
icon: 'G',
level: 'Advanced',
blurb: 'Models that create rather than classify: autoencoders and VAEs, generative adversarial networks, and the diffusion models behind modern image generation.',
intro: `
## Discriminative versus generative

~~~text
DISCRIMINATIVE          learns P(y | x)      "given this image, which class?"
GENERATIVE              learns P(x)          "what does a valid image look like?"
                                             -> and can therefore SAMPLE new ones
~~~

## The four families

| Model | Idea | Strength | Weakness |
|---|---|---|---|
| **Autoencoder** | Compress and reconstruct | Simple; great for denoising and anomaly detection | Not truly generative |
| **VAE** | A probabilistic latent space | Smooth, controllable latent space; stable training | Blurry samples |
| **GAN** | Generator versus discriminator | Extremely sharp images | Unstable; mode collapse |
| **Diffusion** | Learn to reverse gradual noising | **Best quality and diversity** | Slow sampling |

~~~text
                 quality   diversity   training   sampling speed
  VAE              low       high        easy         fast
  GAN             high        low        hard         fast
  Diffusion       high       high      medium         slow
~~~

Diffusion won because it is the only family that gets **both** quality and diversity with
stable training. The slow sampling is being solved separately (distillation, consistency
models, flow matching).
`,
topics: [

/* ============================================================ */
{
id: 'autoencoders',
title: 'Autoencoders and VAEs',
summary: 'Compression as learning, the reparameterisation trick, the KL term, and what a smooth latent space actually buys you.',
tags: ['generative', 'vae', 'autoencoder'],
intro: `
## Autoencoder: learn by compressing

~~~text
  input  ->  ENCODER  ->  z  ->  DECODER  ->  reconstruction
  (784)                 (32)                    (784)
                         ^
                    the BOTTLENECK forces the network to keep
                    only what is needed to rebuild the input
~~~

Train it to minimise reconstruction error. The bottleneck **z** becomes a compressed,
learned representation.

**Uses**: denoising, anomaly detection (high reconstruction error = anomaly), dimensionality
reduction, pretraining. It is **not** generative - sample a random z and the decoder
produces noise, because the latent space has holes.

## VAE: make the latent space continuous

The encoder outputs a **distribution** (a mean and a variance) rather than a point, and the
loss adds a term pushing that distribution toward a standard normal.

:::math The VAE loss
**L = reconstruction loss + beta * KL( q(z|x) || N(0, I) )**

- **Reconstruction** - the decoder must rebuild the input.
- **KL divergence** - the encoded distributions must stay close to a standard normal,
  which fills the latent space and removes the holes.

The two terms fight. Reconstruction wants distinct, spread-out codes; KL wants everything
collapsed to N(0,1). **beta** sets the balance - and beta > 1 gives more disentangled but
blurrier results.
:::

## The reparameterisation trick

You cannot backpropagate through random sampling. So instead of sampling z from
N(mu, sigma), write:

:::math Reparameterisation
**z = mu + sigma * epsilon**, where **epsilon ~ N(0, 1)**

The randomness now lives in **epsilon**, which has no parameters. Gradients flow cleanly
through mu and sigma. This one substitution is what makes VAEs trainable.
:::
`,
keyPoints: [
  'An autoencoder learns compression; it is not generative on its own.',
  'A VAE encodes a distribution and regularises it toward N(0,1), making the space samplable.',
  'The reparameterisation trick moves randomness out of the gradient path.',
  'VAE samples are characteristically blurry because reconstruction loss rewards averaging.'
],
pitfalls: [
  'Posterior collapse - the KL term wins entirely and the decoder ignores z.',
  'Forgetting that the encoder outputs log-variance, not variance, for numerical stability.',
  'Expecting sharp images from a VAE - blurriness is inherent to pixel-wise reconstruction loss.',
  'Using an ordinary autoencoder for generation.'
],
levels: [
{
name: 'Autoencoder to VAE',
goal: 'Build both, see why a plain autoencoder cannot generate, and fix it with the VAE formulation.',
md: `
~~~python autoencoder_vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# DATA
# =====================================================================
transform = T.Compose([T.ToTensor()])
train_set = torchvision.datasets.MNIST("./data", train=True, download=True,
                                       transform=transform)
test_set = torchvision.datasets.MNIST("./data", train=False, download=True,
                                      transform=transform)
train_loader = DataLoader(train_set, 256, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, 256, num_workers=2)

LATENT = 2          # 2-D so we can PLOT the latent space


# =====================================================================
# 1. PLAIN AUTOENCODER
# =====================================================================
class Autoencoder(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent),                 # a POINT, not a distribution
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# =====================================================================
# 2. VARIATIONAL AUTOENCODER
# =====================================================================
class VAE(nn.Module):
    def __init__(self, latent=LATENT):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        # TWO heads: the mean and the LOG-variance of q(z|x)
        self.fc_mu = nn.Linear(256, latent)
        self.fc_logvar = nn.Linear(256, latent)

        self.decoder = nn.Sequential(
            nn.Linear(latent, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 784), nn.Sigmoid(),
            nn.Unflatten(1, (1, 28, 28)),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        """THE TRICK. z = mu + sigma * epsilon, epsilon ~ N(0,1).

        We predict LOG-variance so the network output is unconstrained;
        exp() then guarantees a positive standard deviation."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)              # the randomness lives HERE
        return mu + eps * std                    # gradients flow through mu and std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        return self.decoder(z), mu, logvar, z


def vae_loss(recon, x, mu, logvar, beta=1.0):
    """Reconstruction + beta * KL divergence."""
    # binary cross-entropy summed over pixels (MNIST pixels are in [0,1])
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)

    # closed-form KL between N(mu, sigma) and N(0, 1)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    return recon_loss + beta * kl, recon_loss, kl


# =====================================================================
# TRAIN BOTH
# =====================================================================
def train(model, epochs=20, is_vae=False, beta=1.0):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(epochs):
        model.train()
        totals = np.zeros(3)
        for x, _ in train_loader:
            x = x.to(device)
            opt.zero_grad()
            if is_vae:
                recon, mu, logvar, _ = model(x)
                loss, rl, kl = vae_loss(recon, x, mu, logvar, beta)
            else:
                recon, _ = model(x)
                loss = F.binary_cross_entropy(recon, x, reduction="sum") / x.size(0)
                rl, kl = loss, torch.tensor(0.0)
            loss.backward()
            opt.step()
            totals += [loss.item(), rl.item(), float(kl)]
        totals /= len(train_loader)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  epoch {ep:2d}  loss {totals[0]:8.3f}  "
                  f"recon {totals[1]:8.3f}  KL {totals[2]:7.3f}")
    return model


print("TRAINING PLAIN AUTOENCODER")
ae = train(Autoencoder(), epochs=20, is_vae=False)
print("\\nTRAINING VAE")
vae = train(VAE(), epochs=20, is_vae=True, beta=1.0)


# =====================================================================
# THE CRITICAL COMPARISON: can they GENERATE?
# =====================================================================
@torch.no_grad()
def sample_from_prior(model, n=8, is_vae=False):
    """Sample z from N(0,1) and decode. A generative model should produce
    plausible outputs; a plain autoencoder will not."""
    model.eval()
    z = torch.randn(n, LATENT).to(device)
    return model.decoder(z).cpu()

fig, axes = plt.subplots(2, 8, figsize=(15, 4.2))
for j, img in enumerate(sample_from_prior(ae, 8)):
    axes[0, j].imshow(img.squeeze(), cmap="gray"); axes[0, j].axis("off")
for j, img in enumerate(sample_from_prior(vae, 8)):
    axes[1, j].imshow(img.squeeze(), cmap="gray"); axes[1, j].axis("off")
axes[0, 0].set_title("Autoencoder: random z -> noise", loc="left", fontsize=11)
axes[1, 0].set_title("VAE: random z -> digits", loc="left", fontsize=11)
plt.tight_layout(); plt.show()


# =====================================================================
# WHY - look at the latent spaces
# =====================================================================
@torch.no_grad()
def encode_test_set(model, is_vae=False):
    model.eval()
    zs, labels = [], []
    for x, y in test_loader:
        x = x.to(device)
        if is_vae:
            mu, _ = model.encode(x)
            zs.append(mu.cpu())
        else:
            zs.append(model.encoder(x).cpu())
        labels.append(y)
    return torch.cat(zs).numpy(), torch.cat(labels).numpy()

z_ae, y_ae = encode_test_set(ae)
z_vae, y_vae = encode_test_set(vae, is_vae=True)

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
for a, (z, y, title) in zip(ax, [(z_ae, y_ae, "Autoencoder latent space"),
                                 (z_vae, y_vae, "VAE latent space")]):
    sc = a.scatter(z[:, 0], z[:, 1], c=y, cmap="tab10", s=3, alpha=0.6)
    circle = plt.Circle((0, 0), 3, fill=False, color="red", ls="--", lw=2)
    a.add_patch(circle)
    a.set_title(f"{title}\\nred circle = where N(0,1) samples land")
    a.set_aspect("equal")
plt.colorbar(sc, ax=ax, label="digit", fraction=0.02)
plt.show()

print(f"""
THE ANSWER IS IN THOSE PLOTS

Autoencoder latent range: x [{z_ae[:,0].min():.1f}, {z_ae[:,0].max():.1f}]
VAE latent range        : x [{z_vae[:,0].min():.1f}, {z_vae[:,0].max():.1f}]

The autoencoder spreads codes wherever it likes, with large empty gaps.
Sampling z ~ N(0,1) lands in a gap, and the decoder has never seen anything
there, so it outputs noise.

The KL term forces the VAE's codes to fill a roughly standard-normal ball.
Every point in that ball decodes to something plausible. THAT is what makes
it generative.
""")

# ---- walk through the latent space -----------------------------------
@torch.no_grad()
def latent_grid(model, n=18, span=2.5):
    model.eval()
    grid = torch.zeros(n * 28, n * 28)
    xs = torch.linspace(-span, span, n)
    ys = torch.linspace(span, -span, n)
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            z = torch.tensor([[xv, yv]], dtype=torch.float32).to(device)
            img = model.decoder(z).cpu().squeeze()
            grid[i*28:(i+1)*28, j*28:(j+1)*28] = img
    return grid

plt.figure(figsize=(10, 10))
plt.imshow(latent_grid(vae), cmap="gray")
plt.axis("off")
plt.title("A grid over the VAE latent space - digits morph smoothly into each other")
plt.tight_layout(); plt.show()

# ---- interpolate between two real digits -----------------------------
@torch.no_grad()
def interpolate(model, x1, x2, steps=10):
    model.eval()
    mu1, _ = model.encode(x1.unsqueeze(0).to(device))
    mu2, _ = model.encode(x2.unsqueeze(0).to(device))
    out = []
    for t in torch.linspace(0, 1, steps):
        z = (1 - t) * mu1 + t * mu2
        out.append(model.decoder(z).cpu().squeeze())
    return out

x1, _ = test_set[0]
x2, _ = test_set[1]
seq = interpolate(vae, x1, x2, 10)
fig, axes = plt.subplots(1, 10, figsize=(16, 2))
for a, img in zip(axes, seq):
    a.imshow(img, cmap="gray"); a.axis("off")
plt.suptitle("Latent interpolation: every intermediate point is a valid digit", y=1.15)
plt.tight_layout(); plt.show()
~~~

### beta and posterior collapse

~~~python beta_vae.py
"""beta controls the reconstruction / regularity trade-off."""
import torch

print(f"{'beta':>6} {'recon loss':>12} {'KL':>8} {'behaviour'}")
print("-" * 68)
for beta in [0.1, 0.5, 1.0, 4.0, 20.0]:
    m = train(VAE(latent=8), epochs=8, is_vae=True, beta=beta)
    m.eval()
    with torch.no_grad():
        x, _ = next(iter(test_loader))
        x = x.to(device)
        recon, mu, logvar, _ = m(x)
        _, rl, kl = vae_loss(recon, x, mu, logvar, beta)
    note = ("sharp reconstructions, ragged latent space" if beta < 0.5 else
            "balanced" if beta <= 4 else
            "POSTERIOR COLLAPSE - the decoder ignores z entirely")
    print(f"{beta:>6.1f} {rl.item():>12.2f} {kl.item():>8.3f} {note}")

print("""
POSTERIOR COLLAPSE

When beta is too large, the cheapest way to minimise the loss is to make
q(z|x) EXACTLY N(0,1) for every input - KL becomes 0, z carries no
information, and the decoder learns to output the average digit regardless.

Symptom: KL divergence goes to ~0 and all reconstructions look the same.

FIXES
  - KL ANNEALING: start beta at 0 and ramp it up over training
  - FREE BITS: do not penalise KL below a small threshold per dimension
  - a weaker decoder, so it cannot ignore z and still do well
""")


def kl_annealing_schedule(epoch, total_epochs, warmup_fraction=0.3):
    """Ramp beta from 0 to 1 over the first 30% of training."""
    warmup = total_epochs * warmup_fraction
    return min(1.0, epoch / warmup) if warmup > 0 else 1.0


def free_bits_kl(mu, logvar, free_bits=0.5):
    """Do not penalise KL below free_bits nats per latent dimension."""
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    return torch.clamp(kl_per_dim, min=free_bits).sum(dim=1).mean()
~~~

### Where autoencoders earn their keep

~~~python ae_applications.py
"""The plain autoencoder is not generative, but it is genuinely useful."""
import torch, torch.nn as nn, numpy as np
import matplotlib.pyplot as plt

# =====================================================================
# 1. DENOISING - train on (noisy input -> clean target)
# =====================================================================
class DenoisingAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid())

    def forward(self, x):
        return self.dec(self.enc(x))


def train_denoiser(epochs=10, noise=0.4):
    m = DenoisingAE().to(device)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(epochs):
        for x, _ in train_loader:
            x = x.to(device)
            noisy = torch.clamp(x + noise * torch.randn_like(x), 0, 1)
            opt.zero_grad()
            loss = nn.functional.mse_loss(m(noisy), x)   # target is the CLEAN image
            loss.backward(); opt.step()
        if ep % 3 == 0:
            print(f"  denoiser epoch {ep}: {loss.item():.5f}")
    return m


# =====================================================================
# 2. ANOMALY DETECTION - train on NORMAL data only
# =====================================================================
def anomaly_scores(model, loader):
    """Reconstruction error IS the anomaly score. An autoencoder trained
    only on normal data reconstructs normal data well and anomalies badly."""
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            err = ((model(x) - x) ** 2).mean(dim=(1, 2, 3))
            scores.append(err.cpu()); labels.append(y)
    return torch.cat(scores).numpy(), torch.cat(labels).numpy()


print("""
AUTOENCODER APPLICATIONS THAT ARE NOT GENERATION

  DENOISING          train noisy -> clean. Works for images, audio, sensors.
  ANOMALY DETECTION  train on normal data only; high reconstruction error
                     flags an anomaly. Widely used in manufacturing and
                     network monitoring.
  DIMENSIONALITY     a non-linear PCA. The bottleneck is a learned embedding.
  PRETRAINING        learn representations from unlabelled data, then attach
                     a small supervised head.
  COMPRESSION        the latent code IS the compressed representation. This
                     is exactly what Stable Diffusion's VAE does - it works
                     in a 64x64 latent space instead of 512x512 pixels,
                     which is a 48x reduction in compute.
""")
~~~
`
}
],
quiz: [
{
q: 'Why can a plain autoencoder not generate new samples?',
options: [
  'It is too small',
  'Its latent space has large empty regions, so a random z decodes to noise',
  'It has no decoder',
  'It only works on images'
],
answer: 1,
why: 'Nothing constrains where the encoder places codes. The VAE KL term pushes them toward a standard normal, filling the space so every sampled point decodes to something plausible.'
},
{
q: 'What is the reparameterisation trick and why is it needed?',
options: [
  'A way to reduce parameters',
  'Writing z = mu + sigma * epsilon so randomness sits in epsilon and gradients can flow through mu and sigma',
  'A learning-rate schedule',
  'A regularisation method'
],
answer: 1,
why: 'You cannot backpropagate through a sampling operation. Moving the randomness into a parameter-free noise variable makes the sampling differentiable with respect to mu and sigma.'
},
{
q: 'Your VAE KL divergence drops to nearly zero and all reconstructions look identical. What happened?',
options: [
  'It converged perfectly',
  'Posterior collapse - the KL term dominated, so z carries no information and the decoder ignores it',
  'The learning rate is too low',
  'The latent dimension is too large'
],
answer: 1,
why: 'Setting q(z|x) exactly to N(0,1) makes KL zero at the cost of all information. Fix with KL annealing, free bits, or a weaker decoder.'
},
{
q: 'Why are VAE samples characteristically blurry?',
options: [
  'The model is too small',
  'Pixel-wise reconstruction loss rewards predicting the average of plausible outputs, and the average of sharp images is blurry',
  'The latent space is too small',
  'They are not blurry'
],
answer: 1,
why: 'When several sharp outputs are equally plausible, the loss-minimising prediction is their mean. GANs avoid this by using a learned discriminator instead of a pixel-wise loss.'
}
]
},

/* ============================================================ */
{
id: 'gans',
title: 'Generative adversarial networks',
summary: 'Two networks competing - the minimax game, why training is unstable, mode collapse, and the tricks that make GANs work.',
tags: ['generative', 'gan', 'advanced'],
intro: `
## The adversarial game

~~~text
  random noise z  ->  GENERATOR  ->  fake image  \\
                                                  ->  DISCRIMINATOR  ->  real or fake?
                       real image  --------------/

  GENERATOR wants:      the discriminator to say "real" about its fakes
  DISCRIMINATOR wants:  to tell real from fake correctly

  They are optimising DIRECTLY AGAINST each other.
~~~

:::math The minimax objective
**min over G, max over D of: E[log D(x)] + E[log(1 - D(G(z)))]**

At the theoretical optimum the generator's distribution equals the data distribution and
the discriminator outputs 0.5 everywhere - it cannot tell them apart.
:::

## Why GANs are hard

| Problem | What you see | Cause |
|---|---|---|
| **Mode collapse** | The generator produces only a few outputs | It found something that reliably fools D and stopped exploring |
| **Vanishing gradient** | The generator stops improving | D became too good; log(1 - D(G(z))) saturates |
| **Oscillation** | The losses cycle and never converge | The two objectives chase each other |
| **No useful metric** | Loss values tell you nothing | It is a game, not a minimisation - low loss is not good |

:::warn GAN losses are not a progress indicator
In ordinary training, falling loss means progress. In a GAN, a falling generator loss may
simply mean the discriminator got worse. **Look at the samples**, and use FID.
:::

## The fixes that actually work

1. **Non-saturating generator loss** - maximise log(D(G(z))) instead of minimising
   log(1 - D(G(z))). Same optimum, far better gradients early on.
2. **Wasserstein loss with gradient penalty (WGAN-GP)** - a meaningful loss and stable training.
3. **Spectral normalisation** - constrain the discriminator's Lipschitz constant.
4. **Two time-scale update rule** - a higher learning rate for D than G.
5. **Label smoothing** - use 0.9 rather than 1.0 for real labels.
`,
keyPoints: [
  'GAN training is a game between two networks, not a single minimisation.',
  'Mode collapse means the generator produces few distinct outputs.',
  'Use the non-saturating generator loss; the original saturates early in training.',
  'Judge GANs by samples and FID, never by the loss curves.'
],
pitfalls: [
  'Reading GAN losses as progress.',
  'Letting the discriminator become too strong, which starves the generator of gradient.',
  'Using batch normalisation in the discriminator with a gradient penalty - they conflict.',
  'Not checking sample diversity, and shipping a mode-collapsed model.'
],
levels: [
{
name: 'Building a DCGAN, and diagnosing it',
goal: 'Implement a working GAN with the stabilisation tricks, and detect mode collapse.',
md: `
~~~python dcgan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LATENT = 100
BATCH = 128

transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])   # to [-1, 1]
dataset = torchvision.datasets.MNIST("./data", train=True, download=True,
                                     transform=transform)
loader = DataLoader(dataset, BATCH, shuffle=True, num_workers=2, drop_last=True)


# =====================================================================
# GENERATOR: noise -> image
# =====================================================================
class Generator(nn.Module):
    def __init__(self, latent=LATENT, features=64):
        super().__init__()
        self.net = nn.Sequential(
            # (latent, 1, 1) -> (features*4, 7, 7)
            nn.ConvTranspose2d(latent, features * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(features * 4), nn.ReLU(True),
            # -> (features*2, 14, 14)
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2), nn.ReLU(True),
            # -> (1, 28, 28)
            nn.ConvTranspose2d(features * 2, 1, 4, 2, 1, bias=False),
            nn.Tanh(),                      # output in [-1, 1], matching the data
        )

    def forward(self, z):
        return self.net(z.view(z.size(0), -1, 1, 1))


# =====================================================================
# DISCRIMINATOR: image -> is it real?
# =====================================================================
class Discriminator(nn.Module):
    def __init__(self, features=64):
        super().__init__()
        self.net = nn.Sequential(
            # SPECTRAL NORM constrains the Lipschitz constant - a strong stabiliser
            nn.utils.spectral_norm(nn.Conv2d(1, features, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),      # LeakyReLU, not ReLU, in D
            nn.utils.spectral_norm(nn.Conv2d(features, features * 2, 4, 2, 1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Conv2d(features * 2, 1, 7, 1, 0)),
        )

    def forward(self, x):
        return self.net(x).view(-1)               # LOGITS, no sigmoid


G = Generator().to(device)
D = Discriminator().to(device)


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)

G.apply(weights_init)
print(f"G parameters: {sum(p.numel() for p in G.parameters()):,}")
print(f"D parameters: {sum(p.numel() for p in D.parameters()):,}")


# =====================================================================
# TRAINING - the two-time-scale update rule (TTUR)
# =====================================================================
opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=4e-4, betas=(0.5, 0.999))
criterion = nn.BCEWithLogitsLoss()

fixed_noise = torch.randn(64, LATENT, device=device)     # to watch progress
history = {"d_loss": [], "g_loss": [], "d_real": [], "d_fake": []}

EPOCHS = 20
LABEL_SMOOTH = 0.9        # use 0.9 instead of 1.0 for real labels

for epoch in range(EPOCHS):
    for i, (real, _) in enumerate(loader):
        real = real.to(device)
        b = real.size(0)

        # =============================================================
        # TRAIN D: maximise log D(real) + log(1 - D(fake))
        # =============================================================
        opt_D.zero_grad()

        out_real = D(real)
        loss_real = criterion(out_real, torch.full((b,), LABEL_SMOOTH, device=device))

        z = torch.randn(b, LATENT, device=device)
        fake = G(z)
        out_fake = D(fake.detach())          # DETACH - do not update G here
        loss_fake = criterion(out_fake, torch.zeros(b, device=device))

        loss_D = loss_real + loss_fake
        loss_D.backward()
        opt_D.step()

        # =============================================================
        # TRAIN G: the NON-SATURATING loss.
        #   original: minimise log(1 - D(G(z)))   -> vanishing gradient early
        #   used:     maximise log D(G(z))        -> strong gradient when D wins
        # =============================================================
        opt_G.zero_grad()
        out = D(fake)
        loss_G = criterion(out, torch.ones(b, device=device))    # label them REAL
        loss_G.backward()
        opt_G.step()

        if i % 100 == 0:
            history["d_loss"].append(loss_D.item())
            history["g_loss"].append(loss_G.item())
            history["d_real"].append(torch.sigmoid(out_real).mean().item())
            history["d_fake"].append(torch.sigmoid(out_fake).mean().item())

    print(f"epoch {epoch+1:2d}/{EPOCHS}  D {loss_D.item():.4f}  G {loss_G.item():.4f}  "
          f"D(real) {torch.sigmoid(out_real).mean():.3f}  "
          f"D(fake) {torch.sigmoid(out_fake).mean():.3f}")

    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            samples = G(fixed_noise).cpu()
        grid = torchvision.utils.make_grid(samples, nrow=8, normalize=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
        plt.axis("off"); plt.title(f"epoch {epoch+1}")
        plt.show()
~~~

### Reading the training signals

~~~python gan_diagnostics.py
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
ax[0].plot(history["d_loss"], label="discriminator")
ax[0].plot(history["g_loss"], label="generator")
ax[0].set_xlabel("iteration (x100)"); ax[0].set_ylabel("loss")
ax[0].legend(); ax[0].grid(alpha=0.3); ax[0].set_title("Losses - NOT a progress metric")

ax[1].plot(history["d_real"], label="D(real)")
ax[1].plot(history["d_fake"], label="D(fake)")
ax[1].axhline(0.5, color="k", ls="--", lw=1, label="equilibrium")
ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].grid(alpha=0.3)
ax[1].set_title("D outputs - THIS is what to watch")
plt.tight_layout(); plt.show()

print("""
WHAT THE D-OUTPUT PLOT TELLS YOU

  D(real) ~ 0.5 and D(fake) ~ 0.5      HEALTHY. D cannot distinguish them.
  D(real) ~ 1.0 and D(fake) ~ 0.0      D IS WINNING. G gets no useful gradient.
                                       -> weaken D: lower its lr, fewer D steps,
                                          add dropout or noise to its inputs
  D(real) ~ 0.5 and D(fake) ~ 0.9      G IS WINNING (or D collapsed).
                                       -> strengthen D
  Both oscillating wildly              unstable. Lower both learning rates,
                                       add spectral norm, try WGAN-GP.
""")


# =====================================================================
# DETECTING MODE COLLAPSE
# =====================================================================
@torch.no_grad()
def diversity_report(G, n=2000, latent=LATENT):
    """Mode collapse = the generator produces few distinct outputs."""
    G.eval()
    z = torch.randn(n, latent, device=device)
    samples = G(z).cpu().view(n, -1).numpy()

    # 1. pairwise distance between samples - low means they are all alike
    idx = np.random.default_rng(0).choice(n, 400, replace=False)
    sub = samples[idx]
    dists = np.sqrt(((sub[:, None] - sub[None, :]) ** 2).sum(-1))
    mean_dist = dists[np.triu_indices(len(sub), 1)].mean()

    # 2. how many distinct clusters do the samples fall into?
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    reduced = PCA(20, random_state=0).fit_transform(samples)
    km = KMeans(10, n_init=10, random_state=0).fit(reduced)
    sizes = np.bincount(km.labels_, minlength=10)
    # entropy of the cluster distribution: 1.0 = perfectly even
    p = sizes / sizes.sum()
    evenness = -np.sum(p * np.log(p + 1e-12)) / np.log(10)

    # 3. classify the samples and check the class distribution
    print(f"  mean pairwise distance : {mean_dist:.3f}")
    print(f"  cluster sizes          : {sizes}")
    print(f"  cluster evenness       : {evenness:.3f}   "
          f"(1.0 = all modes covered, <0.6 = COLLAPSE)")

    if evenness < 0.6:
        print("  >>> MODE COLLAPSE DETECTED")
        print("      fixes: minibatch discrimination, unrolled GAN, WGAN-GP,")
        print("             feature matching, or a lower generator learning rate")
    return evenness


print("DIVERSITY CHECK")
diversity_report(G)


# =====================================================================
# FID - the standard GAN quality metric
# =====================================================================
FID_EXPLANATION = """
FRECHET INCEPTION DISTANCE

  1. Push real images and generated images through a pretrained Inception network
  2. Take the 2048-dimensional pool3 activations for each set
  3. Fit a Gaussian to each set (mean vector + covariance matrix)
  4. FID = the Frechet distance between those two Gaussians

  LOWER IS BETTER.
    FID <  10   excellent
    FID 10-30   good
    FID 30-60   recognisable but flawed
    FID >  100  poor

  It captures BOTH quality and diversity: a mode-collapsed generator has a
  small covariance and therefore a large FID even if each sample is sharp.

  CAVEATS
    - Needs 10,000+ samples for a stable estimate
    - Only comparable within the same implementation and image resolution
    - Uses an ImageNet-trained network, so it is biased toward natural images

  pip install pytorch-fid  /  torchmetrics
      from torchmetrics.image.fid import FrechetInceptionDistance
      fid = FrechetInceptionDistance(feature=2048)
      fid.update(real_uint8, real=True)
      fid.update(fake_uint8, real=False)
      print(fid.compute())
"""
print(FID_EXPLANATION)
~~~

### WGAN-GP: the stable alternative

~~~python wgan_gp.py
"""Wasserstein GAN with gradient penalty - a meaningful loss and stable training."""
import torch
import torch.nn as nn


def gradient_penalty(D, real, fake, device, lambda_gp=10.0):
    """Penalise the critic's gradient norm away from 1, at random points
    between real and fake samples. This enforces the Lipschitz constraint
    that the Wasserstein formulation requires."""
    b = real.size(0)
    epsilon = torch.rand(b, 1, 1, 1, device=device)
    interpolated = (epsilon * real + (1 - epsilon) * fake).requires_grad_(True)

    scores = D(interpolated)
    grads = torch.autograd.grad(
        outputs=scores, inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True)[0]

    grads = grads.view(b, -1)
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gp


def train_wgan_gp(G, D, loader, epochs=20, n_critic=5, latent=100):
    """The critic is trained n_critic times per generator step."""
    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))

    for epoch in range(epochs):
        for i, (real, _) in enumerate(loader):
            real = real.to(device)
            b = real.size(0)

            # ---- train the CRITIC n_critic times ---------------------
            for _ in range(n_critic):
                z = torch.randn(b, latent, device=device)
                fake = G(z).detach()
                opt_D.zero_grad()
                # Wasserstein loss: maximise D(real) - D(fake)
                loss_D = (-D(real).mean() + D(fake).mean()
                          + gradient_penalty(D, real, fake, device))
                loss_D.backward()
                opt_D.step()

            # ---- train the GENERATOR once ----------------------------
            opt_G.zero_grad()
            z = torch.randn(b, latent, device=device)
            loss_G = -D(G(z)).mean()
            loss_G.backward()
            opt_G.step()

        # THE NEGATIVE CRITIC LOSS APPROXIMATES THE WASSERSTEIN DISTANCE,
        # so unlike a standard GAN, this number IS meaningful and should fall.
        print(f"epoch {epoch}: W-distance estimate {-loss_D.item():.4f}")


print("""
WHY WGAN-GP IS EASIER TO TRAIN

  1. The critic loss ESTIMATES the Wasserstein distance, so it correlates
     with sample quality. You finally have a meaningful progress metric.
  2. The Wasserstein distance gives useful gradients even when the real and
     fake distributions do not overlap - which is exactly when a standard
     GAN's gradient vanishes.
  3. Mode collapse is far rarer.

  COSTS: 5x more critic steps per generator step, and no batch normalisation
  in the critic (it breaks the per-sample gradient penalty - use LayerNorm,
  InstanceNorm or spectral norm instead).
""")
~~~

:::tip Where GANs still win, and where they lost
**Still competitive:** real-time generation (one forward pass, versus 20-50 for diffusion),
super-resolution, image-to-image translation, and face generation (StyleGAN quality remains
outstanding).

**Lost to diffusion:** general text-to-image, controllable generation, and anything needing
broad diversity. Diffusion is easier to train, scales better, and conditions more naturally.

Learn GANs for the adversarial idea - it appears everywhere from domain adaptation to
robustness research - but reach for diffusion when you want to generate images today.
:::
`
}
],
quiz: [
{
q: 'Your GAN generator loss is falling steadily. Is that good?',
options: [
  'Yes, lower loss is better',
  'Not necessarily - it may just mean the discriminator got worse. Judge by samples and FID',
  'It means the model converged',
  'It means mode collapse'
],
answer: 1,
why: 'A GAN is a game, not a minimisation. The losses are relative to an opponent that is also changing. Watch D(real) and D(fake) approaching 0.5, and look at the samples.'
},
{
q: 'What is mode collapse?',
options: [
  'The discriminator stops learning',
  'The generator produces only a few distinct outputs, ignoring most of the data distribution',
  'The loss becomes NaN',
  'Training is too slow'
],
answer: 1,
why: 'The generator finds a narrow output that reliably fools the discriminator and stops exploring. Detect it by measuring sample diversity; fix with WGAN-GP, minibatch discrimination or unrolling.'
},
{
q: 'Why use the non-saturating generator loss?',
options: [
  'It is faster',
  'The original log(1 - D(G(z))) has almost no gradient early on when D easily wins',
  'It gives a different optimum',
  'It uses less memory'
],
answer: 1,
why: 'Early in training D(G(z)) is near 0, so log(1-D(G(z))) is flat and the generator barely learns. Maximising log D(G(z)) has the same optimum but a strong gradient exactly when it is needed.'
},
{
q: 'What does FID measure?',
options: [
  'Only image sharpness',
  'The distance between Gaussians fitted to Inception features of real and generated images - capturing both quality and diversity',
  'The generator loss',
  'Training speed'
],
answer: 1,
why: 'Because it compares full covariance structures, a mode-collapsed generator scores badly even if its individual samples are sharp - which is exactly why FID replaced eyeballing.'
}
]
},

/* ============================================================ */
{
id: 'diffusion',
title: 'Diffusion models',
summary: 'Add noise gradually, then learn to remove it - the mathematics, a working implementation, and how Stable Diffusion assembles the pieces.',
tags: ['generative', 'diffusion', 'advanced'],
intro: `
## The idea

~~~text
FORWARD PROCESS (fixed, no learning)
  x0 -> x1 -> x2 -> ... -> xT
  clean       add a little gaussian noise at each step      pure noise

REVERSE PROCESS (learned)
  xT -> ... -> x2 -> x1 -> x0
  noise        the network predicts and removes the noise    clean image
~~~

The training task is deceptively simple: **given a noisy image and the timestep, predict
the noise that was added.** That is it - a regression problem with an MSE loss.

:::math The forward process, in closed form
You can jump to any timestep directly, without simulating the steps:

**x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon**

where **alpha_bar_t** decreases from ~1 (barely any noise) to ~0 (pure noise), and
**epsilon ~ N(0, I)**.

This is what makes training efficient: sample a random t, noise the image in one step,
and ask the network to predict epsilon.
:::

:::math The training loss
**L = E over t, x0, epsilon of || epsilon - model(x_t, t) || squared**

An MSE between the true noise and the predicted noise. Nothing adversarial, nothing
unstable - it just works.
:::

## Why diffusion won

| | GAN | Diffusion |
|---|---|---|
| Training | Adversarial, unstable | Simple MSE regression, stable |
| Diversity | Mode collapse is common | Excellent coverage |
| Quality | Sharp | Sharp |
| Conditioning | Awkward | Natural (classifier-free guidance) |
| Sampling | 1 forward pass | 20-1000 forward passes |
| Scaling | Plateaus | Keeps improving |

The one weakness is sampling speed, and that is being attacked directly by DDIM, DPM-Solver,
consistency models and distillation - which now reach usable quality in 1-4 steps.
`,
keyPoints: [
  'Training is a simple MSE: predict the noise added at a random timestep.',
  'The closed-form forward process lets you jump to any timestep in one operation.',
  'Classifier-free guidance trades diversity for prompt adherence via a single scale parameter.',
  'Latent diffusion runs in a compressed VAE space, cutting compute by roughly 50x.'
],
pitfalls: [
  'Forgetting the timestep embedding - the model cannot know how much noise to remove.',
  'Using too few sampling steps with a basic DDPM sampler.',
  'Setting the guidance scale too high, which produces saturated, low-diversity images.',
  'Expecting a diffusion model to train quickly - it needs many epochs.'
],
levels: [
{
name: 'A diffusion model from scratch',
goal: 'Implement the forward and reverse processes, train a small denoiser, and generate images.',
md: `
~~~python diffusion.py
"""A DDPM (denoising diffusion probabilistic model), from scratch."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# 1. THE NOISE SCHEDULE
# =====================================================================
class NoiseSchedule:
    def __init__(self, timesteps=1000, schedule="cosine"):
        self.T = timesteps

        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, timesteps)
        else:
            # COSINE schedule - adds noise more slowly at the start,
            # which preserves information for longer and works better.
            s = 0.008
            steps = torch.arange(timesteps + 1, dtype=torch.float32) / timesteps
            alphas_cumprod = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = torch.clip(1 - alphas_cumprod[1:] / alphas_cumprod[:-1], 0, 0.999)

        self.betas = betas.to(device)
        self.alphas = (1.0 - betas).to(device)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (self.betas * (1 - self.alphas_cumprod_prev)
                                   / (1 - self.alphas_cumprod))

    def add_noise(self, x0, t, noise=None):
        """THE FORWARD PROCESS, in closed form - jump straight to timestep t."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1mac = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_ac * x0 + sqrt_1mac * noise, noise


schedule = NoiseSchedule(timesteps=1000, schedule="cosine")

# ---- see what the schedule does --------------------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
for name in ["linear", "cosine"]:
    s = NoiseSchedule(1000, name)
    plt.plot(s.alphas_cumprod.cpu(), lw=2, label=name)
plt.xlabel("timestep t"); plt.ylabel("alpha_bar (signal remaining)")
plt.legend(); plt.grid(alpha=0.3); plt.title("Noise schedules")

plt.subplot(1, 2, 2)
for name in ["linear", "cosine"]:
    s = NoiseSchedule(1000, name)
    plt.plot(s.betas.cpu(), lw=2, label=name)
plt.xlabel("timestep t"); plt.ylabel("beta (noise added per step)")
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()


# =====================================================================
# 2. VISUALISE THE FORWARD PROCESS
# =====================================================================
transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))])
dataset = torchvision.datasets.MNIST("./data", train=True, download=True,
                                     transform=transform)
loader = DataLoader(dataset, 128, shuffle=True, num_workers=2, drop_last=True)

x0, _ = dataset[7]
x0 = x0.unsqueeze(0).to(device)

steps_to_show = [0, 50, 150, 300, 500, 700, 850, 999]
fig, axes = plt.subplots(1, len(steps_to_show), figsize=(17, 2.6))
for ax, t in zip(axes, steps_to_show):
    xt, _ = schedule.add_noise(x0, torch.tensor([t], device=device))
    ax.imshow(xt.cpu().squeeze(), cmap="gray", vmin=-1, vmax=1)
    ax.set_title(f"t={t}\\nsignal {schedule.alphas_cumprod[t]:.3f}", fontsize=8)
    ax.axis("off")
plt.suptitle("The forward process: a digit dissolves into pure noise", y=1.12)
plt.tight_layout(); plt.show()


# =====================================================================
# 3. THE DENOISING NETWORK - a small U-Net with timestep conditioning
# =====================================================================
class SinusoidalTimeEmbedding(nn.Module):
    """The model MUST know which timestep it is at, to know how much noise
    to remove. Same encoding as transformer positions."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_ch)      # inject the timestep
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = (nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch
                     else nn.Identity())

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]   # ADD the time signal
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class SimpleUNet(nn.Module):
    def __init__(self, base=64, time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base),
            nn.Linear(base, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.in_conv = nn.Conv2d(1, base, 3, padding=1)

        self.down1 = ResBlock(base, base, time_dim)
        self.down2 = ResBlock(base, base * 2, time_dim)
        self.down3 = ResBlock(base * 2, base * 4, time_dim)
        self.pool = nn.AvgPool2d(2)

        self.mid = ResBlock(base * 4, base * 4, time_dim)

        self.up3 = ResBlock(base * 8, base * 2, time_dim)
        self.up2 = ResBlock(base * 4, base, time_dim)
        self.up1 = ResBlock(base * 2, base, time_dim)

        self.out_norm = nn.GroupNorm(8, base)
        self.out_conv = nn.Conv2d(base, 1, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        h1 = self.down1(self.in_conv(x), t_emb)             # 28
        h2 = self.down2(self.pool(h1), t_emb)               # 14
        h3 = self.down3(self.pool(h2), t_emb)               # 7

        m = self.mid(h3, t_emb)

        u3 = self.up3(torch.cat([m, h3], 1), t_emb)
        u3 = F.interpolate(u3, size=h2.shape[-2:], mode="nearest")
        u2 = self.up2(torch.cat([u3, h2], 1), t_emb)
        u2 = F.interpolate(u2, size=h1.shape[-2:], mode="nearest")
        u1 = self.up1(torch.cat([u2, h1], 1), t_emb)

        return self.out_conv(F.silu(self.out_norm(u1)))     # PREDICTED NOISE


model = SimpleUNet().to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")


# =====================================================================
# 4. TRAIN - the entire loss is one MSE
# =====================================================================
optimiser = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
EPOCHS = 30
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser,
                                                       T_max=EPOCHS * len(loader))

# EMA of the weights - almost always improves diffusion sample quality
ema_model = SimpleUNet().to(device)
ema_model.load_state_dict(model.state_dict())

@torch.no_grad()
def update_ema(ema, current, decay=0.999):
    for pe, pc in zip(ema.parameters(), current.parameters()):
        pe.mul_(decay).add_(pc, alpha=1 - decay)


for epoch in range(EPOCHS):
    model.train()
    total = 0.0
    for x0, _ in loader:
        x0 = x0.to(device)
        b = x0.size(0)

        # 1. a RANDOM timestep per example
        t = torch.randint(0, schedule.T, (b,), device=device)

        # 2. noise the image to that timestep, in one step
        xt, noise = schedule.add_noise(x0, t)

        # 3. predict the noise, and take the MSE. That is the whole objective.
        predicted = model(xt, t)
        loss = F.mse_loss(predicted, noise)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        scheduler.step()
        update_ema(ema_model, model)

        total += loss.item()

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch:2d}  loss {total/len(loader):.5f}")
~~~

### Sampling

~~~python sampling_diffusion.py
@torch.no_grad()
def ddpm_sample(model, schedule, n=16, shape=(1, 28, 28), record=False):
    """The original sampler: walk backwards one timestep at a time."""
    model.eval()
    x = torch.randn(n, *shape, device=device)          # start from pure noise
    snapshots = []

    for i in reversed(range(schedule.T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)

        predicted_noise = model(x, t)

        alpha = schedule.alphas[i]
        alpha_bar = schedule.alphas_cumprod[i]
        beta = schedule.betas[i]

        # remove the predicted noise, scaled correctly
        mean = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise)

        if i > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(schedule.posterior_variance[i]) * noise
        else:
            x = mean                                   # no noise on the last step

        if record and i % 100 == 0:
            snapshots.append(x.clone().cpu())

    return (x.cpu(), snapshots) if record else x.cpu()


@torch.no_grad()
def ddim_sample(model, schedule, n=16, shape=(1, 28, 28), steps=50, eta=0.0):
    """DDIM: a DETERMINISTIC sampler that skips timesteps.
    50 steps instead of 1000, with very little quality loss."""
    model.eval()
    timesteps = torch.linspace(schedule.T - 1, 0, steps).long().to(device)
    x = torch.randn(n, *shape, device=device)

    for i in range(len(timesteps)):
        t = timesteps[i]
        t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(-1)

        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        predicted_noise = model(x, t_batch)

        alpha_bar = schedule.alphas_cumprod[t]
        alpha_bar_prev = (schedule.alphas_cumprod[t_prev] if t_prev >= 0
                          else torch.tensor(1.0, device=device))

        # estimate the clean image x0 from the current noisy x and the prediction
        x0_pred = (x - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
        x0_pred = x0_pred.clamp(-1, 1)

        # then step back toward it
        sigma = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) *
                                 (1 - alpha_bar / alpha_bar_prev))
        direction = torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * predicted_noise
        x = torch.sqrt(alpha_bar_prev) * x0_pred + direction
        if eta > 0 and t_prev >= 0:
            x = x + sigma * torch.randn_like(x)

    return x.cpu()


import time

print(f"{'sampler':22s} {'steps':>7s} {'time':>9s}")
print("-" * 40)
t0 = time.perf_counter()
ddpm_out = ddpm_sample(ema_model, schedule, n=16)
print(f"{'DDPM':22s} {schedule.T:>7} {time.perf_counter()-t0:>8.1f}s")

for steps in [10, 20, 50, 100]:
    t0 = time.perf_counter()
    out = ddim_sample(ema_model, schedule, n=16, steps=steps)
    print(f"{'DDIM':22s} {steps:>7} {time.perf_counter()-t0:>8.1f}s")

# ---- see the reverse process ----------------------------------------
_, snapshots = ddpm_sample(ema_model, schedule, n=8, record=True)
fig, axes = plt.subplots(len(snapshots), 8, figsize=(12, 1.5 * len(snapshots)))
for row, snap in enumerate(snapshots):
    for col in range(8):
        axes[row, col].imshow(snap[col].squeeze(), cmap="gray", vmin=-1, vmax=1)
        axes[row, col].axis("off")
    axes[row, 0].set_ylabel(f"t={900 - row*100}")
plt.suptitle("The reverse process: noise gradually resolves into digits", y=1.0)
plt.tight_layout(); plt.show()

grid = torchvision.utils.make_grid(ddim_sample(ema_model, schedule, 64, steps=50),
                                   nrow=8, normalize=True)
plt.figure(figsize=(9, 9))
plt.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
plt.axis("off"); plt.title("Generated samples, DDIM 50 steps")
plt.tight_layout(); plt.show()
~~~

### Conditioning and classifier-free guidance

~~~python guidance.py
"""How text-to-image models actually follow prompts."""
import torch
import torch.nn as nn


class ConditionalUNet(SimpleUNet):
    """Add class conditioning, with a NULL class for unconditional training."""

    def __init__(self, n_classes=10, base=64, time_dim=256):
        super().__init__(base, time_dim)
        # n_classes + 1: the extra slot is the NULL (unconditional) embedding
        self.class_emb = nn.Embedding(n_classes + 1, time_dim)
        self.null_class = n_classes

    def forward(self, x, t, y=None):
        t_emb = self.time_mlp(t)
        if y is None:
            y = torch.full((x.size(0),), self.null_class, device=x.device)
        t_emb = t_emb + self.class_emb(y)          # combine time and class
        # ... the rest of the forward pass is unchanged
        return super().forward(x, t)


def train_step_with_cfg(model, schedule, x0, y, drop_prob=0.1):
    """CLASSIFIER-FREE GUIDANCE TRAINING

    Randomly replace the label with the null class 10% of the time. The
    SAME network therefore learns both the conditional and unconditional
    score - no separate classifier is needed."""
    b = x0.size(0)
    t = torch.randint(0, schedule.T, (b,), device=x0.device)
    xt, noise = schedule.add_noise(x0, t)

    mask = torch.rand(b, device=x0.device) < drop_prob
    y = y.clone()
    y[mask] = model.null_class                     # drop the condition

    return F.mse_loss(model(xt, t, y), noise)


@torch.no_grad()
def sample_with_guidance(model, schedule, class_id, n=8, steps=50, guidance=3.0):
    """CLASSIFIER-FREE GUIDANCE SAMPLING

    Run the model twice per step - once with the condition, once without -
    and extrapolate AWAY from the unconditional prediction:

        noise = uncond + guidance * (cond - uncond)

    guidance = 1.0  no guidance, maximum diversity
    guidance = 3-8  the usual range - strong prompt adherence
    guidance > 15   saturated, over-contrasted, low-diversity images
    """
    model.eval()
    x = torch.randn(n, 1, 28, 28, device=device)
    y_cond = torch.full((n,), class_id, device=device, dtype=torch.long)
    y_null = torch.full((n,), model.null_class, device=device, dtype=torch.long)

    timesteps = torch.linspace(schedule.T - 1, 0, steps).long().to(device)
    for i, t in enumerate(timesteps):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)

        noise_cond = model(x, t_batch, y_cond)
        noise_uncond = model(x, t_batch, y_null)
        noise = noise_uncond + guidance * (noise_cond - noise_uncond)

        # ... the DDIM update, using this combined noise prediction
    return x.cpu()


print("""
THE GUIDANCE SCALE IS THE MOST IMPORTANT KNOB IN TEXT-TO-IMAGE

  1.0    ignores the prompt; maximum diversity
  3.0    loose adherence, natural-looking
  7.5    the Stable Diffusion default - a good balance
  12.0   very literal, starting to look over-processed
  20.0+  saturated, high-contrast, low diversity, often distorted

It trades DIVERSITY for PROMPT ADHERENCE. There is no free lunch.
""")
~~~

### How Stable Diffusion assembles the pieces

~~~text
STABLE DIFFUSION = LATENT DIFFUSION + TEXT CONDITIONING

  1. VAE ENCODER      512x512x3 image -> 64x64x4 latent
                      A 48x reduction in the number of values. Diffusion then
                      runs in that small latent space, which is why it fits
                      on a consumer GPU at all.

  2. TEXT ENCODER     the prompt -> CLIP text embeddings

  3. U-NET            denoises the LATENT, conditioned on the text embeddings
                      via CROSS-ATTENTION at multiple resolutions
                      (query from the image latent, key/value from the text)

  4. SCHEDULER        DDIM / DPM-Solver / Euler - decides the timestep path

  5. VAE DECODER      the denoised 64x64x4 latent -> a 512x512x3 image

WHY LATENT SPACE MATTERS
  pixel diffusion at 512x512:  786,432 values per step
  latent diffusion at 64x64x4:  16,384 values per step
  -> 48x less compute per step, which is the difference between a data
     centre and a laptop.
~~~

~~~python stable_diffusion.py
"""Using a real diffusion model."""
# pip install diffusers transformers accelerate
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

# a faster scheduler - good quality in 20-25 steps instead of 50
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()          # lower VRAM

image = pipe(
    prompt="a photograph of an astronaut riding a horse on mars, 50mm lens",
    negative_prompt="blurry, low quality, distorted, watermark",
    num_inference_steps=25,
    guidance_scale=7.5,
    height=512, width=512,
    generator=torch.Generator("cuda").manual_seed(42),   # reproducible
).images[0]
image.save("output.png")

print("""
THE PARAMETERS THAT MATTER

  num_inference_steps  20-30 with DPM-Solver is usually enough.
                       More steps beyond ~50 rarely helps.
  guidance_scale       7-8 is the sweet spot. Above 12 looks over-processed.
  negative_prompt      surprisingly effective - describe what you do NOT want.
  seed                 fix it for reproducibility and for A/B testing prompts.

RELATED PIPELINES
  StableDiffusionImg2ImgPipeline     transform an existing image
  StableDiffusionInpaintPipeline     edit a masked region
  ControlNet                         condition on pose, depth, edges, scribbles
  LoRA / DreamBooth                  teach it a specific subject or style
""")
~~~
`
}
],
quiz: [
{
q: 'What is a diffusion model actually trained to predict?',
options: [
  'The clean image directly',
  'The noise that was added at a given timestep - a simple MSE regression',
  'Whether an image is real or fake',
  'The next pixel'
],
answer: 1,
why: 'Sample a random timestep, noise the image in closed form, and regress the noise. That simplicity is why diffusion trains so much more stably than a GAN.'
},
{
q: 'Why does Stable Diffusion run in a latent space rather than on pixels?',
options: [
  'Latent images look better',
  '64x64x4 is 48x fewer values than 512x512x3, so each denoising step costs 48x less compute',
  'Pixels cannot be denoised',
  'It improves the prompt following'
],
answer: 1,
why: 'Latent diffusion is what made high-resolution generation feasible on consumer hardware. A VAE compresses to the latent space and decodes back at the end.'
},
{
q: 'What does raising the classifier-free guidance scale do?',
options: [
  'Speeds up sampling',
  'Increases prompt adherence at the cost of diversity - too high gives saturated, distorted images',
  'Improves resolution',
  'Reduces memory'
],
answer: 1,
why: 'It extrapolates away from the unconditional prediction. 7-8 is the usual balance; above about 12 images become over-contrasted and repetitive.'
},
{
q: 'Why did diffusion largely replace GANs for image generation?',
options: [
  'Diffusion samples faster',
  'It achieves both high quality AND high diversity with stable, non-adversarial training',
  'GANs cannot generate colour',
  'Diffusion uses fewer parameters'
],
answer: 1,
why: 'GANs trade diversity for sharpness and are hard to train. Diffusion gets both from an ordinary MSE objective. Its one weakness - slow sampling - is being solved by better solvers and distillation.'
}
]
}

]
});
