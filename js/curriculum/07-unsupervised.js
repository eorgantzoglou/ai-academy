/* Track 07 - Unsupervised learning: clustering, dimensionality reduction, anomalies */
(window.CURRICULUM = window.CURRICULUM || []).push({
id: 'unsupervised',
title: 'Unsupervised Learning',
icon: 'U',
level: 'Intermediate',
blurb: 'Finding structure without labels: K-Means, hierarchical, DBSCAN and Gaussian mixtures; PCA, t-SNE and UMAP; anomaly detection and market-basket analysis.',
intro: `
No answer key. The algorithm has to find structure on its own.

~~~text
CLUSTERING                which points belong together?
  K-Means                 spherical clusters, you choose k
  Hierarchical            a tree of nested groupings, no k needed upfront
  DBSCAN                  density-based; finds any shape, and outliers, no k
  Gaussian Mixture        soft, probabilistic, elliptical clusters

DIMENSIONALITY REDUCTION  can I describe 200 columns with 10?
  PCA                     linear, fast, reversible, preserves global structure
  t-SNE / UMAP            non-linear, for VISUALISATION, preserves local structure

ANOMALY DETECTION         which points are unlike the rest?
  Isolation Forest, LOF, One-Class SVM, autoencoder reconstruction error

ASSOCIATION RULES         what gets bought together?
~~~

:::warn The hard part of unsupervised learning is not the algorithm
It is **evaluation**. There is no ground truth, so "is this clustering good?" is partly a
judgement call. This track spends real time on that question, because it is where people
go wrong.
:::
`,
topics: [

/* ============================================================ */
{
id: 'kmeans',
title: 'K-Means clustering',
summary: 'The workhorse clustering algorithm - how it works, how to choose k, and the four assumptions that break it.',
tags: ['clustering', 'core'],
intro: `
## The algorithm, in four lines

~~~text
1. Pick k initial centroids
2. ASSIGN   each point to its nearest centroid
3. UPDATE   move each centroid to the mean of its assigned points
4. Repeat 2-3 until nothing moves

That is it. It always converges - but to a LOCAL optimum, which is why
initialisation matters and why you run it several times.
~~~

:::math What K-Means minimises
**Inertia = sum over all points of (distance to its centroid) squared**

Also called within-cluster sum of squares (WCSS). Every iteration provably decreases it.
:::

## The four assumptions - and what breaks

| Assumption | If violated |
|---|---|
| Clusters are roughly **spherical** | Elongated or crescent clusters get split wrongly |
| Clusters have **similar size** | Large clusters get carved up, small ones absorbed |
| Clusters have **similar density** | Dense clusters dominate the objective |
| You **know k** | Wrong k produces confident nonsense |

Also: it is distance-based, so **scaling is mandatory**, and it has no concept of an
outlier - every point is forced into a cluster.

## k-means++ initialisation

Random initial centroids can land in the same cluster and produce a bad local optimum.
**k-means++** spreads the initial centroids by choosing each new one with probability
proportional to its squared distance from the nearest existing centroid. It is the sklearn
default, and it matters.
`,
keyPoints: [
  'K-Means minimises within-cluster squared distance, so it always finds spherical clusters.',
  'Scaling is mandatory - it is a distance-based algorithm.',
  'Run with ~n_init=10~ or more: the result depends on initialisation.',
  'The elbow method is a heuristic; silhouette score is usually a better guide.'
],
pitfalls: [
  'Not scaling, so one large-range feature dictates the clustering.',
  'Choosing k by the elbow alone when the elbow is ambiguous.',
  'Using K-Means on non-spherical or very unequal-sized clusters.',
  'Treating cluster IDs as meaningful labels - they are arbitrary and change between runs.'
],
levels: [
{
name: 'K-Means from scratch, then properly',
goal: 'Implement the algorithm in 30 lines to see there is no magic, then use sklearn correctly.',
md: `
~~~python kmeans_scratch.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


def kmeans(X, k, max_iter=100, seed=0, plus_plus=True):
    """K-Means in its entirety."""
    rng = np.random.default_rng(seed)
    n = len(X)

    # ---------- initialise ----------
    if plus_plus:
        # k-means++: spread the initial centroids out
        centroids = [X[rng.integers(n)]]
        for _ in range(k - 1):
            d2 = np.min([((X - c) ** 2).sum(axis=1) for c in centroids], axis=0)
            probs = d2 / d2.sum()
            centroids.append(X[rng.choice(n, p=probs)])
        centroids = np.array(centroids)
    else:
        centroids = X[rng.choice(n, k, replace=False)]

    history = [centroids.copy()]
    for it in range(max_iter):
        # ---------- ASSIGN ----------
        # distances: (n, k) - every point to every centroid
        distances = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        labels = distances.argmin(axis=1)

        # ---------- UPDATE ----------
        new_centroids = np.array([
            X[labels == j].mean(axis=0) if (labels == j).any() else centroids[j]
            for j in range(k)
        ])
        history.append(new_centroids.copy())

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    inertia = ((X - centroids[labels]) ** 2).sum()
    return labels, centroids, inertia, it + 1, history


# =====================================================================
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=1.0, random_state=42)

labels, centroids, inertia, iters, history = kmeans(X, k=4)
print(f"converged in {iters} iterations, inertia {inertia:.2f}")

# ---- watch it converge ----------------------------------------------
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, step in zip(axes, [0, 1, 2, 3, len(history) - 1]):
    c = history[min(step, len(history) - 1)]
    d = np.linalg.norm(X[:, None, :] - c[None, :, :], axis=2)
    lab = d.argmin(axis=1)
    ax.scatter(X[:, 0], X[:, 1], c=lab, cmap="viridis", s=18, alpha=0.7)
    ax.scatter(c[:, 0], c[:, 1], c="red", marker="X", s=250, edgecolor="k")
    ax.set_title(f"iteration {step}" if step < len(history) - 1 else "converged")
plt.tight_layout(); plt.show()

# ---- compare to sklearn ---------------------------------------------
from sklearn.cluster import KMeans
sk = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
print(f"my inertia     : {inertia:.4f}")
print(f"sklearn inertia: {sk.inertia_:.4f}")

# ---- why n_init matters ---------------------------------------------
print("\\nRANDOM vs K-MEANS++ INITIALISATION (20 runs each)")
for name, pp in [("random", False), ("k-means++", True)]:
    inertias = [kmeans(X, 4, seed=s, plus_plus=pp)[2] for s in range(20)]
    print(f"  {name:12s} best {min(inertias):8.2f}  worst {max(inertias):8.2f}  "
          f"spread {max(inertias)-min(inertias):7.2f}")
print("\\n  Random initialisation sometimes lands in a bad local optimum.")
print("  k-means++ makes that far less likely. sklearn uses it by default,")
print("  AND runs n_init times, keeping the best.")
~~~

### Using it properly

~~~python kmeans_proper.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.datasets import make_blobs

rng = np.random.default_rng(0)
n = 800
customers = pd.DataFrame({
    "annual_spend": np.concatenate([rng.normal(2000, 400, 300),
                                    rng.normal(9000, 1500, 250),
                                    rng.normal(400, 120, 250)]),
    "visits_per_year": np.concatenate([rng.normal(24, 5, 300),
                                       rng.normal(48, 9, 250),
                                       rng.normal(6, 2, 250)]),
    "avg_basket": np.concatenate([rng.normal(83, 15, 300),
                                  rng.normal(188, 30, 250),
                                  rng.normal(67, 12, 250)]),
})

# =====================================================================
# SCALING IS MANDATORY - demonstrate it
# =====================================================================
raw = KMeans(4, n_init=10, random_state=0).fit(customers)
scaled = make_pipeline(StandardScaler(),
                       KMeans(4, n_init=10, random_state=0)).fit(customers)

print("feature ranges:")
print((customers.max() - customers.min()).round(1).to_string())
print("\\ncluster sizes WITHOUT scaling:", np.bincount(raw.labels_))
print("cluster sizes WITH scaling   :",
      np.bincount(scaled[-1].labels_))
print("\\nWithout scaling, annual_spend (range ~12000) completely dominates")
print("visits_per_year (range ~50). The clustering is effectively 1-D.")

# =====================================================================
# CHOOSING k - three methods, used together
# =====================================================================
Xs = StandardScaler().fit_transform(customers)
ks = range(2, 11)
inertias, silhouettes = [], []
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
ch, db = [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=0).fit(Xs)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(Xs, km.labels_))
    ch.append(calinski_harabasz_score(Xs, km.labels_))
    db.append(davies_bouldin_score(Xs, km.labels_))

fig, ax = plt.subplots(1, 4, figsize=(19, 4))
ax[0].plot(ks, inertias, "o-"); ax[0].set_title("Elbow (inertia)\\nlower is better, look for the bend")
ax[1].plot(ks, silhouettes, "o-", color="darkorange")
ax[1].axvline(list(ks)[int(np.argmax(silhouettes))], color="green", ls="--")
ax[1].set_title("Silhouette\\nHIGHER is better")
ax[2].plot(ks, ch, "o-", color="seagreen")
ax[2].set_title("Calinski-Harabasz\\nHIGHER is better")
ax[3].plot(ks, db, "o-", color="crimson")
ax[3].set_title("Davies-Bouldin\\nLOWER is better")
for a in ax:
    a.set_xlabel("k"); a.grid(alpha=0.3)
plt.tight_layout(); plt.show()

print(f"\\n{'k':>4} {'inertia':>12} {'silhouette':>12} {'CH':>10} {'DB':>8}")
for i, k in enumerate(ks):
    print(f"{k:>4} {inertias[i]:>12.1f} {silhouettes[i]:>12.4f} "
          f"{ch[i]:>10.1f} {db[i]:>8.3f}")
print(f"\\nsilhouette says k = {list(ks)[int(np.argmax(silhouettes))]}")
print(f"the data was generated with 3 groups.")

# =====================================================================
# THE SILHOUETTE PLOT - much more informative than the single number
# =====================================================================
best_k = 3
km = KMeans(best_k, n_init=10, random_state=0).fit(Xs)
sample_sil = silhouette_samples(Xs, km.labels_)

plt.figure(figsize=(9, 6))
y_lower = 10
for i in range(best_k):
    vals = np.sort(sample_sil[km.labels_ == i])
    y_upper = y_lower + len(vals)
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, vals, alpha=0.75)
    plt.text(-0.05, y_lower + 0.5 * len(vals), str(i))
    y_lower = y_upper + 10
plt.axvline(sample_sil.mean(), color="red", ls="--",
            label=f"average = {sample_sil.mean():.3f}")
plt.xlabel("silhouette coefficient"); plt.ylabel("cluster")
plt.legend(); plt.title("Silhouette plot: are any clusters weak?")
plt.tight_layout(); plt.show()

print("\\nREADING IT:")
print("  values near +1 : the point is far from other clusters (good)")
print("  values near  0 : the point sits on a boundary")
print("  values below 0 : the point is probably in the WRONG cluster")
print(f"  points with negative silhouette: {(sample_sil < 0).sum()}")

# =====================================================================
# PROFILE THE CLUSTERS - this is the actual deliverable
# =====================================================================
customers["cluster"] = km.labels_
profile = customers.groupby("cluster").agg(["mean", "count"]).round(1)
print("\\nCLUSTER PROFILES")
print(customers.groupby("cluster").mean().round(1).to_string())
print("\\nsizes:", np.bincount(km.labels_))
print("\\nNaming them is YOUR job, and it is the part that creates value:")
means = customers.groupby("cluster")["annual_spend"].mean().sort_values()
names = ["Occasional (low spend, rare visits)",
         "Regular (moderate spend, frequent)",
         "VIP (high spend, high basket)"]
for (c, _), nm in zip(means.items(), names):
    print(f"  cluster {c} -> {nm}")
~~~

:::tip Clustering output is a starting point, not an answer
The algorithm gives you group IDs. The value comes from **profiling** each group and
giving it a name a business person can act on. If you cannot describe a cluster in one
sentence, either k is wrong or the features are wrong.
:::
`
},
{
name: 'Where K-Means fails',
goal: 'See all four failure modes concretely, and know which algorithm to switch to.',
md: `
~~~python kmeans_failures.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)

# ---- four problem datasets ------------------------------------------
datasets = {}

# 1. non-spherical (moons)
datasets["1. crescent shapes"] = make_moons(n_samples=500, noise=0.06,
                                            random_state=0)[0]

# 2. elongated / anisotropic
Xa, _ = make_blobs(n_samples=500, centers=3, random_state=170)
datasets["2. elongated clusters"] = Xa @ [[0.6, -0.6], [-0.4, 0.8]]

# 3. very different variances
datasets["3. unequal variance"] = make_blobs(
    n_samples=500, centers=3, cluster_std=[1.0, 3.0, 0.4], random_state=170)[0]

# 4. very different sizes
Xb, yb = make_blobs(n_samples=1500, centers=3, random_state=170)
datasets["4. unequal sizes"] = np.vstack([Xb[yb == 0][:500],
                                          Xb[yb == 1][:80],
                                          Xb[yb == 2][:20]])

algorithms = {
    "KMeans": lambda X, k: KMeans(k, n_init=10, random_state=0).fit_predict(X),
    "GaussianMixture": lambda X, k: GaussianMixture(k, random_state=0).fit_predict(X),
    "Agglomerative(ward)": lambda X, k: AgglomerativeClustering(k).fit_predict(X),
    "DBSCAN": lambda X, k: DBSCAN(eps=0.3, min_samples=8).fit_predict(
        StandardScaler().fit_transform(X)),
    "Spectral": lambda X, k: SpectralClustering(
        k, affinity="nearest_neighbors", random_state=0,
        assign_labels="kmeans").fit_predict(X),
}

fig, axes = plt.subplots(len(datasets), len(algorithms),
                         figsize=(4 * len(algorithms), 3.6 * len(datasets)))
for i, (dname, X) in enumerate(datasets.items()):
    Xs = StandardScaler().fit_transform(X)
    for j, (aname, fn) in enumerate(algorithms.items()):
        try:
            labels = fn(Xs, 3 if "1." not in dname else 2)
        except Exception:
            labels = np.zeros(len(Xs))
        ax = axes[i, j]
        ax.scatter(Xs[:, 0], Xs[:, 1], c=labels, cmap="viridis", s=12)
        n_clusters = len(set(labels) - {-1})
        n_noise = int((labels == -1).sum())
        title = f"{aname}" if i == 0 else ""
        ax.set_title(f"{title}\\n{n_clusters} clusters" +
                     (f", {n_noise} noise" if n_noise else ""), fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(dname, fontsize=10)
plt.tight_layout(); plt.show()
~~~

### The failure-to-fix table

| Failure | Why K-Means fails | Use instead |
|---|---|---|
| **Crescent / ring shapes** | It draws straight boundaries (Voronoi cells) | DBSCAN, Spectral clustering |
| **Elongated clusters** | Squared distance is isotropic; it wants spheres | **GaussianMixture** with ~covariance_type="full"~ |
| **Unequal variance** | Large-variance clusters get split | GaussianMixture |
| **Unequal sizes** | The objective favours equal-mass clusters | DBSCAN, or Gaussian mixture |
| **Outliers present** | Every point must join a cluster, and outliers drag centroids | DBSCAN (has a noise label) |
| **Unknown k** | You must specify it | DBSCAN, HDBSCAN, hierarchical |

### The outlier problem, demonstrated

~~~python outlier_effect.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.6, random_state=0)
X_out = np.vstack([X, [[12, 12], [13, 11], [-9, -9]]])    # three far-away points

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

km_clean = KMeans(3, n_init=10, random_state=0).fit(X)
ax[0].scatter(X[:, 0], X[:, 1], c=km_clean.labels_, cmap="viridis", s=18)
ax[0].scatter(*km_clean.cluster_centers_.T, c="red", marker="X", s=250, edgecolor="k")
ax[0].set_title("K-Means, clean data")

km_dirty = KMeans(3, n_init=10, random_state=0).fit(X_out)
ax[1].scatter(X_out[:, 0], X_out[:, 1], c=km_dirty.labels_, cmap="viridis", s=18)
ax[1].scatter(*km_dirty.cluster_centers_.T, c="red", marker="X", s=250, edgecolor="k")
ax[1].set_title("K-Means, 3 outliers added\\nan entire cluster is wasted on them")

db = DBSCAN(eps=0.8, min_samples=6).fit(X_out)
noise = db.labels_ == -1
ax[2].scatter(X_out[~noise, 0], X_out[~noise, 1], c=db.labels_[~noise],
              cmap="viridis", s=18)
ax[2].scatter(X_out[noise, 0], X_out[noise, 1], c="red", marker="x", s=90)
ax[2].set_title(f"DBSCAN\\n{noise.sum()} points labelled NOISE")
plt.tight_layout(); plt.show()

print("K-Means centroid shift caused by 3 outliers out of 303 points:")
print(f"  clean centroids:\\n{np.sort(km_clean.cluster_centers_, axis=0).round(2)}")
print(f"  with outliers  :\\n{np.sort(km_dirty.cluster_centers_, axis=0).round(2)}")
print("\\nK-Means has no concept of 'this point does not belong anywhere'.")

# MiniBatchKMeans for large data
from sklearn.cluster import MiniBatchKMeans
import time
Xbig, _ = make_blobs(n_samples=500000, centers=8, n_features=20, random_state=0)
for name, m in [("KMeans", KMeans(8, n_init=3, random_state=0)),
                ("MiniBatchKMeans", MiniBatchKMeans(8, n_init=3, batch_size=2048,
                                                    random_state=0))]:
    t0 = time.perf_counter(); m.fit(Xbig); dt = time.perf_counter() - t0
    print(f"\\n{name:18s} {dt:6.2f}s  inertia {m.inertia_:,.0f}")
~~~

:::tip The decision guide
~~~text
Do you know how many clusters you want?
  NO  -> DBSCAN / HDBSCAN (density) or hierarchical (cut the dendrogram later)
  YES -> continue

Are the clusters roughly round and similar in size?
  YES -> KMeans (fast, simple, scales)
  NO  -> continue

Are they elliptical / different variances?
  YES -> GaussianMixture(covariance_type='full')
  NO  -> continue

Arbitrary shapes, or outliers present?
  YES -> DBSCAN / HDBSCAN

More than 100,000 rows?
  -> MiniBatchKMeans, or HDBSCAN with an approximate index
~~~
:::
`
}
],
quiz: [
{
q: 'Your clusters are elongated ellipses. K-Means splits them wrongly. What should you use?',
options: [
  'K-Means with a larger k',
  'GaussianMixture with covariance_type="full"',
  'More iterations',
  'Remove features'
],
answer: 1,
why: 'K-Means minimises isotropic squared distance, so it can only find spherical clusters. A Gaussian mixture with full covariance fits arbitrary ellipses.'
},
{
q: 'Why must you scale features before K-Means?',
options: [
  'To speed it up',
  'It is distance-based, so the largest-range feature would dominate every distance calculation',
  'sklearn requires it',
  'To avoid negative values'
],
answer: 1,
why: 'A feature ranging over thousands swamps one ranging over tens, making the clustering effectively one-dimensional regardless of what the other features contain.'
},
{
q: 'Three extreme outliers are added to your data. What does K-Means do?',
options: [
  'Labels them as noise',
  'Ignores them',
  'Wastes a whole cluster on them and shifts the other centroids',
  'Raises an error'
],
answer: 2,
why: 'Every point must be assigned, and squared distance means far points have enormous pull. DBSCAN, which has an explicit noise label, handles this properly.'
},
{
q: 'The elbow plot is ambiguous. What is a better guide to k?',
options: [
  'Always use k=3',
  'Silhouette score, plus a silhouette plot and business interpretability of the resulting clusters',
  'The number of features',
  'Inertia at k = n'
],
answer: 1,
why: 'Silhouette measures separation directly and has an unambiguous optimum. The silhouette plot also exposes weak clusters, and ultimately the clusters must be nameable to be useful.'
}
]
},

/* ============================================================ */
{
id: 'other-clustering',
title: 'Hierarchical, DBSCAN and Gaussian mixtures',
summary: 'Three algorithms that fix what K-Means cannot do: nested structure without choosing k, arbitrary shapes with outlier detection, and soft probabilistic membership.',
tags: ['clustering', 'density', 'probabilistic'],
intro: `
## Hierarchical clustering

Build a tree of nested clusters, then cut it wherever you like.

~~~text
AGGLOMERATIVE (bottom-up, the common one)
  start: every point is its own cluster
  repeat: merge the two closest clusters
  end: one cluster containing everything
  -> the merge history is a DENDROGRAM; cut it at any height to get k clusters

LINKAGE = how you measure the distance between two CLUSTERS
  ward      minimise the increase in within-cluster variance (default, K-Means-like)
  complete  the distance between the two FARTHEST members  -> compact clusters
  average   the mean pairwise distance                     -> a middle ground
  single    the distance between the two CLOSEST members   -> chains, finds shapes
~~~

## DBSCAN

Density-Based Spatial Clustering. **No k, arbitrary shapes, explicit noise.**

~~~text
Two parameters:
  eps          the neighbourhood radius
  min_samples  how many neighbours make a point a "core point"

CORE point    : has >= min_samples neighbours within eps
BORDER point  : within eps of a core point, but not itself core
NOISE point   : neither  -> labelled -1

Clusters are chains of connected core points.
~~~

## Gaussian Mixture Models

K-Means assumes each cluster is a sphere and assigns hard labels. A GMM models each cluster
as a **Gaussian with its own mean and covariance**, and gives each point a **probability**
of belonging to each one.

:::math The relationship
K-Means is the limiting case of a Gaussian mixture where every covariance is a tiny
identical sphere and assignments are hard. GMM is the general, soft version.
:::
`,
keyPoints: [
  'A dendrogram lets you choose k after seeing the structure, not before.',
  'DBSCAN needs no k, finds arbitrary shapes, and explicitly labels outliers.',
  'DBSCAN eps is chosen from a k-distance plot, not guessed.',
  'GMM gives soft probabilistic membership and fits elliptical clusters.'
],
pitfalls: [
  'Running hierarchical clustering on 100,000 points - it is O(n squared) in memory.',
  'Guessing DBSCAN eps instead of reading it off the k-distance elbow.',
  'Using DBSCAN on data with widely varying density - use HDBSCAN instead.',
  'Forgetting that GMM, like K-Means, needs the number of components specified (use BIC).'
],
levels: [
{
name: 'Dendrograms and density',
goal: 'Read a dendrogram to choose k, and tune DBSCAN properly with a k-distance plot.',
md: `
~~~python hierarchical.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

X, y_true = make_blobs(n_samples=150, centers=4, cluster_std=1.1, random_state=42)
Xs = StandardScaler().fit_transform(X)

# =====================================================================
# THE DENDROGRAM
# =====================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
for ax, method in zip(axes.ravel(), ["ward", "complete", "average", "single"]):
    Z = linkage(Xs, method=method)
    dendrogram(Z, ax=ax, truncate_mode="level", p=5, no_labels=True,
               color_threshold=Z[-3, 2])
    ax.set_title(f"linkage = '{method}'")
    ax.set_ylabel("merge distance")
plt.suptitle("The same data, four linkage criteria", y=1.01)
plt.tight_layout(); plt.show()

# =====================================================================
# READING IT: the biggest vertical gap suggests where to cut
# =====================================================================
Z = linkage(Xs, method="ward")
merge_distances = Z[:, 2]
gaps = np.diff(merge_distances[-10:])
print("last 10 merge distances:", merge_distances[-10:].round(3))
print("gaps between them      :", gaps.round(3))
suggested_k = int(np.argmax(gaps[::-1])) + 2
print(f"\\nbiggest gap suggests cutting into {suggested_k} clusters")

# cut it three ways
print(f"\\n{'k':>4} {'silhouette':>12} {'cluster sizes'}")
for k in range(2, 8):
    labels = fcluster(Z, k, criterion="maxclust")
    print(f"{k:>4} {silhouette_score(Xs, labels):>12.4f}  {np.bincount(labels)[1:]}")

# cut by DISTANCE instead of by count
labels_by_dist = fcluster(Z, t=merge_distances[-4], criterion="distance")
print(f"\\ncutting at distance {merge_distances[-4]:.3f} gives "
      f"{len(set(labels_by_dist))} clusters")

# ---- sklearn version, with distance_threshold instead of n_clusters --
auto = AgglomerativeClustering(n_clusters=None, distance_threshold=8.0,
                               linkage="ward").fit(Xs)
print(f"AgglomerativeClustering with threshold 8.0 found "
      f"{auto.n_clusters_} clusters")

# ---- linkage comparison on non-spherical data -----------------------
from sklearn.datasets import make_moons
Xm, _ = make_moons(n_samples=300, noise=0.06, random_state=0)
Xm = StandardScaler().fit_transform(Xm)

fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, method in zip(ax, ["ward", "complete", "average", "single"]):
    lab = AgglomerativeClustering(2, linkage=method).fit_predict(Xm)
    a.scatter(Xm[:, 0], Xm[:, 1], c=lab, cmap="viridis", s=18)
    a.set_title(f"linkage='{method}'")
plt.suptitle("On crescents, only SINGLE linkage follows the shape", y=1.03)
plt.tight_layout(); plt.show()
print("\\nsingle linkage chains along dense paths, so it finds elongated shapes -")
print("but it is fragile: a single bridging point merges two real clusters.")
~~~

### DBSCAN, tuned properly

~~~python dbscan.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# a mixture: two crescents plus scattered noise
Xm, _ = make_moons(n_samples=400, noise=0.06, random_state=0)
rng = np.random.default_rng(0)
noise_pts = rng.uniform(Xm.min(0) - 0.4, Xm.max(0) + 0.4, (40, 2))
X = StandardScaler().fit_transform(np.vstack([Xm, noise_pts]))

# =====================================================================
# CHOOSING eps: the k-distance plot
# =====================================================================
MIN_SAMPLES = 8      # rule of thumb: 2 * n_features, at least 4
nn = NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(X)
distances, _ = nn.kneighbors(X)
kth = np.sort(distances[:, -1])          # distance to the MIN_SAMPLES-th neighbour

# the elbow: the point of maximum curvature
diffs = np.diff(kth)
elbow_idx = int(np.argmax(diffs[len(diffs)//2:])) + len(diffs)//2
suggested_eps = kth[elbow_idx]

plt.figure(figsize=(9, 5))
plt.plot(kth, lw=2)
plt.axhline(suggested_eps, color="red", ls="--",
            label=f"suggested eps = {suggested_eps:.3f}")
plt.xlabel("points, sorted"); plt.ylabel(f"distance to {MIN_SAMPLES}th neighbour")
plt.legend(); plt.grid(alpha=0.3)
plt.title("k-distance plot: the elbow is your eps")
plt.tight_layout(); plt.show()

# =====================================================================
# SEE WHAT eps DOES
# =====================================================================
fig, axes = plt.subplots(1, 5, figsize=(21, 4.2))
print(f"{'eps':>7} {'clusters':>9} {'noise':>7} {'silhouette':>12}")
print("-" * 40)
for ax, eps in zip(axes, [0.08, 0.15, suggested_eps, 0.4, 0.9]):
    db = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit(X)
    lab = db.labels_
    n_clusters = len(set(lab) - {-1})
    n_noise = int((lab == -1).sum())
    sil = (silhouette_score(X[lab != -1], lab[lab != -1])
           if n_clusters > 1 and (lab != -1).sum() > n_clusters else float("nan"))
    print(f"{eps:>7.3f} {n_clusters:>9} {n_noise:>7} {sil:>12.4f}")

    ax.scatter(X[lab != -1, 0], X[lab != -1, 1], c=lab[lab != -1],
               cmap="viridis", s=16)
    ax.scatter(X[lab == -1, 0], X[lab == -1, 1], c="red", marker="x", s=30)
    ax.set_title(f"eps={eps:.3f}\\n{n_clusters} clusters, {n_noise} noise")
plt.tight_layout(); plt.show()
~~~

~~~text
    eps  clusters   noise   silhouette
----------------------------------------
  0.080        14     271       0.1204
  0.150         4      69       0.3891
  0.213         2      41       0.6109
  0.400         2       4       0.5218
  0.900         1       0            nan
~~~

:::warn DBSCAN's weakness: varying density
DBSCAN uses ONE global eps. If one cluster is dense and another sparse, no single eps works
- you either merge the dense ones or shatter the sparse one.

**HDBSCAN** solves this by building a hierarchy over density levels and extracting the most
stable clusters at each. It has essentially one parameter and no eps at all:
~~~python
from sklearn.cluster import HDBSCAN            # sklearn >= 1.3
labels = HDBSCAN(min_cluster_size=15).fit_predict(X)
~~~
For real-world clustering, HDBSCAN is usually the better first choice.
:::

~~~python varying_density.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=[300, 300, 300], centers=[[0, 0], [6, 6], [0, 8]],
                  cluster_std=[0.3, 1.6, 0.6], random_state=0)

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].scatter(X[:, 0], X[:, 1], s=14, c="grey")
ax[0].set_title("three clusters, very different densities")

for a, (name, lab) in zip(ax[1:], [
        ("DBSCAN(eps=0.5)", DBSCAN(eps=0.5, min_samples=8).fit_predict(X)),
        ("HDBSCAN", HDBSCAN(min_cluster_size=25).fit_predict(X))]):
    a.scatter(X[lab != -1, 0], X[lab != -1, 1], c=lab[lab != -1], cmap="viridis", s=14)
    a.scatter(X[lab == -1, 0], X[lab == -1, 1], c="red", marker="x", s=25)
    a.set_title(f"{name}\\n{len(set(lab) - {-1})} clusters, {(lab==-1).sum()} noise")
plt.tight_layout(); plt.show()
~~~
`
},
{
name: 'Gaussian mixtures: soft, probabilistic clustering',
goal: 'Fit a GMM, read the membership probabilities, and choose the number of components with BIC.',
md: `
~~~python gmm.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)

# anisotropic, overlapping clusters - exactly where K-Means struggles
X, y_true = make_blobs(n_samples=700, centers=3, cluster_std=[1.0, 2.2, 0.7],
                       random_state=170)
X = X @ np.array([[0.7, -0.5], [-0.35, 0.85]])       # stretch and rotate

# =====================================================================
# 1. GMM vs KMEANS
# =====================================================================
km = KMeans(3, n_init=10, random_state=0).fit(X)
gm = GaussianMixture(3, covariance_type="full", random_state=0).fit(X)

def draw_ellipses(gmm, ax):
    for mean, cov, w in zip(gmm.means_, gmm.covariances_, gmm.weights_):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        for n_std in (1, 2):
            ax.add_patch(Ellipse(mean, 2 * n_std * np.sqrt(vals[0]),
                                 2 * n_std * np.sqrt(vals[1]), angle=angle,
                                 fill=False, edgecolor="red", lw=2, alpha=0.7))

fig, ax = plt.subplots(1, 3, figsize=(17, 5))
ax[0].scatter(X[:, 0], X[:, 1], c=km.labels_, cmap="viridis", s=14)
ax[0].scatter(*km.cluster_centers_.T, c="red", marker="X", s=220, edgecolor="k")
ax[0].set_title("K-Means: straight boundaries, spherical assumption")

ax[1].scatter(X[:, 0], X[:, 1], c=gm.predict(X), cmap="viridis", s=14)
draw_ellipses(gm, ax[1])
ax[1].set_title("GMM: fitted ellipses match the real shapes")

# uncertainty: colour by how CONFIDENT the assignment is
probs = gm.predict_proba(X)
confidence = probs.max(axis=1)
sc = ax[2].scatter(X[:, 0], X[:, 1], c=confidence, cmap="RdYlGn", s=18,
                   vmin=0.34, vmax=1.0)
plt.colorbar(sc, ax=ax[2], label="max membership probability")
ax[2].set_title("GMM knows what it is UNSURE about")
plt.tight_layout(); plt.show()

# =====================================================================
# 2. SOFT MEMBERSHIP - the real advantage
# =====================================================================
uncertain = np.argsort(confidence)[:5]
print("THE FIVE MOST AMBIGUOUS POINTS")
print(pd.DataFrame(probs[uncertain].round(3),
                   columns=[f"P(cluster {i})" for i in range(3)]).to_string())
print("\\nK-Means would give each of these a confident hard label.")
print(f"points with max probability below 0.6: {(confidence < 0.6).sum()}")

# =====================================================================
# 3. CHOOSING THE NUMBER OF COMPONENTS with BIC / AIC
# =====================================================================
n_range = range(1, 11)
cov_types = ["spherical", "diag", "tied", "full"]

results = []
for cov in cov_types:
    for n in n_range:
        g = GaussianMixture(n, covariance_type=cov, random_state=0,
                            n_init=3).fit(X)
        results.append({"covariance": cov, "n": n,
                        "BIC": g.bic(X), "AIC": g.aic(X)})
res = pd.DataFrame(results)

plt.figure(figsize=(12, 4.5))
plt.subplot(1, 2, 1)
for cov in cov_types:
    sub = res[res["covariance"] == cov]
    plt.plot(sub["n"], sub["BIC"], "o-", label=cov)
plt.xlabel("number of components"); plt.ylabel("BIC (lower is better)")
plt.legend(); plt.grid(alpha=0.3); plt.title("BIC")

plt.subplot(1, 2, 2)
for cov in cov_types:
    sub = res[res["covariance"] == cov]
    plt.plot(sub["n"], sub["AIC"], "o-", label=cov)
plt.xlabel("number of components"); plt.ylabel("AIC")
plt.legend(); plt.grid(alpha=0.3); plt.title("AIC")
plt.tight_layout(); plt.show()

best = res.loc[res["BIC"].idxmin()]
print(f"\\nBIC selects: {int(best['n'])} components, "
      f"covariance_type='{best['covariance']}'  (true answer: 3, full)")

print("\\nCOVARIANCE TYPES")
print("  spherical : one variance per component      (K-Means-like)")
print("  diag      : axis-aligned ellipses")
print("  tied      : all components share one shape")
print("  full      : each component has its own arbitrary ellipse (most flexible)")

# =====================================================================
# 4. GMM AS A DENSITY MODEL - generate new data, score likelihood
# =====================================================================
new_samples, new_labels = gm.sample(300)
print(f"\\ngenerated {len(new_samples)} synthetic points from the fitted model")

# log-likelihood as an anomaly score
scores = gm.score_samples(X)
threshold = np.percentile(scores, 2)
outliers = scores < threshold
print(f"points below the 2nd percentile of log-likelihood: {outliers.sum()}")
print("A GMM is a full probability model, so it does density estimation,")
print("generation and anomaly detection - not just clustering.")

# =====================================================================
# 5. LETTING THE MODEL CHOOSE k ITSELF
# =====================================================================
bgm = BayesianGaussianMixture(n_components=10, covariance_type="full",
                              weight_concentration_prior=0.01,
                              random_state=0, max_iter=500).fit(X)
active = (bgm.weights_ > 0.02).sum()
print(f"\\nBayesianGaussianMixture given 10 components used only {active}")
print(f"weights: {bgm.weights_.round(3)}")
print("It shrinks unnecessary components toward zero weight automatically.")
~~~

~~~text
THE FIVE MOST AMBIGUOUS POINTS
   P(cluster 0)  P(cluster 1)  P(cluster 2)
0         0.341         0.338         0.321
1         0.371         0.316         0.313
2         0.298         0.354         0.348
...

BIC selects: 3 components, covariance_type='full'  (true answer: 3, full)

BayesianGaussianMixture given 10 components used only 3
weights: [0.    0.    0.331 0.    0.412 0.    0.    0.257 0.    0.   ]
~~~

:::tip When to reach for a GMM
- You need **probabilities**, not hard labels (e.g. "60% likely this segment").
- Clusters are **elliptical** or overlap.
- You want a **generative** model - to sample new data or score density.
- You want a principled way to **choose k** (BIC), rather than an elbow heuristic.

The cost: it assumes each cluster is Gaussian. For genuinely arbitrary shapes, DBSCAN or
HDBSCAN remain the right tools.
:::
`
}
],
quiz: [
{
q: 'What does DBSCAN do that K-Means cannot?',
options: [
  'Run faster',
  'Find arbitrarily-shaped clusters, decide the number of clusters itself, and label outliers as noise',
  'Handle more features',
  'Give probabilities'
],
answer: 1,
why: 'Density-based clustering follows any shape, needs no k, and marks low-density points -1. K-Means forces every point into one of k spherical clusters.'
},
{
q: 'How should you choose the DBSCAN eps parameter?',
options: [
  'Always use 0.5',
  'Plot the sorted distance to the min_samples-th nearest neighbour and take the elbow',
  'Set it to the mean distance between all points',
  'Try random values'
],
answer: 1,
why: 'The k-distance plot shows where density drops off. The elbow separates within-cluster distances from between-cluster ones, which is exactly what eps should be.'
},
{
q: 'Your clusters have very different densities and DBSCAN either merges or shatters them. What helps?',
options: [
  'A larger min_samples',
  'HDBSCAN, which builds a hierarchy over density levels instead of using one global eps',
  'K-Means',
  'More data'
],
answer: 1,
why: 'A single global eps cannot suit both a dense and a sparse cluster. HDBSCAN extracts the most stable clusters across density levels and needs no eps at all.'
},
{
q: 'How do you choose the number of components in a Gaussian mixture?',
options: [
  'The elbow method on inertia',
  'Minimise BIC (or AIC), or use a BayesianGaussianMixture which prunes unused components',
  'Always use 3',
  'Silhouette score only'
],
answer: 1,
why: 'A GMM has a likelihood, so information criteria that penalise parameter count apply directly - a principled model-selection method rather than a visual heuristic.'
}
]
},

/* ============================================================ */
{
id: 'pca',
title: 'PCA and dimensionality reduction',
summary: 'Compressing many correlated features into a few uncorrelated components - the maths, the practice, and when it helps versus hurts.',
tags: ['dimensionality-reduction', 'pca', 'core'],
intro: `
## The idea

If ten features are highly correlated, they are really measuring two or three underlying
things. PCA finds those underlying directions.

~~~text
       x2                            PC2
        |    . . .                    \\    . . .
        |  . . . .                     \\ . . . .
        | . . . .        rotate         \\. . . .  ----> PC1
        |. . . .        ------->         . . . .
        +----------- x1                  the axes now align with the
                                         directions of greatest spread
~~~

PCA finds the **orthogonal directions of maximum variance**, ranks them, and lets you keep
only the top few.

:::math The mechanics
1. **Centre** the data (subtract the mean of each feature). Usually **scale** too.
2. Compute the **covariance matrix**.
3. Take its **eigenvectors** (the principal components) and **eigenvalues** (variance along each).
4. Sort by eigenvalue, keep the top k.
5. Project: **X_reduced = X_centred @ components**

In practice sklearn uses the SVD of X directly - numerically better, same result.
:::

## What PCA is and is not

**Good for**: removing multicollinearity, speeding up training, compression, denoising,
visualisation of global structure, decorrelating features.

**Not good for**: interpretability (components are blends of everything), non-linear
structure (use kernel PCA, UMAP or an autoencoder), or feature *selection* (PCA creates
new features, it does not pick existing ones).

:::danger Scaling changes the answer completely
PCA maximises variance, and variance depends on units. A feature in millimetres has
1,000,000x the variance of the same feature in metres, and will dominate PC1 for no real
reason. **Standardise first**, unless all features are already in the same unit.
:::
`,
keyPoints: [
  'Components are orthogonal directions of decreasing variance.',
  'Standardise before PCA unless every feature shares a unit.',
  'Choose k from the cumulative explained-variance curve (often 90-95%).',
  'PCA is linear - it cannot unroll curved structure.'
],
pitfalls: [
  'Fitting PCA on the full dataset before the train/test split.',
  'Applying PCA and then wondering why the features are uninterpretable.',
  'Assuming PCA improves accuracy - it often costs a little, buying speed and stability.',
  'Using PCA for visualisation when the structure is non-linear; use UMAP or t-SNE.'
],
levels: [
{
name: 'PCA by hand, then in practice',
goal: 'Derive PCA from eigenvectors, then use it for compression, denoising and speed.',
md: `
~~~python pca_scratch.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)

# correlated 2-D data
n = 400
x1 = rng.normal(0, 3, n)
x2 = 0.8 * x1 + rng.normal(0, 1.2, n)
X = np.column_stack([x1, x2])

# =====================================================================
# PCA FROM FIRST PRINCIPLES
# =====================================================================
X_centred = X - X.mean(axis=0)                       # 1. centre

cov = np.cov(X_centred.T)                            # 2. covariance
print("covariance matrix:\\n", cov.round(3))

eigenvalues, eigenvectors = np.linalg.eigh(cov)      # 3. eigen-decomposition
order = eigenvalues.argsort()[::-1]                  # 4. sort, largest first
eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

print("\\neigenvalues (variance along each component):", eigenvalues.round(3))
print("explained variance ratio:", (eigenvalues / eigenvalues.sum()).round(4))
print("components (each column is a direction):\\n", eigenvectors.round(3))

X_pca = X_centred @ eigenvectors                     # 5. project

print("\\nAFTER PROJECTION")
print("  variance along PC1:", X_pca[:, 0].var().round(3))
print("  variance along PC2:", X_pca[:, 1].var().round(3))
print("  correlation between components:",
      round(np.corrcoef(X_pca.T)[0, 1], 10), " <- exactly zero, by construction")

# ---- verify against sklearn -----------------------------------------
sk = PCA(n_components=2).fit(X)
print("\\nsklearn explained_variance_ratio_:", sk.explained_variance_ratio_.round(4))
print("sklearn components_:\\n", sk.components_.round(3))
print("(signs may be flipped - a component and its negative are equivalent)")

# ---- draw it ---------------------------------------------------------
plt.figure(figsize=(13, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, s=18)
mean = X.mean(axis=0)
for i in range(2):
    vec = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 2.5
    plt.arrow(*mean, *vec, color="red", width=0.12, head_width=0.5,
              length_includes_head=True)
    plt.text(*(mean + vec * 1.12), f"PC{i+1}\\n{eigenvalues[i]/eigenvalues.sum():.1%}",
             color="red", fontweight="bold")
plt.axis("equal"); plt.title("The principal directions"); plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=18, color="seagreen")
plt.axhline(0, color="k", lw=0.5); plt.axvline(0, color="k", lw=0.5)
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.axis("equal")
plt.title("Rotated into the new basis"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.show()
~~~

### Choosing how many components

~~~python choose_components.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import time

X, y = load_digits(return_X_y=True)         # 64 features (8x8 images)
Xs = StandardScaler().fit_transform(X)

pca_full = PCA().fit(Xs)
cum = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(13, 4.5))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(cum) + 1), pca_full.explained_variance_ratio_, "o-", ms=3)
plt.xlabel("component"); plt.ylabel("explained variance ratio")
plt.title("Scree plot"); plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(cum) + 1), cum, lw=2)
for pct in [0.80, 0.90, 0.95, 0.99]:
    k = int(np.argmax(cum >= pct)) + 1
    plt.axhline(pct, color="grey", ls=":", lw=1)
    plt.plot(k, cum[k - 1], "ro")
    plt.annotate(f"{pct:.0%} at {k} comp.", (k, cum[k - 1]),
                 textcoords="offset points", xytext=(8, -12), fontsize=9)
plt.xlabel("components kept"); plt.ylabel("cumulative explained variance")
plt.grid(alpha=0.3); plt.title("How many do you need?")
plt.tight_layout(); plt.show()

print(f"{'variance kept':>15} {'components':>12} {'compression':>13}")
for pct in [0.50, 0.80, 0.90, 0.95, 0.99]:
    k = int(np.argmax(cum >= pct)) + 1
    print(f"{pct:>14.0%} {k:>12} {k/X.shape[1]:>12.1%}")

# sklearn can do this directly
auto = PCA(n_components=0.95).fit(Xs)
print(f"\\nPCA(n_components=0.95) kept {auto.n_components_} components")

# =====================================================================
# DOES IT HELP THE MODEL?
# =====================================================================
print(f"\\n{'components':>12} {'CV accuracy':>13} {'fit time':>10}")
print("-" * 38)
for k in [None, 40, 30, 20, 10, 5]:
    if k is None:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
        label = "all 64"
    else:
        pipe = make_pipeline(StandardScaler(), PCA(k),
                             LogisticRegression(max_iter=3000))
        label = str(k)
    t0 = time.perf_counter()
    s = cross_val_score(pipe, X, y, cv=5, n_jobs=-1)
    dt = time.perf_counter() - t0
    print(f"{label:>12} {s.mean():>13.4f} {dt:>9.2f}s")
~~~

~~~text
  variance kept   components   compression
            50%            6         9.4%
            80%           21        32.8%
            90%           29        45.3%
            95%           40        62.5%
            99%           55        85.9%

  components   CV accuracy   fit time
--------------------------------------
      all 64        0.9243      4.11s
          40        0.9260      2.63s
          30        0.9199      2.01s
          20        0.9088      1.44s
          10        0.8280      0.93s
           5        0.6155      0.61s
~~~

**40 components kept 95% of variance, matched the accuracy, and cut fit time by 36%.**
That is the realistic value of PCA: speed and stability, not accuracy.

### Compression and denoising

~~~python pca_denoise.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

digits = load_digits()
X = digits.data
rng = np.random.default_rng(0)
X_noisy = X + rng.normal(0, 4.5, X.shape)

# ---- reconstruction at several compression levels -------------------
fig, axes = plt.subplots(6, 8, figsize=(13, 10))
for j in range(8):
    axes[0, j].imshow(X[j].reshape(8, 8), cmap="gray")
    axes[0, j].axis("off")
axes[0, 0].set_title("original", loc="left", fontsize=9)

for i, k in enumerate([64, 32, 16, 8, 4], start=1):
    p = PCA(n_components=k).fit(X)
    recon = p.inverse_transform(p.transform(X))
    err = np.mean((X - recon) ** 2)
    for j in range(8):
        axes[i, j].imshow(recon[j].reshape(8, 8), cmap="gray")
        axes[i, j].axis("off")
    axes[i, 0].set_title(f"{k} comp, MSE {err:.1f}", loc="left", fontsize=9)
plt.tight_layout(); plt.show()

# ---- denoising: keep only the components that carry signal ----------
p = PCA(n_components=0.85).fit(X_noisy)
denoised = p.inverse_transform(p.transform(X_noisy))
print(f"components used for denoising: {p.n_components_}")
print(f"MSE noisy vs clean   : {np.mean((X_noisy - X) ** 2):.2f}")
print(f"MSE denoised vs clean: {np.mean((denoised - X) ** 2):.2f}")
print("\\nNoise is spread thinly across ALL components; signal concentrates in")
print("the first few. Dropping the tail removes noise preferentially.")

fig, ax = plt.subplots(3, 8, figsize=(13, 5.2))
for j in range(8):
    ax[0, j].imshow(X[j].reshape(8, 8), cmap="gray"); ax[0, j].axis("off")
    ax[1, j].imshow(X_noisy[j].reshape(8, 8), cmap="gray"); ax[1, j].axis("off")
    ax[2, j].imshow(denoised[j].reshape(8, 8), cmap="gray"); ax[2, j].axis("off")
for i, t in enumerate(["clean", "noisy", "PCA denoised"]):
    ax[i, 0].set_title(t, loc="left", fontsize=9)
plt.tight_layout(); plt.show()
~~~

:::warn PCA inside a pipeline, always
~~~python
# WRONG - the components were learned from test data too
X_pca = PCA(30).fit_transform(X)
X_tr, X_te = train_test_split(X_pca, ...)

# RIGHT
pipe = make_pipeline(StandardScaler(), PCA(30), LogisticRegression())
~~~
:::

### Interpreting the components

~~~python pca_loadings.py
import numpy as np, pandas as pd
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = load_wine()
Xs = StandardScaler().fit_transform(data.data)
p = PCA(n_components=3).fit(Xs)

loadings = pd.DataFrame(p.components_.T,
                        columns=[f"PC{i+1}" for i in range(3)],
                        index=data.feature_names)
print("LOADINGS (how much each original feature contributes)")
print(loadings.round(3).to_string())

print("\\nWhat each component seems to represent:")
for i in range(3):
    top = loadings[f"PC{i+1}"].abs().nlargest(4)
    print(f"\\n  PC{i+1} ({p.explained_variance_ratio_[i]:.1%} of variance)")
    for feat in top.index:
        v = loadings.loc[feat, f'PC{i+1}']
        print(f"    {feat:28s} {v:+.3f}")
~~~
`
}
],
quiz: [
{
q: 'Why must you standardise before PCA?',
options: [
  'PCA cannot handle negative numbers',
  'PCA maximises variance, which depends on units - a feature in millimetres would dominate PC1 for no real reason',
  'It makes it faster',
  'You do not need to'
],
answer: 1,
why: 'Variance is unit-dependent. Without standardisation the component ranking reflects measurement scales rather than actual structure. The exception is when every feature shares a unit (e.g. pixel intensities).'
},
{
q: 'You keep 40 of 64 components. Accuracy is unchanged and training is 36% faster. Is PCA worth it?',
options: [
  'No - accuracy did not improve',
  'Yes - the value is speed, decorrelation and stability, not accuracy',
  'No - you lost information',
  'Only if accuracy improves'
],
answer: 1,
why: 'PCA rarely improves accuracy on clean data. Its wins are compute, removal of multicollinearity, denoising and compression. Judge it against those goals.'
},
{
q: 'Your data lies on a curved manifold (a Swiss roll). What does PCA do?',
options: [
  'Unrolls it correctly',
  'Fails - PCA is a linear projection and will flatten the roll, mixing distant points',
  'Raises an error',
  'Works but is slow'
],
answer: 1,
why: 'PCA can only rotate and project linearly. For curved structure use kernel PCA, t-SNE, UMAP or an autoencoder.'
},
{
q: 'Why are PCA components hard to interpret?',
options: [
  'They are randomly ordered',
  'Each component is a weighted blend of every original feature, not one of them',
  'They have no units',
  'sklearn hides them'
],
answer: 1,
why: 'A component is a linear combination of all features. You can inspect loadings to characterise it, but you can no longer say "this coefficient is the effect of income".'
}
]
},

/* ============================================================ */
{
id: 'manifold',
title: 't-SNE and UMAP',
summary: 'Non-linear dimensionality reduction for visualisation - what they show, what they hide, and how to avoid over-interpreting the picture.',
tags: ['dimensionality-reduction', 'visualisation', 'advanced'],
intro: `
## Why not just use PCA for plots?

PCA preserves **global** structure with a linear projection. If your data lies on a curved
manifold, PCA flattens it and distant points overlap.

t-SNE and UMAP are **non-linear** and preserve **local neighbourhoods** - points that were
close stay close. The result looks dramatically better.

~~~text
                PCA                 t-SNE / UMAP
structure       global               local
transform       linear               non-linear
new data        transform() works    t-SNE: no; UMAP: yes
reversible      yes                  no (UMAP roughly)
speed           very fast            slow / moderate
determinism     yes                  no (random init)
distances       meaningful           NOT meaningful between clusters
cluster sizes   meaningful           NOT meaningful
~~~

:::danger What a t-SNE plot does NOT tell you
1. **Cluster sizes are meaningless.** t-SNE expands dense clusters and contracts sparse ones.
2. **Distances between clusters are meaningless.** Two clusters far apart on the plot may be
   adjacent in the real space.
3. **Apparent clusters may not be real.** t-SNE produces cluster-looking blobs even from
   pure random noise.

Use it to **generate hypotheses**, never to prove one.
:::

## The key parameter

**perplexity** (t-SNE) or **n_neighbors** (UMAP) sets the size of the local neighbourhood
each point considers. Small values emphasise fine local structure; large values emphasise
global layout. **Always try several.**
`,
keyPoints: [
  'These are visualisation tools, not preprocessing steps for a model.',
  'Distances between clusters and cluster sizes carry no meaning.',
  'Always run several perplexity / n_neighbors values before concluding anything.',
  'UMAP is faster, preserves more global structure, and can transform new data.'
],
pitfalls: [
  'Feeding t-SNE output into a classifier - use PCA or UMAP if you need a reusable transform.',
  'Reading distance between blobs as similarity.',
  'Running t-SNE on 200 raw features - reduce with PCA to about 50 first.',
  'Reporting one t-SNE plot as evidence of cluster structure.'
],
levels: [
{
name: 'Using them, and reading them honestly',
goal: 'Produce good embeddings, tune the neighbourhood parameter, and see the traps first-hand.',
md: `
~~~bash
pip install umap-learn
~~~

~~~python tsne_umap.py
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
print(f"data: {X.shape}")

# =====================================================================
# BEST PRACTICE: PCA first, THEN t-SNE
#   - removes noise, speeds t-SNE up enormously
#   - recommended whenever you have more than ~50 features
# =====================================================================
X50 = PCA(n_components=30, random_state=0).fit_transform(StandardScaler().fit_transform(X))
print(f"after PCA: {X50.shape}, "
      f"variance kept {PCA(30).fit(StandardScaler().fit_transform(X)).explained_variance_ratio_.sum():.1%}")

embeddings = {}

t0 = time.perf_counter()
embeddings["PCA (2 components)"] = PCA(2, random_state=0).fit_transform(X50)
print(f"PCA   : {time.perf_counter()-t0:.2f}s")

t0 = time.perf_counter()
embeddings["t-SNE (perplexity 30)"] = TSNE(
    n_components=2, perplexity=30, init="pca", random_state=0,
    max_iter=1000).fit_transform(X50)
print(f"t-SNE : {time.perf_counter()-t0:.2f}s")

try:
    import umap
    t0 = time.perf_counter()
    embeddings["UMAP (n_neighbors 15)"] = umap.UMAP(
        n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(X50)
    print(f"UMAP  : {time.perf_counter()-t0:.2f}s")
except ImportError:
    print("(umap-learn not installed)")

fig, axes = plt.subplots(1, len(embeddings), figsize=(6 * len(embeddings), 5.5))
axes = np.atleast_1d(axes)
for ax, (name, emb) in zip(axes, embeddings.items()):
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=8, alpha=0.8)
    ax.set_title(name)
    ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(sc, ax=axes, label="digit", fraction=0.02)
plt.show()
~~~

### The perplexity sweep - never trust a single run

~~~python perplexity.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

perplexities = [2, 5, 15, 30, 50, 100]
fig, axes = plt.subplots(1, len(perplexities), figsize=(4 * len(perplexities), 4.2))
for ax, p in zip(axes, perplexities):
    emb = TSNE(2, perplexity=p, init="pca", random_state=0,
               max_iter=800).fit_transform(X50)
    ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=6, alpha=0.8)
    ax.set_title(f"perplexity = {p}")
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Same data, six perplexities - the picture changes a lot", y=1.03)
plt.tight_layout(); plt.show()

print("perplexity roughly = how many neighbours each point considers")
print("  too small (2-5)  : fragments real clusters into shards")
print("  good (5-50)      : the usual range; try at least three values")
print("  too large (>n/3) : everything merges into one blob")
print("\\nRule of thumb: perplexity should be well below n/3.")
~~~

### The trap: t-SNE finds clusters in pure noise

~~~python tsne_trap.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

rng = np.random.default_rng(0)

# COMPLETELY random data - no structure whatsoever
noise = rng.normal(size=(600, 40))

fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
for ax, p in zip(axes, [5, 15, 30, 60]):
    emb = TSNE(2, perplexity=p, init="random", random_state=0,
               max_iter=800).fit_transform(noise)
    ax.scatter(emb[:, 0], emb[:, 1], s=8, alpha=0.7, color="steelblue")
    ax.set_title(f"perplexity {p}")
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("This is PURE GAUSSIAN NOISE. t-SNE still produces blobs.", y=1.03)
plt.tight_layout(); plt.show()

print("If you had not been told, you would report 'we found 5 clusters'.")
print("\\nHOW TO PROTECT YOURSELF:")
print("  1. Run several perplexities - real structure persists, artefacts do not")
print("  2. Run several random seeds")
print("  3. VALIDATE with an actual clustering algorithm and a silhouette score")
print("  4. Check whether the clusters differ on real features")

# ---- and cluster SIZE / DISTANCE are meaningless --------------------
from sklearn.datasets import make_blobs
Xd, yd = make_blobs(n_samples=[400, 400, 400],
                    centers=[[0, 0], [20, 0], [22, 0]],   # two are very close!
                    cluster_std=[0.4, 0.4, 0.4], n_features=2, random_state=0)
emb = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Xd)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ax[0].scatter(Xd[:, 0], Xd[:, 1], c=yd, cmap="viridis", s=10)
ax[0].set_title("TRUE geometry: clusters 1 and 2 are adjacent,\\ncluster 0 is far away")
ax[1].scatter(emb[:, 0], emb[:, 1], c=yd, cmap="viridis", s=10)
ax[1].set_title("t-SNE: all three look equally separated")
plt.tight_layout(); plt.show()
~~~

### UMAP: usually the better choice

~~~python umap_demo.py
import numpy as np
import matplotlib.pyplot as plt
try:
    import umap
except ImportError:
    raise SystemExit("pip install umap-learn")

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X, y = load_digits(return_X_y=True)
X50 = PCA(30, random_state=0).fit_transform(StandardScaler().fit_transform(X))

# ---- the two parameters that matter ---------------------------------
fig, axes = plt.subplots(2, 4, figsize=(19, 9))
for j, nn in enumerate([5, 15, 50, 150]):
    emb = umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=0).fit_transform(X50)
    axes[0, j].scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=5)
    axes[0, j].set_title(f"n_neighbors = {nn}\\n(local <-> global)")
    axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
for j, md in enumerate([0.0, 0.1, 0.5, 0.99]):
    emb = umap.UMAP(n_neighbors=15, min_dist=md, random_state=0).fit_transform(X50)
    axes[1, j].scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=5)
    axes[1, j].set_title(f"min_dist = {md}\\n(tight <-> spread out)")
    axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
plt.tight_layout(); plt.show()

# =====================================================================
# UMAP CAN TRANSFORM NEW DATA - t-SNE cannot
# =====================================================================
X_tr, X_te, y_tr, y_te = train_test_split(X50, y, test_size=0.3, random_state=0)
reducer = umap.UMAP(n_neighbors=15, random_state=0).fit(X_tr)
emb_tr = reducer.embedding_
emb_te = reducer.transform(X_te)          # <-- this is the key advantage

plt.figure(figsize=(8, 6.5))
plt.scatter(emb_tr[:, 0], emb_tr[:, 1], c=y_tr, cmap="tab10", s=8, alpha=0.4,
            label="training")
plt.scatter(emb_te[:, 0], emb_te[:, 1], c=y_te, cmap="tab10", s=28,
            edgecolor="k", linewidth=0.4, label="new points")
plt.legend(); plt.title("UMAP projects unseen data into the SAME space")
plt.tight_layout(); plt.show()

# ---- SUPERVISED UMAP: use labels to sharpen the embedding -----------
sup = umap.UMAP(n_neighbors=15, random_state=0).fit_transform(X50, y=y)
plt.figure(figsize=(7, 6))
plt.scatter(sup[:, 0], sup[:, 1], c=y, cmap="tab10", s=6)
plt.title("Supervised UMAP - labels guide the layout")
plt.tight_layout(); plt.show()
~~~

:::tip Which to use
| Goal | Tool |
|---|---|
| Fast, reversible, feed into a model | **PCA** |
| Beautiful cluster visualisation | **UMAP** (or t-SNE) |
| Need to project new data later | **UMAP** or PCA |
| Preserve global structure as well as local | **UMAP** |
| Publication-standard local structure | **t-SNE**, with a perplexity sweep |
| Non-linear compression for a model | **Autoencoder** (see the Generative track) |

**Default recommendation: PCA to 30-50 dimensions, then UMAP to 2 for the picture.**
:::
`
}
],
quiz: [
{
q: 'In a t-SNE plot, two clusters appear very far apart. What can you conclude?',
options: [
  'They are very different',
  'Nothing - distances between clusters in t-SNE are not meaningful',
  'One is larger than the other',
  'They should be merged'
],
answer: 1,
why: 't-SNE optimises local neighbourhood preservation only. Global layout, inter-cluster distance and relative cluster size are all artefacts of the optimisation.'
},
{
q: 'Why should you run PCA before t-SNE on high-dimensional data?',
options: [
  'It is required',
  'It removes noise and cuts runtime dramatically, without losing the structure t-SNE needs',
  'It makes the plot prettier',
  'To standardise the data'
],
answer: 1,
why: 'Reducing to about 30-50 components denoises the input and makes the pairwise distance computation far cheaper. It is standard practice.'
},
{
q: 'You need to embed new, unseen data points into the same 2-D space you built earlier. Which tool?',
options: ['t-SNE', 'UMAP or PCA', 'DBSCAN', 'Any of them'],
answer: 1,
why: 't-SNE has no transform method - the embedding is optimised jointly for the given points only. UMAP and PCA both provide a reusable transform.'
},
{
q: 'You run t-SNE on random Gaussian noise. What do you see?',
options: [
  'One uniform blob',
  'Distinct-looking clusters, which are entirely artefacts',
  'An error',
  'A straight line'
],
answer: 1,
why: 't-SNE will impose apparent structure on structureless data. Always validate with multiple perplexities, multiple seeds, and an actual clustering metric.'
}
]
},

/* ============================================================ */
{
id: 'anomaly-detection',
title: 'Anomaly detection',
summary: 'Finding the unusual when you have almost no examples of it - Isolation Forest, LOF, One-Class SVM, and how to evaluate without labels.',
tags: ['anomaly', 'unsupervised', 'applied'],
intro: `
## When classification will not work

Fraud, equipment failure, network intrusion, manufacturing defects. The positive class is
0.01% of the data, and next month's anomalies will not look like last month's. Supervised
learning struggles; anomaly detection is designed for exactly this.

## Three types of anomaly

~~~text
POINT anomaly       a single value far from the rest
                    -> a 50,000 EUR transaction on a 40 EUR/day card

CONTEXTUAL anomaly  normal in general, abnormal in THIS context
                    -> 30 degrees is fine in July, an anomaly in January

COLLECTIVE anomaly  each point is normal; the SEQUENCE is not
                    -> a heartbeat pattern where every beat is fine but
                       the rhythm is wrong
~~~

## The algorithms

| Method | Idea | Best for |
|---|---|---|
| **Isolation Forest** | Random splits isolate anomalies in fewer steps | **The default.** Fast, scales, few assumptions |
| **Local Outlier Factor** | Compare a point's local density to its neighbours' | Clusters of differing density |
| **One-Class SVM** | Learn a boundary enclosing the normal data | Small, clean training sets |
| **Elliptic Envelope** | Fit a Gaussian; flag low-probability points | Roughly Gaussian data |
| **Autoencoder** | Flag high reconstruction error | Images, sequences, complex data |
| **Statistical (z, IQR)** | Simple thresholds | One dimension, quick checks |
`,
keyPoints: [
  'Isolation Forest is the sensible default - fast, few assumptions, scales well.',
  'The ~contamination~ parameter is your prior on the anomaly rate, and it sets the threshold.',
  'Train on clean data when you can - it makes "normal" much better defined.',
  'Anomaly scores are more useful than binary labels: rank and triage by budget.'
],
pitfalls: [
  'Setting contamination far from the real rate, which makes the threshold meaningless.',
  'Forgetting to scale for distance-based methods (LOF, One-Class SVM).',
  'Evaluating with accuracy - it is meaningless at 0.1% positives.',
  'Assuming an unsupervised detector will find the specific anomalies your business cares about.'
],
levels: [
{
name: 'The detectors, compared and evaluated',
goal: 'Run every method on the same data, tune contamination, and evaluate properly using the few labels you have.',
md: `
~~~python anomaly_detection.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve, classification_report)

rng = np.random.default_rng(42)

# =====================================================================
# 1. VISUAL COMPARISON ON 2-D DATA
# =====================================================================
datasets = {
    "one blob": np.vstack([rng.normal(0, 1, (300, 2)),
                           rng.uniform(-5, 5, (20, 2))]),
    "two blobs": np.vstack([make_blobs(n_samples=300, centers=2, cluster_std=0.5,
                                       random_state=0)[0],
                            rng.uniform(-6, 6, (20, 2))]),
    "moons": np.vstack([make_moons(n_samples=300, noise=0.05, random_state=0)[0],
                        rng.uniform(-2, 3, (20, 2))]),
}

detectors = {
    "IsolationForest": IsolationForest(contamination=0.06, random_state=0),
    "LocalOutlierFactor": LocalOutlierFactor(n_neighbors=20, contamination=0.06,
                                             novelty=True),
    "OneClassSVM": OneClassSVM(nu=0.06, gamma="scale"),
    "EllipticEnvelope": EllipticEnvelope(contamination=0.06, random_state=0),
}

fig, axes = plt.subplots(len(datasets), len(detectors),
                         figsize=(4.3 * len(detectors), 4 * len(datasets)))
for i, (dname, X) in enumerate(datasets.items()):
    Xs = StandardScaler().fit_transform(X)
    xx, yy = np.meshgrid(np.linspace(Xs[:, 0].min()-1, Xs[:, 0].max()+1, 200),
                         np.linspace(Xs[:, 1].min()-1, Xs[:, 1].max()+1, 200))
    for j, (aname, det) in enumerate(detectors.items()):
        det.fit(Xs)
        pred = det.predict(Xs)
        Z = det.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax = axes[i, j]
        ax.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 8), cmap="Blues_r")
        ax.contour(xx, yy, Z, levels=[0], colors="red", linewidths=2)
        ax.scatter(Xs[pred == 1, 0], Xs[pred == 1, 1], c="white", s=14,
                   edgecolor="k", linewidth=0.4)
        ax.scatter(Xs[pred == -1, 0], Xs[pred == -1, 1], c="red", marker="x", s=40)
        ax.set_title(aname if i == 0 else "", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(dname, fontsize=10)
plt.tight_layout(); plt.show()
~~~

### Evaluating with the few labels you have

~~~python anomaly_evaluate.py
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

# realistic: 1% anomalies, and we happen to have labels for evaluation only
X, y = make_classification(n_samples=20000, n_features=20, n_informative=8,
                           weights=[0.99], flip_y=0.0, class_sep=1.4, random_state=0)
Xs = StandardScaler().fit_transform(X)
print(f"anomaly rate: {y.mean():.2%}  ({y.sum()} of {len(y)})")

detectors = {
    "IsolationForest":  IsolationForest(contamination=0.01, random_state=0, n_jobs=-1),
    "LOF":              LocalOutlierFactor(n_neighbors=35, contamination=0.01),
    "OneClassSVM":      OneClassSVM(nu=0.01, gamma="scale"),
    "EllipticEnvelope": EllipticEnvelope(contamination=0.01, random_state=0,
                                         support_fraction=0.9),
}

print(f"\\n{'method':20s} {'ROC-AUC':>9s} {'PR-AUC':>9s} {'prec@1%':>9s} {'recall@1%':>10s}")
print("-" * 62)
rows = []
for name, det in detectors.items():
    if isinstance(det, LocalOutlierFactor):
        det.fit(Xs)
        scores = -det.negative_outlier_factor_       # higher = more anomalous
    else:
        det.fit(Xs)
        scores = -det.decision_function(Xs)

    roc = roc_auc_score(y, scores)
    pr = average_precision_score(y, scores)
    # what if we investigate the top 1% by score?
    k = int(0.01 * len(y))
    top = np.argsort(scores)[-k:]
    prec = y[top].mean()
    rec = y[top].sum() / y.sum()
    print(f"{name:20s} {roc:>9.4f} {pr:>9.4f} {prec:>9.3f} {rec:>10.3f}")
    rows.append((name, scores))

# =====================================================================
# THE contamination PARAMETER
# =====================================================================
print("\\nHOW contamination CHANGES THE OUTCOME (IsolationForest)")
print(f"{'contamination':>14} {'flagged':>9} {'precision':>11} {'recall':>9}")
print("-" * 46)
for c in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
    det = IsolationForest(contamination=c, random_state=0, n_jobs=-1).fit(Xs)
    pred = det.predict(Xs) == -1
    prec = y[pred].mean() if pred.sum() else 0
    rec = y[pred].sum() / y.sum()
    print(f"{c:>14.3f} {pred.sum():>9} {prec:>11.3f} {rec:>9.3f}")

print("\\ncontamination does NOT change the SCORES - only where the cut falls.")
print("Set it to your best estimate of the true rate, or ignore it and")
print("threshold the scores yourself using your investigation budget.")

# =====================================================================
# THE RIGHT WAY TO DEPLOY: rank, do not label
# =====================================================================
det = IsolationForest(n_estimators=300, contamination="auto",
                      random_state=0, n_jobs=-1).fit(Xs)
scores = -det.score_samples(Xs)

BUDGET = 150            # cases the team can review per day
top = np.argsort(scores)[-BUDGET:][::-1]
print(f"\\nDAILY TRIAGE with a budget of {BUDGET} reviews:")
print(f"  true anomalies caught : {int(y[top].sum())} of {int(y.sum())} "
      f"({y[top].sum()/y.sum():.1%} recall)")
print(f"  precision of the queue: {y[top].mean():.3f}")
print(f"  lift over random      : {y[top].mean()/y.mean():.1f}x")
~~~

~~~text
anomaly rate: 1.00%  (200 of 20000)

method                 ROC-AUC    PR-AUC   prec@1%  recall@1%
--------------------------------------------------------------
IsolationForest         0.9241    0.4108     0.505      0.505
LOF                     0.8317    0.1682     0.250      0.250
OneClassSVM             0.9105    0.3521     0.440      0.440
EllipticEnvelope        0.9403    0.4820     0.575      0.575

DAILY TRIAGE with a budget of 150 reviews:
  true anomalies caught : 88 of 200 (44.0% recall)
  precision of the queue: 0.587
  lift over random      : 58.7x
~~~

:::tip The deployment pattern that works
Do not ship a binary "anomaly / normal" label. Ship a **ranked queue**:

1. Score every case.
2. Sort descending.
3. Send the top N to human review, where N is the team's actual capacity.
4. Feed the review outcomes back as labels.
5. Once you have a few thousand labels, **train a supervised model** - it will beat the
   unsupervised detector substantially.

The unsupervised phase is how you bootstrap labels, not the final system.
:::

### Semi-supervised: train on normal data only

~~~python semi_supervised.py
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# If you can identify a clean period (before the fraud started, or verified-good
# equipment logs), train on THAT. "Normal" becomes far better defined.
X_tr, X_te, y_tr, y_te = train_test_split(Xs, y, test_size=0.4, random_state=0,
                                          stratify=y)

contaminated = IsolationForest(random_state=0, n_jobs=-1).fit(X_tr)
clean_only = IsolationForest(random_state=0, n_jobs=-1).fit(X_tr[y_tr == 0])

print("training data          ROC-AUC on the test set")
print(f"  everything          {roc_auc_score(y_te, -contaminated.score_samples(X_te)):.4f}")
print(f"  verified-normal only {roc_auc_score(y_te, -clean_only.score_samples(X_te)):.4f}")
print("\\nTraining on clean data alone is 'novelty detection' and it is almost")
print("always better - if you can get clean data.")
~~~

### Time-series anomalies

~~~python ts_anomaly.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
n = 1000
t = np.arange(n)
signal = (50 + 10 * np.sin(t / 24 * 2 * np.pi) + 5 * np.sin(t / 168 * 2 * np.pi)
          + rng.normal(0, 1.5, n))
# inject three kinds of anomaly
signal[300:305] += 30            # a spike
signal[600:650] += 12            # a level shift
signal[850:900] *= 0.3           # a collapse
s = pd.Series(signal)

# ---- method 1: rolling z-score (contextual) -------------------------
W = 48
roll_mean = s.shift(1).rolling(W).mean()
roll_std = s.shift(1).rolling(W).std()
z = (s - roll_mean) / roll_std
anom_z = z.abs() > 3.5

# ---- method 2: residual from a seasonal decomposition ---------------
try:
    from statsmodels.tsa.seasonal import STL
    stl = STL(s, period=24, robust=True).fit()
    resid = stl.resid
    anom_stl = resid.abs() > 3.5 * resid.std()
except ImportError:
    anom_stl = pd.Series(False, index=s.index)

fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
ax[0].plot(s, lw=0.8); ax[0].set_title("signal with three injected anomalies")
for start, end, label in [(300, 305, "spike"), (600, 650, "level shift"),
                          (850, 900, "collapse")]:
    ax[0].axvspan(start, end, color="red", alpha=0.15)
    ax[0].text(start, s.max(), label, fontsize=8, color="red")

ax[1].plot(s, lw=0.8, color="grey")
ax[1].scatter(s.index[anom_z], s[anom_z], color="red", s=14)
ax[1].set_title(f"rolling z-score: {anom_z.sum()} flagged")

ax[2].plot(s, lw=0.8, color="grey")
ax[2].scatter(s.index[anom_stl], s[anom_stl], color="red", s=14)
ax[2].set_title(f"STL residual: {anom_stl.sum()} flagged")
plt.tight_layout(); plt.show()

print("Rolling statistics catch SPIKES well but adapt to level shifts and")
print("stop flagging them after a while. Seasonal decomposition handles the")
print("periodic structure explicitly and catches sustained shifts better.")
~~~
`
}
],
quiz: [
{
q: 'Which anomaly detector is the sensible default for tabular data?',
options: ['One-Class SVM', 'Isolation Forest', 'z-score threshold', 'K-Means'],
answer: 1,
why: 'Isolation Forest is fast, scales to large datasets, needs no distance metric or scaling assumption, and makes few distributional assumptions.'
},
{
q: 'What does the contamination parameter actually control?',
options: [
  'How anomalous each point is scored',
  'Where the threshold on the scores falls - the fraction flagged as anomalies',
  'The number of trees',
  'The learning rate'
],
answer: 1,
why: 'Scores are computed independently of contamination; it only sets the cut-off percentile. You can ignore it entirely and threshold the scores by your review budget.'
},
{
q: 'You have 20,000 unlabelled transactions and can review 150 per day. What should you ship?',
options: [
  'A binary fraud / not-fraud label for every transaction',
  'A ranked queue of the 150 highest-scoring cases, feeding review outcomes back as labels',
  'All transactions above a z-score of 3',
  'A clustering of the transactions'
],
answer: 1,
why: 'Ranking matches the real constraint (human capacity) and bootstraps labelled data. Once a few thousand reviewed labels exist, a supervised model will outperform the detector.'
},
{
q: 'Why is training an anomaly detector on verified-normal data only usually better?',
options: [
  'It is faster',
  '"Normal" becomes precisely defined instead of being contaminated by the anomalies you want to find',
  'It uses less memory',
  'It is not better'
],
answer: 1,
why: 'This is novelty detection. If anomalies sit in the training data, the model partly learns to consider them normal, blunting exactly the signal you need.'
}
]
}

]
});
