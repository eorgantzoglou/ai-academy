/* Track 16 - MLOps: shipping and running models */
(window.CURRICULUM = window.CURRICULUM || []).push({
id: 'mlops',
title: 'MLOps & Deployment',
icon: 'Op',
level: 'Advanced',
blurb: 'Getting a model out of the notebook: saving artefacts, serving with FastAPI, containerising with Docker, testing ML code and data, and monitoring for drift once it is live.',
intro: `
## A model that is not deployed has zero value

~~~text
NOTEBOOK                         PRODUCTION
runs once, by you                runs thousands of times, by strangers
you fix errors interactively     it must handle errors itself
your laptop's library versions   pinned, reproducible, containerised
accuracy is the only metric      latency, cost, uptime, drift also matter
data is a fixed CSV              data arrives live and CHANGES
~~~

## The pieces

~~~text
  TRAIN                SERVE                  OPERATE
  |                    |                      |
  +-- versioned data   +-- a saved PIPELINE   +-- monitor inputs for drift
  +-- tracked runs     +-- an API             +-- monitor predictions
  +-- reproducible     +-- validation         +-- monitor accuracy when
  +-- tested           +-- containerised          labels eventually arrive
                       +-- monitored          +-- retrain on a schedule
                                              +-- roll back when needed
~~~

:::warn Training-serving skew
The single most common production failure: preprocessing at serving time differs from
preprocessing at training time. The model receives inputs in a distribution it never saw
and quietly produces nonsense.

**The fix is structural**: save the entire preprocessing pipeline with the model, so
there is only ever one implementation.
:::
`,
topics: [

/* ============================================================ */
{
id: 'serving',
title: 'Saving and serving models',
summary: 'Persisting a complete pipeline, building a FastAPI service with validation, and the patterns that make it fast and safe.',
tags: ['mlops', 'fastapi', 'deployment'],
intro: `
## What to save

~~~text
NOT this               joblib.dump(model, "model.pkl")

BUT this               the WHOLE pipeline: imputer + scaler + encoder + model
                       + the feature names, in order
                       + the metrics it achieved
                       + the training date and data version
                       + the library versions
                       + the decision threshold
                       + known limitations
~~~

## Formats

| Format | Use | Notes |
|---|---|---|
| **joblib** | scikit-learn | Fast for numpy arrays. **Version-fragile.** |
| **pickle** | Python objects | Same fragility, plus a security risk |
| **ONNX** | Cross-framework | Portable, fast, language-independent |
| **TorchScript / torch.save** | PyTorch | ~state_dict~ plus the class definition |
| **SafeTensors** | Large models | Safe to load from untrusted sources |
| **PMML** | Legacy enterprise | Widely supported by older tooling |

:::danger Never unpickle an untrusted file
Pickle executes arbitrary code on load. A malicious ~.pkl~ is remote code execution.
For anything you did not create yourself, use ONNX or SafeTensors.
:::
`,
keyPoints: [
  'Save the whole pipeline, not the estimator - it eliminates training-serving skew.',
  'Store metadata: feature order, metrics, versions, threshold and limitations.',
  'Validate every incoming request with a schema before it reaches the model.',
  'Load the model once at startup, never per request.'
],
pitfalls: [
  'Loading the model inside the request handler, adding hundreds of milliseconds per call.',
  'Not pinning library versions - a joblib file may not load on a different scikit-learn.',
  'Returning a raw exception message to the caller, which leaks internals.',
  'No health check, so the orchestrator cannot tell whether the service is ready.'
],
levels: [
{
name: 'A production model artefact and API',
goal: 'Package a model properly, then serve it with a validated, observable FastAPI service.',
md: `
~~~python save_model.py
"""Package a model as a complete, self-describing artefact."""
import json
import platform
import subprocess
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report


@dataclass
class ModelCard:
    """Everything a future maintainer needs to know."""
    name: str
    version: str
    task: str
    trained_at: str
    training_rows: int
    feature_names: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    target: str
    positive_class: str
    decision_threshold: float
    metrics: Dict[str, float]
    library_versions: Dict[str, str]
    intended_use: str
    out_of_scope: str
    limitations: List[str]
    git_commit: str = ""
    data_version: str = ""

    def to_json(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @staticmethod
    def load(path):
        return ModelCard(**json.loads(Path(path).read_text()))


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


# =====================================================================
# TRAIN
# =====================================================================
rng = np.random.default_rng(0)
n = 8000
df = pd.DataFrame({
    "age": rng.integers(18, 80, n).astype(float),
    "tenure_months": rng.integers(1, 120, n).astype(float),
    "monthly_spend": rng.lognormal(3.6, 0.7, n).round(2),
    "support_tickets": rng.poisson(1.1, n).astype(float),
    "plan": rng.choice(["basic", "standard", "premium"], n, p=[0.5, 0.35, 0.15]),
    "region": rng.choice(["north", "south", "east", "west"], n),
})
logit = (-1.4 + 0.9 * (df["plan"] == "basic") + 0.35 * df["support_tickets"]
         - 0.012 * df["tenure_months"] + rng.normal(0, 0.6, n))
df["churned"] = (1 / (1 + np.exp(-logit)) > rng.random(n)).astype(int)

NUMERIC = ["age", "tenure_months", "monthly_spend", "support_tickets"]
CATEGORICAL = ["plan", "region"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churned"

X, y = df[FEATURES], df[TARGET]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)

pipeline = Pipeline([
    ("prep", ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   sparse_output=False))]),
         CATEGORICAL),
    ], remainder="drop")),
    ("model", HistGradientBoostingClassifier(random_state=42)),
])
pipeline.fit(X_tr, y_tr)

proba = pipeline.predict_proba(X_te)[:, 1]
THRESHOLD = 0.35            # chosen from the business cost of each error
metrics = {
    "roc_auc": float(roc_auc_score(y_te, proba)),
    "pr_auc": float(average_precision_score(y_te, proba)),
    "threshold": THRESHOLD,
    "test_rows": int(len(y_te)),
}
print(classification_report(y_te, (proba >= THRESHOLD).astype(int), digits=3))

# =====================================================================
# SAVE THE ARTEFACT
# =====================================================================
OUT = Path("artefacts/churn/v1.2.0")
OUT.mkdir(parents=True, exist_ok=True)

joblib.dump(pipeline, OUT / "pipeline.joblib", compress=3)

card = ModelCard(
    name="customer-churn",
    version="1.2.0",
    task="binary classification",
    trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    training_rows=len(X_tr),
    feature_names=FEATURES,
    numeric_features=NUMERIC,
    categorical_features=CATEGORICAL,
    target=TARGET,
    positive_class="churned",
    decision_threshold=THRESHOLD,
    metrics=metrics,
    library_versions={
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "joblib": joblib.__version__,
    },
    intended_use=("Score active subscribers monthly to prioritise retention "
                  "outreach. Output is a probability, not a decision."),
    out_of_scope=("Not for individual adverse action without human review. "
                  "Not validated for accounts under 1 month old."),
    limitations=[
        "Trained on 2023-2025 data; retrain if the product mix changes.",
        "The 'west' region is only 5% of training data - lower reliability there.",
        "Probabilities are calibrated on the training distribution only.",
    ],
    git_commit=git_commit(),
    data_version="snapshot-2026-05-01",
)
card.to_json(OUT / "model_card.json")

# a SMOKE TEST fixture - proves the artefact loads and predicts correctly
sample = X_te.head(5)
expected = pipeline.predict_proba(sample)[:, 1]
np.save(OUT / "smoke_input.npy", sample.to_numpy(dtype=object), allow_pickle=True)
sample.to_json(OUT / "smoke_input.json", orient="records")
np.save(OUT / "smoke_expected.npy", expected)

print(f"\\nsaved to {OUT}")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name:24s} {f.stat().st_size/1024:8.1f} KB")


# =====================================================================
# VERIFY IT LOADS AND REPRODUCES
# =====================================================================
def verify_artefact(directory):
    directory = Path(directory)
    loaded = joblib.load(directory / "pipeline.joblib")
    card = ModelCard.load(directory / "model_card.json")
    smoke_in = pd.read_json(directory / "smoke_input.json")[card.feature_names]
    smoke_expected = np.load(directory / "smoke_expected.npy")

    actual = loaded.predict_proba(smoke_in)[:, 1]
    ok = np.allclose(actual, smoke_expected, atol=1e-6)
    print(f"\\nSMOKE TEST: {'PASS' if ok else 'FAIL'}")
    print(f"  expected {smoke_expected.round(5)}")
    print(f"  actual   {actual.round(5)}")
    if sklearn.__version__ != card.library_versions["scikit-learn"]:
        print(f"  WARNING: trained with sklearn "
              f"{card.library_versions['scikit-learn']}, "
              f"loading with {sklearn.__version__}")
    return ok

verify_artefact(OUT)
~~~

### The FastAPI service

~~~bash
pip install "fastapi[standard]" uvicorn pydantic joblib prometheus-client
~~~

~~~python app.py
"""A production-shaped model service."""
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}')
log = logging.getLogger("model-service")

ARTEFACT_DIR = Path("artefacts/churn/v1.2.0")
STATE = {}


# =====================================================================
# LIFESPAN - load the model ONCE at startup, never per request
# =====================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading model artefact")
    STATE["pipeline"] = joblib.load(ARTEFACT_DIR / "pipeline.joblib")
    STATE["card"] = json.loads((ARTEFACT_DIR / "model_card.json").read_text())
    STATE["features"] = STATE["card"]["feature_names"]
    STATE["threshold"] = STATE["card"]["decision_threshold"]

    # warm up - the first prediction is always slower (lazy imports, JIT)
    smoke = pd.read_json(ARTEFACT_DIR / "smoke_input.json")[STATE["features"]]
    STATE["pipeline"].predict_proba(smoke)

    log.info(f"ready: {STATE['card']['name']} v{STATE['card']['version']}")
    yield
    log.info("shutting down")
    STATE.clear()


app = FastAPI(
    title="Churn Prediction Service",
    version="1.2.0",
    description="Scores active subscribers for churn risk.",
    lifespan=lifespan,
)


# =====================================================================
# SCHEMAS - validation is the first line of defence
# =====================================================================
class Customer(BaseModel):
    age: float = Field(..., ge=18, le=120, description="Age in years")
    tenure_months: float = Field(..., ge=0, le=600)
    monthly_spend: float = Field(..., ge=0, le=100000)
    support_tickets: float = Field(..., ge=0, le=1000)
    plan: Literal["basic", "standard", "premium"]
    region: Literal["north", "south", "east", "west"]

    @field_validator("monthly_spend")
    @classmethod
    def flag_extreme(cls, v):
        if v > 5000:
            log.warning(f"unusually high monthly_spend: {v}")
        return v

    model_config = {"json_schema_extra": {"examples": [{
        "age": 34, "tenure_months": 8, "monthly_spend": 42.5,
        "support_tickets": 3, "plan": "basic", "region": "north"}]}}


class BatchRequest(BaseModel):
    customers: List[Customer] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    churn_probability: float
    risk_band: Literal["low", "medium", "high"]
    flagged: bool
    threshold_used: float


class PredictResponse(BaseModel):
    request_id: str
    model_version: str
    predictions: List[Prediction]
    latency_ms: float


# =====================================================================
# ENDPOINTS
# =====================================================================
@app.get("/health")
def health():
    """LIVENESS - is the process up? Kept trivially cheap."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """READINESS - can it actually serve? The orchestrator uses this to
    decide whether to route traffic here."""
    if "pipeline" not in STATE:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail="model not loaded")
    return {"status": "ready", "model": STATE["card"]["name"],
            "version": STATE["card"]["version"]}


@app.get("/model-card")
def model_card():
    """Transparency: expose what the model is, and is not, for."""
    return STATE["card"]


def to_band(p: float) -> str:
    return "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"


@app.post("/predict", response_model=PredictResponse)
def predict(request: BatchRequest):
    t0 = time.perf_counter()
    request_id = str(uuid.uuid4())

    try:
        # build the frame with the columns IN THE TRAINED ORDER
        frame = pd.DataFrame([c.model_dump() for c in request.customers])
        frame = frame[STATE["features"]]

        proba = STATE["pipeline"].predict_proba(frame)[:, 1]
    except Exception as e:
        # log the detail, return something generic - never leak internals
        log.exception(f"prediction failed request_id={request_id}")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="prediction failed") from e

    threshold = STATE["threshold"]
    latency = (time.perf_counter() - t0) * 1000

    # LOG EVERY PREDICTION - this is your drift-monitoring data source
    log.info(json.dumps({
        "event": "prediction",
        "request_id": request_id,
        "n": len(proba),
        "mean_proba": float(np.mean(proba)),
        "flagged": int((proba >= threshold).sum()),
        "latency_ms": round(latency, 2),
        "model_version": STATE["card"]["version"],
    }))

    return PredictResponse(
        request_id=request_id,
        model_version=STATE["card"]["version"],
        predictions=[Prediction(churn_probability=round(float(p), 5),
                                risk_band=to_band(float(p)),
                                flagged=bool(p >= threshold),
                                threshold_used=threshold) for p in proba],
        latency_ms=round(latency, 2),
    )


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled exception")
    return JSONResponse(status_code=500,
                        content={"detail": "internal server error"})


# run with:  uvicorn app:app --host 0.0.0.0 --port 8000 --workers 4
# docs at:   http://localhost:8000/docs
~~~

### Testing the service

~~~python test_api.py
"""Every one of these has caught a real bug at some point."""
from fastapi.testclient import TestClient
import pytest
from app import app

client = TestClient(app)

VALID = {"age": 34, "tenure_months": 8, "monthly_spend": 42.5,
         "support_tickets": 3, "plan": "basic", "region": "north"}


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_ready_after_startup():
    assert client.get("/ready").status_code == 200


def test_valid_prediction():
    r = client.post("/predict", json={"customers": [VALID]})
    assert r.status_code == 200
    body = r.json()
    assert len(body["predictions"]) == 1
    p = body["predictions"][0]["churn_probability"]
    assert 0.0 <= p <= 1.0


def test_batch():
    r = client.post("/predict", json={"customers": [VALID] * 50})
    assert len(r.json()["predictions"]) == 50


@pytest.mark.parametrize("field,value", [
    ("age", 5),                       # below the minimum
    ("age", 200),                     # above the maximum
    ("monthly_spend", -10),           # negative
    ("plan", "platinum"),             # not in the enum
    ("region", "middle"),
])
def test_rejects_invalid(field, value):
    bad = {**VALID, field: value}
    assert client.post("/predict", json={"customers": [bad]}).status_code == 422


def test_rejects_missing_field():
    incomplete = {k: v for k, v in VALID.items() if k != "plan"}
    assert client.post("/predict",
                       json={"customers": [incomplete]}).status_code == 422


def test_rejects_empty_batch():
    assert client.post("/predict", json={"customers": []}).status_code == 422


def test_rejects_oversized_batch():
    assert client.post("/predict",
                       json={"customers": [VALID] * 5000}).status_code == 422


def test_deterministic():
    """The SAME input must give the SAME output."""
    a = client.post("/predict", json={"customers": [VALID]}).json()
    b = client.post("/predict", json={"customers": [VALID]}).json()
    assert (a["predictions"][0]["churn_probability"]
            == b["predictions"][0]["churn_probability"])


def test_no_internal_leak_on_error():
    """A failure must not expose stack traces or file paths."""
    r = client.post("/predict", json={"customers": [{**VALID, "age": "abc"}]})
    assert "Traceback" not in r.text
    assert "site-packages" not in r.text
~~~

### Making it fast

~~~python performance.py
"""Latency patterns that matter in serving."""

NOTES = """
1. LOAD ONCE, AT STARTUP
   Loading a joblib pipeline takes 50-500 ms. Doing it per request is the
   single most common serving mistake.

2. BATCH
   Predicting 100 rows in one call is ~50x faster than 100 separate calls,
   because the per-call overhead dominates. If callers send single items,
   consider a micro-batching queue that accumulates for 5-20 ms.

3. WARM UP AT STARTUP
   The first prediction is always slower - lazy imports, memory allocation,
   JIT. Run a smoke prediction before reporting ready.

4. USE ONNX RUNTIME FOR CPU INFERENCE
   Typically 2-5x faster than scikit-learn for tree ensembles, and it
   removes the sklearn version fragility entirely.

5. WORKERS, NOT THREADS, FOR CPU-BOUND WORK
   uvicorn --workers 4. Python's GIL means threads will not help.
   Set the worker count to roughly the CPU count.

6. RESPONSE MODELS COST TIME
   Pydantic validation of a 1000-row response is not free. For very
   high-throughput endpoints, return a plain dict.

7. CACHE WHAT REPEATS
   If the same entity is scored repeatedly within a short window, an
   LRU cache keyed on the input hash is nearly free.
"""
print(NOTES)


# ---- export to ONNX --------------------------------------------------
def export_to_onnx(pipeline, X_sample, path="model.onnx"):
    """ONNX: portable, fast, and immune to sklearn version drift."""
    from skl2onnx import to_onnx
    onx = to_onnx(pipeline, X_sample[:1], options={id(pipeline): {"zipmap": False}})
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())
    return path


def onnx_predict(path, X):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    inputs = {i.name: X[[i.name]].to_numpy() for i in sess.get_inputs()}
    return sess.run(None, inputs)


# ---- a micro-batching queue ------------------------------------------
import asyncio
from collections import deque

class MicroBatcher:
    """Accumulate requests for a few milliseconds, then predict as one batch.
    Trades a little latency for a large throughput gain."""

    def __init__(self, predict_fn, max_batch=64, max_wait_ms=10):
        self.predict_fn = predict_fn
        self.max_batch = max_batch
        self.max_wait = max_wait_ms / 1000
        self.queue = deque()
        self._task = None

    async def predict(self, item):
        future = asyncio.get_running_loop().create_future()
        self.queue.append((item, future))
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._flush_loop())
        return await future

    async def _flush_loop(self):
        await asyncio.sleep(self.max_wait)
        while self.queue:
            batch = [self.queue.popleft() for _ in
                     range(min(self.max_batch, len(self.queue)))]
            items = [b[0] for b in batch]
            results = self.predict_fn(items)
            for (_, fut), r in zip(batch, results):
                if not fut.done():
                    fut.set_result(r)
~~~
`
}
],
quiz: [
{
q: 'What is training-serving skew, and what prevents it?',
options: [
  'Different hardware - use the same GPU',
  'Preprocessing differs between training and serving - prevent it by saving the whole pipeline as one artefact',
  'A random seed problem',
  'Different batch sizes'
],
answer: 1,
why: 'When preprocessing is reimplemented in the serving code, the two implementations drift apart. Saving imputer, scaler, encoder and model as one object means there is only ever one implementation.'
},
{
q: 'Where should the model be loaded in a FastAPI service?',
options: [
  'Inside the request handler',
  'Once at startup, in the lifespan handler',
  'In a background thread per request',
  'It does not matter'
],
answer: 1,
why: 'Loading takes 50-500 ms. Per request that dominates your latency budget entirely. Load once, and warm it up with a smoke prediction before reporting ready.'
},
{
q: 'Why should the API never return the raw exception message?',
options: [
  'It is too long',
  'It leaks stack traces, file paths and internal structure to the caller',
  'Exceptions cannot be serialised',
  'It slows the response'
],
answer: 1,
why: 'Log the detail server-side with a request id; return a generic message and that id. Internal paths and library versions are useful reconnaissance for an attacker.'
},
{
q: 'What is the difference between /health and /ready?',
options: [
  'They are the same',
  'Health is liveness (is the process up); ready is readiness (can it actually serve, e.g. is the model loaded)',
  'Health is for humans, ready is for machines',
  'Ready is deprecated'
],
answer: 1,
why: 'An orchestrator restarts a container that fails liveness, but only stops routing traffic to one that fails readiness. Conflating them causes restart loops during slow model loads.'
}
]
},

/* ============================================================ */
{
id: 'containers-cicd',
title: 'Docker, CI and reproducibility',
summary: 'Containerising a model service properly, pinning dependencies, and a CI pipeline that tests the model as well as the code.',
tags: ['mlops', 'docker', 'ci'],
intro: `
## Why containers

~~~text
"it works on my machine"        ->  ship the machine

A container image bundles the OS libraries, the Python version, every
pinned dependency, your code and the model artefact. It runs identically
on your laptop, in CI, and in production.
~~~

## The layers that matter

~~~text
FROM python:3.12-slim          <- pin the MINOR version, not just 3
COPY requirements.txt          <- copy dependencies FIRST
RUN pip install -r ...         <- this layer caches; it only rebuilds when
                                  requirements change
COPY src/ ./src/               <- code changes rebuild only from here down
COPY artefacts/ ./artefacts/
~~~

Ordering matters enormously. Copying your source before installing dependencies means every
code change reinstalls everything - turning a 5-second rebuild into a 5-minute one.

## Reproducibility

| Level | What it pins |
|---|---|
| ~requirements.txt~ with ~==~ | Direct dependency versions |
| ~pip-compile~ / ~uv lock~ | Transitive dependencies too, with hashes |
| Docker image digest | The OS and system libraries |
| A data version | The training data itself |
| A random seed | The training run |

All five are needed to genuinely reproduce a model.
`,
keyPoints: [
  'Copy requirements before source so the dependency layer caches.',
  'Use a multi-stage build and a non-root user for smaller, safer images.',
  'Pin transitive dependencies with a lock file, not just direct ones.',
  'CI must test the data and the model, not only the code.'
],
pitfalls: [
  'Running as root inside the container.',
  'Using the ~latest~ tag, which makes builds non-reproducible.',
  'Baking secrets into the image - they persist in the layer history.',
  'Copying the whole context, including data and the .git directory.'
],
levels: [
{
name: 'Containerising and automating',
goal: 'Write a production Dockerfile, compose the stack, and build a CI pipeline that tests the model.',
md: `
~~~text Dockerfile
# =====================================================================
# STAGE 1: BUILD - compile wheels here, so the final image stays small
# =====================================================================
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# build-time-only system packages
RUN apt-get update && apt-get install -y --no-install-recommends \\
        build-essential \\
    && rm -rf /var/lib/apt/lists/*

# COPY REQUIREMENTS FIRST - this layer caches until requirements change
COPY requirements.lock .
RUN python -m venv /opt/venv \\
    && /opt/venv/bin/pip install --upgrade pip \\
    && /opt/venv/bin/pip install --require-hashes -r requirements.lock


# =====================================================================
# STAGE 2: RUNTIME - only what is needed to RUN
# =====================================================================
FROM python:3.12-slim AS runtime

# a NON-ROOT user - never run a service as root
RUN groupadd --gid 1000 appuser \\
    && useradd --uid 1000 --gid 1000 --create-home appuser

# copy the prepared virtualenv from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \\
    PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# code and artefacts last - they change most often
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser artefacts/ ./artefacts/

USER appuser
EXPOSE 8000

# the orchestrator uses this to decide whether the container is alive
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \\
    CMD python -c "import urllib.request,sys; \\
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
~~~

~~~text .dockerignore
# NEVER copy these into the image
.git/
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
.ipynb_checkpoints/
notebooks/
data/raw/
data/processed/
tests/
*.md
.env
.env.*
*.log
mlruns/
wandb/
~~~

~~~bash build.sh
# ---- pin EVERYTHING, including transitive dependencies ---------------
pip install pip-tools
pip-compile requirements.in --generate-hashes --output-file requirements.lock

# ---- build ------------------------------------------------------------
docker build -t churn-service:1.2.0 -t churn-service:latest .

docker images churn-service
# a slim multi-stage image should be roughly 300-500 MB

# ---- run --------------------------------------------------------------
docker run -d --name churn \\
    -p 8000:8000 \\
    --memory=2g --cpus=2 \\
    --restart=unless-stopped \\
    --read-only --tmpfs /tmp \\
    churn-service:1.2.0

# ---- verify -----------------------------------------------------------
curl -s http://localhost:8000/ready | python -m json.tool

curl -s -X POST http://localhost:8000/predict \\
    -H "Content-Type: application/json" \\
    -d '{"customers":[{"age":34,"tenure_months":8,"monthly_spend":42.5,
         "support_tickets":3,"plan":"basic","region":"north"}]}' \\
    | python -m json.tool

docker logs churn --tail 20
docker stats churn --no-stream
~~~

~~~text docker-compose.yml
services:
  api:
    build: .
    image: churn-service:1.2.0
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - MODEL_DIR=/app/artefacts/churn/v1.2.0
    volumes:
      - ./logs:/app/logs
    deploy:
      resources:
        limits:
          cpus: "2"
          memory: 2G
    healthcheck:
      test: ["CMD", "python", "-c",
             "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 40s
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:v2.54.1
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro

  grafana:
    image: grafana/grafana:11.2.0
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=__set_via_secret__
    volumes:
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus

volumes:
  grafana-data:
~~~

### CI that tests the model, not just the code

~~~text .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install
        run: |
          pip install -r requirements.lock
          pip install pytest pytest-cov ruff mypy

      # ---- CODE QUALITY -------------------------------------------
      - name: Lint
        run: ruff check src/ tests/

      - name: Format check
        run: ruff format --check src/ tests/

      - name: Type check
        run: mypy src/ --ignore-missing-imports

      # ---- TESTS ---------------------------------------------------
      - name: Unit and API tests
        run: pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=80

      # ---- ML-SPECIFIC CHECKS --------------------------------------
      - name: Data validation
        run: pytest tests/test_data.py -v

      - name: Model artefact smoke test
        run: python -m src.verify_artefact artefacts/churn/v1.2.0

      - name: Performance regression gate
        run: python -m src.check_metrics --min-auc 0.78 --max-latency-ms 50

      # ---- SECURITY ------------------------------------------------
      - name: Dependency audit
        run: |
          pip install pip-audit
          pip-audit --requirement requirements.lock

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Build image
        run: docker build -t churn-service:\${{ github.sha }} .
      - name: Scan image
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: churn-service:\${{ github.sha }}
          severity: CRITICAL,HIGH
          exit-code: "1"
~~~

~~~python src/check_metrics.py
"""A CI gate: fail the build if the model regressed."""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artefact", default="artefacts/churn/v1.2.0")
    parser.add_argument("--min-auc", type=float, required=True)
    parser.add_argument("--max-latency-ms", type=float, required=True)
    args = parser.parse_args()

    directory = Path(args.artefact)
    card = json.loads((directory / "model_card.json").read_text())
    pipeline = joblib.load(directory / "pipeline.joblib")

    failures = []

    # 1. the recorded metric must clear the bar
    auc = card["metrics"]["roc_auc"]
    if auc < args.min_auc:
        failures.append(f"ROC-AUC {auc:.4f} is below the minimum {args.min_auc}")

    # 2. latency must be acceptable
    sample = pd.read_json(directory / "smoke_input.json")[card["feature_names"]]
    pipeline.predict_proba(sample)                     # warm up
    t0 = time.perf_counter()
    for _ in range(100):
        pipeline.predict_proba(sample)
    latency = (time.perf_counter() - t0) / 100 * 1000
    if latency > args.max_latency_ms:
        failures.append(f"latency {latency:.1f} ms exceeds "
                        f"{args.max_latency_ms} ms")

    # 3. the artefact must reproduce its recorded predictions exactly
    expected = np.load(directory / "smoke_expected.npy")
    actual = pipeline.predict_proba(sample)[:, 1]
    if not np.allclose(actual, expected, atol=1e-6):
        failures.append("smoke test predictions do not match the stored values")

    # 4. the model card must be complete
    for field in ["intended_use", "limitations", "decision_threshold",
                  "library_versions"]:
        if not card.get(field):
            failures.append(f"model card is missing '{field}'")

    print(f"ROC-AUC   : {auc:.4f}  (minimum {args.min_auc})")
    print(f"latency   : {latency:.2f} ms  (maximum {args.max_latency_ms})")
    print(f"smoke test: {'pass' if not any('smoke' in f for f in failures) else 'FAIL'}")

    if failures:
        print("\\nGATE FAILED")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\\nall gates passed")


if __name__ == "__main__":
    main()
~~~

:::tip Deployment strategies, in increasing safety
| Strategy | How it works | Risk |
|---|---|---|
| **Recreate** | Stop the old, start the new | Downtime |
| **Rolling** | Replace instances gradually | Two versions live at once |
| **Blue-green** | Full parallel environment, switch traffic | Doubles infrastructure cost |
| **Canary** | 5% of traffic to the new version, watch, ramp up | **The safest for models** |
| **Shadow** | New model receives copies of live traffic, its output is discarded | **Zero risk** - the only way to validate on real traffic before committing |

For a model, **shadow first, then canary** is the responsible sequence. Shadow mode
answers "does it behave sanely on real production data?" without any user impact at all.
:::
`
}
],
quiz: [
{
q: 'Why copy requirements.txt before the source code in a Dockerfile?',
options: [
  'Alphabetical ordering',
  'So the expensive dependency-install layer caches and only rebuilds when requirements change',
  'It is required by Docker',
  'To reduce the image size'
],
answer: 1,
why: 'Docker caches layers up to the first change. Copying source first invalidates the install layer on every code edit, turning a 5-second rebuild into minutes.'
},
{
q: 'What is shadow deployment?',
options: [
  'Deploying at night',
  'The new model receives copies of live traffic but its predictions are discarded - zero user risk',
  'A backup model',
  'Deploying to a staging environment'
],
answer: 1,
why: 'It is the only way to validate a model on genuine production data before it can affect anyone. Shadow first, then canary, is the responsible sequence.'
},
{
q: 'Why should a container not run as root?',
options: [
  'It is slower',
  'A container escape or code-execution bug then has root on the host - run as an unprivileged user',
  'Docker forbids it',
  'It uses more memory'
],
answer: 1,
why: 'Defence in depth. Create a user with a fixed UID, chown the application files to it, and add USER before CMD. Combine with --read-only where possible.'
},
{
q: 'What should an ML CI pipeline test that ordinary software CI does not?',
options: [
  'Nothing extra',
  'Data validation, model artefact reproducibility, and metric regression gates',
  'Only the tests run faster',
  'Just linting'
],
answer: 1,
why: 'Code can be correct while the model has silently regressed. Gate on a minimum metric, verify the artefact reproduces its stored predictions, and validate the data schema.'
}
]
},

/* ============================================================ */
{
id: 'monitoring',
title: 'Monitoring, drift and retraining',
summary: 'What to watch after deployment, how to detect data and concept drift before accuracy collapses, and when to retrain.',
tags: ['mlops', 'monitoring', 'drift'],
intro: `
## Models decay

Every deployed model degrades. Not because the code broke, but because **the world moved**.

~~~text
DATA DRIFT (covariate shift)
  P(X) changes, P(y|X) does not
  -> your customers changed; the relationship still holds
  -> detectable IMMEDIATELY from the inputs alone

CONCEPT DRIFT
  P(y|X) changes
  -> the relationship itself changed
  -> detectable ONLY once labels arrive, which may be weeks later

LABEL DRIFT
  P(y) changes - the base rate moved

UPSTREAM DATA ISSUES
  A pipeline change, a renamed field, a unit change, a new default value.
  -> the most common cause in practice, and the most preventable
~~~

:::danger The monitoring gap
Accuracy needs labels, and labels arrive late - sometimes months late for churn or default
prediction. By the time accuracy drops in your dashboard, the model has been wrong for
weeks.

**That is why you monitor the inputs.** Input drift is observable immediately and is the
early-warning signal.
:::

## The monitoring hierarchy

~~~text
1. SYSTEM     latency, error rate, throughput, memory      immediate
2. DATA       schema, missing rates, distributions         immediate  <- the early warning
3. PREDICTION distribution of outputs, flag rate            immediate
4. QUALITY    accuracy, AUC, calibration                    delayed
5. BUSINESS   revenue, retention, cost per decision         delayed
~~~
`,
keyPoints: [
  'Monitor inputs, not just accuracy - inputs are observable immediately.',
  'PSI above 0.2 or a significant KS test indicates meaningful drift.',
  'The prediction distribution shifting is a strong early warning.',
  'Retrain on a schedule and on a trigger, with a shadow evaluation before promoting.'
],
pitfalls: [
  'Alerting on every statistical test - with enough features something is always significant.',
  'Retraining automatically on drift without checking the new model is actually better.',
  'Monitoring aggregate metrics only, so a failing segment is invisible.',
  'Not versioning the reference distribution the comparison is made against.'
],
levels: [
{
name: 'Drift detection and a retraining policy',
goal: 'Implement PSI, KS and prediction-drift monitors, then design the retraining decision.',
md: `
~~~python drift.py
"""Detect drift before it becomes an accuracy problem."""
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional
import json


# =====================================================================
# 1. POPULATION STABILITY INDEX - the industry standard
# =====================================================================
def population_stability_index(reference, current, bins=10):
    """PSI measures how much a distribution has shifted.

    PSI < 0.10   no significant change
    0.10 - 0.25  moderate shift - investigate
    PSI > 0.25   significant shift - act
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) == 0 or len(current) == 0:
        return np.nan

    # bin edges come from the REFERENCE, so both are compared on one scale
    edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf

    ref_pct = np.histogram(reference, edges)[0] / len(reference)
    cur_pct = np.histogram(current, edges)[0] / len(current)

    # avoid log(0)
    eps = 1e-6
    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def categorical_psi(reference, current):
    """PSI for a categorical column, including unseen categories."""
    ref = pd.Series(reference).value_counts(normalize=True)
    cur = pd.Series(current).value_counts(normalize=True)
    categories = set(ref.index) | set(cur.index)
    eps = 1e-6
    total = 0.0
    for c in categories:
        r = max(ref.get(c, 0.0), eps)
        u = max(cur.get(c, 0.0), eps)
        total += (u - r) * np.log(u / r)
    return float(total)


# =====================================================================
# 2. THE MONITOR
# =====================================================================
@dataclass
class DriftResult:
    feature: str
    test: str
    statistic: float
    p_value: Optional[float]
    severity: str
    detail: str


class DriftMonitor:
    """Compare a live window against a stored reference window."""

    PSI_MODERATE = 0.10
    PSI_SIGNIFICANT = 0.25
    KS_ALPHA = 0.01                 # strict, because we test many features

    def __init__(self, reference: pd.DataFrame, numeric: List[str],
                 categorical: List[str]):
        self.reference = reference
        self.numeric = numeric
        self.categorical = categorical
        # store the reference SUMMARY, so you can persist it cheaply
        self.summary = {
            "n": len(reference),
            "numeric": {c: {"mean": float(reference[c].mean()),
                            "std": float(reference[c].std()),
                            "p01": float(reference[c].quantile(0.01)),
                            "p99": float(reference[c].quantile(0.99)),
                            "missing_rate": float(reference[c].isna().mean())}
                        for c in numeric},
            "categorical": {c: reference[c].value_counts(normalize=True).to_dict()
                            for c in categorical},
        }

    def check(self, current: pd.DataFrame) -> List[DriftResult]:
        results = []

        # ---- SCHEMA first. A missing column is not drift, it is a break.
        missing = set(self.reference.columns) - set(current.columns)
        extra = set(current.columns) - set(self.reference.columns)
        if missing:
            results.append(DriftResult("SCHEMA", "columns", np.nan, None,
                                       "CRITICAL", f"missing columns: {sorted(missing)}"))
        if extra:
            results.append(DriftResult("SCHEMA", "columns", np.nan, None,
                                       "warning", f"unexpected columns: {sorted(extra)}"))

        # ---- NUMERIC features ---------------------------------------
        for col in self.numeric:
            if col not in current.columns:
                continue
            ref, cur = self.reference[col].dropna(), current[col].dropna()
            if len(cur) < 30:
                continue

            psi = population_stability_index(ref, cur)
            severity = ("CRITICAL" if psi > self.PSI_SIGNIFICANT else
                        "warning" if psi > self.PSI_MODERATE else "ok")
            results.append(DriftResult(
                col, "PSI", psi, None, severity,
                f"mean {ref.mean():.3f} -> {cur.mean():.3f}"))

            # Kolmogorov-Smirnov - distribution-shape change
            ks_stat, p = stats.ks_2samp(ref, cur)
            results.append(DriftResult(
                col, "KS", float(ks_stat), float(p),
                "CRITICAL" if p < self.KS_ALPHA else "ok",
                f"p={p:.2e}"))

            # RANGE VIOLATIONS - often a unit change or a pipeline bug
            lo, hi = self.summary["numeric"][col]["p01"], self.summary["numeric"][col]["p99"]
            out_of_range = ((cur < lo - 3 * (hi - lo)) |
                            (cur > hi + 3 * (hi - lo))).mean()
            if out_of_range > 0.01:
                results.append(DriftResult(
                    col, "range", float(out_of_range), None, "CRITICAL",
                    f"{out_of_range:.1%} far outside the training range - "
                    f"check for a unit or scale change"))

            # MISSING RATE change - usually an upstream problem
            ref_missing = self.summary["numeric"][col]["missing_rate"]
            cur_missing = current[col].isna().mean()
            if abs(cur_missing - ref_missing) > 0.05:
                results.append(DriftResult(
                    col, "missing_rate", float(cur_missing), None, "CRITICAL",
                    f"missing rate {ref_missing:.1%} -> {cur_missing:.1%}"))

        # ---- CATEGORICAL features -----------------------------------
        for col in self.categorical:
            if col not in current.columns:
                continue
            psi = categorical_psi(self.reference[col], current[col])
            severity = ("CRITICAL" if psi > self.PSI_SIGNIFICANT else
                        "warning" if psi > self.PSI_MODERATE else "ok")
            results.append(DriftResult(col, "PSI (categorical)", psi, None,
                                       severity, ""))

            # NEW CATEGORIES the model has never seen
            unseen = set(current[col].dropna().unique()) - set(
                self.summary["categorical"][col])
            if unseen:
                results.append(DriftResult(
                    col, "new_categories", float(len(unseen)), None, "warning",
                    f"unseen values: {sorted(unseen)[:5]}"))

        return results

    def report(self, current: pd.DataFrame):
        results = self.check(current)
        critical = [r for r in results if r.severity == "CRITICAL"]
        warnings = [r for r in results if r.severity == "warning"]

        print(f"{'feature':22s} {'test':20s} {'statistic':>11s} {'severity':>10s}  detail")
        print("-" * 96)
        for r in results:
            if r.severity == "ok":
                continue
            stat = f"{r.statistic:.4f}" if np.isfinite(r.statistic) else "-"
            print(f"{r.feature:22s} {r.test:20s} {stat:>11s} "
                  f"{r.severity:>10s}  {r.detail}")

        print(f"\\n{len(critical)} critical, {len(warnings)} warnings, "
              f"{len(results) - len(critical) - len(warnings)} ok")
        return results, len(critical) > 0


# =====================================================================
# 3. SIMULATE PRODUCTION DRIFT
# =====================================================================
rng = np.random.default_rng(0)

def make_batch(n, shift=0.0, new_category=False, unit_bug=False, missing=0.02):
    df = pd.DataFrame({
        "age": rng.normal(45 + shift * 12, 14, n).clip(18, 90),
        "tenure_months": rng.gamma(3, 14, n).clip(1, 200),
        "monthly_spend": rng.lognormal(3.6 + shift * 0.4, 0.7, n),
        "support_tickets": rng.poisson(1.1 + shift * 1.5, n).astype(float),
        "plan": rng.choice(["basic", "standard", "premium"], n,
                           p=[0.5 - shift*0.25, 0.35, 0.15 + shift*0.25]),
        "region": rng.choice(
            ["north", "south", "east", "west"] + (["central"] if new_category else []),
            n),
    })
    if unit_bug:
        df["monthly_spend"] *= 100         # a currency-unit change upstream
    mask = rng.random(n) < missing
    df.loc[mask, "age"] = np.nan
    return df


NUMERIC = ["age", "tenure_months", "monthly_spend", "support_tickets"]
CATEGORICAL = ["plan", "region"]

reference = make_batch(20000, shift=0.0)
monitor = DriftMonitor(reference, NUMERIC, CATEGORICAL)

scenarios = {
    "week 1 - stable":            make_batch(3000, shift=0.0),
    "week 8 - gradual shift":     make_batch(3000, shift=0.35),
    "week 16 - large shift":      make_batch(3000, shift=0.9),
    "a new region appears":       make_batch(3000, shift=0.1, new_category=True),
    "UPSTREAM UNIT BUG":          make_batch(3000, shift=0.0, unit_bug=True),
    "missing data spike":         make_batch(3000, shift=0.0, missing=0.35),
}

for name, batch in scenarios.items():
    print(f"\\n{'=' * 96}\\n{name}\\n{'=' * 96}")
    monitor.report(batch)
~~~

~~~text
================================================================================
UPSTREAM UNIT BUG
================================================================================
feature                test                   statistic   severity  detail
------------------------------------------------------------------------------------------------
monthly_spend          PSI                       9.8412   CRITICAL  mean 45.3 -> 4531.2
monthly_spend          KS                        1.0000   CRITICAL  p=0.00e+00
monthly_spend          range                     0.9987   CRITICAL  99.9% far outside the
                                                                    training range - check for
                                                                    a unit or scale change

3 critical, 0 warnings, 17 ok
~~~

**That is the most valuable alert in the whole system.** A currency-unit change upstream
would have silently destroyed every prediction. Input monitoring caught it the same day,
without a single label.

### Prediction drift and delayed accuracy

~~~python prediction_monitoring.py
import numpy as np
import pandas as pd
from scipy import stats


class PredictionMonitor:
    """Watching the OUTPUT distribution is the cheapest early warning
    available - it requires no labels at all."""

    def __init__(self, reference_predictions, threshold):
        self.reference = np.asarray(reference_predictions)
        self.threshold = threshold
        self.ref_mean = float(self.reference.mean())
        self.ref_flag_rate = float((self.reference >= threshold).mean())

    def check(self, current_predictions):
        cur = np.asarray(current_predictions)
        psi = population_stability_index(self.reference, cur)
        ks, p = stats.ks_2samp(self.reference, cur)
        flag_rate = float((cur >= self.threshold).mean())

        alerts = []
        if psi > 0.25:
            alerts.append(f"prediction PSI {psi:.3f} - the output distribution shifted")
        if abs(flag_rate - self.ref_flag_rate) > 0.5 * self.ref_flag_rate:
            alerts.append(f"flag rate {self.ref_flag_rate:.3f} -> {flag_rate:.3f}")
        if abs(cur.mean() - self.ref_mean) > 0.1:
            alerts.append(f"mean prediction {self.ref_mean:.3f} -> {cur.mean():.3f}")

        return {"psi": psi, "ks_p": float(p), "flag_rate": flag_rate,
                "mean": float(cur.mean()), "alerts": alerts}


# =====================================================================
# DELAYED LABELS - the realistic accuracy monitoring problem
# =====================================================================
class DelayedLabelTracker:
    """Predictions are made now; the truth arrives in 30-90 days.

    Store predictions, join labels as they arrive, and compute a
    ROLLING accuracy over whatever has matured."""

    def __init__(self):
        self.predictions = {}          # id -> (timestamp, proba, features_hash)
        self.labels = {}               # id -> (timestamp, label)

    def log_prediction(self, entity_id, timestamp, proba):
        self.predictions[entity_id] = (timestamp, proba)

    def log_label(self, entity_id, timestamp, label):
        self.labels[entity_id] = (timestamp, label)

    def matured_metrics(self, window_days=30):
        """Compute metrics on the predictions that now have labels."""
        from sklearn.metrics import roc_auc_score, average_precision_score
        joined = [(self.predictions[i][1], self.labels[i][1])
                  for i in self.labels if i in self.predictions]
        if len(joined) < 100:
            return {"status": "insufficient labels", "n": len(joined)}
        proba, y = zip(*joined)
        return {
            "n": len(joined),
            "roc_auc": float(roc_auc_score(y, proba)),
            "pr_auc": float(average_precision_score(y, proba)),
            "positive_rate": float(np.mean(y)),
        }


MONITORING_PLAN = """
THE COMPLETE MONITORING PLAN

IMMEDIATE (no labels needed)
  system     p50/p95/p99 latency, error rate, throughput, memory
  schema     column presence, dtypes, allowed ranges, enum values
  data       PSI and KS per feature, missing rates, new categories
  prediction output distribution, flag rate, confidence distribution

DELAYED (as labels arrive)
  quality    ROC-AUC, PR-AUC, calibration, on matured predictions
  by segment the SAME metrics per region, plan, cohort - an aggregate
             number hides a failing segment completely

BUSINESS
  the metric the model was built to move, versus a holdout control group

ALERTING
  Do NOT alert on every significant test - with 40 features something is
  always significant. Alert on:
    - any CRITICAL schema or range violation           immediately
    - PSI > 0.25 on a top-10 important feature         immediately
    - PSI > 0.10 on any feature, sustained for 3 days  daily digest
    - prediction flag rate moving more than 50%        immediately
    - quality metric dropping more than 5% relative    on maturity

  Every alert must name the feature, the magnitude and the suggested action.
  An alert nobody can act on will be muted within a week.
"""
print(MONITORING_PLAN)
~~~

### The retraining decision

~~~python retraining.py
"""When to retrain, and how to do it safely."""
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np


@dataclass
class RetrainingPolicy:
    """Both scheduled AND triggered. Neither alone is sufficient."""
    schedule_days: int = 30
    psi_threshold: float = 0.25
    metric_drop_relative: float = 0.05
    min_new_labels: int = 1000
    min_days_between: int = 7

    def should_retrain(self, state) -> tuple[bool, str]:
        # never retrain more often than the minimum interval
        if state["days_since_last"] < self.min_days_between:
            return False, "too soon since the last retrain"

        # never retrain without enough new labelled data
        if state["new_labels"] < self.min_new_labels:
            return False, (f"only {state['new_labels']} new labels, "
                           f"need {self.min_new_labels}")

        reasons = []
        if state["days_since_last"] >= self.schedule_days:
            reasons.append(f"scheduled ({state['days_since_last']} days)")
        if state["max_feature_psi"] > self.psi_threshold:
            reasons.append(f"data drift (PSI {state['max_feature_psi']:.3f})")
        if state["metric_drop"] > self.metric_drop_relative:
            reasons.append(f"performance drop ({state['metric_drop']:.1%})")
        if state["schema_violation"]:
            reasons.append("schema violation - investigate BEFORE retraining")

        return (bool(reasons), "; ".join(reasons) if reasons else "no trigger")


RETRAINING_PROCEDURE = """
THE SAFE RETRAINING PROCEDURE

1. TRIGGER
   Scheduled interval OR a drift/performance trigger fires.

2. INVESTIGATE FIRST
   Is this real drift, or an upstream bug? Retraining on corrupted data
   BAKES THE BUG IN. A schema violation or a range violation is a data
   incident, not a retraining trigger.

3. BUILD THE NEW TRAINING SET
   Decide the window deliberately:
     - all history          most data, but includes stale patterns
     - a rolling window     adapts fast, forgets rare events
     - time-weighted        recent rows weighted higher. Usually best.

4. TRAIN AND VALIDATE
   Same pipeline, same CV protocol. Evaluate on the MOST RECENT period,
   held out chronologically - not a random split.

5. COMPARE AGAINST THE INCUMBENT
   On the SAME recent test window:
     - is the new model better overall?
     - is it better on EVERY important segment, or did it trade one for another?
     - is it still calibrated?
     - is the latency acceptable?
   IF IT IS NOT BETTER, DO NOT PROMOTE IT. Investigate why.

6. SHADOW
   Run it alongside the incumbent on live traffic, discarding its output.
   Compare the prediction distributions. Look for disagreements.

7. CANARY
   5% of traffic, then 25%, then 100%, watching the metrics at each stage.

8. KEEP THE PREVIOUS VERSION DEPLOYABLE
   Rollback must be one command. Version everything: model, data snapshot,
   code commit, and the config.

9. DOCUMENT
   Why it was retrained, what changed, what improved, what regressed.
"""
print(RETRAINING_PROCEDURE)


def compare_models(incumbent, challenger, X_recent, y_recent, segments):
    """The promotion gate. A single aggregate number is not enough."""
    from sklearn.metrics import roc_auc_score, brier_score_loss

    p_old = incumbent.predict_proba(X_recent)[:, 1]
    p_new = challenger.predict_proba(X_recent)[:, 1]

    print(f"{'metric':28s} {'incumbent':>11s} {'challenger':>12s} {'change':>10s}")
    print("-" * 66)

    results = {}
    for name, fn in [("ROC-AUC", roc_auc_score),
                     ("Brier (lower better)", brier_score_loss)]:
        old, new = fn(y_recent, p_old), fn(y_recent, p_new)
        results[name] = (old, new)
        print(f"{name:28s} {old:>11.4f} {new:>12.4f} {new-old:>+10.4f}")

    print(f"\\n{'segment':20s} {'n':>7s} {'incumbent':>11s} {'challenger':>12s} "
          f"{'change':>10s}")
    print("-" * 66)
    regressions = []
    for seg_name, mask in segments.items():
        if mask.sum() < 100 or len(np.unique(y_recent[mask])) < 2:
            continue
        old = roc_auc_score(y_recent[mask], p_old[mask])
        new = roc_auc_score(y_recent[mask], p_new[mask])
        flag = "  <- REGRESSION" if new < old - 0.02 else ""
        if flag:
            regressions.append(seg_name)
        print(f"{seg_name:20s} {mask.sum():>7} {old:>11.4f} {new:>12.4f} "
              f"{new-old:>+10.4f}{flag}")

    promote = (results["ROC-AUC"][1] > results["ROC-AUC"][0]) and not regressions
    print(f"\\nDECISION: {'PROMOTE' if promote else 'DO NOT PROMOTE'}")
    if regressions:
        print(f"  segments that regressed: {regressions}")
    return promote
~~~

:::warn Never retrain automatically on drift alone
Drift may be a genuine shift in the world - or a broken upstream pipeline. Retraining on
corrupted data bakes the corruption into the model and makes the problem permanent and much
harder to diagnose.

**Investigate, then retrain, then compare, then shadow, then canary.** The automation should
prepare and evaluate the candidate; a human should approve the promotion.
:::
`
}
],
quiz: [
{
q: 'Why monitor input distributions rather than waiting for accuracy to drop?',
options: [
  'Inputs are easier to plot',
  'Accuracy needs labels, which may arrive weeks or months later - input drift is observable immediately',
  'Accuracy is unreliable',
  'Inputs matter more than outputs'
],
answer: 1,
why: 'For churn or default prediction, labels mature in 30-90 days. By the time accuracy drops on a dashboard the model has been wrong for weeks. Input drift is the early-warning signal.'
},
{
q: 'A PSI of 0.31 on an important feature means:',
options: [
  'No change',
  'A significant distribution shift - investigate and consider retraining',
  'The model improved',
  'A calculation error'
],
answer: 1,
why: 'The conventional bands are PSI below 0.10 stable, 0.10 to 0.25 moderate, above 0.25 significant. Combine with the KS test and a range check to distinguish real drift from an upstream bug.'
},
{
q: 'Drift is detected. Should you retrain automatically?',
options: [
  'Yes, immediately',
  'No - investigate first, because drift may be an upstream data bug, and retraining on corrupted data bakes it in',
  'Only on weekends',
  'Only if accuracy also dropped'
],
answer: 1,
why: 'A currency-unit change or a renamed field looks exactly like drift. Retraining on it makes the corruption permanent. Investigate, then train a candidate, compare per segment, shadow, canary.'
},
{
q: 'Your challenger model has better overall AUC but is worse on the "west" segment. Should you promote it?',
options: [
  'Yes, overall AUC is what matters',
  'Not without a deliberate decision - an aggregate gain can hide a segment regression that harms real users',
  'Yes, if the difference is under 5%',
  'Never promote a new model'
],
answer: 1,
why: 'Aggregate metrics conceal segment failures. Compare per segment as part of the promotion gate, and make the trade-off explicit and documented rather than accidental.'
}
]
}

]
});
