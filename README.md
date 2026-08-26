# AI Academy

An offline, browser-based course covering Data Science, Machine Learning, Deep Learning,
Computer Vision and AI fundamentals — from "what is machine learning" to deploying a
monitored model in production.

**No server, no internet, no runtime to install.** Double-click `install.bat` and it
becomes a real desktop app — its own icon, its own window, Start Menu and Desktop
shortcuts, and an entry in *Settings → Apps*. Or skip installing entirely and just open
`index.html` in a browser.

---

## Install (Windows)

```
install.bat
```

That is the whole thing. It:

- copies the app to `%LOCALAPPDATA%\Programs\AI Academy` (no administrator rights needed),
- creates **Desktop** and **Start Menu** shortcuts using `assets/ai-academy.ico`,
- registers an uninstall entry so it appears in *Settings → Apps → Installed apps*,
- offers to launch it straight away.

The shortcut opens the app in a **chromeless window** via Edge/Chrome's `--app` mode, so
there is no address bar, no tabs, no browser chrome — it looks and behaves like a native
application while still being a single folder of HTML you can read. If no Chromium browser
is present it falls back to your default browser.

To remove it: *Settings → Apps → AI Academy → Uninstall*, or run `uninstall.bat`. It deletes
the program folder, the shortcuts and the registry entry, and leaves your saved progress
alone.

**Running it without installing:** open `index.html` directly, or run `launch.vbs` for the
app-window experience from wherever the folder happens to live.

### Regenerating the icon

The icon is drawn in code, not committed as hand-made art:

```
powershell -ExecutionPolicy Bypass -File tools\make-icon.ps1
```

This writes `assets/ai-academy.ico` (nine sizes, 16–256px; BMP frames up to 64px for
compatibility, PNG above) plus the two PNGs the page uses as its favicon.

---

## What is in it

| | |
|---|---|
| Tracks | 17 |
| Topics | 80 |
| Guided levels | 132 |
| Quiz questions | 300 |
| Runnable Python examples | 637 code blocks |
| Written content | ~1.2 MB |

### The tracks

| # | Track | Level | What it covers |
|---|---|---|---|
| 1 | Foundations | Beginner | What AI/ML/DL/DS actually are, types of learning, the workflow, overfitting, your first model five times over |
| 2 | The Python Toolkit | Beginner | NumPy, pandas, matplotlib, seaborn |
| 3 | Math for ML | Beginner | Linear algebra, calculus, probability, statistics, information theory |
| 4 | Data Engineering & EDA | Intermediate | EDA, cleaning, encoding, scaling, feature engineering, splitting, imbalance, leak-proof pipelines |
| 5 | Regression | Beginner | Linear, polynomial, Ridge/Lasso/ElasticNet, metrics, diagnostics, trees, a full project |
| 6 | Classification | Intermediate | Logistic, KNN, Naive Bayes, SVM, trees, ensembles, metrics, calibration |
| 7 | Unsupervised Learning | Intermediate | K-Means, hierarchical, DBSCAN, GMM, PCA, t-SNE/UMAP, anomaly detection |
| 8 | Evaluation & Tuning | Intermediate | Bias-variance, hyperparameter search, SHAP, ensembling, error analysis |
| 9 | Deep Learning | Intermediate | Neurons, backprop from scratch in NumPy, activations, optimisers, PyTorch, debugging |
| 10 | CNNs | Advanced | Convolution, pooling, LeNet→ResNet, training on CIFAR-10, transfer learning |
| 11 | Computer Vision | Advanced | OpenCV, object detection with YOLO, U-Net segmentation, ViT and CLIP |
| 12 | Sequence Models | Advanced | RNN, vanishing gradients, LSTM/GRU, seq2seq with attention, time-series forecasting |
| 13 | Transformers & NLP | Advanced | Tokenisation, embeddings, attention from scratch, a full GPT, BERT, LoRA, RAG |
| 14 | Generative Models | Advanced | Autoencoders, VAEs, GANs, diffusion models |
| 15 | AI Fundamentals | Intermediate | Search and A\*, minimax, MCTS, CSPs, reinforcement learning, agents, ethics |
| 16 | MLOps & Deployment | Advanced | Model artefacts, FastAPI, Docker, CI, drift monitoring, retraining |
| 17 | Paths & Projects | Beginner | Six learning paths, ten portfolio project briefs, how to keep going |

---

## How to use it

1. Launch **AI Academy** from the Desktop or Start Menu (or open `index.html` in any
   browser if you did not install it).
2. Pick a track from the home page, or use the sidebar.
3. Each topic follows the same shape:
   - **Explanation** — what it is and how it actually works, in plain language
   - **Level 1** — the absolute simplest version that works
   - **Levels 2, 3, 4…** — each adds one real-world concern (scaling, EDA, validation,
     tuning, deployment) and measures what it was worth
   - **Check yourself** — a short quiz with explanations
4. Mark topics complete. Progress is saved in your browser (`localStorage`).

### Keyboard

| Key | Action |
|---|---|
| `/` | Focus search |
| `Esc` | Clear focus / close the mobile menu |
| `Alt` + `←` / `→` | Previous / next lesson |

### If you are new

Go to **Paths & Projects → Choose your path** first. It gives you an ordered route through
the course for six different goals, with realistic timelines.

---

## Design principles

The content follows a few rules consistently:

- **Every topic starts with intuition, not notation.** The maths comes after you know what
  the thing is for.
- **Every level escalates deliberately.** Level 1 is the naive version; each later level
  adds one concern and *measures* the improvement, so you learn why the habit exists.
- **Baselines everywhere.** A score without a baseline is meaningless, so the code always
  computes one.
- **The failure modes are taught, not hidden.** Leakage, mode collapse, posterior collapse,
  dying ReLU, dead heads, drift — each gets a demonstration you can run.
- **The code is complete and runnable**, not fragments. Outputs are shown so you can check
  your results.

---

## Running the Python examples

The examples are meant to be typed and run. A minimal environment:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1
# Unix:     source .venv/bin/activate

pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Per-track extras, installed when you reach them:

```bash
# Deep learning (tracks 9-14)
pip install torch torchvision

# Computer vision (tracks 10-11)
pip install opencv-python ultralytics timm

# NLP / transformers (track 13)
pip install transformers datasets sentence-transformers peft accelerate

# Gradient boosting (tracks 5-8)
pip install xgboost lightgbm

# MLOps (track 16)
pip install fastapi uvicorn joblib

# Misc
pip install shap optuna imbalanced-learn statsmodels gymnasium umap-learn
```

No GPU is required for tracks 1–8. Tracks 9–14 run on CPU but are far more comfortable with
one — [Google Colab](https://colab.research.google.com) gives you a free GPU if you do not
have one locally.

---

## Project layout

```
ai-academy/
  install.bat                 installs it as a desktop app (per user, no admin)
  uninstall.bat               removes it again
  launch.vbs                  opens the app window without installing
  index.html                  the app shell
  assets/
    ai-academy.ico            the app icon, 16-256px
    icon-256.png              favicon, also the window icon in app mode
    icon-64.png
  css/style.css               styling, light and dark themes
  js/
    render.js                 markdown + Python syntax highlighter (no dependencies)
    app.js                    routing, navigation, progress, search
    curriculum/
      01-foundations.js       ... one file per track
      ...
      17-projects.js
  tools/
    setup.vbs                 shortcut + registry work for install/uninstall
    make-icon.ps1             draws the icon and packs the .ico
    validate.py               content linter (syntax, fences, callouts, structure)
    render-test.js            renders every markdown block and checks the output
    app-test.js               structural checks against the app's assumptions
  README.md
```

### Adding or editing content

Content lives in JavaScript template literals, so it uses tilde syntax instead of
backticks:

- ` ~~~python ... ~~~ ` for fenced code (a filename after the language becomes a label)
- `~inline code~`
- `:::tip Title ... :::` for callouts — `tip`, `warn`, `danger`, `math`, `note`

After any edit, run all three checks:

```bash
python tools/validate.py     # backticks, ${, fences, callouts, required fields
node tools/render-test.js    # every block renders cleanly
node tools/app-test.js       # ids, links, quizzes, search
```

`validate.py` exists because a stray backtick or `${` inside a template literal silently
breaks the whole file — it catches both, along with unbalanced fences and malformed quizzes.

---

## Technical notes

- **Zero dependencies.** No CDN, no build step, no framework. Everything is vanilla
  HTML/CSS/JS and runs from `file://`.
- **The markdown renderer and Python syntax highlighter are custom** (`js/render.js`,
  ~250 lines) so the app stays fully offline.
- **Progress and theme** are stored in `localStorage`, per browser.
- **Content is loaded as classic scripts**, not ES modules, specifically so `file://` works
  without a server.
- **The installer is plain batch and VBScript** — no Electron, no packaging toolchain, no
  bundled browser, nothing to download. That keeps the whole thing about 1.7 MB instead of
  the ~200 MB an Electron build would cost, and VBScript is used instead of PowerShell so
  installing never trips over an execution policy.

---

*Built to be read start to finish, or dipped into anywhere.*
