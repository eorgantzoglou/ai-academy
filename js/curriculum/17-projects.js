/* Track 17 - Learning paths and portfolio projects */
(window.CURRICULUM = window.CURRICULUM || []).push({
id: 'projects',
title: 'Paths & Projects',
icon: 'Pr',
level: 'Beginner',
blurb: 'How to work through this course depending on your goal, ten portfolio projects with full briefs, and how to keep learning once you finish.',
intro: `
## Two ways to use this course

**Linear** - start at Foundations and work through to MLOps. Roughly 3-6 months at a few
hours a week, and you will finish genuinely competent.

**Targeted** - pick the path below that matches your goal and take only what it needs.
Faster, and you can always come back for the rest.

Either way: **build things**. Reading about gradient descent and implementing it are
different experiences, and only one of them sticks.

:::tip The rule that decides whether this works
For every topic, type the code out and run it. Change something and see what breaks.
Copy-pasting teaches you nothing; typing and debugging teaches you everything.
:::
`,
topics: [

/* ============================================================ */
{
id: 'learning-paths',
title: 'Choose your path',
summary: 'Ordered routes through the course for six different goals, with realistic time estimates.',
tags: ['guide', 'planning'],
intro: `
## The dependency map

~~~text
FOUNDATIONS ---> TOOLKIT ---> DATA ENGINEERING ---+--> REGRESSION ---+
     |              |                             |                  |
     |              |                             +--> CLASSIFICATION+--> EVAL & TUNING
     |              |                             |                  |
     |              |                             +--> UNSUPERVISED -+
     |              |                                                |
     |              +--> MATH ------------------------------------+  |
     |                                                            |  |
     +------------------------------------------------------------  |
                                                                 |  |
                                       DEEP LEARNING <-----------+--+
                                            |
                       +--------------------+--------------------+
                       |                    |                    |
                      CNNs             SEQUENCES          TRANSFORMERS
                       |                    |                    |
                  COMPUTER VISION           +------> GENERATIVE <-+
                       |                                 |
                       +------------> MLOPS <------------+

  AI FUNDAMENTALS is independent - take it whenever you like.
~~~

## Do not skip these

Whatever your path:

1. **Foundations / Overfitting** - the concept everything else defends against
2. **Data Engineering / Splitting** and **Pipelines** - leakage will ruin your work otherwise
3. **Classification / Metrics** - accuracy is a trap
4. **Model Eval / Error analysis** - the habit that separates practitioners from tutorial-followers
`,
keyPoints: [
  'Pick a path, but never skip overfitting, splitting, metrics or error analysis.',
  'Build a project after every two or three tracks - passive reading fades fast.',
  'Classical ML pays the bills; learn it properly before deep learning.',
  'Consistency beats intensity: an hour a day beats a weekend binge.'
],
pitfalls: [
  'Starting with deep learning because it sounds more impressive.',
  'Reading everything and building nothing.',
  'Collecting tutorials instead of finishing one project end to end.',
  'Skipping the maths entirely, then being unable to debug anything.'
],
levels: [
{
name: 'The six paths',
goal: 'Find the route that matches your goal, with the tracks in order and a realistic timeline.',
md: `
## Path 1 - Complete beginner to employable data scientist

**Time: 4-6 months at 8-10 hours a week.** The full route, in order.

~~~text
MONTH 1   FOUNDATIONS (all 7 topics)
          TOOLKIT (all 4)
          -> project: EDA on a dataset you personally care about

MONTH 2   MATH (linear algebra, calculus, statistics - skim information theory)
          DATA ENGINEERING (all 8)
          -> project: a full cleaning and feature-engineering pipeline

MONTH 3   REGRESSION (all 6)
          CLASSIFICATION (all 5)
          -> project: the housing project, then a classification project

MONTH 4   UNSUPERVISED (all 5)
          EVALUATION & TUNING (all 5)
          -> project: customer segmentation, end to end

MONTH 5   DEEP LEARNING (all 5)
          CNNs (all 4)
          -> project: an image classifier with transfer learning

MONTH 6   MLOPS (all 3)
          -> project: deploy one earlier model as a real API in Docker

  Optional afterwards, by interest: computer vision, sequences,
  transformers, generative, AI fundamentals.
~~~

---

## Path 2 - I can code, I want machine learning fast

**Time: 6-8 weeks.** Skip the Python basics; keep everything that prevents mistakes.

~~~text
WEEK 1    FOUNDATIONS: what is AI, types of learning, ML workflow, OVERFITTING
          TOOLKIT: numpy, pandas (skim), visualisation

WEEK 2    DATA ENGINEERING: EDA, cleaning, encoding, scaling
          -> the leakage lesson is NOT optional

WEEK 3    DATA ENGINEERING: splitting, pipelines, feature engineering, imbalance
          REGRESSION: linear, regularised, metrics

WEEK 4    CLASSIFICATION: logistic, trees & ensembles, METRICS & CALIBRATION

WEEK 5    EVALUATION & TUNING: bias-variance, tuning, error analysis
          -> project: a complete tabular project with a real dataset

WEEK 6    UNSUPERVISED: k-means, PCA
          MLOPS: serving

WEEKS 7-8 DEEP LEARNING, if you need it
          -> project: something you would actually put on a CV
~~~

---

## Path 3 - Deep learning and computer vision

**Time: 8-10 weeks.** Assumes you already know Python and basic ML.

~~~text
WEEK 1    MATH: linear algebra, calculus (essential for debugging networks)
          FOUNDATIONS: overfitting

WEEK 2    DEEP LEARNING: perceptron, backprop from scratch
          -> implement the NumPy network yourself. Do not skip this.

WEEK 3    DEEP LEARNING: activations/losses/optimisers, PyTorch
WEEK 4    DEEP LEARNING: regularisation and debugging
          -> project: a tabular neural network, beaten by gradient boosting,
             and understand why

WEEK 5    CNNs: convolution, architecture
WEEK 6    CNNs: ResNet, transfer learning
          -> project: fine-tune a pretrained model on your own images

WEEK 7    VISION: image basics and OpenCV
WEEK 8    VISION: object detection with YOLO
WEEK 9    VISION: segmentation, ViT and CLIP
WEEK 10   MLOPS: serving a vision model
          -> project: a deployed detection or classification service
~~~

---

## Path 4 - NLP and large language models

**Time: 8-10 weeks.**

~~~text
WEEK 1    FOUNDATIONS + TOOLKIT (skim), MATH: linear algebra
WEEK 2    DEEP LEARNING: the whole track
WEEK 3    TRANSFORMERS: text representation
          -> build the BPE tokeniser yourself

WEEK 4    SEQUENCES: RNN and LSTM
          -> you need the vanishing-gradient problem to appreciate attention

WEEK 5    TRANSFORMERS: attention from scratch
WEEK 6    TRANSFORMERS: the full architecture
          -> train the character-level GPT. It is the single most
             instructive exercise in the track.

WEEK 7    TRANSFORMERS: BERT, GPT, fine-tuning
WEEK 8    TRANSFORMERS: prompting, LoRA
WEEK 9    TRANSFORMERS: RAG and evaluation
          -> project: a RAG system over documents you care about,
             WITH a retrieval-recall evaluation

WEEK 10   MLOPS + AI FUNDAMENTALS: agents and ethics
~~~

---

## Path 5 - I have an interview in three weeks

**Time: 3 weeks, intensive.** Cover what gets asked, and be able to defend it.

~~~text
WEEK 1  CONCEPTS THEY WILL ASK ABOUT
        Foundations: overfitting, bias-variance, the ML workflow
        Data eng: leakage, splitting, imbalance
        Classification: metrics, precision/recall trade-off, ROC vs PR
        Evaluation: cross-validation, why nested CV exists

WEEK 2  THINGS THEY WILL ASK YOU TO EXPLAIN
        Regression: regularisation, L1 vs L2, multicollinearity
        Classification: how a tree splits, bagging vs boosting
        Unsupervised: k-means assumptions, PCA
        Deep learning: backprop, vanishing gradients, batch norm vs layer norm

WEEK 3  BUILD AND REHEARSE
        Take ONE project end to end and be able to talk about:
          - why you framed it that way
          - what the baseline was
          - what you tried that did NOT work
          - what the errors looked like, and what you learned from them
          - what you would do next with more time

        That last set of answers is what actually distinguishes candidates.
~~~

**The questions that come up most:**

1. Explain the bias-variance trade-off. -> Foundations / Overfitting
2. Your model is 99% accurate. Is it good? -> Classification / Metrics
3. What is data leakage, and give an example. -> Data Engineering / Pipelines
4. L1 versus L2 regularisation. -> Regression / Regularised
5. How do you handle imbalanced data? -> Data Engineering / Imbalance
6. Random forest versus gradient boosting. -> Classification / Trees & ensembles
7. Explain backpropagation. -> Deep Learning / Backprop from scratch
8. What is attention? -> Transformers / Self-attention
9. How would you deploy and monitor this? -> MLOps
10. Tell me about a project that went wrong. -> your own experience

---

## Path 6 - Classical AI, not machine learning

**Time: 3-4 weeks.** For robotics, operations research, games or planning.

~~~text
WEEK 1  AI FUNDAMENTALS: search algorithms
        -> implement A* and use it for something real
WEEK 2  AI FUNDAMENTALS: games and constraint satisfaction
        -> build a solver for a scheduling problem you actually have
WEEK 3  AI FUNDAMENTALS: reinforcement learning
        -> train a DQN agent
WEEK 4  AI FUNDAMENTALS: agents and ethics
        MATH: probability
~~~

---

## How to study so it sticks

~~~text
FOR EVERY TOPIC

  1. Read the introduction. Do not skip it - it is the mental model.
  2. TYPE the Level 1 code. Do not copy-paste. Run it.
  3. BREAK IT deliberately. Change a parameter. Remove a line.
     Predict what will happen, then check.
  4. Work up through the levels.
  5. Answer the quiz WITHOUT looking back.
  6. Write two or three sentences in your own words about what you learned.

  If you cannot explain it simply, you have not learned it yet.

EVERY TWO OR THREE TRACKS: build something with your own data.
  A project you chose, on data you care about, is worth ten tutorials.

SPACING BEATS CRAMMING
  Four hours across four days beats four hours in one sitting, by a
  wide margin. Come back to old topics after a week and re-answer the quiz.
~~~
`
}
],
quiz: [
{
q: 'You want to work in industry on tabular business data. Which should you master first?',
options: [
  'Transformers and LLMs',
  'Data engineering plus classical ML (regression, classification, ensembles)',
  'Computer vision',
  'Reinforcement learning'
],
answer: 1,
why: 'The majority of deployed models are gradient boosting on tabular data. Data quality and feature work move the metric far more than model choice, and that is what most jobs actually involve.'
},
{
q: 'What is the most effective way to use this course?',
options: [
  'Read everything first, then start coding',
  'Type and run the code for each topic, break it deliberately, and build a project every few tracks',
  'Copy the code into a file for reference',
  'Watch videos instead'
],
answer: 1,
why: 'Passive reading fades within days. Typing forces attention, running reveals what you misunderstood, and deliberately breaking things teaches you what each part is actually doing.'
},
{
q: 'Which topics should nobody skip, regardless of path?',
options: [
  'Transformers and CNNs',
  'Overfitting, splitting/pipelines, classification metrics, and error analysis',
  'Reinforcement learning',
  'Generative models'
],
answer: 1,
why: 'These four are the ones that prevent you from producing confidently wrong work. Everything else is a specialisation; these are the professional baseline.'
}
]
},

/* ============================================================ */
{
id: 'portfolio-projects',
title: 'Ten portfolio projects',
summary: 'Full project briefs - the question, the data, the required steps, the deliverable, and what makes each one impressive rather than ordinary.',
tags: ['projects', 'portfolio'],
intro: `
## What makes a project impressive

Almost every beginner portfolio contains the same three notebooks: Titanic, iris, and a
housing regression, each ending at ~model.fit~ with an accuracy number.

Here is what actually distinguishes a portfolio:

~~~text
ORDINARY                              IMPRESSIVE
"I got 94% accuracy"                  "I got 94% against a 78% baseline, and
                                       the errors cluster in one segment -
                                       here is why, and here is the fix"

a clean notebook                      a notebook AND a deployed API AND a README
                                       that explains the decisions

used a famous dataset                 collected or combined your own data

tried five models                     framed the problem, chose the metric from
                                       the business cost, and can defend both

no mention of failure                 "I tried X, it did not work, here is what
                                       I learned and what I did instead"
~~~

:::tip The single highest-value addition
**Deploy one project.** A working URL where someone can send a request and get a
prediction puts you ahead of the large majority of applicants, because it proves you
understand everything between a notebook and a running system.
:::
`,
keyPoints: [
  'Always report a baseline - a score without one means nothing.',
  'Error analysis and a documented limitation are more impressive than a higher score.',
  'Deploy at least one project so there is something a person can actually use.',
  'Write the README for a reader who has five minutes.'
],
pitfalls: [
  'Ending at model.fit with no evaluation, no baseline and no discussion.',
  'Using only famous tutorial datasets.',
  'Hiding what did not work - it is the most interesting part.',
  'A README that only says how to install dependencies.'
],
levels: [
{
name: 'The project briefs',
goal: 'Ten complete specifications, ordered by difficulty, each with a defined deliverable.',
md: `
## 1. Exploratory analysis of something you care about

**Level: beginner. Time: 1 week. Tracks needed: Foundations, Toolkit.**

~~~text
THE BRIEF
  Choose a dataset about a subject you personally find interesting - your
  city's open data, a sport, a game you play, music listening history,
  public health, climate, your own bank statements.

REQUIRED
  [ ] A clear question stated at the top: what are you trying to find out?
  [ ] A complete column profile: types, missing, distributions, outliers
  [ ] At least five charts, each answering a specific question
  [ ] At least three findings a non-technical reader would find surprising
  [ ] Explicit limitations: what this data CANNOT tell you

DELIVERABLE
  A notebook that reads like an article, and a 300-word summary at the top.

WHAT MAKES IT GOOD
  The findings. Anyone can plot a histogram. Finding something genuinely
  non-obvious and explaining WHY it is true is the skill on display.
~~~

---

## 2. A regression model with an honest evaluation

**Level: beginner. Time: 1-2 weeks. Tracks: + Data Engineering, Regression.**

~~~text
THE BRIEF
  Predict a continuous outcome. House prices, bike-share demand, energy
  consumption, delivery time, crop yield.

REQUIRED
  [ ] The framing checklist filled in BEFORE modelling
  [ ] Baselines: mean, median, and one simple heuristic
  [ ] At least three model families, cross-validated
  [ ] Feature engineering, with a MEASURED improvement
  [ ] Residual analysis: what does the model systematically get wrong?
  [ ] Test-set evaluation, opened once
  [ ] Metrics in business units, not just R2

DELIVERABLE
  A repo with src/, notebooks/, a saved model and a README.

WHAT MAKES IT GOOD
  The residual analysis. Showing "the model underestimates in this range,
  and here is why" demonstrates real understanding.
~~~

---

## 3. Classification with a real cost trade-off

**Level: intermediate. Time: 2 weeks. Tracks: + Classification, Evaluation.**

~~~text
THE BRIEF
  A binary classification problem where the two errors have DIFFERENT costs.
  Fraud, medical screening, equipment failure, loan default, churn.

REQUIRED
  [ ] State the cost of a false positive and a false negative, in units
  [ ] A DummyClassifier baseline
  [ ] Handle the class imbalance, and justify how
  [ ] The threshold chosen from the cost function, not left at 0.5
  [ ] A calibration curve, and calibration if needed
  [ ] Per-segment performance, not just aggregate
  [ ] SHAP explanations for individual predictions

DELIVERABLE
  Notebook plus a one-page summary written for a decision-maker.

WHAT MAKES IT GOOD
  The threshold analysis. Show the cost curve and state, in euros, what
  moving the threshold saved.
~~~

---

## 4. Customer segmentation that a business could act on

**Level: intermediate. Time: 2 weeks. Tracks: + Unsupervised.**

~~~text
THE BRIEF
  Cluster customers, users or products into segments, and make each one
  actionable.

REQUIRED
  [ ] RFM or equivalent domain features, engineered deliberately
  [ ] k chosen with silhouette AND business interpretability
  [ ] At least two algorithms compared (k-means and one other)
  [ ] Every cluster PROFILED and NAMED in one sentence
  [ ] A recommended action per segment
  [ ] Stability check: does the clustering survive a resample?

DELIVERABLE
  A segment report with a profile card per cluster.

WHAT MAKES IT GOOD
  The naming and the actions. "Cluster 2" is worthless; "High-value
  at-risk: spends 3x average but has not purchased in 90 days - target
  with a win-back offer" is a business deliverable.
~~~

---

## 5. An image classifier with transfer learning

**Level: intermediate. Time: 2 weeks. Tracks: + Deep Learning, CNNs.**

~~~text
THE BRIEF
  Classify images in a domain you choose. Ideally COLLECT YOUR OWN -
  plant species from your garden, your own photo library, product
  categories, recyclable materials.

REQUIRED
  [ ] 500+ images, at least 5 classes
  [ ] A from-scratch baseline versus a fine-tuned pretrained model
  [ ] An augmentation ablation: measure what each transform is worth
  [ ] A confusion matrix, with the confusions explained
  [ ] Grad-CAM on both correct and incorrect predictions
  [ ] Discussion of what would break in deployment

DELIVERABLE
  A repo, a saved model, and a small demo (Gradio or Streamlit).

WHAT MAKES IT GOOD
  Your own data, and the Grad-CAM analysis. Showing that the model
  attends to the object rather than the background is exactly the check
  a professional would run.
~~~

---

## 6. Time-series forecasting with honest baselines

**Level: intermediate. Time: 2 weeks. Tracks: + Sequences.**

~~~text
THE BRIEF
  Forecast something with a real seasonal structure. Energy demand, sales,
  website traffic, air quality, bike-share usage.

REQUIRED
  [ ] STL decomposition and a stationarity discussion
  [ ] Naive and seasonal-naive baselines, with MASE reported
  [ ] Classical (Holt-Winters or SARIMA) AND ML (gradient boosting on lags)
  [ ] STRICTLY causal features - every rolling window shifted
  [ ] Expanding-window cross-validation
  [ ] Error reported PER HORIZON
  [ ] Prediction intervals, with measured coverage

DELIVERABLE
  A notebook and a forecast function that takes history and returns a
  forecast with intervals.

WHAT MAKES IT GOOD
  Reporting MASE and beating the seasonal-naive baseline. A great many
  published forecasting projects quietly do not.
~~~

---

## 7. A RAG system over documents you care about

**Level: advanced. Time: 2-3 weeks. Tracks: + Transformers.**

~~~text
THE BRIEF
  Build question answering over a document collection - your company
  handbook, a textbook, legislation, research papers, game rules.

REQUIRED
  [ ] Structure-aware chunking, with the strategy justified
  [ ] Hybrid search (dense + BM25) and a reranker
  [ ] A TEST SET of 50+ questions with known answer locations
  [ ] Retrieval recall@k measured and reported
  [ ] Citations in every answer
  [ ] An explicit "not found in sources" path
  [ ] Faithfulness evaluation on the generated answers
  [ ] Prompt-injection defence documented

DELIVERABLE
  A working app plus an evaluation report with the retrieval metrics.

WHAT MAKES IT GOOD
  The evaluation. Almost every RAG demo skips it entirely. Showing
  recall@5 = 0.91 and faithfulness = 0.94 puts you in a small minority.
~~~

---

## 8. Fine-tune a language model with LoRA

**Level: advanced. Time: 2 weeks. Tracks: + Transformers.**

~~~text
THE BRIEF
  Adapt a small open model to a specific style, format or narrow task
  that prompting alone cannot achieve.

REQUIRED
  [ ] Justify WHY fine-tuning rather than prompting or RAG
  [ ] A curated dataset of 500-5000 examples, quality over quantity
  [ ] QLoRA on a consumer GPU, with the settings explained
  [ ] Evaluation against the BASE model on a held-out set
  [ ] Both automatic metrics and human judgement
  [ ] Cost and time reported

DELIVERABLE
  The adapter on Hugging Face Hub, plus an evaluation report.

WHAT MAKES IT GOOD
  Comparing against the base model with prompting. If prompting matches
  your fine-tune, say so - that is a genuinely valuable finding, and
  reporting it honestly is more impressive than hiding it.
~~~

---

## 9. Deploy a model as a real service

**Level: advanced. Time: 1-2 weeks. Tracks: + MLOps.**

~~~text
THE BRIEF
  Take any earlier project and make it a running service someone else
  can use.

REQUIRED
  [ ] The whole pipeline saved as one artefact, with a model card
  [ ] A FastAPI service with Pydantic validation
  [ ] Health and readiness endpoints
  [ ] A test suite covering valid input, invalid input and error handling
  [ ] A multi-stage Dockerfile, non-root user
  [ ] Deployed somewhere public (Render, Railway, Fly.io, Hugging Face
      Spaces, or a small cloud VM)
  [ ] Structured request logging
  [ ] A load test with reported p50/p95 latency

DELIVERABLE
  A LIVE URL, plus the repo.

WHAT MAKES IT GOOD
  That it is live. Include the curl command in your README so a reviewer
  can try it in ten seconds.
~~~

---

## 10. An end-to-end system with monitoring

**Level: advanced. Time: 3-4 weeks. The capstone.**

~~~text
THE BRIEF
  A complete production system: ingest, train, serve, monitor, retrain.

REQUIRED
  [ ] Automated data ingestion on a schedule
  [ ] A reproducible training pipeline, with tracked experiments
  [ ] A model registry with versioning
  [ ] The serving API, containerised
  [ ] Drift monitoring on inputs and predictions
  [ ] A dashboard showing system and model health
  [ ] A documented retraining policy, with the promotion gate
  [ ] CI that gates on data validation and metric regression
  [ ] A written runbook: what to do when each alert fires

DELIVERABLE
  A repo, an architecture diagram, a live system, and the runbook.

WHAT MAKES IT GOOD
  The runbook and the monitoring. Anyone can train a model. Very few
  beginners can explain what they would do when it starts drifting at
  3am, and that is exactly what employers are hiring for.
~~~

---

## Where to find data

~~~text
GENERAL
  Kaggle Datasets           huge variety, usually clean
  UCI ML Repository         classic benchmarks
  Hugging Face Datasets     text, image, audio, multimodal
  Papers With Code          datasets attached to published results
  Google Dataset Search     a search engine for datasets

GOVERNMENT AND OPEN DATA
  data.gov / data.gov.uk / data.europa.eu
  Eurostat, World Bank, WHO, OECD
  your own city's open data portal    <- underused and locally relevant

DOMAIN-SPECIFIC
  finance     Yahoo Finance API, FRED, Quandl
  sport       football-data.co.uk, Statsbomb open data
  health      MIMIC-IV (with credentialing), PhysioNet
  climate     NOAA, Copernicus, ERA5
  text        Common Crawl, Wikipedia dumps, arXiv
  vision      COCO, Open Images, ImageNet

BEST OF ALL: COLLECT YOUR OWN
  an API you use, a public site you may scrape, sensors, your own
  photos or logs. A project on data you gathered yourself is
  immediately more interesting than the thousandth Titanic notebook.
~~~

## The README template

~~~text README.md
# Project title

One sentence: what it does and why anyone should care.

## The question
What were you trying to find out or predict, and for whom?

## Result
The headline number, WITH ITS BASELINE.
  "MAE 27,900 dollars, versus 91,100 for a mean baseline - a 69% reduction.
   82% of predictions land within the 50,000 dollar threshold the business set."

## Data
Source, size, time period, licence. Anything unusual about it.

## Approach
1. How you framed it and why
2. What you engineered, and what it was worth
3. Models compared, and how you chose
4. How you validated

## What did not work
The most interesting section. Be specific.

## Limitations
What this model should NOT be used for. Where it fails.

## Run it
    git clone ...
    pip install -r requirements.txt
    python src/train.py

    curl -X POST https://your-live-url/predict -d '{...}'

## Repository layout
Brief, so a reader knows where to look.
~~~

:::warn Three things that lose you an interview
1. **A notebook with no baseline.** The reviewer cannot tell whether 0.87 is good.
2. **No mention of anything that went wrong.** It reads as either dishonest or inexperienced.
3. **A model that predicts the target from a leaked feature.** Reviewers check for this,
   and finding it ends the conversation.
:::
`
}
],
quiz: [
{
q: 'What single addition most improves a portfolio?',
options: [
  'More models compared',
  'Deploying one project as a live service someone can actually call',
  'A higher accuracy number',
  'More visualisations'
],
answer: 1,
why: 'It proves you understand everything between a notebook and a running system - packaging, validation, containers, latency, errors. Very few beginner portfolios have it.'
},
{
q: 'Your project reports 94% accuracy. What must accompany it?',
options: [
  'The training time',
  'The baseline - what a trivial model achieves on the same data',
  'The number of parameters',
  'A confusion matrix only'
],
answer: 1,
why: '94% against a 93% majority-class baseline is nothing; 94% against 61% is substantial. Without the baseline the number is uninterpretable.'
},
{
q: 'Why include a "what did not work" section in your README?',
options: [
  'To fill space',
  'It demonstrates real experimentation and judgement, which is what distinguishes candidates',
  'It is required by convention',
  'To lower expectations'
],
answer: 1,
why: 'Anyone can present a clean success. Explaining what you tried, why it failed and what you concluded shows the thinking that a polished final notebook hides.'
}
]
},

/* ============================================================ */
{
id: 'next-steps',
title: 'After this course',
summary: 'How to keep learning, where the field is going, how to read papers, and how to stay current without drowning.',
tags: ['guide', 'resources'],
intro: `
## You are not finished, and that is correct

This course covers the foundations thoroughly and the frontier partially. The field moves;
the foundations do not. Everything in tracks 1 through 9 will still be true in ten years.
Tracks 13 and 14 will look dated in three.

**That is the right ratio to have learned them in.**
`,
keyPoints: [
  'Foundations age slowly; frontier techniques age fast. Weight your effort accordingly.',
  'Read papers with a three-pass method rather than linearly.',
  'Depth in one domain beats shallow familiarity with all of them.',
  'Build in public - writing forces you to actually understand.'
],
pitfalls: [
  'Chasing every new model release instead of deepening fundamentals.',
  'Reading papers linearly from the abstract, and giving up.',
  'Collecting courses instead of finishing projects.'
],
levels: [
{
name: 'Staying current, and going deeper',
goal: 'A concrete plan for continuing: what to read, how to read it, and what to specialise in.',
md: `
## How to read a paper

Most people read papers linearly and give up on page three. Do this instead:

~~~text
PASS 1 - five minutes. Decide whether to keep reading.
  Read: title, abstract, section headings, all figures and their captions,
        the conclusion.
  Answer: what problem, what approach, what result, is it relevant to me?

PASS 2 - one hour. Understand the content, not the proofs.
  Read: everything except the proofs and the deepest technical details.
  Look carefully at every figure, table and graph. Note the datasets and
  metrics. Mark what you do not understand.
  Answer: could I explain this to a colleague?

PASS 3 - four hours or more. Only for papers you intend to build on.
  Re-derive the method as if you were the author. Question every choice.
  Identify what would break. Try to reimplement the core idea.
  Answer: could I reproduce this?

MOST PAPERS DESERVE ONLY PASS 1.
~~~

**Read the papers that matter, not the newest ones.** A reading list of genuine landmarks:

~~~text
FOUNDATIONS
  Breiman (2001)          Random Forests
  Breiman (2001)          Statistical Modeling: The Two Cultures  <- read this one
  Friedman (2001)         Greedy Function Approximation (gradient boosting)
  Chen & Guestrin (2016)  XGBoost

DEEP LEARNING
  Krizhevsky et al (2012) AlexNet
  Simonyan & Zisserman    VGG
  He et al (2015)         Deep Residual Learning (ResNet)
  Ioffe & Szegedy (2015)  Batch Normalization
  Srivastava et al (2014) Dropout
  Kingma & Ba (2014)      Adam

SEQUENCES AND TRANSFORMERS
  Hochreiter & Schmidhuber (1997)  LSTM
  Bahdanau et al (2014)   Neural MT by Jointly Learning to Align and Translate
  Vaswani et al (2017)    Attention Is All You Need
  Devlin et al (2018)     BERT
  Brown et al (2020)      Language Models are Few-Shot Learners (GPT-3)
  Hu et al (2021)         LoRA
  Lewis et al (2020)      Retrieval-Augmented Generation

GENERATIVE
  Kingma & Welling (2013) Auto-Encoding Variational Bayes
  Goodfellow et al (2014) Generative Adversarial Networks
  Ho et al (2020)         Denoising Diffusion Probabilistic Models
  Rombach et al (2021)    Latent Diffusion (Stable Diffusion)

PRACTICE AND JUDGEMENT
  Sculley et al (2015)    Hidden Technical Debt in Machine Learning Systems
  Kapoor & Narayanan      Leakage and the Reproducibility Crisis in ML-based Science
  Mitchell et al (2019)   Model Cards for Model Reporting
  Gebru et al (2018)      Datasheets for Datasets
~~~

:::tip Start with two
If you read only two of those, make them **"Statistical Modeling: The Two Cultures"** and
**"Hidden Technical Debt in Machine Learning Systems."** Neither has an equation you need to
work through, and both will change how you think about the work.
:::

---

## Books worth owning

~~~text
PRACTICAL
  Hands-On Machine Learning (Geron)            the best single ML book
  Python for Data Analysis (McKinney)          by the author of pandas
  Designing Machine Learning Systems (Huyen)   the production perspective
  Deep Learning with PyTorch (Stevens et al)

THEORETICAL
  An Introduction to Statistical Learning       free, and outstanding
  The Elements of Statistical Learning          free, and hard
  Pattern Recognition and ML (Bishop)           the Bayesian view
  Deep Learning (Goodfellow, Bengio, Courville) free online

SPECIALISED
  Speech and Language Processing (Jurafsky & Martin)   free, NLP
  Reinforcement Learning (Sutton & Barto)              free, definitive
  Artificial Intelligence: A Modern Approach (Russell & Norvig)
  Interpretable Machine Learning (Molnar)              free
~~~

---

## Staying current without drowning

~~~text
WEEKLY (30 minutes)
  - one newsletter: Import AI, The Batch, or Ahead of AI
  - scan Papers With Code trending

MONTHLY (2 hours)
  - read ONE paper properly (pass 2)
  - try ONE new library or technique on an existing project

QUARTERLY (a weekend)
  - build something with a technique you have not used
  - revisit an old project and improve it with what you now know

IGNORE
  - most model release announcements
  - benchmark leaderboard churn
  - anything promising that a technique makes everything else obsolete

The signal-to-noise ratio in AI news is poor. The fundamentals you learned
here will still apply when this year's models are forgotten.
~~~

---

## Choosing a specialisation

Breadth got you here. **Depth is what gets you hired and keeps you interesting.**

~~~text
TABULAR / BUSINESS ML          the largest job market by far
  go deeper: feature stores, causal inference, uplift modelling,
             survival analysis, experimentation platforms

COMPUTER VISION
  go deeper: 3D vision, video understanding, multimodal models,
             efficient inference on edge devices

NLP / LLM ENGINEERING
  go deeper: retrieval systems, evaluation, agent architectures,
             inference optimisation, alignment

MLOPS / PLATFORM
  go deeper: distributed training, feature stores, orchestration,
             cost optimisation, observability at scale

RESEARCH
  go deeper: read constantly, reproduce papers, publish, do a PhD if
             you want to work on frontier problems

DOMAIN SPECIALIST
  medicine, finance, climate, biology, law. DOMAIN KNOWLEDGE PLUS ML
  is rarer and more valuable than ML alone. If you already have a
  domain, this is almost certainly your highest-leverage path.
~~~

---

## Build in public

~~~text
WHY IT WORKS
  Writing forces you to actually understand. You cannot explain a concept
  clearly while still confused about it - the act of writing exposes the gaps.

WHAT TO DO
  - a blog post per project, explaining a decision rather than the code
  - answer questions on forums; teaching is the fastest way to learn
  - contribute to an open-source library you use, starting with docs
  - present at a local meetup

WHAT IT GETS YOU
  Better understanding. A public record of your thinking. And a
  professional network, which is how most jobs actually happen.
~~~

---

## A final word

You now know how to:

- frame a problem, choose a metric, and build a baseline
- clean data, engineer features, and avoid leakage
- train, tune, evaluate and interpret every major model family
- build neural networks from scratch and in a framework
- work with images, sequences and text
- deploy a model and keep it healthy
- ask whether you should be building it at all

That is a genuinely broad foundation. The remaining work is not more tracks - it is
**repetition on real problems**, which is the only thing that turns knowledge into skill.

~~~text
  Pick a problem you care about.
  Build the simplest thing that could work.
  Measure it against a baseline.
  Look at what it gets wrong.
  Fix the biggest thing.
  Repeat.

  That loop is the whole job.
~~~
`
}
],
quiz: [
{
q: 'What is the most efficient way to read a research paper?',
options: [
  'Linearly from abstract to conclusion',
  'Three passes: five minutes for relevance, an hour for content, hours only for papers you will build on',
  'Read only the abstract',
  'Read the code first'
],
answer: 1,
why: 'Most papers deserve only the first pass. The three-pass method lets you triage quickly and spend depth only where it pays.'
},
{
q: 'Which is more valuable for most careers?',
options: [
  'Shallow familiarity with every technique',
  'Depth in one area, especially combined with domain knowledge',
  'Reading every new model announcement',
  'Memorising algorithm details'
],
answer: 1,
why: 'Domain expertise plus ML is rarer and more valuable than ML alone. Breadth gets you started; depth is what makes you genuinely useful on hard problems.'
},
{
q: 'Which parts of what you learned here will age fastest?',
options: [
  'Bias-variance, leakage, metrics, evaluation',
  'Specific model architectures and current LLM tooling',
  'Linear algebra and probability',
  'The scientific method'
],
answer: 1,
why: 'Architectures and tooling turn over every couple of years. The evaluation discipline, the statistical reasoning and the engineering practices are durable - which is why they got the most space here.'
}
]
}

]
});
