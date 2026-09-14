# Hiver AI Support Agent for `@AmazonHelp`

[![GitHub](https://img.shields.io/badge/GitHub-AyushDocs%2Fhiver--support--agent-181717?logo=github)](https://github.com/AyushDocs/hiver-support-agent)
[![HF Models](https://img.shields.io/badge/HF%20Models-hiver--support--agent-FFD21E?logo=huggingface)](https://huggingface.co/24f2004275/hiver-support-agent)
[![HF Datasets](https://img.shields.io/badge/HF%20Dataset-hiver--support--golden-FFD21E?logo=huggingface)](https://huggingface.co/datasets/24f2004275/hiver-support-golden)

An end-to-end support agent that, given a customer tweet, **classifies the
intent**, **drafts a reply grounded in how AmazonHelp historically resolved the
same kind of issue**, and **decides whether to auto-handle or escalate to a
human** — with a stated reason. No LLM is required for inference; everything is
retrieval + trained bag-of-words embeddings + a transparent rule.

```
tweet ──► 01 clean ─► 02 intent (NMF topics) ─► 03 classify (TF-IDF or word2vec + LR)
                         │                           │
                         ▼                           ▼
      golden_eval.csv / golden_hand.csv        embeddings + models
                         │                           ▼
                         └──────────────► agent.py: classify → retrieve reply → route
                                                       │
              eval.py (metrics + baselines + LLM judge) ◄─┘
```

Headline results (see [`REPORT.md`](REPORT.md) — read §7 for what is misleading).
TF-IDF engine is the default (`--engine w2v` reproduces the notebooks):

| Metric | TF-IDF | w2v | Baseline |
|---|---:|---:|---:|
| Intent accuracy (auto-gold N=200) | **0.930** | 0.775* | 0.400 |
| Routing accuracy vs auto-gold | **0.975** | 0.915 | 0.425 |
| Agent vs human labels (N=50, intent) | **0.625** | 0.583 | 0.520 silver-vs-human |
| Corpus auto-route rate | **48.34%** | ≈48.3%* | — |

\* w2v is trained by the mirror pipeline with `workers=4` (gensim) and drifts
~1.5 pts per retrain; TF-IDF is deterministic. See REPORT.md §9.

## Repository layout

```
agent.py                 live agent (classify + history-grounded draft + route)
build_golden.py          builds data/processed/golden_eval.csv (N=200, auto-labeled)
build_golden_hand.py     builds data/processed/golden_hand.csv (N=50, human-labeled)
hand_labels.py           the 50 human labels (reviewer evidence for audit)
eval.py                  evaluation harness (metrics, baselines, LLM judge, kappa)
train_tfidf.py           trains the default TF-IDF intent engine (0.924 test acc)
local_preprocess.py      01 mirror (needs raw twcs.csv)
local_run.py             02 mirror (NMF intent discovery)
local_classify.py        03 mirror (word2vec + logistic regression classifier)
local_route.py           04 mirror (auto-route rule)
reconstruct_preprocessed.py  rebuilds 01 output from raw linkage + labeled text
train_tfidf.py           trains the default TF-IDF intent engine (0.924 test acc)
pyproject.toml           pip-installable package (`hiver` console script)
data/hf_upload.py        publishes golden sets + models to the Hugging Face Hub
notebooks/01..04_*.ipynb executed Kaggle notebooks (canonical pipeline)
data/
  raw/twcs.csv           raw corpus (~516 MB, download step)      [optional]
  processed/             01–05 outputs: preprocessed/labeled/classified/routed,
                          golden_eval/_hand, eval_report[_w2v|_tfidf], routing_report etc.
  models/                word2vec + tfidf models, logistic-regression engines
REPORT.md                ≤6-page report (framing, baselines, failure modes, caveats)
DECISION_LOG.md          20 decisions with alternatives & rationale
```

## Setup

```bash
# as a pip package (installs the `hiver` CLI)
pip install -e .            # dev install from this repo
hiver "my order never arrived" --author-id 172791877

# or plain venv + requirements
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`: `nltk`, `pyspellchecker`, `emoji`, `pandas`, `tqdm`,
`scikit-learn`, `joblib`, `openai`, `anthropic`.

Optional extras (`pip install -e '.[judge]'`, `'.[pipeline]'`, `'.[all]'`) add
the LLM judge clients, the gensim/matplotlib pipeline mirrors, or both.

LLM judge (optional): set `OPENAI_API_KEY` (used by default, `gpt-4o-mini`) or
`ANTHROPIC_API_KEY` / `ANTHROPIC_BASE_URL`. If no endpoint is reachable, `eval.py`
falls back to a deterministic judge and records `llm_judge.mode` accordingly.
Judge counts as an extra ~2 min for `--judge-n 30`.

Models & eval data are mirrored on the Hugging Face Hub
([models](https://huggingface.co/24f2004275/hiver-support-agent),
[dataset](https://huggingface.co/datasets/24f2004275/hiver-support-golden));
re-upload with `HF_TOKEN=... python data/hf_upload.py`.

## 15-minute reproduction

All pipeline *outputs* are already in `data/processed`. The commands below
re-run every stage to verify the chain end-to-end.

> Stage 01 (`local_preprocess.py`) needs the 516 MB raw file and is the slow
> step (~hours). **Skip it:** `data/processed/data_preprocessed.csv` is shipped
> (reconstructed losslessly from the labeled set + raw linkage). Re-run 01 only
> if you have the raw file and a few hours.

```bash
# Step 0 — data (skip if data/raw/twcs.csv already present)
#   Required only for local_route.py / agent cache rebuild:
mkdir -p data/raw
kaggle datasets download -d thoughtvector/customer-support-on-twitter   # unzip twcs.csv into data/raw/

# Step 1 — intent discovery (02)          ~1–2 min
python local_run.py                       # -> amazon_help_labeled.csv, intent_mapping.json

# Step 2 — classifier (03)                ~2–3 min
python local_classify.py                  # -> embeddings.npz, models, amazon_help_classified.csv

# Step 3 — auto-route (04)                ~1–2 min (reads raw twcs.csv)
python local_route.py                     # -> amazon_help_routed.csv, routing_report.json, routing_plot.png

# Step 4 — golden sets + evaluation          ~2 min (+ judge ~2 min)
python build_golden.py                       # -> golden_eval.csv
python build_golden_hand.py                  # -> golden_hand.csv (human labels, already shipped)
python eval.py --judge-n 30                 # -> eval_report.json, eval_predictions.csv
                                             #    (+ judge vs gold + vs human labels, kappa)
# Step 5 — try the live agent                seconds
python agent.py "my package said out for delivery today but it never arrived" --author-id 172791877
```

Total re-run ≈ 8–10 min (excluding the raw-file download). The notebooks in
`notebooks/` contain the same code (pre-executed) for reference.

## Using the agent

```bash
python agent.py "customer tweet" [--author-id <id>] [--k 3]
```

Returns intent, confidence, route (`auto`/`assist`), route reason, draft, and
the retrieval evidence (historical tweets + the brand's actual replies used).
Importable too:

```python
from agent import Agent
ag = Agent()
r = ag.respond('@AmazonHelp cancel my order #445566 please', author_id='172791877')
# r['pred_intent'], r['pred_confidence'], r['route'], r['route_reason'], r['draft'], r['evidence']
```

History caches (`amazon_replies.csv`, `author_history.csv`) are built once from
the raw corpus and reused; the agent also runs without the raw file once they
exist. Routing constants live at the top of `agent.py`/`local_route.py`.

## Evaluation, honest

`golden_eval.csv` (N=200) is **stratified by intent** and **auto-labeled**
(silver topic intent + routing rule), with `gold_intent`/`gold_route` columns a
human can flip before re-running `eval.py`. Because the labels come from the same
pipeline, the headline accuracies are optimistic by construction — see
[`REPORT.md` §7](REPORT.md) and top-5 failure modes §6.

`golden_hand.csv` (N=50) is the one **human-independent** label set. `eval.py`
scores the agent against it and reports **judge↔human agreement (Cohen's κ)** so
the LLM judge cannot silently validate the pipeline circularly. The honest
finding: silver-vs-human intent agreement is only 0.52, and the judge's routing κ
vs humans is ≈ 0 — treat all single-label "accuracies" as upper bounds until the
golden set is fully human-labeled.