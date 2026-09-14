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

---

## Quick start

```bash
# as a pip package (installs the `hiver` CLI)
pip install -e .
hiver "my order never arrived" --author-id 172791877

# or plain venv + requirements
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python agent.py "my order never arrived" --author-id 172791877
```

Optional extras (`pip install -e '.[judge]'`, `'.[pipeline]'`, `'.[all]'`) add
the LLM judge clients, the gensim/matplotlib pipeline notebooks, or both.

Models and eval data are mirrored on the Hugging Face Hub
([models](https://huggingface.co/24f2004275/hiver-support-agent),
[dataset](https://huggingface.co/datasets/24f2004275/hiver-support-golden));
re-upload with `HF_TOKEN=... python data/hf_upload.py`.

---

## 15-minute reproduction

All pipeline outputs are already in `data/processed`. The commands below
re-run every stage to verify the chain end-to-end.

> The raw corpus (`data/raw/twcs.csv`, 516 MB) is a Kaggle download step and
> **not shipped**. Notebook 01 is the slow step (~hours). Everything else
> re-runs in under 5 minutes.

```bash
# Step 1 — intent discovery (02)          ~1 min
#   Requires pre-executed notebook 02 or its output in data/processed.

# Step 2 — classifier (03)                ~2–3 min
#   Requires notebook 03 output (embeddings.npz + models).

# Step 3 — auto-route (04)                ~2 min
#   Requires notebook 04 output (amazon_help_routed.csv).

# Step 4 — golden sets + evaluation       ~2 min (+ judge ~2 min)
python build_golden.py                       # -> golden_eval.csv
python build_golden_hand.py                  # -> golden_hand.csv
python eval.py --judge-n 30                 # -> eval_report.json, eval_predictions.csv

# Step 5 — try the live agent              seconds
python agent.py "my package said out for delivery today but it never arrived" --author-id 172791877
```

The notebooks in `notebooks/` contain the same code (pre-executed) for reference.

---

## Headline results

TF-IDF engine is the default (`--engine w2v` reproduces the notebooks).

| Metric | TF-IDF | w2v | Baseline |
|---|---:|---:|---:|
| Intent accuracy (auto-gold N=200) | **0.930** | 0.775* | 0.400 |
| Routing accuracy vs auto-gold | **0.975** | 0.915 | 0.425 |
| Agent vs human labels (N=50, intent) | **0.625** | 0.583 | 0.520 silver-vs-human |
| Corpus auto-route rate | **48.34%** | ≈48.3%* | — |

\* w2v is trained with gensim `workers=4` (non-deterministic) and drifts ~1.5 pts
per retrain; TF-IDF is fully deterministic. See **Limitations** below.

---

## Data

| Stage | Rows | Notes |
|---|---:|---|
| Raw `twcs.csv` | 3,002,523 | full customer-support corpus |
| Inbound `@AmazonHelp` mentions | 31,972 | `inbound==1` + text contains "amazonhelp" |
| Classified (model-ready) | 31,862 | after denoising / ≥4-word / author filters |
| Unique authors | 18,777 | author history drives routing |

The corpus reply linkage (`response_tweet_id`, `in_response_to_tweet_id`) is
what makes history-grounded drafting possible. `text` is cleaned (URLs/handles
normalised, lemmatised) in notebook 01.

---

## Approach (pipeline)

1. **01 — Preprocessing.** Filter to English `@AmazonHelp` inbound mentions,
   remove noise, keep ≥4-word tweets, normalise text. Output
   `data_preprocessed.csv` (7 columns incl. reply linkage).
2. **02 — Intent discovery.** TF-IDF + NMF topic model, `k` swept 5–15 by
   coherence; 5 topics mapped to production intents:
   `delivery_issue, customer_service, order_status, email_contact, appreciation`.
   Topic-confidence below the 25th percentile flagged `low_confidence`.
3. **03 — Intent classifier.** Two engines, same split and same Logistic
   Regression; the tensor features are built independently per engine:
   - Word2Vec (size 100) mean-pooled, L2-normalised (legacy, matches notebooks).
   - **TF-IDF bag-of-words (shipped default)** — same LR, test **acc 0.924**
     (vs 0.767 for Word2Vec) on an identical `train/val/test` split.
4. **04 — Auto-route.** Deterministic rule over classifier confidence, intent
   precedent, and author reply history (constants below).
5. **`agent.py`** — end-to-end entry point: clean → classify → retrieve
   history-grounded draft → route with reason.
   `--engine w2v|tfidf` (default **tfidf**). `build_golden.py` builds the eval
   set; `eval.py` scores it, including a real LLM judge (see Evaluation).

### Routing rule

| Condition | Route | Reason |
|---|---|---|
| intent `appreciation` | auto | `acknowledge_and_close` |
| `pred_confidence < 0.30` | assist | `low_prediction_confidence` |
| `low_confidence` label | assist | `unclear_intent` |
| intent prevalence `< 2%` | assist | `novel_intent_no_precedent` |
| author history not sufficient* | assist | `insufficient_history` |
| intent ∈ {`order_status`, `email_contact`} | auto | `routine_intent_history` |
| `history_replied ≥ 2` and rate `≥ 0.7` | auto | `consistent_resolution_history` |
| otherwise | assist | `needs_judgment` |

\* sufficient = `history_volume ≥ 3` and `history_reply_rate ≥ 0.3`.

**Corpus routing result:** **auto 15,403 (48.34%) / assist 16,459 (51.66%)**.
Reason breakdown: `insufficient_history` 6,455 · `unclear_intent` 5,695 ·
`consistent_resolution_history` 5,571 · `routine_intent_history` 5,331 ·
`acknowledge_and_close` 4,501 · `needs_judgment` 4,156 · `low_prediction_confidence` 153.

---

## Evaluation setup

### Golden sets

* **Auto-labeled golden — N = 200** (`golden_eval.csv`). Stratified by silver
  intent (seed=42), author history retained because routing consumes it.
  `gold_intent` comes from the topic model, `gold_route` from the 04 rule
  applied to the *gold* intent. **Provisional** by construction; columns are
  flip-ready for an expert.
* **Human-labeled subset — N = 50** (`golden_hand.csv`). A reviewer labelled
  `hand_intent` / `hand_route` by reading each tweet **plus its author's
  conversation history**. This is the only label set independent of the pipeline.
  Intent split: `order_status` 14 · `delivery_issue` 13 · `customer_service` 13 ·
  `email_contact` 7 · `appreciation` 1 · `other` 2. Route: 44 assist / 6 auto.

### Harness (`eval.py`)

Scores the live agent (either engine) over both golden sets and reports:

1. **Intent** — accuracy + macro-F1 vs `gold_intent` / `hand_intent`.
2. **Routing** — accuracy vs `gold_route` / `hand_route`.
3. **Baselines** — majority intent; all-assist routing; measured TF-IDF+LR test
   accuracy on the held-out split.
4. **Draft** — retrieval evidence depth + fraction with a non-empty draft.
5. **LLM-as-judge** — `gpt-4o-mini` rates intent, route and draft sanity on a
   30-tweet subsample **and on the full 50-tweet hand subset**, reporting
   judge↔label agreement **and judge↔human (Cohen's κ)**. The prompt has **no**
   planted silver/gold labels, so the judge is un-anchored. Fallbacks
   (Anthropic, deterministic keyword) exist if the OpenAI endpoint is unreachable
   and `llm_judge.mode` records which one ran.

---

## Results

### Agent vs auto-labeled golden (N=200)

| Metric | w2v engine | **TF-IDF engine** | Baseline |
|---|---:|---:|---:|
| Intent accuracy | 0.775 | **0.930** | 0.400 (always `customer_service`) |
| Intent macro-F1 | 0.763 | **0.929** | — |
| Routing accuracy | 0.915 | **0.975** | 0.425 (always `assist`) |
| Draft present | 100% | 100% | — |
| Mean retrieval similarity | 0.9949 | 0.9949 | — |

### Agent vs human labels (N=50) — the honest picture

| Comparison | Intent acc | Route acc | Note |
|---|---:|---:|---|
| **Silver gold vs human** | **0.52** | **0.44** | the auto-labels barely agree with a human |
| w2v agent vs human | 0.583 | 0.540 | |
| **TF-IDF agent vs human** | **0.625** | **0.520** | human assist rate is 0.88 (baseline) |
| **LLM judge vs human** | **0.46–0.52** | **0.84** | route κ = **−0.06 to −0.09** (≈ chance) |
| LLM judge vs auto-gold | 0.30–0.37 | 0.433 | n=30 subsample |

### Judge verdicts

`gpt-4o-mini`, n=30 gold subsample + n=50 hand subsample, run with both engines
(judge agreement was engine-independent by construction):

* **Drafts:** **~1/30 rated `yes`** (28–30/30 `no`, 0–1 `partial`) — even after
  `@handle` stripping, the judge never considered a raw historical draft
  send-ready. Grounded ≠ send-ready.
* **Route:** the judge routed **30/30 → `assist`**. Its agreement with the human
  label set (0.84) is mostly an artifact of both being assist-heavy; κ ≈ −0.06
  shows no real agreement on *which* tweets are auto-safe.

---

## Top failure modes (in order of impact)

1. **The labels do not agree with each other.** Silver-vs-human intent
   agreement is 0.52, judge-vs-human 0.46–0.52, silver-vs-judge 0.30 — three
   "ground truths" that pairwise agree at ≈ chance. Every accuracy number is
   therefore an *upper bound* on real performance.
2. **Drafts are grounded but not send-ready.** Replies are the brand's actual
   historical tweets: they open with another customer's handle and carry
   templated signature tokens (`^AC`). The agent now strips every `@handle`
   (`sanitize_reply`) and blocks auto-sending any draft without prior history
   (safety gate), but the strict LLM judge still flagged ~100% of samples —
   the remaining gap is relevance/format, not just de-identification.
3. **Retrieval similarity is inflated/compressed.** Mean top-k cosine similarity
   is 0.9949 for essentially *all* tweets (low angular spread in the mean-pooled
   space), so similarity does not certify relevance.
4. **Routing is driven by `unclear_intent` and `insufficient_history`**
   (12,150 of 16,459 assists, 74%) — most escalations are "ambiguous topic" or
   "new author", not "hard problem". The auto-rate mostly tracks label clarity
   and author tenure.
5. **Short/OOV tweets fall through** — pure-abbreviation / emoji / typo-heavy
   tweets have no Word2Vec embedding → `unclear_intent`. (TF-IDF handles these
   better via char-level n-grams, one reason it wins by 15 pts.)

---

## What is misleading about the headline number

The cleanest headline is **"97.5% routing accuracy / 48.34% auto-resolution."**
Both halves are misleading:

* **The 97.5% is circular.** `gold_route` is generated by the *same* rule (04)
  applied to the *gold* intent, so the metric measures the rule's
  self-consistency, not whether the route was right for the customer. The 50-row
  human subset shows real routing agreement ≈ 0.52 — a 40-pt gap.
* **The 93.0% intent accuracy is agreement with auto-labels.** Training and
  evaluation both stand on the NMF topic model's silver labels. A perfect score
  means "reproduces the topic model", not "understands the customer." Silver-vs-
  human agreement (0.52) is the true ceiling for this number.
* **Even the celebrated judge is not trustworthy.** judge↔human intent agreement
  is ≈ 0.5 ≈ chance, and its routing κ is negative. Using the judge to *validate*
  the agent would be as circular as using the auto-gold — this is precisely why
  the evaluation reports judge↔human agreement explicitly instead of just
  judge↔agent agreement.
* **0.9949 similarity and 100% draft-present flatter the drafter.** The strict
  judge rated ~1/30 drafts send-ready, so "draft present" says nothing about
  draft quality.
* **TF-IDF winning by 15 pts flips the story.** Our headline classifier result
  (w2v 0.78) is *below* what a standard bag-of-words + LR achieves (0.924 feature
  baseline). The honest technical headline is: "the retrieval-grounded
  architecture works; the intended-feature choice did not — shipped TF-IDF."
* **The 48.34% auto-rate is a policy knob, not a quality result.**
  `CONF_LO` alone moves it 48.5%→43.1% (0.50); the defensible statement is not
  "we automate 48%" but "at the chosen risk posture we automate 48% with an
  auto-route precision that is currently unmeasured" — the very measurement the
  human-labeled eval exists to produce.

---

## Next-week plan

1. **Human-label a routing-quality sample (n≈100–200)** and report true
   **auto-route precision**: share of auto-handled tweets genuinely safe to
   auto-answer. All other numbers are secondary until this exists.
2. **Benchmark the judge as a labeler before trusting it as a QA gate.** At
   κ ≈ 0 vs humans it can currently only catch *format* defects, not routing
   correctness; if a prompt/version is found with judge↔human κ ≥ 0.6 it becomes
   a cheap every-run gate.
3. **Make drafts send-ready for real.** Beyond handle-stripping, re-address to
   the current customer, drop signature tokens, and substitute slot values; keep
   the safety gate that blocks any draft containing another user's handle/PII.
4. **Replace fixed thresholds with a cost model.** Route by expected cost
   (wrong-auto ≫ assist) and publish an auto-resolve/trade-off curve, not a point.
5. **Fix retrieval geometry.** Compare mean-pooled Word2Vec against
   sentence-embedding / TF-IDF indexes and score retrieval with recall@k on
   human-judged "was this historical reply appropriate?" instead of cosine sim.

---

## Decision log

Every meaningful choice, with the alternative considered and the reasoning.
Decisions appear roughly in the order they were made.

**D1 — Draft by retrieval from brand history, not by generative LLM.**
Chose retrieval over LLM generation. No API assumed available; retrieval
replies are traceable to real brand responses; generation risks hallucinating
shipping/refund policy.

**D2 — Build the full pipeline now, not a verticals-then-integrate plan.**
Chose one-pass 01→04+agent+eval. The deliverables need one coherent runnable
chain; stages share artifacts and a later integration would redo work.

**D3 — NMF over TF-IDF for silver intent discovery.**
Chose NMF (`k` swept 5–15 by coherence, `k=5`). No pre-existing labels; NMF is
fast, deterministic, and its topics mapped cleanly onto 5 intents. LDA weaker
coherence; BERTopic needs a model + embedding budget.

**D4 — Five production intents from five topics.**
`delivery_issue, customer_service, order_status, email_contact, appreciation`.
Coherence sweep supported exactly 5 stable topics; finer classes would have been
label-sketchy and under-supported.

**D5 — Word2Vec mean-pooling + Logistic Regression.**
Chose `word2vec (d=100)` → LR (test acc 0.767, macro-F1 0.747).
Matches the retrieval embedding space so one vector space serves both
classification and nearest-neighbour drafting; LR is cheap and interpretable.
*Caveat:* the same LR on plain TF-IDF features scores **0.924** — 15+ pts
above word2vec. Swapping features to TF-IDF is the top model-level fix.

**D6 — Train classifier on high-confidence silver labels only.**
Dropped bottom-quartile topic-confidence labels before training; classified all
rows at inference. Bottom-quartile labels are topic-mixing noise; training on
clean labels measurably raised validation accuracy/F1.

**D7 — Routing is a deterministic, transparent rule.**
Ordered rule over confidence → low-confidence flag → intent precedent → author
history → routine intents → consistent-resolver. No ground-truth route labels
exist; a rule is auditable, tunable, and the "stated reason" deliverable comes
free. Corpus auto-rate 48.34%.

**D8 — Routing constants tuned to a cautious risk posture.**
`CONF_LO=0.30, PREV_MIN=0.02, HIST_VOL_MIN=3, HIST_REPLY_MIN=0.3,
AUTO_INTENTS={order_status,email_contact}, AUTO_REPLY_MIN={replied≥2, rate≥0.7}`.
A wrong auto-reply is costly; sensitivity sweep (0.15→0.50 CONF_LO moves
auto-rate 48.5%→43.1%) is documented so the owner can re-tune.

**D9 — Golden set: auto-label now, human spot-check later (N=200, stratified).**
Auto-labeled `gold_intent` from topic model and `gold_route` from rule+gold
intent+author history; shipped with flip-able columns. No labeler scheduled;
the harness must run now. The report loudly flags the circularity.

**D10 — Eval includes trivial baselines on every metric.**
Majority-class intent and all-assist routing baselines; similarity and
draft-presence signals reported alongside. Deltas vs "always customer_service"
(0.40) and "always assist" (0.425) frame the real gain honestly.

**D11 — LLM-as-judge harness with a sane failure mode.**
Judge prompt asks for intent/route/draft verdicts on a 30-tweet subsample;
falls back to a deterministic keyword judge and records `mode:"heuristic"` if
the API is unreachable. Honest labeling of the fallback keeps numbers from being
misread as LLM-judge results.

**D12 — Reuse of replies for drafting: keep raw reply but de-PII later.**
Draft = first brand reply in the nearest retrieved thread, verbatim, catalogued
as `amazon_replies.csv`. Grounding and traceability first; handle-stripping is a
failure-mode fix.

**D13 — Ship derived data, not the 516 MB raw corpus.**
The repo carries `data_preprocessed.csv` (+ models, embeddings,
labeled/classified/routed, eval artifacts); raw `twcs.csv` is a download step.
Reproduction must stay under 15 min; the derived table is 60× smaller.

**D14 — Headline number is reported with its caveats.**
Report **48.34% auto / 97.5% routing-acc / 93.0% intent-acc** as the headline
trio but dedicate a section to what is misleading about each. The golden labels
are auto-generated, so the headline accuracy numbers are circular by
construction.

**D15 — Human-labeled subset (N=50) as the one pipeline-independent ground truth.**
A reviewer labelled `hand_intent`/`hand_route` from tweet + author history;
`eval.py` scores both engines against it. The first run proved the point:
silver-vs-human intent agreement is only **0.52** (route 0.44) — the auto-gold
barely agrees with a human.

**D16 — Real LLM judge (OpenAI) + judge↔human agreement, un-anchored.**
Judge on `gpt-4o-mini`, fallback chain OpenAI→Anthropic→keyword; the prompt
carries **no** gold or agent labels so the judge is un-anchored; evaluate the
judge *against the human labels* (accuracy + Cohen's κ). Result: judge-vs-human
intent ≈ 0.46–0.52, routing κ ≈ −0.06 to −0.09 — the judge is not (yet) a
valid labeler.

**D17 — TF-IDF becomes the default intent engine.**
`agent.py`/`eval.py` default `--engine tfidf`. TF-IDF scores **0.930 vs 0.775**
(intent, auto-gold) and **0.625 vs 0.583** (vs human labels N=50) — strictly
better and far better on typo/OOV tweets (char-level n-grams).

**D18 — Drafts: sanitised handles + auto-route safety gate.**
`sanitize_reply()` strips `@handles`; routing downgrades `auto` to `assist` /
`draft_not_send_ready` when no send-ready draft exists. The strict LLM judge
rated **29/30 drafts `not ok`** even after sanitisation — the failure is now
honestly measured.

**D19 — Accept (and document) nondeterminism.**
The gensim Word2Vec train uses `workers=4`, so a full re-run drifts w2v results
by ~1.5 pts / <1 pp auto-rate. TF-IDF is fully deterministic. Reproducibility
matters less than the numbers being honest **right now**.

---

## Limitations

* Auto-gold labels are provisional; all accuracy figures on them are upper
  bounds. The only independent ground truth is the 50-row human subset.
* The LLM judge is a strict, un-calibrated single model (`gpt-4o-mini`); its
  routing shows a strong assist prior and near-chance agreement with humans.
* Preprocessing is the slow step (~hours over 3M rows); the submission ships
  the derived `data_preprocessed.csv`, so the <15-min reproduction starts at 02.
* Routing is a hand-tuned deterministic rule, not a learned policy; thresholds
  are not calibrated to a cost target.
* Executed notebooks 02–04 show near-identical but not byte-identical numbers to
  the local re-run (sklearn-version drift, ~0.3% of labels) — noted for
  transparency, behaviourally irrelevant.
* Word2Vec training is non-deterministic (`workers=4`): a full re-run drifts
  the w2v auto-rate by <1 pp and golden intent accuracy by ~1.5 pts. The
  shipped TF-IDF engine is fully deterministic.

---

## Repository layout

```
agent.py                 live agent (classify + history-grounded draft + route)
build_golden.py          builds golden_eval.csv (N=200, auto-labeled)
build_golden_hand.py     builds golden_hand.csv (N=50, human-labeled)
hand_labels.py           the 50 human labels (reviewer evidence for audit)
eval.py                  evaluation harness (metrics, baselines, LLM judge, kappa)
train_tfidf.py           trains the default TF-IDF intent engine (0.924 test acc)
pyproject.toml           pip-installable package (`hiver` console script)
requirements.txt         dependencies
data/hf_upload.py        publishes golden sets + models to the Hugging Face Hub
notebooks/01..04_*.ipynb executed Kaggle notebooks (canonical pipeline)
data/
  raw/twcs.csv           raw corpus (~516 MB, download step)      [not shipped]
  processed/             01–05 outputs: preprocessed/labeled/classified/routed,
                         golden_eval/_hand, eval_report[_w2v|_tfidf], routing_report etc.
  models/                word2vec + tfidf models, logistic-regression engines
```
