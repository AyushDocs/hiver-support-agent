# Hiver Support Agent — Report

*AmazonHelp intent classification, history-grounded reply drafting, and
auto-route / human-escalation for customer-support tweets.*

---

## 1. Framing

**Goal.** Given an inbound customer tweet addressed to `@AmazonHelp`, the agent
must (1) classify the **intent**, (2) draft a reply **grounded in how AmazonHelp
has actually resolved similar issues historically**, and (3) decide whether the
request can be **auto-handled** or must **escalate to a human**, stating the
reason.

**Why history-grounded?** Support replies are highly templated and the brand has
already answered hundreds of thousands of similar tweets. Rather than generating
free-form text (which risks hallucinated policies), the agent retrieves the
nearest historical customer issues in an embedding space and reuses the brand's
own observed reply as the draft. Every draft is therefore traceable to a real,
previously-sent response.

**Success criteria.** High intent accuracy; drafts that are relevant and safe;
an auto-route policy that maximizes automation *without* silently mis-handling
customers; and, crucially, an evaluation that is honest about how much of the
measured quality is real versus an artifact of the labels.

---

## 2. Data

| Stage | Rows | Notes |
|---|---:|---|
| Raw `twcs.csv` | 3,002,523 | full customer-support corpus |
| Inbound `@AmazonHelp` mentions | 31,972 | `inbound==1` + text contains "amazonhelp" |
| Classified (model-ready) | 31,862 | after denoising / ≥4-word / author filters |
| Unique authors | 18,777 | author history drives routing |

The corpus already contains the reply linkage (`response_tweet_id`,
`in_response_to_tweet_id`), which is what makes history-grounded drafting
possible. `text` is cleaned (URLs/handles/emojis normalised, lemmatised) in
notebook 01.

---

## 3. Approach (pipeline)

1. **01 — Preprocessing.** Filter to English `@AmazonHelp` inbound mentions,
   remove noise, keep ≥4-word tweets, normalise text. Output `data_preprocessed.csv`
   (7 columns incl. reply linkage).
2. **02 — Intent discovery.** TF-IDF + NMF topic model, `k` swept 5–15 by
   coherence; 5 topics mapped to production intents:
   `delivery_issue, customer_service, order_status, email_contact, appreciation`.
   Topic-confidence below the 25th percentile flagged `low_confidence`.
3. **03 — Intent classifier.** Two engines, same split and same Linear
   Regression; the tensor features are built independently per engine:
   - Word2Vec (size 100) mean-pooled, L2-normalised (legacy, matches notebooks).
   - **TF-IDF bag-of-words (shipped default)** — same LR, test **acc 0.924**
     (vs 0.767 for Word2Vec) on an identical `train/val/test` split.
4. **04 — Auto-route.** Deterministic rule over classifier confidence, intent
   precedent, and author reply history (constants below).
5. **`agent.py`** — end-to-end entry point: clean → classify → retrieve
   history-grounded draft → route with reason.
   `--engine w2v|tfidf` (default **tfidf**). `build_golden.py` builds the eval
   set; `eval.py` scores it, including a real LLM judge (see §4).

### Routing rule (mirrors notebook 04)

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

## 4. Evaluation setup

### 4.1 Golden sets

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

### 4.2 Harness (`eval.py`)

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

## 5. Results

### 5.1 Agent vs auto-labeled golden (N=200)

| Metric | w2v engine | **TF-IDF engine** | Baseline |
|---|---:|---:|---:|
| Intent accuracy | 0.775 | **0.930** | 0.400 (always `customer_service`) |
| Intent macro-F1 | 0.763 | **0.929** | — |
| Routing accuracy | 0.915 | **0.975** | 0.425 (always `assist`) |
| Draft present | 100% | 100% | — |
| Mean retrieval similarity | 0.9949 | 0.9949 | — |

TF-IDF engine on the golden-200 reasons: `consistent_resolution_history` 54 ·
`insufficient_history` 52 · `routine_intent_history` 38 · `needs_judgment` 37 ·
`acknowledge_and_close` 18 · `low_prediction_confidence` 1.

### 5.2 Agent vs human labels (N=50) — the honest picture

| Comparison | Intent acc | Route acc | Note |
|---|---:|---:|---|
| **Silver gold vs human** | **0.52** | **0.44** | the auto-labels barely agree with a human |
| w2v agent vs human | 0.583 | 0.540 | |
| **TF-IDF agent vs human** | **0.625** | **0.520** | human assist rate is 0.88 (baseline) |
| **LLM judge vs human** | **0.46–0.52** | **0.84** | route κ = **−0.06 to −0.09** (≈ chance) |
| LLM judge vs auto-gold | 0.30–0.37 | 0.433 | n=30 subsample |

Judge numbers are quoted as run-to-run ranges: the LLM's own decisions are not
bit-reproducible across evals (±0.02 intent). Implemented `--judge-n 30` samples
are journaled in `eval_report_w2v.json` / `eval_report_tfidf.json`.

### 5.3 Judge verdicts

`gpt-4o-mini`, n=30 gold subsample + n=50 hand subsample, run with both engines
(judge agreement was engine-independent by construction):

* **Drafts:** **~1/30 rated `yes`** (28–30/30 `no`, 0–1 `partial`) — even after
  `@handle` stripping, the judge never considered a raw historical draft
  send-ready. Grounded ≠ send-ready (§6.2).
* **Route:** the judge routed **30/30 → `assist`**. Its agreement with the human
  label set (0.84) is mostly an artifact of both being assist-heavy; κ ≈ −0.06
  shows no real agreement on *which* tweets are auto-safe.

---

## 6. Top failure modes (in order of impact)

1. **The labels do not agree with each other.** Silver-vs-human intent
   agreement is 0.52, judge-vs-human 0.46–0.52, silver-vs-judge 0.30 — three
   "ground truths" that pairwise agree at ≈ chance. Every accuracy number in §5.1
   is therefore an *upper bound* on real performance, and the difference between
   w2v (0.78) and TF-IDF (0.93) on auto-labels is likely inflated by
   auto-label echo (§7).
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

## 7. What is misleading about the headline number

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
  §4 reports judge↔human agreement explicitly instead of just judge↔agent
  agreement.
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

## 8. Next-week plan (reprioritised after the judge run)

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

## 9. Limitations

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
* The mirror pipeline (`local_run.py` → `local_classify.py`) is **not
  bit-reproducible**: gensim trains Word2Vec with `workers=4`, so a full
  re-run drifts the w2v auto-rate by <1 pp and w2v golden intent accuracy by
  ~1.5 pts. The shipped TF-IDF engine is fully deterministic; treat w2v rows as
  "≈" and re-run `eval.py` (a 5-minute job) after any retrain.