# Decision Log

Every meaningful choice made while building the AmazonHelp support agent, with
the alternative considered and the reasoning. Decisions appear roughly in the
order they were made.

---

### D1 — Draft by retrieval from brand history, not by generative LLM
- **Chosen:** Retrieve the nearest historical customer tweet in embedding space
  and reuse AmazonHelp's *actual* reply as the draft.
- **Rejected:** Free-form summary generation via an LLM.
- **Why:** No LLM API was assumed available; retrieval replies are traceable to a
  real, previously-sent brand response (defensible in a support setting);
  generation risks hallucinating shipping/refund policy.

### D2 — Build the full pipeline now, not a verticals-then-integrate plan
- **Chosen:** Implement 01→04 + agent + eval in one pass and iterate.
- **Rejected:** A staged delivery (classifier first, drafting later).
- **Why:** The assignment's deliverables need one coherent runnable chain; stages
  share artifacts and a later integration would redo work.

### D3 — Use NMF over TF-IDF for silver intent discovery
- **Chosen:** NMF topic model, `k` swept 5–15 on coherence, `k=5` selected.
- **Rejected:** LDA / BERTopic / hand-labeled intents.
- **Why:** No pre-existing labels; NMF is fast, deterministic, reproducible on
  32k tweets, and its topics mapped cleanly onto 5 understandable intents. LDA
  was ruled out for weaker coherence; BERTopic needs a model + embedding budget.

### D4 — Five production intents from the five topics
- **Chosen:** `delivery_issue, customer_service, order_status, email_contact,
  appreciation`.
- **Rejected:** Finer-grained classes (e.g. `refund`, `cancellation` split out).
- **Why:** Coherence sweep and topic word-top-10 supported exactly 5 stable
  topics; finer classes would have been label-sketchy and under-supported.

### D5 — Word2Vec mean-pooling + Logistic Regression as the classifier
- **Chosen:** `word2vec (d=100)` mean-pooled, L2-normalized, → `LogisticRegression`
  (test acc 0.767, macro-F1 0.747).
- **Rejected:** TF-IDF+SVM, neural classifiers.
- **Why:** Matches the retrieval embedding space so one vector space serves both
  classification and nearest-neighbour drafting; LR is cheap and interpretable.
- **Measured caveat:** Eval of the same LR on the same split with plain TF-IDF
  features scores **0.924** — 15+ pts above word2vec. The classifier is a
  ship-ready-now placeholder; swapping its features to TF-IDF (or a concat) is
  the top model-level fix in next-week (report §8.2). Retrieval stays on word2vec.

### D6 — Train the classifier on high-confidence silver labels only
- **Chosen:** Drop tweets whose topic confidence is in the bottom quartile before
  training; classify *all* rows at inference.
- **Rejected:** Train on all silver labels.
- **Why:** Bottom-quartile labels are topic-mixing noise; training on clean labels
  measurably raised validation accuracy/F1 (see notebook 03).

### D7 — Routing is a deterministic, transparent rule — not a learned policy
- **Chosen:** Ordered rule over confidence → low-confidence flag → intent
  precedent → author history → routine intents → consistent-resolver.
- **Rejected:** A trained classifier / RL over route outcomes.
- **Why:** No ground-truth route labels exist; a rule is auditable, tunable, and
  the "stated reason" deliverable comes free. Corpus auto-rate 48.34%.

### D8 — Routing constants tuned to a cautious risk posture
- **Chosen:** `CONF_LO=0.30, PREV_MIN=0.02, HIST_VOL_MIN=3, HIST_REPLY_MIN=0.3,
  AUTO_INTENTS={order_status,email_contact}, AUTO_REPLY_MIN={replied≥2, rate≥0.7}`.
- **Rejected:** Looser thresholds (higher auto-rate).
- **Why:** A wrong auto-reply to a support tweet is costly; sensitivity sweep
  (0.15→0.50 CONF_LO moves auto-rate 48.5%→43.1%) is documented so the owner can
  re-tune against a real cost model.

### D9 — Golden set: auto-label now, human spot-check later (N=200, stratified)
- **Chosen:** Auto-label `gold_intent` from the topic model and `gold_route` from
  rule+gold intent+author history; ship as `golden_eval.csv` with flip-able
  columns.
- **Rejected:** Waiting for a full human labeling pass before delivering eval.
- **Why:** No labeler scheduled; the harness must run now for the evaluation
  deliverable. The report (§7) loudly flags this circularity. Spot-check columns
  make the human pass a 30-minute task.

### D10 — Eval includes trivial baselines on every metric
- **Chosen:** Majority-class intent and all-assist routing baselines; similarity
  and draft-presence signals reported alongside.
- **Rejected:** Reporting raw accuracy alone (or worse, only aggregated F1).
- **Why:** The assignment requires ≥2 baselines incl. a trivial one; deltas vs
  "always customer_service" (0.40) and "always assist" (0.415) frame the real
  gain honestly.

### D11 — LLM-as-judge harness with a sane failure mode
- **Chosen:** Judge prompt asks for intent/route/draft verdicts on a 30-tweet
  subsample, computing judge↔human agreement; **fall back to a deterministic
  keyword judge and record `mode:"heuristic"`** if the API is unreachable.
- **Rejected:** Hard-failing the eval when the network is down.
- **Why:** The judge framework is a required deliverable and must run in this
  environment (endpoint was unreachable). Honest labeling of the fallback keeps
  the numbers from being misread as LLM-judge results.

### D12 — Reuse of replies for drafting: keep raw reply but de-PII later
- **Chosen:** Draft = first brand reply in the nearest retrieved thread, verbatim,
  catalogued as `amazon_replies.csv`.
- **Rejected:** Post-processing/redaction in this iteration.
- **Why:** Grounding and traceability first; handle-stripping is a failure-mode
  fix scheduled next week (report §8.3) once the evals can verify it doesn't
  degrade drafts.

### D13 — Reconstruct `data_preprocessed.csv` from downstream artifacts
- **Chosen:** Re-derive the 7-column 01 output by merging cleaned text from
  `amazon_help_labeled.csv` with the linkage columns re-extracted from raw
  `twcs.csv` (21s).
- **Rejected:** Re-running notebook-01 preprocessing locally (~hours over 3M
  rows, background jobs kept dying on shell timeouts).
- **Why:** The linkage columns are exactly what was lost; the merge is lossless
  for this pipeline and makes the 15-minute reproduction feasible.

### D14 — Ship derived data, not the 516 MB raw corpus
- **Chosen:** The repo carries `data_preprocessed.csv` (+ models, embeddings,
  labeled/classified/routed, eval artifacts); raw `twcs.csv` is a download step.
- **Rejected:** Committing the ~516 MB raw file.
- **Why:** Reproduction must stay under 15 min; the derived table is 60× smaller
  and every downstream stage (02–05) runs on it.

### D15 — Headline number is reported *with* its caveats
- **Chosen:** Report **48.34% auto / 97.5% routing-acc / 93.0% intent-acc** as
  the headline trio but dedicate report §7 to what is misleading about each.
- **Rejected:** Advertising a single clean percentage.
- **Why:** The golden labels are auto-generated, so the headline accuracy numbers
  are circular by construction; the graders explicitly asked for
  "what's misleading about my headline number".

### D16 — Human-labeled subset (N=50) as the one pipeline-independent ground truth
- **Chosen:** A reviewer labelled `hand_intent`/`hand_route` from tweet + author
  history (`golden_hand.csv`, labels in `hand_labels.py` for audit); `eval.py`
  now scores both engines against it.
- **Rejected:** Trusting the auto-gold as the sole label set.
- **Why:** The first run proved the point: silver-vs-human intent agreement is
  only **0.52** (route 0.44) — the auto-gold barely agrees with a human, so every
  accuracy on it is an upper bound on real performance.

### D17 — Real LLM judge (OpenAI) + judge↔human agreement, un-anchored
- **Chosen:** Judge on `gpt-4o-mini` (OpenAI reachable; `.env`+
  `load_env()`), fallback chain OpenAI→Anthropic→keyword; the prompt carries **no**
  gold or agent labels so the judge is un-anchored; evaluate the judge *against the
  human labels* (accuracy + Cohen's κ), not only against the pipeline's gold.
- **Rejected:** Measuring only judge↔auto-gold agreement (a circular validity
  check); showing the agent's predicted intent in the prompt (anchoring artifact —
  observed to shift judge labels).
- **Why:** The assignment requires judge↔human agreement. Result: judge-vs-human
  intent ≈ 0.46–0.52, routing κ ≈ **−0.06 to −0.09** — the judge is not (yet) a
  valid labeler; better to report that than to let the judge rubber-stamp the
  auto-gold. (Judge outputs vary ±0.02 run-to-run, so the log quotes ranges.)

### D18 — TF-IDF becomes the default intent engine
- **Chosen:** `agent.py`/`eval.py` default `--engine tfidf`; ship
  `tfidf_{vectorizer,classifier,label_encoder}.joblib`; keep `+= w2v` working and
  documented as the notebook-legacy engine.
- **Rejected:** Keeping word2vec the default.
- **Why:** TF-IDF scores **0.930 vs 0.775** (intent, auto-gold) and **0.625 vs
  0.583** (vs human labels N=50) — strictly better and far better on typo/OOV
  tweets (char-level n-grams). The assignment asked for an honest re-check of the
  headline result; shipping the better classifier is the concrete action from
  report §8.

### D19 — Drafts: sanitised handles + auto-route safety gate, still pre-send-ready
- **Chosen:** `agent.sanitize_reply()` strips `@handles` before a draft can be
  sent, and routing now has a guard: `auto` is downgraded to `assist` /
  `draft_not_send_ready` when no send-ready draft exists for that row.
- **Rejected:** Sending raw historical replies (they carried other customers'
  handles and `^AC` signature tokens); over-engineering re-personalisation now.
- **Why:** The strict LLM judge rated **29/30 drafts `not ok`** even after
  sanitisation — the failure is now honestly measured (relevance/format, not just
  PII) and is the top remaining item in report §8.3.

### D20 — Accept (and document) mirror-pipeline nondeterminism
- **Chosen:** Ship the regenerated pipeline outputs as canonical
  (auto-rate 48.34%, w2v golden intent 0.775); document in REPORT §9 that
  `local_classify.py` retrains gensim Word2Vec with `workers=4`, so a full re-run
  drifts w2v results by ~1.5 pts / <1 pp auto-rate. The TF-IDF engine and its
  eval are deterministic; `eval.py --engine w2v` recovers current numbers in
  ~1 min.
- **Rejected:** Enforcing bit-reproducibility by retraining with `workers=1`
  (new results, another full re-run) or silently reusing stale report numbers.
- **Why:** Reproducibility matters less than the numbers being honest **right
  now**; the ~1.5-pt w2v drift does not change a single conclusion in REPORT.md."