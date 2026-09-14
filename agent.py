#!/usr/bin/env python3
"""Hiver support agent for AmazonHelp.

End-to-end agent that, given a raw customer tweet, produces:

  1. intent          — classifier trained in notebook 03 (word2vec + logistic reg)
  2. draft           — a reply drafted from how AmazonHelp HISTORICALLY resolved
                       similar issues (nearest-neighbor retrieval over the
                       customer corpus, grounded in the brand's actual replies)
  3. route           — auto-handle vs escalate to a human, with a stated reason
                       (rule set mirroring notebook 04)

Usage:
    .venv/bin/python agent.py "customer tweet" [--author-id N] [--k 3]
"""
import argparse
import json
import os
import re
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, 'data/models')
PROC = os.path.join(ROOT, 'data/processed')
RAW = os.path.join(ROOT, 'data/raw/twcs.csv')

# Rule constants (mirror notebook 04)
CONF_LO = 0.30
PREV_MIN = 0.02
HIST_VOL_MIN = 3
HIST_REPLY_MIN = 0.3
AUTO_INTENTS = {'order_status', 'email_contact'}
AUTO_REPLY_MIN = {'history_replied': 2, 'history_reply_rate': 0.7}

CAND_RAW = ['/kaggle/input/datasets/thoughtvector/customer-support-on-twitter/twcs/twcs.csv',
            '/kaggle/input/twcs/twcs.csv', '/kaggle/working/twcs.csv', RAW]
RAW_PATH = next((p for p in CAND_RAW if os.path.exists(p)), RAW)


def tokenize(text):
    """Mirror notebook 03's tokenizer exactly."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    stop = set(ENGLISH_STOP_WORDS)
    tokens = re.findall(r"[a-z]+", str(text).lower())
    return [t for t in tokens if len(t) > 1 and t not in stop]


def clean_raw(text):
    """Apply the notebook-01 pipeline to a NEW raw message before modelling."""
    text = re.sub(r'https?\S+|@\w+|#\w+', ' ', str(text))
    return re.sub(r'\s+', ' ', text).lower().strip()


HANDLE_RE = re.compile(r'@\w+')


def sanitize_reply(reply):
    """Strip other-customer handles from a historical brand reply so the draft
    is send-ready (no `@<other-user>` leak)."""
    if not reply:
        return ''
    text = HANDLE_RE.sub(' ', str(reply))
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class Agent:
    def __init__(self, verbose=False, autoresume=True, engine='tfidf'):
        import joblib
        from gensim.models import Word2Vec
        self.t0 = time.time()
        self.w2v = Word2Vec.load(os.path.join(MODELS, 'word2vec.model'))
        self.engine = engine
        if engine == 'tfidf':
            self.clf = joblib.load(os.path.join(MODELS, 'tfidf_classifier.joblib'))
            self.tfidf = joblib.load(os.path.join(MODELS, 'tfidf_vectorizer.joblib'))
            self.le = joblib.load(os.path.join(MODELS, 'label_encoder_tfidf.joblib'))
        else:
            self.engine = 'w2v'
            self.clf = joblib.load(os.path.join(MODELS, 'intent_classifier.joblib'))
            self.le = joblib.load(os.path.join(MODELS, 'label_encoder.joblib'))
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        self.stop = set(ENGLISH_STOP_WORDS)

        # retrieval index: embeddings + tweet metadata (including reply linkage)
        d = np.load(os.path.join(PROC, 'embeddings.npz'), allow_pickle=True)
        self.emb = d['embedding'].astype(np.float32)          # (n, 100) L2-normed
        self.emb_ids = np.asarray(d['tweet_id']).astype(str)
        self.emb_intent = np.asarray(d['intent']).astype(str)
        self.emb_conf = d['topic_confidence']
        idx = {}
        for i, tid in enumerate(self.emb_ids):
            idx.setdefault(tid, i)
        self.idx = idx
        cls_df = pd.read_csv(os.path.join(PROC, 'amazon_help_classified.csv'),
                             usecols=['pred_intent'], dtype=str)
        self.prev = cls_df['pred_intent'].value_counts(normalize=True).to_dict()

        meta = pd.read_csv(os.path.join(PROC, 'data_preprocessed.csv'),
                           dtype={'tweet_id': str, 'author_id': str,
                                  'response_tweet_id': str,
                                  'in_response_to_tweet_id': str})
        self.meta = meta.set_index('tweet_id', drop=False)

        # warm caches (single raw-twcs pass) if missing
        self.replies = {}
        self.author_hist = {}
        self._warm_cache(autoresume)
        if verbose:
            print(f'agent ready in {time.time() - self.t0:.1f}s '
                  f'(index={self.emb.shape[0]:,}, replies={len(self.replies):,})')

    # ---- historical grounding caches -------------------------------------
    def _warm_cache(self, autoresume):
        rp = os.path.join(PROC, 'amazon_replies.csv')
        hp = os.path.join(PROC, 'author_history.csv')
        if autoresume and os.path.exists(rp) and os.path.exists(hp):
            r = pd.read_csv(rp, dtype={'tweet_id': str})
            self.replies = dict(zip(r['tweet_id'], r['text']))
            h = pd.read_csv(hp, dtype={'author_id': str},
                            index_col='author_id')
            self.author_hist = h.to_dict('index')
            return

        print(f'Building history caches from {RAW_PATH} ... (one time)')
        t = time.time()
        raw = pd.read_csv(RAW_PATH, dtype={'tweet_id': str, 'author_id': str,
                                           'response_tweet_id': str,
                                           'in_response_to_tweet_id': str})
        amz_mask = raw['inbound'] == 1
        mention = raw.loc[amz_mask, 'text'].str.lower().str.contains('amazonhelp', na=False)
        amz_in = raw.loc[amz_mask & mention].copy()
        responded = amz_in[amz_in['response_tweet_id'].notna()]

        # outbound replies that correspond to AmazonHelp mentions
        resp_ids = set()
        for v in responded['response_tweet_id']:
            resp_ids.update(x for x in str(v).split(',') if x.strip())
        out = raw[raw['inbound'] == 0]
        replies = out[out['tweet_id'].isin(resp_ids)][['tweet_id', 'text', 'created_at']]
        replies.to_csv(rp, index=False)
        self.replies = dict(zip(replies['tweet_id'], replies['text']))

        # author history (mirror notebook 04 cell 6)
        lat = responded[['tweet_id', 'author_id', 'created_at', 'response_tweet_id']].merge(
            out[['tweet_id', 'created_at']], left_on='response_tweet_id', right_on='tweet_id',
            suffixes=('_in', '_out'))
        lat['wait_h'] = (pd.to_datetime(lat['created_at_out']) -
                         pd.to_datetime(lat['created_at_in'])).dt.total_seconds() / 3600
        lat = lat[lat['wait_h'].between(0, 72)]
        hist = amz_in.groupby('author_id').agg(
            history_volume=('tweet_id', 'count'),
            history_replied=('response_tweet_id', lambda s: s.notna().sum()),
        ).reset_index()
        hist['history_reply_rate'] = (hist['history_replied'] / hist['history_volume']).clip(upper=1.0)
        avg_wait = lat.groupby('author_id')['wait_h'].mean().rename('avg_reply_hours')
        hist = hist.merge(avg_wait, on='author_id', how='left')
        hist['history_sufficient'] = ((hist['history_volume'] >= HIST_VOL_MIN) &
                                      (hist['history_reply_rate'] >= HIST_REPLY_MIN))
        hist.set_index('author_id', inplace=True)
        hist.to_csv(hp)
        self.author_hist = hist.to_dict('index')
        print(f'  caches built in {time.time() - t:.0f}s '
              f'(replies={len(self.replies)}, authors={len(self.author_hist)})')

    # ---- step 1: classify -------------------------------------------------
    def embed(self, tokens):
        vecs = [self.w2v.wv[t] for t in tokens if t in self.w2v.wv]
        if not vecs:
            return None
        v = np.mean(vecs, axis=0)
        v = v / np.linalg.norm(v)
        return v.astype(np.float32)

    def classify(self, raw_text, preprocess=True):
        text = clean_raw(raw_text) if preprocess else raw_text
        tokens = tokenize(text)
        if self.engine == 'tfidf':
            if not str(text).strip():
                return {'intent': None, 'confidence': 0.0, 'tokens': tokens,
                        'text': text, 'oov': True}
            x = self.tfidf.transform([str(text)])
            probs = self.clf.predict_proba(x)[0]
        else:
            q = self.embed(tokens)
            if q is None:
                return {'intent': None, 'confidence': 0.0, 'tokens': tokens,
                        'text': text, 'oov': True}
            probs = self.clf.predict_proba(q.reshape(1, -1))[0]
        i = int(np.argmax(probs))
        return {'intent': self.le.classes_[i], 'confidence': float(probs[i]),
                'tokens': tokens, 'text': text, 'oov': False}

    # ---- step 2: draft from history --------------------------------------
    def retrieve(self, q, k=3):
        """Nearest neighbors in the indexed corpus (only threads we can ground)."""
        sims = self.emb @ q
        meta_ids = set(self.meta.index)
        scored = []
        order = np.argsort(-sims)
        for rank in order:
            tid = self.emb_ids[rank]
            if tid not in meta_ids:
                continue
            row = self.meta.loc[tid]
            resp = row.get('response_tweet_id', None)
            repl = self._first_reply(resp)
            if repl is None:
                continue
            scored.append((rank, float(sims[rank]), tid, self.emb_intent[rank], repl))
            if len(scored) >= k:
                break
        return scored

    def _first_reply(self, response_tweet_id):
        if response_tweet_id is None or pd.isna(response_tweet_id):
            return None
        for cand in str(response_tweet_id).split(','):
            cand = cand.strip()
            if cand in self.replies:
                return cand
        return None

    def draft(self, raw_text, k=3):
        cls = self.classify(raw_text)
        if cls['oov']:
            return {'draft': None, 'evidence': [], 'classify': cls}
        q = self.embed(cls['tokens'])
        ranked = self.retrieve(q, k=k)
        evidence = []
        for rank, sim, tid, intent, rep_id in ranked:
            row = self.meta.loc[tid].to_dict()
            raw_reply = self.replies[rep_id]
            evidence.append({
                'sim': round(sim, 4), 'customer_tweet_id': tid,
                'customer_text': row.get('text', ''), 'intent': intent,
                'brand_reply_id': rep_id, 'brand_reply': raw_reply,
                'brand_reply_sanitized': sanitize_reply(raw_reply),
            })
        draft = sanitize_reply(evidence[0]['brand_reply']) if evidence else None
        return {'draft': draft, 'evidence': evidence, 'classify': cls}

    # ---- step 3: route ----------------------------------------------------
    def route(self, cls, author_id=None, draft=None):
        intent = cls.get('intent')
        conf = cls.get('confidence', 0.0)
        if intent == 'appreciation':
            return 'auto', 'acknowledge_and_close'
        if conf < 0.30:
            return 'assist', 'low_prediction_confidence'
        if intent is None:
            return 'assist', 'unclear_intent'
        # prevalence of the predicted intent in the training distribution
        if self.prev.get(intent, 0.0) < PREV_MIN:
            return 'assist', 'novel_intent_no_precedent'
        if author_id is not None and author_id in self.author_hist:
            h = self.author_hist[author_id]
            if not h['history_sufficient']:
                return 'assist', 'insufficient_history'
        if intent in AUTO_INTENTS:
            route, reason = 'auto', 'routine_intent_history'
        elif author_id is not None and author_id in self.author_hist:
            h = self.author_hist[author_id]
            if (h['history_replied'] >= AUTO_REPLY_MIN['history_replied'] and
                    h['history_reply_rate'] >= AUTO_REPLY_MIN['history_reply_rate']):
                route, reason = 'auto', 'consistent_resolution_history'
            else:
                route, reason = 'assist', 'needs_judgment'
        else:
            route, reason = 'assist', 'needs_judgment'
        # safety gate (agent-level, beyond notebook 04): never auto-send without
        # a send-ready grounded draft
        if route == 'auto' and not (draft and sanitize_reply(draft).strip()):
            return 'assist', 'draft_not_send_ready'
        return route, reason

    # ---- full response ----------------------------------------------------
    def respond(self, raw_text, author_id=None, k=3):
        res = self.draft(raw_text, k=k)
        route, reason = self.route(res['classify'], author_id, draft=res['draft'])
        res['route'] = route
        res['route_reason'] = reason
        res['pred_confidence'] = res['classify']['confidence']
        res['pred_intent'] = res['classify']['intent']
        return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('text', help='raw customer tweet')
    ap.add_argument('--author-id', default=None, help='Twitter author id (for history)')
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--engine', default='tfidf', choices=['w2v', 'tfidf'],
                    help="classifier engine: 'w2v' (default) or 'tfidf'")
    ap.add_argument('--no-preprocess', action='store_true',
                    help='treat text as already preprocessed')
    args = ap.parse_args()

    ag = Agent(verbose=True, engine=args.engine)
    res = ag.respond(args.text, author_id=args.author_id, k=args.k)
    # strip heavy internals from the CLI dump
    print(json.dumps({
        'input': args.text,
        'intent': res['pred_intent'],
        'confidence': round(res['pred_confidence'], 4),
        'route': res['route'],
        'route_reason': res['route_reason'],
        'draft': res['draft'],
        'evidence': res['evidence'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()