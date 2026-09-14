#!/usr/bin/env python3
"""Build the golden evaluation set for the AmazonHelp agent.

Sampling note
-------------
* Universe: 31,862 @AmazonHelp customer tweets (classified set, 03).
* Sample:   N=200, stratified by the silver intent label (02) so every intent
            is represented roughly in proportion to its prevalence, then
            within each stratum a deterministic random draw (seed=42).
* Author spread: sampling is without replacement on tweet, so a single author
            can appear (their full history is used for routing); we cap such
            repeats implicitly via the tweet-level draw.  Recall author history
            is what the routing rules consume, so repeats are informative.
* Labels:  GOLDEN labels are AUTO-LABELED from the pipeline's silver signals
            (topic-model intent from 02, routing rule from 04 re-derived from
            the GOLD intent + the author's observed history).  These are
            PROVISIONAL ground truth pending expert spot-check: a reviewer can
            flip `gold_intent` / `gold_route_*` columns and re-run `eval.py`.
* Output:  data/processed/golden_eval.csv
"""
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, 'data/processed')

SEED = 42
N = 200
PREV_MIN = 0.02
HIST_VOL_MIN = 3
HIST_REPLY_MIN = 0.3
AUTO_INTENTS = {'order_status', 'email_contact'}
AUTO_REPLY_MIN = {'history_replied': 2, 'history_reply_rate': 0.7}


def golden_route(intent, hist_row):
    """The 'ideal' route the support lead would pick for a tweet whose intent
    is known.  Mirrors notebook 04's rule but consumes the GOLD intent."""
    if intent == 'appreciation':
        return 'auto', 'acknowledge_and_close'
    if hist_row is None or not hist_row['history_sufficient']:
        return 'assist', 'insufficient_history' if intent != 'appreciation' else 'assist'
    if intent in AUTO_INTENTS:
        return 'auto', 'routine_intent_history'
    if (hist_row['history_replied'] >= AUTO_REPLY_MIN['history_replied'] and
            hist_row['history_reply_rate'] >= AUTO_REPLY_MIN['history_reply_rate']):
        return 'auto', 'consistent_resolution_history'
    return 'assist', 'needs_judgment'


def main():
    df = pd.read_csv(os.path.join(PROC, 'amazon_help_classified.csv'), dtype=str)
    hist = pd.read_csv(os.path.join(PROC, 'author_history.csv'),
                       dtype={'author_id': str}, index_col='author_id')

    rng = np.random.RandomState(SEED)
    pick = []
    for intent, grp in df.groupby('intent'):
        n = max(1, int(round(N * len(grp) / len(df))))
        pick.append(grp.sample(n=n, random_state=SEED))
    golden = pd.concat(pick).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    golden = golden.head(N).copy()

    gr, greason, histkeys = [], [], []
    for _, row in golden.iterrows():
        h = hist.loc[row['author_id']] if row['author_id'] in hist.index else None
        r, why = golden_route(row['intent'], h)
        gr.append(r)
        greason.append(why)
        histkeys.append({'history_volume': int(h['history_volume']) if h is not None else 0,
                         'history_replied': int(h['history_replied']) if h is not None else 0,
                         'history_reply_rate': float(h['history_reply_rate']) if h is not None else None,
                         'history_sufficient': bool(h['history_sufficient']) if h is not None else False}
                        if h is not None else {})

    sr = pd.DataFrame(histkeys)
    golden['gold_intent'] = golden['intent']
    golden['gold_route'] = gr
    golden['gold_route_reason'] = greason
    golden['history_volume'] = sr['history_volume']
    golden['history_replied'] = sr['history_replied']
    golden['history_reply_rate'] = sr['history_reply_rate']
    golden['history_sufficient'] = sr['history_sufficient']

    keep = ['tweet_id', 'author_id', 'created_at', 'text',
            'gold_intent', 'gold_route', 'gold_route_reason',
            'pred_intent', 'pred_confidence', 'topic_confidence',
            'history_volume', 'history_replied', 'history_reply_rate', 'history_sufficient']
    golden = golden[keep]
    assert len(golden) == N
    assert golden['gold_intent'].isna().sum() == 0 and golden['gold_route'].isna().sum() == 0

    golden.to_csv(os.path.join(PROC, 'golden_eval.csv'), index=False)
    print(f'Saved golden_eval.csv: {len(golden)} rows, '
          f'route split {golden["gold_route"].value_counts().to_dict()}')
    print('Intent split:', golden['gold_intent'].value_counts().to_dict())
    print('Full-history authors in sample:', int(golden['history_volume'].gt(0).sum()))


if __name__ == '__main__':
    main()