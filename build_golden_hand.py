#!/usr/bin/env python3
"""Build a genuinely hand-labeled gold subset (golden_hand.csv).

Takes the 50-tweet stratified pool from golden_eval.csv and attaches the human
labels in hand_labels.py.  Also carries the silver auto-labels so downstream
metrics can report BOTH auto-gold and hand-gold agreement, and the agent
predictions from eval_predictions.csv when present.

Output: data/processed/golden_hand.csv
"""
import os

import pandas as pd

from hand_labels import HAND_LABELS

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, 'data/processed')


def main():
    golden = pd.read_csv(os.path.join(PROC, 'golden_eval.csv'), dtype=str)
    sub = golden[golden['tweet_id'].isin(HAND_LABELS)].copy()
    sub = sub.set_index('tweet_id')
    qi = sub.index.duplicated(keep='first')
    if qi.any():
        raise ValueError('duplicate tweet_ids in hand pool')
    for tid, (intent, route) in HAND_LABELS.items():
        if tid not in sub.index:
            raise ValueError(f'hand label {tid} missing from golden_eval')
        sub.loc[tid, 'hand_intent'] = intent
        sub.loc[tid, 'hand_route'] = route
    sub = sub.reset_index()
    sub = sub[['tweet_id', 'author_id', 'created_at', 'text',
               'gold_intent', 'gold_route', 'hand_intent', 'hand_route',
               'history_volume', 'history_replied', 'history_reply_rate',
               'history_sufficient']]
    sub.to_csv(os.path.join(PROC, 'golden_hand.csv'), index=False)
    n = len(sub)
    print(f'Saved golden_hand.csv: {n} rows')
    print('hand intent:', sub['hand_intent'].value_counts().to_dict())
    print('hand route:', sub['hand_route'].value_counts().to_dict())
    print('silver==hand intent agreement:',
          round(float((sub['gold_intent'] == sub['hand_intent']).mean()), 3))
    print('silver==hand route agreement: ',
          round(float((sub['gold_route'] == sub['hand_route']).mean()), 3))


if __name__ == '__main__':
    main()