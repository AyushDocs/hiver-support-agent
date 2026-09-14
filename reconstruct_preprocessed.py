#!/usr/bin/env python3
"""Reconstruct data/processed/data_preprocessed.csv (01 output) from raw
twcs.csv linkage + the cleaned text already stored in amazon_help_labeled.csv.

01's output had columns: tweet_id,author_id,inbound,created_at,text,
response_tweet_id,in_response_to_tweet_id.  The labeled CSV (02) carries the
cleaned `text` and author/created_at but drops the linkage columns, so those
two linkage columns are re-extracted straight from the raw file for the exact
same tweet ids (no cleaning needed at this stage).
"""
import os
import time

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, 'data/processed')
RAW = os.path.join(ROOT, 'data/raw/twcs.csv')
CAND_RAW = ['/kaggle/input/datasets/thoughtvector/customer-support-on-twitter/twcs/twcs.csv',
            '/kaggle/input/twcs/twcs.csv', '/kaggle/working/twcs.csv', RAW]
RAW_PATH = next((p for p in CAND_RAW if os.path.exists(p)), RAW)

t0 = time.time()
lab = pd.read_csv(os.path.join(PROC, 'amazon_help_labeled.csv'), dtype=str)
wanted = set(lab['tweet_id'].astype(str))
print(f'labeled rows: {len(lab):,} | want linkage for {len(wanted):,} ids')

linkage = []
t = time.time()
for chunk in pd.read_csv(RAW_PATH, dtype=str, usecols=[
        'tweet_id', 'response_tweet_id', 'in_response_to_tweet_id'],
        chunksize=500000):
    chunk = chunk[chunk['tweet_id'].isin(wanted)]
    linkage.append(chunk)
    if len(linkage) and sum(c.shape[0] for c in linkage) % 5000 == 0:
        print(f'  found {sum(c.shape[0] for c in linkage):,} '
              f'({time.time()-t:.0f}s)')
link = pd.concat(linkage, ignore_index=True)
print(f'raw linkage scan done in {time.time()-t:.0f}s '
      f'(found {len(link):,})')

df = lab.merge(link, on='tweet_id', how='left')
df.insert(2, 'inbound', '1')
df = df[['tweet_id', 'author_id', 'inbound', 'created_at', 'text',
         'response_tweet_id', 'in_response_to_tweet_id']]
df.sort_values('tweet_id', inplace=True)
df.to_csv(os.path.join(PROC, 'data_preprocessed.csv'), index=False)
print(f'wrote data_preprocessed.csv: {len(df):,} rows x {len(df.columns)} cols '
      f'in {time.time()-t0:.0f}s')
print(df.head(3).to_string())