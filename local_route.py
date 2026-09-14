#!/usr/bin/env python3
"""Local run of the auto-route notebook (mirrors 04_auto_route.ipynb).

Decides, for every incoming request, whether it can be resolved AUTOMATICALLY
(sufficient historical precedent to reuse) or needs HUMAN assistance.

Inputs: data/processed/amazon_help_classified.csv  (from 03)
        data/raw/twcs.csv                         (full history incl. responses)
Output: data/processed/amazon_help_routed.csv
        data/processed/routing_report.json
        data/processed/routing_plot.png
"""
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
CLASSIFIED_PATH = os.path.join(ROOT, 'data/processed/amazon_help_classified.csv')
RAW_PATH = os.path.join(ROOT, 'data/raw/twcs.csv')
OUT_ROUTED = os.path.join(ROOT, 'data/processed/amazon_help_routed.csv')
OUT_REPORT = os.path.join(ROOT, 'data/processed/routing_report.json')
OUT_PLOT = os.path.join(ROOT, 'data/processed/routing_plot.png')

t0 = time.time()

# --- Rule constants (tunable, documented in the report) ---
CONF_LO = 0.30          # below classifier confidence -> ambiguous -> human
PREV_MIN = 0.02         # intent must make up >= 2% of history to have precedent
HIST_VOL_MIN = 3        # author must have >= this many prior inbound tweets
HIST_REPLY_MIN = 0.3    # ... and a resolved-reply rate above this
AUTO_INTENTS = {'order_status', 'email_contact'}   # routine, template-able
AUTO_REPLY_MIN = {'history_replied': 2, 'history_reply_rate': 0.7}  # consistent resolver

# --- Cell 1: load classified requests ---
df = pd.read_csv(CLASSIFIED_PATH)
print(f'Requests loaded: {len(df)}')
print(df['pred_intent'].value_counts().to_dict())

# --- Cell 2: author history from full raw dataset ---
print('\nBuilding author history from raw twcs...')
raw = pd.read_csv(RAW_PATH, dtype={'tweet_id': str, 'author_id': str,
                                   'response_tweet_id': str,
                                   'in_response_to_tweet_id': str})
print(f'Raw rows: {len(raw):,}')

amz_mask = raw['inbound'] == 1
mention = raw.loc[amz_mask, 'text'].str.lower().str.contains('amazonhelp', na=False)
amz_in = raw.loc[amz_mask & mention].copy()   # AmazonHelp inbound mentions
print(f'AmazonHelp inbound mentions: {len(amz_in):,}')

responded = amz_in[amz_in['response_tweet_id'].notna()]
print(f'... of which got a reply: {len(responded):,}')

# reply latency: inbound -> matched outbound reply
out = raw[raw['inbound'] == 0][['tweet_id', 'created_at']]
lat = responded[['tweet_id', 'author_id', 'created_at', 'response_tweet_id']].merge(
    out, left_on='response_tweet_id', right_on='tweet_id', suffixes=('_in', '_out'))
lat['wait_h'] = (pd.to_datetime(lat['created_at_out']) - pd.to_datetime(lat['created_at_in'])).dt.total_seconds() / 3600
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
print(f'Authors with history: {len(hist):,}')

df['author_id'] = df['author_id'].astype(str)
df = df.merge(hist, on='author_id', how='left')
for c in ['history_volume', 'history_replied']:
    df[c] = df[c].fillna(0).astype(int)
df['history_reply_rate'] = df['history_reply_rate'].fillna(0.0)
df['avg_reply_hours'] = df['avg_reply_hours'].fillna(np.nan)
df['history_sufficient'] = df['history_sufficient'].fillna(False)

# --- Cell 3: intent-level precedent ---
N = len(df)
prev_share = df['pred_intent'].value_counts(normalize=True)
df['intent_prevalence'] = df['pred_intent'].map(prev_share)
df['has_precedent'] = df['intent_prevalence'] >= PREV_MIN
print('\nIntent prevalence:')
print((prev_share * 100).round(1).to_dict())

# --- Cell 4: routing rule ---
def route_request(row, conf_lo=CONF_LO):
    if row['pred_intent'] == 'appreciation':
        return 'auto', 'acknowledge_and_close'
    if row['pred_confidence'] < conf_lo:
        return 'assist', 'low_prediction_confidence'
    if row['low_confidence']:
        return 'assist', 'unclear_intent'
    if not row['has_precedent']:
        return 'assist', 'novel_intent_no_precedent'
    if not row['history_sufficient']:
        return 'assist', 'insufficient_history'
    if row['pred_intent'] in AUTO_INTENTS:
        return 'auto', 'routine_intent_history'
    if (row['history_replied'] >= AUTO_REPLY_MIN['history_replied'] and
            row['history_reply_rate'] >= AUTO_REPLY_MIN['history_reply_rate']):
        return 'auto', 'consistent_resolution_history'
    return 'assist', 'needs_judgment'

routs = df.apply(lambda r: route_request(r), axis=1)
df['route'] = [r[0] for r in routs]
df['route_reason'] = [r[1] for r in routs]

print('\n=== Route distribution ===')
print(df['route'].value_counts())
print('\nRoutes by intent:')
print(pd.crosstab(df['pred_intent'], df['route']))
print('\nReasons:')
print(df['route_reason'].value_counts())

# --- Cell 5: diagnostics ---
auto = df[df['route'] == 'auto']
assist = df[df['route'] == 'assist']
print(f'\nAuto:        {len(auto):,}  avg conf={auto["pred_confidence"].mean():.3f}  avg history={auto["history_volume"].mean():.1f}')
print(f'Human assist:{len(assist):,}  avg conf={assist["pred_confidence"].mean():.3f}  avg history={assist["history_volume"].mean():.1f}')

print('\nSample AUTO (template-able requests):')
print(auto[['text', 'pred_intent', 'route_reason', 'history_volume', 'history_reply_rate']].head(6).to_string(index=False))
print('\nSample ASSIST (human needed):')
print(assist[['text', 'pred_intent', 'route_reason', 'history_volume', 'history_reply_rate']].head(6).to_string(index=False))

# --- Cell 6: sensitivity of the auto rate vs confidence threshold ---
print('\nSensitivity (auto-rate as CONF_LO varies):')
sens = {}
for c in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    rr = df.apply(lambda r: route_request(r, conf_lo=c), axis=1)
    sens[str(c)] = round(float(np.mean([x[0] == 'auto' for x in rr])), 4)
    print(f'  CONF_LO={c:.2f}  auto_rate={sens[str(c)]:.1%}')

# --- Cell 7: plot ---
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
colors = {'auto': '#2e7d32', 'assist': '#c62828'}
big = df.groupby(['pred_intent', 'route']).size().unstack(fill_value=0)
for r in ['auto', 'assist']:
    if r not in big.columns:
        big[r] = 0
big_pct = big.div(big.sum(axis=1), axis=0)
bottom = np.zeros(len(big_pct))
for r in ['auto', 'assist']:
    axes[0].bar(big_pct.index, big_pct[r], bottom=bottom, label=r, color=colors[r], alpha=0.9)
    bottom += big_pct[r]
axes[0].set_title('Auto-resolve share by intent')
axes[0].set_ylabel('share'); axes[0].set_xticklabels(big_pct.index, rotation=35, ha='right', fontsize=8)
axes[0].legend()

axes[1].pie([len(auto), len(assist)], labels=[f'auto\n{len(auto)}', f'assist\n{len(assist)}'],
            colors=[colors['auto'], colors['assist']], startangle=90, autopct='%1.1f%%')
axes[1].set_title('Overall routing split')
plt.tight_layout()
plt.savefig(OUT_PLOT)
print(f'\nSaved: {OUT_PLOT}')

# --- Cell 8: save ---
df.to_csv(OUT_ROUTED, index=False)
print(f'Saved: {OUT_ROUTED} ({len(df)} rows)')

report = {
    'rule': {
        'appreciation': 'auto (acknowledge & close)',
        'pred_confidence < %s' % CONF_LO: 'assist (low prediction confidence)',
        'low_confidence': 'assist (unclear intent)',
        'intent_prevalence < %s' % PREV_MIN: 'assist (no precedent)',
        'history_sufficient == False': 'assist (insufficient history)',
        'routine intent': 'auto (order_status / email_contact)',
        'consistent resolver': 'auto (>=%d replied, rate>=%s)' % (AUTO_REPLY_MIN['history_replied'], AUTO_REPLY_MIN['history_reply_rate']),
        'else': 'assist (needs judgment)',
    },
    'constants': {
        'conf_lo': CONF_LO, 'prev_min': PREV_MIN, 'hist_vol_min': HIST_VOL_MIN,
        'hist_reply_min': HIST_REPLY_MIN, 'auto_intents': sorted(AUTO_INTENTS),
        'auto_reply_min': AUTO_REPLY_MIN,
    },
    'routing': df['route'].value_counts().to_dict(),
    'routing_by_intent': {i: r.to_dict() for i, r in
                          df.groupby(['pred_intent', 'route']).size().unstack(fill_value=0).iterrows()},
    'reasons': df['route_reason'].value_counts().to_dict(),
    'auto_rate': round(float((df['route'] == 'auto').mean()), 4),
    'avg_confidence_by_route': df.groupby('route')['pred_confidence'].mean().round(4).to_dict(),
    'avg_history_by_route': df.groupby('route')['history_volume'].mean().round(2).to_dict(),
    'sensitivity_conf_lo': sens,
    'authors': int(df['author_id'].nunique()),
}
with open(OUT_REPORT, 'w') as f:
    json.dump(report, f, indent=2)
print(f'Saved: {OUT_REPORT}')

print(f'\nTotal time: {time.time() - t0:.0f}s')