#!/usr/bin/env python3
"""Evaluation harness for the AmazonHelp support agent.

Scores the live agent (w2v or tfidf engine) over:

* golden_eval.csv  — N=200, stratified, AUTO-labeled (silver topic + rule)
* golden_hand.csv  — N=50 sub-set, HAND-labeled by a human reviewer

and reports intent/routing metrics vs BOTH label sets, trivial baselines, a
real OpenAI gpt-4o-mini LLM judge (via .env, keyword heuristic fallback),
judge↔human agreement incl. Cohen's kappa, and draft acceptance.

Engine comes from $HIVER_ENGINE (w2v|tfidf) or --engine; local runs only.

Outputs: data/processed/eval_report.json, eval_predictions.csv,
         eval_judge_results.csv
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score

from agent import Agent

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, 'data/processed')
JUDGE_PROMPT_TEMPLATE = """\
You are a strict QA evaluator for an automated Amazon customer-support agent.
Decide independently — do not assume any suggested label is correct.

CUSTOMER TWEET:
{text}

AGENT'S DRAFT REPLY:
{draft}

TASK — answer each part on its own line, EXACTLY in this format (no extra lines):
INTENT: <your_intent_label>   (one of: appreciation | customer_service | delivery_issue | email_contact | order_status | other)
ROUTE:  <auto|assist>         (should the reply be sent automatically, or go to a human?)
DRAFT_OK: <yes|no|partial>    (is the draft a safe, relevant, grammatical reply?)
WHY: <brief justification>"""


def load_env(path=os.path.join(ROOT, '.env')):
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())


def _parse_judge_output(out):
    parsed = {}
    for line in out.splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        k, v = k.strip().upper(), v.strip()
        if k == 'INTENT':
            parsed['intent'] = v
        elif k == 'ROUTE':
            parsed['route'] = v
        elif k == 'DRAFT_OK':
            parsed['draft_ok'] = v
        elif k == 'WHY':
            parsed['why'] = v
    return parsed


def run_agent(agent, df):
    pred_intents, pred_confs, drafts, ev_sims, routes, reasons = [], [], [], [], [], []
    for _, row in df.iterrows():
        r = agent.respond(str(row['text']), author_id=str(row['author_id']))
        pred_intents.append(r['pred_intent'] or 'UNKNOWN')
        pred_confs.append(r['pred_confidence'])
        drafts.append(r['draft'] or '')
        routes.append(r['route'])
        reasons.append(r['route_reason'])
        sims = [e['sim'] for e in r['evidence']]
        ev_sims.append(np.mean(sims) if sims else 0.0)
    df = df.copy()
    df['pred_intent'] = pred_intents
    df['pred_confidence'] = pred_confs
    df['pred_route'] = routes
    df['pred_route_reason'] = reasons
    df['draft'] = drafts
    df['avg_evidence_sim'] = ev_sims
    return df


# ---- judge plumbing ---------------------------------------------------
def _judge_openai(text, pred_intent, draft, model, client):
    prompt = JUDGE_PROMPT_TEMPLATE.format(text=text, pred_intent=pred_intent,
                                          draft=draft or 'N/A')
    r = client.chat.completions.create(
        model=model, max_tokens=256, temperature=0,
        messages=[{'role': 'user', 'content': prompt}])
    return _parse_judge_output(r.choices[0].message.content)


def _heuristic_judge(text, row):
    t = str(text).lower()
    if any(w in t for w in ['thank', 'appreciate', 'love', 'great', 'awesome']):
        intent = 'appreciation'
    elif any(w in t for w in ['cancel', 'refund', 'return', 'exchange', 'prime']):
        intent = 'order_status'
    elif any(w in t for w in ['ship', 'deliver', 'arrive', 'late', 'tracking',
                              'package', 'ups', 'carrier']):
        intent = 'delivery_issue'
    elif any(w in t for w in ['email', 'contact', 'phone', 'call', 'mail']):
        intent = 'email_contact'
    else:
        intent = 'customer_service'
    hist_suff = str(row.get('history_sufficient', 'False')).lower() == 'true'
    conf = float(row.get('pred_confidence', 0.0))
    route = 'auto' if (intent == 'appreciation' or
                       (hist_suff and conf >= 0.30 and intent in
                        ('order_status', 'email_contact')) or
                       (intent == 'appreciation')) else 'assist'
    draft_ok = 'n/a'
    return {'intent': intent, 'route': route, 'draft_ok': draft_ok,
            'why': 'heuristic-fallback'}


def judge_sample(df_sample, model):
    load_env()
    client = mode = None
    if os.environ.get('OPENAI_API_KEY'):
        try:
            import openai
            client = openai.OpenAI(api_key=os.environ['OPENAI_API_KEY'], timeout=30)
            r = client.chat.completions.create(
                model=model, max_tokens=2, temperature=0,
                messages=[{'role': 'user', 'content': 'OK'}])
            mode = 'openai'
            print(f'  judge: OpenAI reachable (model={model})')
        except Exception as e:
            print(f'  judge: OpenAI failed ({type(e).__name__})')
    if mode is None:
        print('  judge: no LLM reachable — using heuristic fallback')
        mode = 'heuristic'

    results = []
    t0 = time.time()
    for i, (_, row) in enumerate(df_sample.iterrows()):
        if mode == 'openai':
            parsed = _judge_openai(row['text'], row.get('pred_intent', ''),
                                   row.get('draft', ''), model, client)
        else:
            parsed = _heuristic_judge(row['text'], row)
        parsed['tweet_id'] = row['tweet_id']
        results.append(parsed)
        if (i + 1) % 10 == 0:
            print(f'  judge {i+1}/{len(df_sample)} ({time.time()-t0:.0f}s)')
    print(f'  judge done {len(results)} in {time.time()-t0:.0f}s')
    return pd.DataFrame(results), mode


def _acc(pred, true):
    pred = list(pred)
    true = list(true)
    return round(float(accuracy_score(true, pred)), 4)


def main():
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge-n', type=int, default=30)
    ap.add_argument('--engine', default=os.environ.get('HIVER_ENGINE', 'tfidf'),
                    choices=['w2v', 'tfidf'],
                    help="classifier engine (default: $HIVER_ENGINE or tfidf)")
    ap.add_argument('--judge-model', default=None)
    args = ap.parse_args()

    judge_model = (args.judge_model or os.environ.get('OPENAI_MODEL')
                   or 'gpt-4o-mini')

    df = pd.read_csv(os.path.join(PROC, 'golden_eval.csv'), dtype=str)
    print(f'golden_eval: {len(df)} rows | engine={args.engine}')

    agent = Agent(verbose=True, engine=args.engine)
    df = run_agent(agent, df)

    # ---- auto-gold metrics --------------------------------------------
    y_true = df['gold_intent']
    y_pred = df['pred_intent']
    rep = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    intent_acc = accuracy_score(y_true, y_pred)
    maj_base = df['gold_intent'].value_counts().idxmax()
    route_acc = accuracy_score(df['gold_route'], df['pred_route'])
    assist_base = accuracy_score(df['gold_route'], ['assist'] * len(df))

    print(f'\n[auto-gold] intent acc={intent_acc:.4f} (baseline {maj_base}={_acc([maj_base]*len(df), y_true):.4f})')
    print(f'[auto-gold] route acc={route_acc:.4f} (baseline all-assist={assist_base:.4f})')
    print(f'[auto-gold] macro-F1={rep["macro avg"]["f1-score"]:.4f}')
    print('route reasons:', df['pred_route_reason'].value_counts().to_dict())

    report = {
        'engine': args.engine,
        'golden_200_autolabelled': {
            'intent_accuracy': round(float(intent_acc), 4),
            'baseline_majority_intent': maj_base,
            'baseline_accuracy': round(float(_acc([maj_base] * len(df), y_true)), 4),
            'macro_f1': round(float(rep['macro avg']['f1-score']), 4),
            'routing_accuracy': round(float(route_acc), 4),
            'baseline_all_assist_accuracy': round(float(assist_base), 4),
            'route_distribution': df['pred_route'].value_counts().to_dict(),
            'route_reasons': df['pred_route_reason'].value_counts().to_dict(),
            'avg_evidence_sim': round(float(df['avg_evidence_sim'].mean()), 4),
            'pct_with_draft': round(float((df['draft'].str.len() > 0).mean()), 4),
        },
    }

    # ---- hand-gold metrics ---------------------------------------------
    hand = pd.read_csv(os.path.join(PROC, 'golden_hand.csv'), dtype=str)
    if os.path.exists(os.path.join(PROC, 'eval_predictions.csv')):
        pass  # predictions come from df in-memory (hand ⊆ golden)
    h = df[df['tweet_id'].isin(hand['tweet_id'])].copy()
    h = h.merge(hand[['tweet_id', 'hand_intent', 'hand_route']], on='tweet_id')
    h_known = h[h['hand_intent'] != 'other']
    h_metrics = {
        'n': len(h),
        'hand_intent_accuracy': _acc(h_known['pred_intent'], h_known['hand_intent']),
        'hand_route_accuracy': _acc(h['pred_route'], h['hand_route']),
        'hand_route_assist_baseline': _acc(['assist'] * len(h), h['hand_route']),
        'silver_gold_intent_agreement': _acc(h['gold_intent'], h['hand_intent']),
        'silver_gold_route_agreement': _acc(h['gold_route'], h['hand_route']),
        'n_other': int((h['hand_intent'] == 'other').sum()),
    }
    print(f'\n[hand-gold n={len(h)}] intent acc={h_metrics["hand_intent_accuracy"]:.4f} '
          f'| route acc={h_metrics["hand_route_accuracy"]:.4f} '
          f'| silver-vs-hand intent={h_metrics["silver_gold_intent_agreement"]:.4f}')
    report['hand_labeled_50'] = h_metrics

    # ---- judge ----------------------------------------------------------
    if args.judge_n > 0:
        n = min(args.judge_n, len(df))
        idx = np.random.RandomState(42).choice(len(df), size=n, replace=False)
        jdf = df.iloc[idx].reset_index(drop=True)
        print(f'\njudging golden sample n={n}')
        judge_df, jmode = judge_sample(jdf, judge_model)
        jdf = jdf.merge(judge_df, on='tweet_id', how='left')

        ji = np.array([x for x in jdf['intent'] if isinstance(x, str)])
        jtrue = jdf.loc[[isinstance(x, str) for x in jdf['intent']], 'gold_intent']
        jr = np.array([x for x in jdf['route'] if isinstance(x, str)])
        rtrue = jdf.loc[[isinstance(x, str) for x in jdf['route']], 'gold_route']

        judge_gold = {
            'mode': jmode,
            'model': judge_model if jmode != 'heuristic' else None,
            'n': len(jdf),
            'intent_accuracy_vs_autogold': round(float(accuracy_score(jtrue, ji)), 4),
            'route_accuracy_vs_autogold': round(float(accuracy_score(rtrue, jr)), 4),
            'route_kappa_vs_autogold':
                round(float(cohen_kappa_score(rtrue, jr)), 4) if len(set(rtrue)) > 1 else None,
            'draft_yes': int((jdf['draft_ok'] == 'yes').sum()),
            'draft_partial': int((jdf['draft_ok'] == 'partial').sum()),
            'draft_no': int((jdf['draft_ok'] == 'no').sum()),
        }
        print(f'[judge {jmode}] intent-vs-gold={judge_gold["intent_accuracy_vs_autogold"]} '
              f'route-vs-gold={judge_gold["route_accuracy_vs_autogold"]} '
              f'draft y/p/n={judge_gold["draft_yes"]}/{judge_gold["draft_partial"]}/{judge_gold["draft_no"]}')

        judge_hand = {}
        if h is not None and len(h):
            print(f'\njudging hand-labeled subset n={len(h)}')
            hand_judge, _ = judge_sample(
                h[['tweet_id', 'text', 'pred_intent', 'pred_route', 'draft',
                   'history_sufficient', 'pred_confidence']].reset_index(drop=True),
                judge_model)
            hh = h[['tweet_id', 'hand_intent', 'hand_route']].merge(
                hand_judge, on='tweet_id', how='inner')
            if len(hh):
                hk = hh[hh['hand_intent'] != 'other']
                hi = np.array([x for x in hk['intent'] if isinstance(x, str)])
                htrue = hk.loc[[isinstance(x, str) for x in hk['intent']], 'hand_intent']
                hr = np.array([x for x in hh['route'] if isinstance(x, str)])
                rtrue_h = hh.loc[[isinstance(x, str) for x in hh['route']], 'hand_route']
                judge_hand = {
                    'n': len(hh),
                    'intent_accuracy_vs_hand': round(float(accuracy_score(htrue, hi)), 4),
                    'route_accuracy_vs_hand': round(float(accuracy_score(rtrue_h, hr)), 4),
                    'route_kappa_vs_hand':
                        round(float(cohen_kappa_score(rtrue_h, hr)), 4) if len(set(rtrue_h)) > 1 else None,
                    'hand_route_vs_judge_draft_flagged':
                        int((hh['route'] == 'auto').sum()),
                }
                print(f'[judge vs hand n={len(hh)}] intent={judge_hand["intent_accuracy_vs_hand"]} '
                      f'route={judge_hand["route_accuracy_vs_hand"]} '
                      f'kappa={judge_hand["route_kappa_vs_hand"]}')
        report['llm_judge'] = {'vs_gold': judge_gold, 'vs_hand': judge_hand}
        judge_df.to_csv(os.path.join(PROC, 'eval_judge_results.csv'), index=False)
    else:
        report['llm_judge'] = {'note': 'skipped (--judge-n 0)'}

    report['sampling_note'] = (
        f'Golden N={len(df)} stratified by silver intent (seed=42), auto-labeled '
        'in notebook 04. Hand subset N=50 labelled by a human reviewer on '
        'reading tweet + author history (embedded in notebook 04).')

    with open(os.path.join(PROC, 'eval_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)
    df.to_csv(os.path.join(PROC, 'eval_predictions.csv'), index=False)
    print(f'\nSaved eval_report.json + eval_predictions.csv + eval_judge_results.csv')


if __name__ == '__main__':
    main()