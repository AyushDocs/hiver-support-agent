#!/usr/bin/env python3
"""Local run of the intent discovery pipeline (mirrors 02_intent_discovery.ipynb).

Input:  data/processed/data_preprocessed.csv  (output of notebook 01, 'text' already cleaned)
Output: data/processed/amazon_help_labeled.csv
        data/processed/intent_mapping.json
        data/processed/coherence_plot.png
"""
import os
import json
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# --- Cell: imports ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
CLEANED_PATH = os.path.join(ROOT, 'data/processed/data_preprocessed.csv')
OUT_LABELED = os.path.join(ROOT, 'data/processed/amazon_help_labeled.csv')
OUT_MAPPING = os.path.join(ROOT, 'data/processed/intent_mapping.json')
OUT_PLOT = os.path.join(ROOT, 'data/processed/coherence_plot.png')

t0 = time.time()

# --- Cell: load processed data ---
df = pd.read_csv(CLEANED_PATH)
print(f'Loaded: {len(df)} rows')
# 'text' is already cleaned/filtered by notebook 01 (English-only, >=4 words, denoised)
print(f'Total tweets: {len(df)}, unique authors: {df["author_id"].nunique()}')

print('Sample cleaned tweets:')
for i, row in df.sample(10, random_state=42).iterrows():
    print(f'  [{len(row["text"].split())} words] {row["text"][:120]}')

# --- Cell: TF-IDF ---
tfidf = TfidfVectorizer(
    max_features=5000, ngram_range=(1, 1), min_df=3, max_df=0.5,
    stop_words='english', lowercase=True
)
tfidf_matrix = tfidf.fit_transform(df['text'])
feature_names = tfidf.get_feature_names_out()
print(f'\nTF-IDF matrix: {tfidf_matrix.shape}')

# --- Cell: coherence + k sweep ---
def compute_coherence(model, tfidf_matrix, feature_names, top_n=10):
    from sklearn.metrics.pairwise import cosine_similarity
    scores = []
    for topic_idx in range(model.n_components):
        topic = model.components_[topic_idx]
        top_indices = topic.argsort()[:-top_n - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        word_vectors = []
        for word in top_words:
            if word in feature_names:
                idx = list(feature_names).index(word)
                word_vectors.append(tfidf_matrix[:, idx].toarray().flatten())
        if len(word_vectors) > 1:
            sim_matrix = cosine_similarity(word_vectors)
            pairs = [sim_matrix[j][k] for j in range(len(word_vectors)) for k in range(j+1, len(word_vectors))]
            scores.append(np.mean(pairs) if pairs else 0)
        else:
            scores.append(0)
    return np.mean(scores)

print('\n=== Coherence vs k ===')
k_range = range(5, 16)
results = []
for k in k_range:
    nmf = NMF(n_components=k, random_state=42, max_iter=500, init='nndsvda')
    W = nmf.fit_transform(tfidf_matrix)
    coherence = compute_coherence(nmf, tfidf_matrix, feature_names)
    results.append({'k': k, 'coherence': coherence})
    print(f'k={k:2d}  coherence={coherence:.4f}')

results_df = pd.DataFrame(results)
BEST_K = int(results_df.loc[results_df['coherence'].idxmax(), 'k'])
print(f'\nBest k: {BEST_K}')

# --- Cell: plot coherence ---
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(results_df['k'], results_df['coherence'], 'b-o')
ax.set_xlabel('Number of Topics')
ax.set_ylabel('Coherence')
ax.set_title('NMF Coherence vs k')
plt.tight_layout()
plt.savefig(OUT_PLOT, dpi=120)
plt.close()
print(f'Saved plot: {OUT_PLOT}')

# --- Cell: final model ---
nmf_final = NMF(n_components=BEST_K, random_state=42, max_iter=500, init='nndsvda')
W = nmf_final.fit_transform(tfidf_matrix)

topics = []
for i in range(BEST_K):
    topic = nmf_final.components_[i]
    top_words = [feature_names[j] for j in topic.argsort()[:-11:-1]]
    topics.append(top_words)

print(f'\nDiscovered {BEST_K} topics:\n')
for i, topic_words in enumerate(topics):
    print(f'Topic {i}: ' + ' | '.join(topic_words[:10]))

# --- Cell: sample tweets per topic (for manual labeling) ---
print('\n' + '=' * 60)
for i in range(BEST_K):
    print(f'\n{"="*60}')
    print(f'TOPIC {i}: ' + ' | '.join(topics[i][:8]))
    print('=' * 60)
    top_indices = W[:, i].argsort()[-5:][::-1]
    for idx in top_indices:
        text = str(df.iloc[idx]['text'])[:150]
        print(f'  {text}')
    print()

# --- Cell: dynamic labels (rename to production intents) ---
TOPIC_LABELS = {
    0: 'delivery_issue',
    1: 'customer_service',
    2: 'order_status',
    3: 'email_contact',
    4: 'appreciation',
}

# --- Cell: assign intent labels ---
df['topic_id'] = W.argmax(axis=1)
df['topic_confidence'] = W.max(axis=1)
df['intent'] = df['topic_id'].map(TOPIC_LABELS)

CONFIDENCE_THRESHOLD = float(df['topic_confidence'].quantile(0.25))
df['low_confidence'] = df['topic_confidence'] < CONFIDENCE_THRESHOLD

print(f'\nAssigned intents to {len(df)} tweets')
print(f'Low confidence: {df["low_confidence"].sum()} ({df["low_confidence"].mean()*100:.1f}%)')
print('\nIntent distribution:')
print(df['intent'].value_counts())

# --- Cell: save ---
save_cols = ['tweet_id', 'author_id', 'text', 'created_at',
             'topic_id', 'topic_confidence', 'intent', 'low_confidence']
df[save_cols].to_csv(OUT_LABELED, index=False)
print(f'\nSaved: {OUT_LABELED} ({len(df)} rows)')

mapping = {
    'topic_labels': {str(k): v for k, v in TOPIC_LABELS.items()},
    'top_words': {str(i): topics[i][:10] for i in range(BEST_K)},
    'config': {
        'best_k': BEST_K,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'total_tweets': len(df)
    }
}
with open(OUT_MAPPING, 'w') as f:
    json.dump(mapping, f, indent=2)
print(f'Saved: {OUT_MAPPING}')

print(f'\nTotal time: {time.time() - t0:.0f}s')