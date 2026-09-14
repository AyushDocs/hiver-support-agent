#!/usr/bin/env python3
"""Local run of the intent classifier notebook (mirrors 03_intent_classifier.ipynb).

Input:  data/processed/amazon_help_labeled.csv  (from 02 intent discovery)
Output: data/processed/embeddings.npz
        data/processed/embedding_visualization.png
        data/processed/confusion_matrix.png
        data/processed/classification_report.json
        data/models/word2vec.model + intent_classifier.joblib + label_encoder.joblib
        data/processed/amazon_help_classified.csv
"""
import os
import re
import json
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
LABELED_PATH = os.path.join(ROOT, 'data/processed/amazon_help_labeled.csv')
OUT_EMB = os.path.join(ROOT, 'data/processed/embeddings.npz')
OUT_VIZ = os.path.join(ROOT, 'data/processed/embedding_visualization.png')
OUT_CM = os.path.join(ROOT, 'data/processed/confusion_matrix.png')
OUT_REPORT = os.path.join(ROOT, 'data/processed/classification_report.json')
OUT_CLASSIFIED = os.path.join(ROOT, 'data/processed/amazon_help_classified.csv')
os.makedirs(os.path.join(ROOT, 'data/models'), exist_ok=True)
OUT_W2V = os.path.join(ROOT, 'data/models/word2vec.model')
OUT_CLF = os.path.join(ROOT, 'data/models/intent_classifier.joblib')
OUT_LE = os.path.join(ROOT, 'data/models/label_encoder.joblib')

SEED = 42
VEC_SIZE = 100
MIN_COUNT = 2
MAX_FEATURES = 20000

t0 = time.time()
plt.rcParams['figure.dpi'] = 110

# --- Cell: config + load ---
print('Config:')
print(f'  word2vec: vector_size={VEC_SIZE}, window=5, min_count={MIN_COUNT}, seed={SEED}, sg=0 (CBOW)')
print(f'  classifier: logistic regression on mean word2vec embeddings')

df = pd.read_csv(LABELED_PATH)
print(f'Loaded: {len(df)} rows')

# --- Cell: tokenize (lowercase alpha, drop single chars + stopwords) ---
STOP = set(ENGLISH_STOP_WORDS)

def tokenize(text):
    tokens = re.findall(r"[a-z]+", str(text).lower())
    return [t for t in tokens if len(t) > 1 and t not in STOP]

print('Tokenizing...')
with ThreadPoolExecutor(max_workers=8) as ex:
    df['tokens'] = list(ex.map(tokenize, df['text']))

df['n_tokens'] = df['tokens'].str.len()
dropped = int((df['n_tokens'] == 0).sum())
df = df[df['n_tokens'] > 0].reset_index(drop=True)
print(f'Docs with tokens: {len(df)} (dropped {dropped} token-less tweets)')

counts = df.groupby('intent').size()
print('\nIntent distribution before embedding:')
print(counts)
print(f'\nW2V corpus: {df["tokens"].str.len().sum():,} tokens')

# --- Cell: train Word2Vec ---
print('\nTraining Word2Vec...')
w2v = Word2Vec(sentences=df['tokens'], vector_size=VEC_SIZE, window=5,
               min_count=MIN_COUNT, workers=4, epochs=5, seed=SEED)
print(f'Vocab size: {len(w2v.wv)}')

# --- Cell: embed tweets (mean of word vectors) ---
def embed_tweet(tokens):
    vectors = [w2v.wv[t] for t in tokens if t in w2v.wv]
    return np.mean(vectors, axis=0) if vectors else None

print('Embedding tweets...')
with ThreadPoolExecutor(max_workers=8) as ex:
    vecs = list(ex.map(embed_tweet, df['tokens']))

df['embedding'] = vecs
oov_dropped = int(df['embedding'].isna().sum())
df = df[df['embedding'].notna()].reset_index(drop=True)
print(f'Embedded: {len(df)} rows (dropped {oov_dropped} fully-OOV)')
emb = np.array(df['embedding'].tolist(), dtype=np.float32)
emb = normalize(emb, norm='l2')
print(f'Embedding matrix: {emb.shape}')

df = df.drop(columns=['tokens', 'n_tokens', 'embedding'])
np.savez_compressed(OUT_EMB, tweet_id=df['tweet_id'].values, embedding=emb,
                    intent=df['intent'].values, topic_confidence=df['topic_confidence'].values)
print(f'Saved: {OUT_EMB}')

# --- Cell: visualize intents ---
with open(os.path.join(ROOT, 'data/processed/intent_mapping.json')) as f:
    mapping = json.load(f)

top_by_intent = {int(k): v for k, v in mapping['top_words'].items()}
labels_ordered = list(df['intent'].unique())
intent_order = [mapping['topic_labels'][str(t)] for t in sorted(top_by_intent)]
label_to_topic = {v: k for k, v in mapping['topic_labels'].items()}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

colors = plt.cm.tab10(np.linspace(0, 1, len(intent_order)))
color_map = dict(zip(intent_order, colors))

# Panel A: word embedding space (top words per intent)
print('\nWord embedding viz...')
all_words, word_intents, w_vectors = [], [], []
for intent in intent_order:
    topic_id = label_to_topic[intent]
    for w in top_by_intent[int(topic_id)][:8]:
        if w in w2v.wv:
            all_words.append(w)
            word_intents.append(intent)
            w_vectors.append(w2v.wv[w])
w_vectors = normalize(np.array(w_vectors), norm='l2')
w_pca = PCA(n_components=2, random_state=SEED).fit_transform(w_vectors)
for i, (x, y) in enumerate(w_pca):
    axes[0].scatter(x, y, color=color_map[word_intents[i]], s=60,
                    edgecolors='black', linewidths=0.4)
    axes[0].annotate(all_words[i], (x, y), fontsize=8, alpha=0.85)
axes[0].set_title('Word embeddings (PCA) — top words per intent')
axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')

# Panel B: tweet embedding space (stratified subsample, t-SNE)
print('t-SNE on subsample...')
rng = np.random.RandomState(SEED)
sub_idx = np.array([
    idx for intent in intent_order
    for idx in rng.choice(np.where(df['intent'].values == intent)[0],
                          size=min(1200, int((df['intent'].values == intent).sum())), replace=False)
])
sub_emb = emb[sub_idx]
sub_int = df['intent'].values[sub_idx]
tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=SEED, method='barnes_hut')
tsne_xy = tsne.fit_transform(sub_emb)
for intent in intent_order:
    mask = sub_int == intent
    axes[1].scatter(tsne_xy[mask, 0], tsne_xy[mask, 1], s=4, alpha=0.55,
                    color=color_map[intent], label=intent)
axes[1].set_title('Tweet embeddings (t-SNE, subsample) — colored by intent')
axes[1].legend(markerscale=6, fontsize=8, loc='best')

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[i],
                      markersize=8, label=i) for i in intent_order]
fig.suptitle('Embedding Space: Do the NMF intents look separable?', fontsize=13)
plt.tight_layout()
plt.savefig(OUT_VIZ)
plt.close()
print(f'Saved: {OUT_VIZ}')

# --- Cell: split + train classifier on high-confidence rows ---
print('\nClassification setup:')
y = LabelEncoder()
y.fit(df['intent'])
y_enc = y.transform(df['intent'])
X = emb

X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
    X, y_enc, np.arange(len(df)), test_size=0.15, random_state=SEED, stratify=y_enc)
X_tr, X_val, y_tr, y_val, idx_tr, idx_val = train_test_split(
    X_tr, y_tr, idx_tr, test_size=0.1765, random_state=SEED, stratify=y_tr)

train_conf_mask = ~df['low_confidence'].values[idx_tr]
sup_tr, y_sup_tr = X_tr[train_conf_mask], y_tr[train_conf_mask]
print(f'  train: {len(X_tr)} (high-conf subset used: {len(sup_tr)})')
print(f'  val:   {len(X_val)}')
print(f'  test:  {len(X_te)}')

clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
clf.fit(sup_tr, y_sup_tr)

val_acc = accuracy_score(y_val, clf.predict(X_val))
val_f1 = f1_score(y_val, clf.predict(X_val), average='macro')
print(f'Validation accuracy: {val_acc:.4f}  macro-F1: {val_f1:.4f}')

# --- Cell: evaluate on held-out test (all labels) ---
y_pred = clf.predict(X_te)
print('\nTest performance:')
print(f'  accuracy:  {accuracy_score(y_te, y_pred):.4f}')
print(f'  macro-F1:  {f1_score(y_te, y_pred, average="macro"):.4f}')
print('\nClassification report:')
report = classification_report(y_te, y_pred, target_names=y.classes_, output_dict=True, zero_division=0)
print(classification_report(y_te, y_pred, target_names=y.classes_, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_te, y_pred)
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(len(y.classes_))); ax.set_xticklabels(y.classes_, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(y.classes_))); ax.set_yticklabels(y.classes_, fontsize=8)
for i in range(len(y.classes_)):
    for j in range(len(y.classes_)):
        ax.text(j, i, cm[i, j], ha='center', va='center',
                color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=8)
ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
ax.set_title('Confusion matrix — intent classifier (test set)')
plt.tight_layout()
plt.savefig(OUT_CM)
plt.close()
print(f'Saved: {OUT_CM}')

# --- Cell: compare embeddings vs tf-idf baselines ---
print('\nFeature comparison (same LR, validation set):')
compare = {}
tfidf = TfidfVectorizer(max_features=None, ngram_range=(1, 1), min_df=2, max_df=0.5,
                        lowercase=True, stop_words='english')
X_tr_tf = tfidf.fit_transform([str(t) for t in df['text'].values[idx_tr]])
X_val_tf = tfidf.transform([str(t) for t in df['text'].values[idx_val]])
X_te_tf = tfidf.transform([str(t) for t in df['text'].values[idx_te]])

def lr_acc(Xa, ya, Xb, yb):
    m = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
    m.fit(Xa[train_conf_mask], y_tr[train_conf_mask])
    return accuracy_score(yb, m.predict(Xb))

compare['word2vec_mean'] = {'val_acc': accuracy_score(y_val, clf.predict(X_val)),
                            'test_acc': accuracy_score(y_te, y_pred)}
compare['tfidf'] = {'val_acc': lr_acc(X_tr_tf, y_tr, X_val_tf, y_val),
                    'test_acc': lr_acc(X_tr_tf, y_tr, X_te_tf, y_te)}

from scipy.sparse import hstack
X_tr_mix = hstack([X_tr, X_tr_tf]).tocsr()
X_val_mix = hstack([X_val, X_val_tf]).tocsr()
X_te_mix = hstack([X_te, X_te_tf]).tocsr()
compare['w2v+tfidf'] = {'val_acc': lr_acc(X_tr_mix, y_tr, X_val_mix, y_val),
                        'test_acc': lr_acc(X_tr_mix, y_tr, X_te_mix, y_te)}

for name, scores in compare.items():
    print(f'  {name:14s}  val_acc={scores["val_acc"]:.4f}  test_acc={scores["test_acc"]:.4f}')

# --- Cell: predict all rows + save ---
df['pred_intent'] = y.inverse_transform(clf.predict(X))
df['pred_confidence'] = np.max(clf.predict_proba(X), axis=1)

save_cols = ['tweet_id', 'author_id', 'created_at', 'text',
             'intent', 'low_confidence', 'topic_confidence', 'pred_intent', 'pred_confidence']
df[save_cols].to_csv(OUT_CLASSIFIED, index=False)
print(f'\nSaved: {OUT_CLASSIFIED} ({len(df)} rows)')

import joblib
w2v.save(OUT_W2V)
joblib.dump(clf, OUT_CLF)
joblib.dump(y, OUT_LE)
report_meta = {
    'model': str(clf),
    'features': 'word2vec_mean (vector_size=%d) normalized L2' % VEC_SIZE,
    'training': 'high-confidence silver labels only',
    'val_accuracy': round(val_acc, 4), 'val_macro_f1': round(val_f1, 4),
    'test_accuracy': round(accuracy_score(y_te, y_pred), 4),
    'test_macro_f1': round(f1_score(y_te, y_pred, average='macro'), 4),
    'per_class': {k: v for k, v in report.items() if k != 'accuracy'},
    'feature_comparison': compare,
    'label_mapping': {str(i): c for i, c in enumerate(y.classes_)},
}
with open(OUT_REPORT, 'w') as f:
    json.dump(report_meta, f, indent=2)
print(f'Saved: {OUT_REPORT}')
print(f'Saved: {OUT_W2V}, {OUT_CLF}, {OUT_LE}')

print(f'\nTotal time: {time.time() - t0:.0f}s')