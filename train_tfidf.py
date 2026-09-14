#!/usr/bin/env python3
"""Train an optional TF-IDF + Logistic Regression intent classifier.

Mirrors local_classify.py's exact split/high-confidence-training protocol but
replaces the word2vec features with TF-IDF.  Measured on the same split it
outperforms word2vec (test acc ~0.924 vs ~0.767), so it is shipped as a
drop-in alternative engine for agent.py (`--engine tfidf`).

Input : data/processed/amazon_help_classified.csv
Output: data/models/tfidf_vectorizer.joblib
        data/models/tfidf_classifier.joblib
        data/processed/classification_report_tfidf.json
"""
import json
import os
import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import joblib

ROOT = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(ROOT, 'data/processed')
MODELS = os.path.join(ROOT, 'data/models')
SEED = 42

t0 = time.time()
df = pd.read_csv(os.path.join(PROC, 'amazon_help_classified.csv'), dtype=str)
print(f'Rows: {len(df)}')

y = LabelEncoder()
y.fit(df['intent'])
y_enc = y.transform(df['intent'])

texts = np.asarray([str(t) for t in df['text'].values])
idx_all = np.arange(len(df))

# identical split to local_classify.py (03)
idx_tr, idx_te, y_tr, y_te = train_test_split(
    idx_all, y_enc, test_size=0.15, random_state=SEED, stratify=y_enc)
idx_tr, idx_val, y_tr, y_val = train_test_split(
    idx_tr, y_tr, test_size=0.1765, random_state=SEED, stratify=y_tr)

low_flag = (df['low_confidence'].astype(str).str.lower() == 'true')
train_conf_mask = ~low_flag.values[idx_tr]

tfidf = TfidfVectorizer(max_features=None, ngram_range=(1, 1), min_df=2,
                        max_df=0.5, lowercase=True, stop_words='english')
print('Fitting TF-IDF on training text...')
X_tr_tf = tfidf.fit_transform(texts[idx_tr])
X_val_tf = tfidf.transform(texts[idx_val])
X_te_tf = tfidf.transform(texts[idx_te])
print(f'  vocab: {len(tfidf.vocabulary_):,} | train: {X_tr_tf.shape}')

clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED)
clf.fit(X_tr_tf[train_conf_mask], y_tr[train_conf_mask])

val_acc = accuracy_score(y_val, clf.predict(X_val_tf))
val_f1 = f1_score(y_val, clf.predict(X_val_tf), average='macro')
y_pred = clf.predict(X_te_tf)
test_acc = accuracy_score(y_te, y_pred)
test_f1 = f1_score(y_te, y_pred, average='macro')
print(f'Validation accuracy: {val_acc:.4f}  macro-F1: {val_f1:.4f}')
print(f'Test accuracy: {test_acc:.4f}  macro-F1: {test_f1:.4f}')
print(classification_report(y_te, y_pred, target_names=y.classes_, zero_division=0))

os.makedirs(MODELS, exist_ok=True)
joblib.dump(tfidf, os.path.join(MODELS, 'tfidf_vectorizer.joblib'))
joblib.dump(clf, os.path.join(MODELS, 'tfidf_classifier.joblib'))
joblib.dump(y, os.path.join(MODELS, 'label_encoder_tfidf.joblib'))

report = {
    'model': str(clf),
    'features': 'TfidfVectorizer(min_df=2, max_df=0.5, stop_words=english, unigrams)',
    'training': 'high-confidence silver labels only (mirror of 03)',
    'val_accuracy': float(val_acc), 'val_macro_f1': float(val_f1),
    'test_accuracy': float(test_acc), 'test_macro_f1': float(test_f1),
    'per_class': {k: {mk: round(mv, 4) for mk, mv in v.items()}
                  for k, v in classification_report(
                      y_te, y_pred, target_names=y.classes_, output_dict=True,
                      zero_division=0).items()
                  if k not in ('accuracy', 'macro avg', 'weighted avg')},
    'word2vec_test_accuracy': 0.7672,  # from classification_report.json (03)
}
with open(os.path.join(PROC, 'classification_report_tfidf.json'), 'w') as f:
    json.dump(report, f, indent=2)
print(f'Saved: data/models/tfidf_*.joblib + classification_report_tfidf.json '
      f'in {time.time()-t0:.0f}s')