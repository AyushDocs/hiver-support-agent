#!/usr/bin/env python3
"""Local mirror of notebooks/01_preprocessing.ipynb.

Reads the full twcs dataset (data/raw/twcs.csv or Kaggle input), keeps customer
tweets mentioning @AmazonHelp, runs the exact preprocessing pipeline (parallel,
checkpointed) and the strict English + noise quality gate, then writes
data/processed/data_preprocessed.csv with the same 7-column schema the Kaggle
notebook produces (tweet_id, author_id, inbound, created_at, text, ...).

Usage:
    .venv/bin/python local_preprocess.py [--resume] [--save-every 500]

--resume    continue from an existing checkpoint instead of deleting it.
--save-every N   checkpoint interval (default 500, same as the notebook).
"""
import argparse
import os
import sys

import pandas as pd
import numpy as np
import re
import string
import emoji
import multiprocessing
import warnings

from langdetect import detect
warnings.filterwarnings('ignore')

import nltk
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

ROOT = os.path.dirname(os.path.abspath(__file__))
N_JOBS = multiprocessing.cpu_count()


# --- exact preprocess_worker from notebook cell 14 ---------------------------
def preprocess_worker(text):
    """Self-contained worker: imports everything it needs (thread-safe)."""
    from spellchecker import SpellChecker
    from nltk.corpus import wordnet
    from nltk.stem import WordNetLemmatizer
    import nltk, re, string, emoji

    spell = SpellChecker()
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(tag):
        if tag.startswith('J'): return wordnet.ADJ
        elif tag.startswith('V'): return wordnet.VERB
        elif tag.startswith('N'): return wordnet.NOUN
        elif tag.startswith('R'): return wordnet.ADV
        return wordnet.NOUN

    CONTRACTIONS = {
        "ain't": "am not", "aren't": "are not", "can't": "cannot",
        "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
        "don't": "do not", "hadn't": "had not", "hasn't": "has not",
        "haven't": "have not", "he'd": "he would", "he'll": "he will",
        "he's": "he is", "i'd": "i would", "i'll": "i will",
        "i'm": "i am", "i've": "i have", "isn't": "is not",
        "it's": "it is", "let's": "let us", "mustn't": "must not",
        "shan't": "shall not", "she'd": "she would", "she'll": "she will",
        "she's": "she is", "shouldn't": "should not", "that's": "that is",
        "there's": "there is", "they'd": "they would", "they'll": "they will",
        "they're": "they are", "they've": "they have", "wasn't": "was not",
        "we'd": "we would", "we're": "we are", "we've": "we have",
        "weren't": "were not", "what's": "what is", "won't": "will not",
        "wouldn't": "would not", "you'd": "you would", "you'll": "you will",
        "you're": "you are", "you've": "you have", "could've": "could have",
        "should've": "should have", "would've": "would have",
        "y'all": "you all", "ma'am": "madam"
    }

    CHAT_WORDS = {
        "dm": "direct message", "pm": "private message",
        "fyi": "for your information", "asap": "as soon as possible",
        "brb": "be right back", "btw": "by the way",
        "omg": "oh my god", "tbh": "to be honest",
        "smh": "shaking my head", "rn": "right now",
        "pls": "please", "plz": "please", "thx": "thanks",
        "ty": "thank you", "np": "no problem", "acct": "account"
    }

    # Pipeline
    text = re.sub(r'@\w+', '@__sn__', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'^RT @\w+:\s*', '', text)
    text = text.lower()
    for c, e in CONTRACTIONS.items():
        text = re.sub(re.escape(c), e, text, flags=re.IGNORECASE)
    text = emoji.demojize(text, delimiters=(" ", " "))
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)
    text = ' '.join([lemmatizer.lemmatize(w, get_wordnet_pos(t)) for w, t in pos_tags])
    text = ' '.join([CHAT_WORDS.get(w.lower(), w) for w in text.split()])
    return ' '.join(text.split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resume', action='store_true', help='resume from existing checkpoint')
    ap.add_argument('--save-every', type=int, default=500)
    ap.add_argument('--sample-size', type=int, default=50_000)
    ap.add_argument('--brand', default='AmazonHelp')
    args = ap.parse_args()

    from joblib import Parallel, delayed
    from tqdm import tqdm
    import kagglehub

    print(f'All imports successful. {N_JOBS} CPU cores available.')

    # --- notebook cell 6: locate data --------------------------------
    KAGGLE_PATH = '/kaggle/input/datasets/thoughtvector/customer-support-on-twitter/twcs/twcs.csv'
    LOCAL_PATH = os.path.join(ROOT, 'data/raw/twcs.csv')

    if os.path.exists(KAGGLE_PATH):
        DATA_PATH = KAGGLE_PATH
        print(f'Using Kaggle input: {DATA_PATH}')
    elif os.path.exists(LOCAL_PATH):
        DATA_PATH = LOCAL_PATH
        print(f'Using local: {DATA_PATH}')
    else:
        print('Dataset not found locally. Downloading via kagglehub...')
        path = kagglehub.dataset_download('thoughtvector/customer-support-on-twitter')
        DATA_PATH = os.path.join(path, 'twcs', 'twcs.csv')
        print(f'Downloaded to: {DATA_PATH}')

    df = pd.read_csv(DATA_PATH)
    print(f'Dataset shape: {df.shape}')

    # --- notebook cell 8: brand + inbound filter ----------------------
    BRAND = args.brand
    df = df[
        (df['text'].str.contains(f'@{BRAND}', case=False, na=False)) &
        (df['inbound'] == True)
    ].copy()
    print(f'Customer tweets mentioning @{BRAND}: {len(df)}')

    # --- notebook cells 9-10: notna + english heuristic --------------
    df = df[df['text'].notna()].copy()

    def is_english(text):
        if pd.isna(text):
            return False
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        return ascii_chars / max(len(text), 1) > 0.8

    before = len(df)
    df = df[df['text'].apply(is_english)].copy()
    print(f'After ASCII english heuristic: {len(df)} (removed {before - len(df)})')

    # --- notebook cell 18: checkpoint setup + optional sampling ------
    CHECKPOINT_DIR = os.path.join(ROOT, 'data/processed')
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, 'preprocessing_checkpoint.csv')
    SAVE_EVERY = args.save_every

    if len(df) > args.sample_size:
        df = df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)
        print(f'Sampled: {len(df)} rows (of {len(df)} total)')
    else:
        df = df.copy()
        print(f'Using all: {len(df)} rows')

    if args.resume and os.path.exists(CHECKPOINT_FILE):
        checkpoint = pd.read_csv(CHECKPOINT_FILE)
        processed_indices = set(checkpoint['original_index'].tolist())
        print(f'Resumed from checkpoint: {len(processed_indices)} already processed')
    else:
        if os.path.exists(CHECKPOINT_FILE):
            print('Removing stale checkpoint (use --resume to keep).')
            os.remove(CHECKPOINT_FILE)
        processed_indices = set()
        checkpoint = pd.DataFrame(columns=['original_index', 'text_clean'])
        print('No checkpoint found, starting fresh')

    # --- notebook cell 19: parallel processing -----------------------
    texts = df['text'].tolist()
    todo_indices = [i for i in range(len(texts)) if i not in processed_indices]
    print(f'Remaining: {len(todo_indices)} tweets to process')
    print(f'Using {N_JOBS} parallel workers')

    for chunk_start in tqdm(range(0, len(todo_indices), SAVE_EVERY), desc='Chunks'):
        chunk_indices = todo_indices[chunk_start:chunk_start + SAVE_EVERY]
        chunk_texts = [texts[i] for i in chunk_indices]
        results = Parallel(n_jobs=N_JOBS, backend='threading')(
            delayed(preprocess_worker)(text) for text in chunk_texts
        )
        batch = pd.DataFrame({'original_index': chunk_indices, 'text_clean': results})
        checkpoint = pd.concat([checkpoint, batch], ignore_index=True)
        checkpoint.to_csv(CHECKPOINT_FILE, index=False)
    print(f'\nCheckpoint complete: {len(checkpoint)} total rows saved')

    # --- notebook cell 21 + 23: merge and keep cleaned text ----------
    df = df.merge(
        checkpoint[['original_index', 'text_clean']],
        left_index=True,
        right_on='original_index',
        how='left'
    )
    df['text'] = df.iloc[:, 8]
    df = df.iloc[:, :7]

    # --- notebook cell 28: strict English verification ---------------
    df = df[df['text'].str.strip().str.len() > 0]
    df = df[~df['text'].str.contains(r'[^\x00-\x7F]', regex=True, na=False)].copy()

    def detect_language(text):
        try:
            return detect(text)
        except Exception:
            return 'unknown'

    df['lang'] = df['text'].apply(detect_language)
    df = df[df['lang'] == 'en'].copy()
    df = df.drop(columns=['lang'])

    # --- notebook cell 29: noise removal + word filters --------------
    df['text'] = df['text'].str.replace(r'\bsn\b', '', regex=True)
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()

    def remove_noise(text):
        if pd.isna(text):
            return ''
        text = re.sub(r'\b\d{1,2}(?:st|nd|rd|th)\b', '', text)
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'\b\d+[ap]m\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['text'] = df['text'].apply(remove_noise)
    df = df[df['text'].str.split().str.len() >= 2]

    df['word_count'] = df['text'].str.split().str.len()
    df = df[df['word_count'] >= 4].drop(columns=['word_count']).copy()
    df = df[df['text'].str.match(r'[a-zA-Z]', na=False)]

    # --- notebook cell 30: save --------------------------------------
    output_path = os.path.join(CHECKPOINT_DIR, 'data_preprocessed.csv')
    df.to_csv(output_path, index=False)
    print(f'Saved: {output_path} ({len(df)} rows)')
    print(f'Columns: {list(df.columns)}')


if __name__ == '__main__':
    main()