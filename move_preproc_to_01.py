#!/usr/bin/env python3
"""Move the preprocessing steps (language filter + noise removal) that lived in
notebook 02 into notebook 01, aligning all four notebooks. Mirrors: user reruns
01 on Kaggle, then 02/03 consume the already-filtered clean_text directly.
"""
import json, re, sys

BASE = '/home/ayush/Desktop/code/hiver/notebooks'
F02 = f'{BASE}/02_intent_discovery.ipynb'
RUN = f'{BASE}/notebook49929ade53.ipynb'
F01A = f'{BASE}/01_preprocessing.ipynb'
F01B = f'{BASE}/01_preprocessing_kaggle.ipynb'


def to_str(cell):
    src = cell['source']
    return src if isinstance(src, str) else ''.join(src)


def set_src(cell, s):
    cell['source'] = [l + '\n' for l in s.splitlines()]


def is_md(cell):
    return cell['cell_type'] == 'markdown'


def first(cell):
    s = to_str(cell).lstrip('\n')
    return s.split('\n', 1)[0].strip()


def flip_imports(source):
    """Drop lines we no longer need (re / langdetect) from an import block."""
    out = []
    for line in source.splitlines():
        t = line.strip()
        if t == 'import re':
            continue
        if t.startswith('from langdetect') and 'import detect' in line:
            continue
        out.append(line)
    # collapse 3+ blank lines to 1
    res, blanks = [], 0
    for l in out:
        if l.strip() == '':
            blanks += 1
            if blanks <= 1:
                res.append(l)
        else:
            blanks = 0
            res.append(l)
    return '\n'.join(res)


def renumber_md(cells):
    n = 0
    for c in cells:
        if not is_md(c):
            continue
        first_line = first(c)
        m = re.match(r'^## \d+\.\s*(.*)$', first_line)
        if m:
            n += 1
            new_header = f'## {n}. {m.group(1)}'
            rest = to_str(c).split('\n', 1)
            if len(rest) == 2:
                set_src(c, new_header + '\n' + rest[1])
            else:
                set_src(c, new_header)
    return n


def compile_check(path, label):
    nb = json.load(open(path))
    bad = 0
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        src = '\n'.join(l for l in to_str(c).split('\n')
                        if not l.strip().startswith('!') and not l.strip().startswith('%'))
        try:
            compile(src, f'<{label} cell {i}>', 'exec')
        except SyntaxError as e:
            bad += 1
            print(f'  SYNTAX ERROR {label} cell {i}: {e}')
    print(f'{path.split("/")[-1]}: compile {"OK" if bad == 0 else f"{bad} FAILED"}')
    return nb


def dump(path, label):
    nb = compile_check(path, label)
    print(f'  {path.split("/")[-1]} -> {len(nb["cells"])} cells')
    for i, c in enumerate(nb['cells']):
        print(f'    [{i:02d}] {c["cell_type"][:4]}{" "*(4-len(c["cell_type"][:4]))} :: {first(c)[:95]}')
    if not ('markdown' in 'check'):
        pass
    return nb


# ============================================================= Part A: 02 family
def edit_02_template(path):
    nb = json.load(open(path))
    cells = nb['cells']

    # 1) remove language-filter code cell + its markdown
    cells = [c for c in cells if not (c['cell_type'] == 'code' and 'Vectorized language filtering' in to_str(c))]
    cells = [c for c in cells if not (is_md(c) and first(c) == '## 4. Filter for English Only')]
    # 2) remove remove_noise code cell + its markdown
    cells = [c for c in cells if not (c['cell_type'] == 'code' and 'def remove_noise(' in to_str(c))]
    cells = [c for c in cells if not (is_md(c) and '## 5. Clean Text' in first(c))]
    nb['cells'] = cells

    # 3) tidy imports
    for c in cells:
        if c['cell_type'] == 'code' and 'from langdetect import detect' in to_str(c):
            set_src(c, flip_imports(to_str(c)))
            break
    # 4) update intro steps
    for c in cells:
        if is_md(c) and '**Steps:**' in to_str(c):
            set_src(c, INTRO_02)
            break
    renumber_md(cells)
    json.dump(nb, open(path, 'w'), indent=1)
    return nb


def edit_runner(path):
    nb = json.load(open(path))
    cells = nb['cells']

    # remove filter code cell (46 lines) + its markdown (comes after code here)
    cells = [c for c in cells if not (c['cell_type'] == 'code' and 'Vectorized language filtering' in to_str(c))]
    cells = [c for c in cells if not (is_md(c) and first(c) == '## 4. Filter for English Only')]

    # cell 9: keep sentiment mapping only (drop remove_noise def + apply + re/pd imports)
    for c in cells:
        if c['cell_type'] == 'code' and 'detect_sentiment' in to_str(c):
            src = to_str(c)
            head = "import pandas as pd\nimport re\n# Clean text: remove numbers/date patterns\n"
            body = '# Sentiment keyword mapping\n'
            idx = src.index('# Sentiment keyword mapping')
            body += src[idx:]
            set_src(c, body)
            nb['cells'] = cells
            break

    # tidy imports (re + langdetect no longer used)
    for c in cells:
        if c['cell_type'] == 'code' and 'from langdetect import detect' in to_str(c):
            set_src(c, flip_imports(to_str(c)))
            break
    # update intro
    for c in cells:
        if is_md(c) and '**Steps:**' in to_str(c):
            set_src(c, INTRO_02)
            break
    renumber_md(cells)
    json.dump(nb, open(path, 'w'), indent=1)
    return nb


INTRO_02 = """# Intent Discovery — AmazonHelp

NMF topic modeling to discover customer support intents from Amazon tweets.

**Steps:**
1. Load cleaned data (already filtered/denoised by notebook 01 — English-verified, numbers/dates removed, `clean_text`)
2. TF-IDF vectorization
3. NMF topic modeling with auto-k
4. Label topics manually
5. Assign intent labels & confidence
6. Save labeled dataset"""

# ============================================================= Part B: 01 family

FILTER_BLOCK = '''# --- Strict English verification + noise removal (moved from Intent Discovery) ---
# Tier 1 (vectorized): drop any rows with non-ASCII characters (handles/emojis already stripped)
before_tier1 = len(df_final)
df_final = df_final[df_final['text_clean'].notna() & (df_final['text_clean'].str.strip().str.len() > 0)].copy()
df_final = df_final[~df_final['text_clean'].str.contains(r'[^\\x00-\\x7F]', regex=True, na=False)].copy()
print(f'After non-ASCII filter: {len(df_final)} rows (removed {before_tier1 - len(df_final)})')

# Tier 2 (precise): langdetect confirms English on the remaining rows
def detect_language(text):
    try:
        return detect(text)
    except:
        return 'unknown'

before_tier2 = len(df_final)
df_final['lang'] = df_final['text_clean'].apply(detect_language)
df_final = df_final[df_final['lang'] == 'en'].copy()
df_final = df_final.drop(columns=['lang'])
print(f'After langdetect English filter: {len(df_final)} rows (removed {before_tier2 - len(df_final)})')

# Remove 'sn' placeholder tokens and collapse whitespace
df_final['text_clean'] = df_final['text_clean'].str.replace(r'\\bsn\\b', '', regex=True)
df_final['text_clean'] = df_final['text_clean'].str.replace(r'\\s+', ' ', regex=True).str.strip()

# Remove noise: numbers, date suffixes, and time patterns
def remove_noise(text):
    if pd.isna(text):
        return ''
    text = re.sub(r'\\b\\d{1,2}(?:st|nd|rd|th)\\b', '', text)
    text = re.sub(r'\\b\\d+\\b', '', text)
    text = re.sub(r'\\b\\d+[ap]m\\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\\s+', ' ', text).strip()
    return text

df_final['text_clean'] = df_final['text_clean'].apply(remove_noise)
df_final = df_final[df_final['text_clean'].str.split().str.len() >= 2]

# Keep tweets with >= 4 words and a leading alphabetic character
df_final['word_count'] = df_final['text_clean'].str.split().str.len()
df_final = df_final[df_final['word_count'] >= 4].drop(columns=['word_count']).copy()
df_final = df_final[df_final['text_clean'].str.match(r'[a-zA-Z]', na=False)]

# Canonical column name for downstream notebooks (02, 03, 04)
df_final = df_final.rename(columns={'text_clean': 'clean_text'})

print(f'Final dataset: {len(df_final)} tweets')
print(f"Unique authors: {df_final['author_id'].nunique()}")'''

FILTER_MD = '## 10. Strict English Verification + Noise Removal (moved from Intent Discovery)'


def edit_01(path):
    nb = json.load(open(path))
    cells = nb['cells']

    # 1) add langdetect imports to import cell
    for c in cells:
        if c['cell_type'] == 'code' and 'import warnings' in to_str(c):
            src = to_str(c)
            src = src.replace('import warnings\n', 'import warnings\n\nfrom langdetect import detect\nfrom langdetect.lang_detect_exception import LangDetectException\n', 1)
            set_src(c, src)
            break

    # 2) insert new section before the Quick Stats markdown
    insert_at = None
    for i, c in enumerate(cells):
        if is_md(c) and first(c).startswith('## 10. Quick Stats'):
            insert_at = i
            break
    md_cell = {'cell_type': 'markdown', 'metadata': {}, 'source': FILTER_MD + '\n'}
    code_cell = {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
                 'source': [l + '\n' for l in FILTER_BLOCK.splitlines()]}
    cells[insert_at:insert_at] = [md_cell, code_cell]

    # 3) Quick Stats cell -> clean_text
    for c in cells:
        if c['cell_type'] == 'code' and "'text_clean'" in to_str(c) and 'orig_len' in to_str(c):
            set_src(c, to_str(c).replace("'text_clean'", "'clean_text'"))
            break

    # 4) Save cell -> clean_text
    for c in cells:
        if c['cell_type'] == 'code' and 'amazon_help_cleaned.csv' in to_str(c):
            set_src(c, to_str(c).replace("'text_clean'", "'clean_text'"))
            break

    # 5) Summary filters line
    for c in cells:
        if is_md(c) and '**Filters:**' in to_str(c):
            set_src(c, to_str(c).replace(
                '**Filters:** `@AmazonHelp` customer tweets, English only, inbound',
                '**Filters:** `@AmazonHelp` customer tweets, ASCII + langdetect English check, inbound, >4 words, alphabetic start, numbers/dates/times removed'))
            break

    renumber_md(cells)
    json.dump(nb, open(path, 'w'), indent=1)
    return nb


if __name__ == '__main__':
    print('### Part A — 02 family (slim to discovery)')
    edit_02_template(F02)
    dump(F02, '02_template')
    print()
    edit_runner(RUN)
    dump(RUN, 'runner')
    print()
    print('### Part B — 01 family (absorbed preprocessing)')
    edit_01(F01A)
    dump(F01A, '01_base')
    print()
    edit_01(F01B)
    dump(F01B, '01_kaggle')
    print('DONE')