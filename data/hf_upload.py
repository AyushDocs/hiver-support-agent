"""Publish the derived dataset + trained engines to the Hugging Face Hub.

Usage:
    HF_TOKEN=... python data/hf_upload.py

Creates (or updates):
  - dataset: <user>/hiver-support-golden   (golden labels, eval harness outputs)
  - model:   <user>/hiver-support-agent    (TF-IDF + w2v engines, reply index)
"""
import json
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "data" / "models"

DATASET_README = """\
# Hiver AmazonHelp Support — Golden Eval Sets

Human-checkable labels + evaluation artifacts for the `hiver-support-agent`
(see [GitHub](https://github.com/AyushDocs/hiver-support-agent)).

## Files
- `golden_eval.csv` — N=200, **auto-labeled** (silver topic intent + routing
  rule). Stratified by intent, seed 42. `gold_intent`/`gold_route` columns are
  designed to be flipped by a human, then `eval.py` re-run.
- `golden_hand.csv` — N=50, **human-labeled** (`hand_intent`/`hand_route`),
  labelled by a reviewer reading tweet + author history. The only label set
  independent of the pipeline.
- `hand_labels.py` — the reviewer's evidence/audit trail.
- `eval_report_{w2v,tfidf}.json`, `eval_predictions_{w2v,tfidf}.csv`,
  `eval_judge_results.csv`, `routing_report.json` — harness outputs.

## Honest findings
| Comparison | Intent acc | Route acc |
|---|---:|---:|
| Silver-vs-human | 0.52 | 0.44 |
| TF-IDF agent vs human | 0.625 | 0.52 |
| LLM judge vs human | 0.46–0.52 | κ ≈ −0.06 |

On the auto-gold the TF-IDF agent scores intent 0.930 / route 0.975 — treat as
an **upper bound** (auto-labels are produced by the same pipeline; see REPORT §7).
"""

MODEL_README = """\
# hiver-support-agent — models & reply index

Pretrained engines + retrieval index for the AmazonHelp support agent
(examples: out-of-delivery, order status, appreciation).

## Contents
- `tfidf_vectorizer.joblib`, `tfidf_classifier.joblib`,
  `label_encoder_tfidf.joblib` — **default engine**, deterministic.
- `word2vec.model`, `intent_classifier.joblib`, `label_encoder.joblib` —
  legacy engine (not bit-reproducible across retrains).
- `amazon_replies.csv`, `author_history.csv` — retrieval evidence store.

## Use
```bash
pip install hiver-support-agent    # or: pip install -e . ; hiver
python agent.py "my order never arrived" --author-id 172791877
```

| Engine | Intent acc (auto-gold) | Intent acc (hand-gold) |
|---|---:|---:|
| TF-IDF | 0.930 | 0.625 |
| w2v | 0.775 | 0.583 |

Run `eval.py --judge-n 30` to reproduce the numbers.
"""


def main() -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    ds_id, model_id = f"{user}/hiver-support-golden", f"{user}/hiver-support-agent"
    print(f"user={user}")

    api.create_repo(ds_id, repo_type="dataset", exist_ok=True)
    ds_files = {
        "README.md": DATASET_README,
        "golden_eval.csv": PROC / "golden_eval.csv",
        "golden_hand.csv": PROC / "golden_hand.csv",
        "hand_labels.py": ROOT / "hand_labels.py",
        "build_golden.py": ROOT / "build_golden.py",
        "build_golden_hand.py": ROOT / "build_golden_hand.py",
        "eval_report_w2v.json": PROC / "eval_report_w2v.json",
        "eval_report_tfidf.json": PROC / "eval_report_tfidf.json",
        "eval_predictions_w2v.csv": PROC / "eval_predictions_w2v.csv",
        "eval_predictions_tfidf.csv": PROC / "eval_predictions_tfidf.csv",
        "eval_judge_results.csv": PROC / "eval_judge_results.csv",
        "routing_report.json": PROC / "routing_report.json",
    }
    api.upload_folder(
        repo_id=ds_id,
        repo_type="dataset",
        folder_path=str(PROC),
        allow_patterns=[Path(p).name for p in ds_files.values() if isinstance(p, Path)],
        delete_patterns=[".*", "!README.md",
                         "!golden_eval.csv", "!golden_hand.csv",
                         "!hand_labels.py", "!build_golden.py", "!build_golden_hand.py",
                         "!eval_report_w2v.json", "!eval_report_tfidf.json",
                         "!eval_predictions_w2v.csv", "!eval_predictions_tfidf.csv",
                         "!eval_judge_results.csv", "!routing_report.json"],
    )
    for name, path in ds_files.items():
        if isinstance(path, Path):
            api.upload_file(repo_id=ds_id, repo_type="dataset", path_or_fileobj=str(path), path_in_repo=name)
    for name, content in ds_files.items():
        if isinstance(content, str):
            api.upload_file(repo_id=ds_id, repo_type="dataset", path_or_fileobj=content.encode(), path_in_repo=name)
    print(f"dataset  -> https://huggingface.co/datasets/{ds_id}")

    api.create_repo(model_id, repo_type="model", exist_ok=True)
    model_files = [
        MODELS / "tfidf_vectorizer.joblib",
        MODELS / "tfidf_classifier.joblib",
        MODELS / "label_encoder_tfidf.joblib",
        MODELS / "word2vec.model",
        MODELS / "intent_classifier.joblib",
        MODELS / "label_encoder.joblib",
        PROC / "amazon_replies.csv",
        PROC / "author_history.csv",
    ]
    for f in model_files:
        api.upload_file(repo_id=model_id, repo_type="model", path_or_fileobj=str(f), path_in_repo=f.name)
    api.upload_file(repo_id=model_id, repo_type="model", path_or_fileobj=MODEL_README.encode(), path_in_repo="README.md")
    print(f"model     -> https://huggingface.co/{model_id}")


if __name__ == "__main__":
    main()