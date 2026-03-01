from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset JSON must be a list of objects")
    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_unique_pairs(text: str) -> list[tuple[str, str]]:
    pairs = re.findall(
        r"\[Classification:\s*([^\]]+)\]\s*\[Priority:\s*([^\]]+)\]",
        text,
        flags=re.IGNORECASE,
    )
    seen: set[tuple[str, str]] = set()
    unique_pairs: list[tuple[str, str]] = []
    for cls, pri in pairs:
        pair = (cls.strip(), pri.strip())
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)
    return unique_pairs


def normalize_classification(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_tactic(instruction: str) -> str | None:
    match = re.search(r"MITRE ATT&CK tactic:\s*([^\n\r\.]+)", instruction)
    if not match:
        return None
    return match.group(1).strip()


def compute_precision_recall(pred_text: str, label_text: str) -> tuple[float, float]:
    pred_pairs = extract_unique_pairs(pred_text)
    label_pairs = extract_unique_pairs(label_text)

    pred_set = {
        (normalize_classification(cls), pri)
        for cls, pri in pred_pairs
        if normalize_classification(cls)
    }
    label_set = {
        (normalize_classification(cls), pri)
        for cls, pri in label_pairs
        if normalize_classification(cls)
    }

    if not pred_set or not label_set:
        return 0.0, 0.0
    overlap = len(pred_set & label_set)
    precision = overlap / len(pred_set) if pred_set else 0.0
    recall = overlap / len(label_set) if label_set else 0.0
    return precision, recall


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute precision/recall for top-5 tactics, sampling up to 100 items per tactic."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path or HF id for the model checkpoint",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to transformed_dataset_cls_pri_all2preprocessing.json",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--per-tactic", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_path = Path(args.dataset)
    data = load_dataset(dataset_path)

    tactic_counts: Counter[str] = Counter()
    for item in data:
        instruction = item.get("instruction", "")
        tactic = extract_tactic(instruction)
        if tactic:
            tactic_counts[tactic] += 1

    top5 = tactic_counts.most_common(5)
    print("Top 5 tactics by frequency:")
    for tactic, count in top5:
        print(f"- {tactic}: {count}")

    adapter_config = Path(args.model) / "adapter_config.json"
    if adapter_config.exists():
        with adapter_config.open("r", encoding="utf-8") as f:
            adapter_meta = json.load(f)
        base_model = adapter_meta.get("base_model_name_or_path")
        if not base_model:
            raise ValueError("adapter_config.json missing base_model_name_or_path")
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    else:
        base_model = None
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if base_model:
        base = AutoModelForCausalLM.from_pretrained(
            base_model, device_map="auto", torch_dtype=torch.bfloat16
        )
        model = PeftModel.from_pretrained(base, args.model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, device_map="auto", torch_dtype=torch.bfloat16
        )
    model.eval()

    for tactic, _count in top5:
        candidates = [
            item
            for item in data
            if extract_tactic(item.get("instruction", "")) == tactic
        ]
        if not candidates:
            print(f"\nTactic: {tactic}")
            print("  No samples found.")
            continue

        sample_n = min(args.per_tactic, len(candidates))
        sample = random.sample(candidates, k=sample_n)

        precisions: list[float] = []
        recalls: list[float] = []

        for idx, item in enumerate(sample, start=1):
            if idx % 10 == 0 or idx == 1 or idx == sample_n:
                print(f"  Progress: {idx}/{sample_n}")
            prompt = item.get("instruction", "")
            label = item.get("output", "")
            if not prompt or not label:
                continue

            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0.0,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if pred.startswith(prompt):
                pred = pred[len(prompt) :]

            precision, recall = compute_precision_recall(pred, label)
            precisions.append(precision)
            recalls.append(recall)

        avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
        avg_recall = sum(recalls) / len(recalls) if recalls else 0.0

        print(f"\nTactic: {tactic}")
        print(f"  Samples: {sample_n}")
        print(f"  Average precision (unique pairs): {avg_precision:.4f}")
        print(f"  Average recall (unique pairs): {avg_recall:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
