"""
Evaluate generation accuracy on the fixed 80-sample val split.

Usage:
    uv run val.py
    uv run val.py --checkpoint checkpoints/dpo/best
    uv run val.py --compare checkpoints/sft/best checkpoints/dpo/best
"""

import argparse
import re
import time
from pathlib import Path

import torch
from peft import PeftModel

from prepare import (
    build_user_content,
    get_splits,
    load_base_model,
    load_tokenizer,
    normalize_output,
    split_stats,
)

MODES = ("sft", "dpo")

DEFAULT_CHECKPOINTS = {
    "sft": "checkpoints/sft/best",
    "dpo": "checkpoints/dpo/best",
}
MAX_NEW_TOKENS = 128
DEGENERATE_RE = re.compile(r"^(.)\1{7,}$")  # same char repeated 8+ times


def parse_tool_call(pred):
    """Extract YES/NO from model output."""
    text = normalize_output(pred, "tool_call").upper()
    if text in ("YES", "NO"):
        return text
    first_line = text.split("\n", 1)[0]
    for token in re.split(r"[^A-Z]+", first_line):
        if token in ("YES", "NO"):
            return token
    return first_line.strip() or text


def split_tool_body(text):
    text = (text or "").strip()
    if not text.startswith("<|tool|>"):
        return "", text
    tool, _, body = text.partition("\n")
    return tool.strip(), body.strip()


def is_degenerate(text):
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 8:
        return False
    return bool(DEGENERATE_RE.match(compact))


def generate_one(model, tokenizer, record, device):
    user_messages = [{"role": "user", "content": build_user_content(record)}]
    prompt = tokenizer.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def score_record(record, pred):
    task = record.get("task")
    expected = normalize_output(record["chosen"], task)
    pred_norm = normalize_output(pred, task)
    degenerate = is_degenerate(pred)

    if task == "tool_call":
        pred_parsed = parse_tool_call(pred)
        exact = pred_parsed == expected
        return {
            "exact": exact,
            "tool_prefix": exact,
            "body_exact": exact,
            "degenerate": degenerate,
            "pred_display": pred_parsed,
        }

    exp_tool, exp_body = split_tool_body(expected)
    pred_tool, pred_body = split_tool_body(pred_norm)
    exact = pred_norm == expected
    return {
        "exact": exact,
        "tool_prefix": pred_tool == exp_tool and bool(exp_tool),
        "body_exact": pred_body == exp_body,
        "degenerate": degenerate,
        "pred_display": pred_norm[:120],
    }


def summarize(name, results):
    n = len(results)
    if n == 0:
        return

    def rate(key):
        return sum(r[key] for r in results) / n

    by_task = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)

    print(f"=== {name} ({n} samples) ===")
    print(f"  exact:        {rate('exact'):.1%} ({sum(r['exact'] for r in results)}/{n})")
    print(f"  tool_prefix:  {rate('tool_prefix'):.1%}")
    print(f"  body_exact:   {rate('body_exact'):.1%}")
    print(f"  degenerate:   {rate('degenerate'):.1%} ({sum(r['degenerate'] for r in results)} bad)")
    for task, task_results in sorted(by_task.items()):
        tn = len(task_results)
        te = sum(r["exact"] for r in task_results)
        print(f"  [{task}] exact: {te / tn:.1%} ({te}/{tn})")
    print()


def load_checkpoint(checkpoint_path, device):
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    model = load_base_model(device=device)
    model = PeftModel.from_pretrained(model, path)
    model.eval()
    return model


@torch.no_grad()
def evaluate_checkpoint(checkpoint_path, device, val_records, show_errors=5):
    tokenizer = load_tokenizer()
    model = load_checkpoint(checkpoint_path, device)

    results = []
    t0 = time.time()
    for i, record in enumerate(val_records):
        pred = generate_one(model, tokenizer, record, device)
        scores = score_record(record, pred)
        results.append(
            {
                "task": record.get("task", "unknown"),
                "expected": normalize_output(record["chosen"], record.get("task")),
                "pred": pred,
                **scores,
            }
        )
        if (i + 1) % 20 == 0:
            print(f"  generated {i + 1}/{len(val_records)}...", flush=True)

    elapsed = time.time() - t0
    summarize(checkpoint_path, results)

    errors = [r for r in results if not r["exact"]]
    if show_errors and errors:
        print(f"--- first {min(show_errors, len(errors))} errors ---")
        for r in errors[:show_errors]:
            print(f"[{r['task']}] expected: {r['expected'][:100]}")
            print(f"         got:      {r['pred'][:100]}")
            if r["degenerate"]:
                print("         (degenerate output)")
        print("--- end errors ---")

    print(f"eval_seconds: {elapsed:.1f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Val-set generation accuracy")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="dpo",
        help="Val split to use: sft (tool_call) or dpo (groundedness)",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="LoRA checkpoint path (default: checkpoints/<mode>/best)",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="PATH",
        help="Compare multiple checkpoints (e.g. sft/best dpo/best)",
    )
    parser.add_argument("--show-errors", type=int, default=5, help="Print N wrong examples")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_records = get_splits(args.mode)
    print(f"Val split ({args.mode}): {len(val_records)} samples {split_stats(val_records)}")
    print(f"Device: {device}")
    print()

    default_ckpt = DEFAULT_CHECKPOINTS[args.mode]
    checkpoints = args.compare if args.compare else [args.checkpoint or default_ckpt]
    for ckpt in checkpoints:
        evaluate_checkpoint(ckpt, device, val_records, show_errors=args.show_errors)
        print()


if __name__ == "__main__":
    main()
