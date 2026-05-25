"""
Evaluate generation accuracy on the fixed 80-sample val split.

Usage:
    uv run val.py
    uv run val.py --checkpoint checkpoints/dpo/best
    uv run val.py --compare checkpoints/sft/best checkpoints/dpo/best
    uv run val.py --mode dpo --log val_dpo.log

All printed output (including tqdm progress bars) is saved to --log (default: val.log).
"""

import argparse
import re
import sys
import time
from datetime import datetime
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
DEFAULT_LOG_PATH = "val.log"
DISPLAY_TRUNC = 200


class _TeeStream:
  """Write to terminal stream and shared log file."""

  def __init__(self, terminal, log_file):
    self.terminal = terminal
    self.log = log_file

  def write(self, message):
    self.log.write(message)
    try:
      self.terminal.write(message)
    except UnicodeEncodeError:
      enc = getattr(self.terminal, "encoding", None) or "utf-8"
      self.terminal.write(message.encode(enc, errors="replace").decode(enc))

  def flush(self):
    self.terminal.flush()
    self.log.flush()

  def isatty(self):
    return self.terminal.isatty()


class TeeLogger:
  """Mirror stdout and stderr to a log file."""

  def __init__(self, filepath):
    self.path = Path(filepath)
    self.terminal_out = sys.stdout
    self.terminal_err = sys.stderr
    self.log = open(self.path, "w", encoding="utf-8")
    self.log.write(f"# val.py log started {datetime.now().isoformat(timespec='seconds')}\n")
    self.log.flush()

  def install(self):
    sys.stdout = _TeeStream(self.terminal_out, self.log)
    sys.stderr = _TeeStream(self.terminal_err, self.log)

  def restore(self):
    sys.stdout = self.terminal_out
    sys.stderr = self.terminal_err

  def close(self):
    self.log.write(f"\n# val.py log ended {datetime.now().isoformat(timespec='seconds')}\n")
    self.log.flush()
    self.log.close()


def extract_query(record):
  """User-facing query text for error display."""
  inp = (record.get("input") or "").strip()
  if inp.startswith("Query:"):
    query_part, _, _ = inp.partition("\n\nKnowledge:")
    return query_part.strip()
  if inp:
    return inp
  return build_user_content(record)[:DISPLAY_TRUNC]


def truncate(text, limit=DISPLAY_TRUNC):
  text = (text or "").replace("\n", " ").strip()
  if len(text) <= limit:
    return text
  return text[: limit - 3] + "..."


def print_errors(errors, limit=None, title="errors"):
  if not errors:
    return
  shown = errors if limit is None else errors[:limit]
  total = len(errors)
  header = f"--- {title} ({len(shown)}/{total}) ---"
  print(header)
  for i, r in enumerate(shown, 1):
    print(f"[{i}] task={r['task']}")
    print(f"    query:    {truncate(r.get('query', ''))}")
    print(f"    expected: {truncate(r['expected'])}")
    print(f"    got:      {truncate(r['pred'])}")
    if r.get("degenerate"):
      print("    (degenerate output)")
  print(f"--- end {title} ---")


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
        "query": extract_query(record),
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
  if errors:
    limit = None if show_errors <= 0 else show_errors
    print_errors(errors, limit=limit, title="errors")

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
  parser.add_argument(
    "--show-errors",
    type=int,
    default=0,
    help="Print N wrong examples (0 = print all errors, default: 0)",
  )
  parser.add_argument(
    "--log",
    default=DEFAULT_LOG_PATH,
    help=f"Save all printed output to this file (default: {DEFAULT_LOG_PATH})",
  )
  parser.add_argument(
    "--no-log",
    action="store_true",
    help="Do not write a log file",
  )
  args = parser.parse_args()

  tee = None
  if not args.no_log:
    tee = TeeLogger(args.log)
    tee.install()

  try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_records = get_splits(args.mode)
    print(f"Val split ({args.mode}): {len(val_records)} samples {split_stats(val_records)}")
    print(f"Device: {device}")
    if tee:
      print(f"Log file: {Path(args.log).resolve()}")
    print()

    default_ckpt = DEFAULT_CHECKPOINTS[args.mode]
    checkpoints = args.compare if args.compare else [args.checkpoint or default_ckpt]
    for ckpt in checkpoints:
      evaluate_checkpoint(ckpt, device, val_records, show_errors=args.show_errors)
      print()

    if tee:
      print(f"Log saved to: {Path(args.log).resolve()}")
  finally:
    if tee:
      tee.restore()
      tee.close()


if __name__ == "__main__":
  main()
