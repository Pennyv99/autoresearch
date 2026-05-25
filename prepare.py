"""
Fixed data prep and runtime utilities for Qwen SFT/DPO autoresearch.

Usage:
    uv run prepare.py          # verify data + cache model/tokenizer
"""

import json
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 512
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

REPO_ROOT = Path(__file__).resolve().parent
DATA_PATH = REPO_ROOT / "data" / "dpo.jsonl"
DATA_MARKER = REPO_ROOT / "data" / ".dataset_ready"
SAMPLE_400_PATH = REPO_ROOT / "data" / "sample_400.jsonl"
TRAIN_PATH = REPO_ROOT / "data" / "train_320.jsonl"
VAL_PATH = REPO_ROOT / "data" / "val_80.jsonl"

TOTAL_SAMPLES = 400
SAMPLE_SEED = 42
YES_NO_RATIO = 0.30
TRAIN_RATIO = 0.80

CACHE_DIR = Path(os.path.expanduser("~")) / ".cache" / "autoresearch"
SPLIT_CACHE = CACHE_DIR / "split_400.json"

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def normalize_output(text, task):
    text = (text or "").strip()
    if task == "tool_call":
        upper = text.upper()
        if upper in ("YES", "NO"):
            return upper
    return text


def build_user_content(record):
    instruction = (record.get("instruction") or "").strip()
    inp = (record.get("input") or "").strip()
    if instruction and inp:
        return f"{instruction}\n\n{inp}"
    return instruction or inp


def build_messages(record, assistant_content=None):
    if record.get("messages"):
        messages = [{"role": m["role"], "content": m["content"]} for m in record["messages"]]
        if assistant_content is not None:
            messages = messages[:-1] + [{"role": "assistant", "content": assistant_content}]
        elif messages[-1]["role"] == "assistant":
            messages[-1]["content"] = normalize_output(messages[-1]["content"], record.get("task"))
        return messages

    user_content = build_user_content(record)
    assistant = assistant_content if assistant_content is not None else record["chosen"]
    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": normalize_output(assistant, record.get("task"))},
    ]


def load_all_records():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    records = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def stratified_sample(records):
    tool_call = [r for r in records if r.get("task") == "tool_call"]
    groundedness = [r for r in records if r.get("task") == "groundedness"]
    n_tool = int(round(TOTAL_SAMPLES * YES_NO_RATIO))
    n_ground = TOTAL_SAMPLES - n_tool
    if len(tool_call) < n_tool or len(groundedness) < n_ground:
        raise ValueError(
            f"Not enough samples: need {n_tool} tool_call and {n_ground} groundedness, "
            f"have {len(tool_call)} and {len(groundedness)}"
        )
    rng = random.Random(SAMPLE_SEED)
    sampled = rng.sample(tool_call, n_tool) + rng.sample(groundedness, n_ground)
    rng.shuffle(sampled)
    return sampled


def split_records(records):
    n_train = int(len(records) * TRAIN_RATIO)
    return records[:n_train], records[n_train:]


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_splits_to_data(train, val):
    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VAL_PATH, val)
    write_jsonl(SAMPLE_400_PATH, train + val)


def get_splits():
    if SPLIT_CACHE.exists():
        with open(SPLIT_CACHE, encoding="utf-8") as f:
            cached = json.load(f)
        train, val = cached["train"], cached["val"]
    else:
        sampled = stratified_sample(load_all_records())
        train, val = split_records(sampled)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(SPLIT_CACHE, "w", encoding="utf-8") as f:
            json.dump({"train": train, "val": val}, f, ensure_ascii=False)

    export_splits_to_data(train, val)
    return train, val


def split_stats(split):
    counts = {}
    for r in split:
        counts[r.get("task", "unknown")] = counts.get(r.get("task", "unknown"), 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Tokenizer / model helpers
# ---------------------------------------------------------------------------

def load_tokenizer(model_name=DEFAULT_MODEL):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_name=DEFAULT_MODEL, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return model.to(device)


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _encode_prompt(tokenizer, record):
    user_messages = [{"role": "user", "content": build_user_content(record)}]
    prompt_text = tokenizer.apply_chat_template(
        user_messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(prompt_text, add_special_tokens=False)


def tokenize_sft_example(tokenizer, record, max_len):
    messages = build_messages(record)
    prompt_ids = _encode_prompt(tokenizer, record)
    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    input_ids = tokenizer.encode(full_text, add_special_tokens=False)[:max_len]
    labels = [-100] * min(len(prompt_ids), len(input_ids))
    labels += input_ids[len(labels):]
    labels = labels[:max_len]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    if len(input_ids) < max_len:
        pad_len = max_len - len(input_ids)
        input_ids = input_ids + [pad_id] * pad_len
        labels = labels + [-100] * pad_len
    return input_ids, labels


def tokenize_dpo_example(tokenizer, record, max_len):
    prompt_ids = _encode_prompt(tokenizer, record)
    prompt_len = len(prompt_ids)

    def response_ids(text):
        text = normalize_output(text, record.get("task"))
        resp_messages = build_messages(record, assistant_content=text)
        full_text = tokenizer.apply_chat_template(
            resp_messages, tokenize=False, add_generation_prompt=False
        )
        full_ids = tokenizer.encode(full_text, add_special_tokens=False)
        return full_ids[prompt_len:]

    chosen_ids = response_ids(record["chosen"])
    rejected_ids = response_ids(record["rejected"])
    max_resp = max(len(chosen_ids), len(rejected_ids), 1)
    max_prompt = max_len - max_resp
    if prompt_len > max_prompt:
        prompt_ids = prompt_ids[:max_prompt]
        prompt_len = len(prompt_ids)

    def pack(response):
        ids = prompt_ids + response
        if len(ids) > max_len:
            ids = ids[:max_len]
        attn = [1] * len(ids)
        labels = [-100] * prompt_len + ids[prompt_len:]
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        if len(ids) < max_len:
            pad_len = max_len - len(ids)
            ids = ids + [pad_id] * pad_len
            attn = attn + [0] * pad_len
            labels = labels + [-100] * pad_len
        return ids, attn, labels

    return pack(chosen_ids), pack(rejected_ids)


def _pad_batch(items, pad_value=0):
    max_len = max(len(x) for x in items)
    return [item + [pad_value] * (max_len - len(item)) for item in items]


def collate_sft(records, tokenizer, max_len):
    pairs = [tokenize_sft_example(tokenizer, r, max_len) for r in records]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    input_ids = torch.tensor(_pad_batch([p[0] for p in pairs], pad_value=pad_id), dtype=torch.long)
    labels = torch.tensor(_pad_batch([p[1] for p in pairs], pad_value=-100), dtype=torch.long)
    attention_mask = (input_ids != pad_id).long()
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def collate_dpo(records, tokenizer, max_len):
    chosen_items, rejected_items = [], []
    for r in records:
        c, rej = tokenize_dpo_example(tokenizer, r, max_len)
        chosen_items.append(c)
        rejected_items.append(rej)

    def stack(side):
        input_ids = torch.tensor(_pad_batch([x[0] for x in side]), dtype=torch.long)
        attention_mask = torch.tensor(_pad_batch([x[1] for x in side]), dtype=torch.long)
        labels = torch.tensor(_pad_batch([x[2] for x in side], pad_value=-100), dtype=torch.long)
        return input_ids, attention_mask, labels

    c_ids, c_mask, c_labels = stack(chosen_items)
    r_ids, r_mask, r_labels = stack(rejected_items)
    return {
        "chosen_input_ids": c_ids,
        "chosen_attention_mask": c_mask,
        "chosen_labels": c_labels,
        "rejected_input_ids": r_ids,
        "rejected_attention_mask": r_mask,
        "rejected_labels": r_labels,
    }


# ---------------------------------------------------------------------------
# Dataloaders (infinite iterators for training)
# ---------------------------------------------------------------------------

def make_sft_dataloader(records, tokenizer, batch_size, max_len, shuffle=True):
    rng = random.Random(SAMPLE_SEED)
    while True:
        order = records[:]
        if shuffle:
            rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            if batch:
                yield collate_sft(batch, tokenizer, max_len)


def make_dpo_dataloader(records, tokenizer, batch_size, max_len, shuffle=True):
    rng = random.Random(SAMPLE_SEED + 1)
    while True:
        order = records[:]
        if shuffle:
            rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            batch = order[i:i + batch_size]
            if batch:
                yield collate_dpo(batch, tokenizer, max_len)


def iter_sft_batches(records, tokenizer, batch_size, max_len):
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        if batch:
            yield collate_sft(batch, tokenizer, max_len)


def iter_dpo_batches(records, tokenizer, batch_size, max_len):
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        if batch:
            yield collate_dpo(batch, tokenizer, max_len)


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — fixed metrics)
# ---------------------------------------------------------------------------

def _sequence_logps(logits, labels):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    log_probs = F.log_softmax(shift_logits, dim=-1)
    mask = shift_labels != -100
    safe_labels = shift_labels.clamp(min=0)
    token_logps = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask
    seq_logps = token_logps.sum(dim=-1)
    return token_logps.sum(dim=-1)


@torch.no_grad()
def evaluate_sft_loss(model, tokenizer, val_records, batch_size, max_len=MAX_SEQ_LEN):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    device = next(model.parameters()).device
    for batch in iter_sft_batches(val_records, tokenizer, batch_size, max_len):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        labels = batch["labels"]
        mask = labels != -100
        n_tokens = mask.sum().item()
        if n_tokens == 0:
            continue
        loss = F.cross_entropy(
            outputs.logits[:, :-1, :].reshape(-1, outputs.logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += n_tokens
    model.train()
    return total_loss / max(total_tokens, 1)


@torch.no_grad()
def evaluate_dpo_loss(
    model, tokenizer, val_records, batch_size, beta=0.1, max_len=MAX_SEQ_LEN, ref_model=None
):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    device = next(model.parameters()).device
    for batch in iter_dpo_batches(val_records, tokenizer, batch_size, max_len):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = dpo_batch_loss(model, batch, beta, ref_model=ref_model)
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


def dpo_batch_loss(policy, batch, beta, ref_model=None):
    def avg_logps(model, input_ids, attention_mask, labels):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        return _sequence_logps(outputs.logits, labels)

    pi_c = avg_logps(policy, batch["chosen_input_ids"], batch["chosen_attention_mask"], batch["chosen_labels"])
    pi_r = avg_logps(policy, batch["rejected_input_ids"], batch["rejected_attention_mask"], batch["rejected_labels"])

    with torch.no_grad():
        if ref_model is not None:
            ref_c = avg_logps(
                ref_model,
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            ref_r = avg_logps(
                ref_model,
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )
        else:
            with policy.disable_adapter():
                ref_c = avg_logps(
                    policy,
                    batch["chosen_input_ids"],
                    batch["chosen_attention_mask"],
                    batch["chosen_labels"],
                )
                ref_r = avg_logps(
                    policy,
                    batch["rejected_input_ids"],
                    batch["rejected_attention_mask"],
                    batch["rejected_labels"],
                )

    logits = beta * ((pi_c - pi_r) - (ref_c - ref_r))
    return (-F.logsigmoid(logits)).mean()


@torch.no_grad()
def preview_val_generations(model, tokenizer, val_records, n_examples=3, max_new_tokens=64):
    """Print a few val-set generations so you can eyeball quality."""
    model.eval()
    device = next(model.parameters()).device
    examples = val_records[:n_examples]
    print("--- val preview ---")
    for i, record in enumerate(examples):
        user_messages = [{"role": "user", "content": build_user_content(record)}]
        prompt = tokenizer.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
        pred = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        expected = normalize_output(record["chosen"], record.get("task"))
        print(f"[{i + 1}] task={record.get('task')}")
        print(f"  expected: {expected[:200]}")
        print(f"  model:    {pred[:200]}")
    print("--- end preview ---")
    model.train()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def verify_and_prepare(model_name=DEFAULT_MODEL):
    records = load_all_records()
    train, val = get_splits()
    print(f"Dataset: data/dpo.jsonl ({len(records)} total rows)")
    print(f"Sampled: {TOTAL_SAMPLES} (seed={SAMPLE_SEED})")
    print(f"Train: {len(train)} {split_stats(train)}")
    print(f"Val:   {len(val)} {split_stats(val)}")
    print(f"Exported: data/train_320.jsonl, data/val_80.jsonl, data/sample_400.jsonl")
    print(f"Split cache: autoresearch/split_400.json (under ~/.cache)")
    print()
    print(f"Loading tokenizer and model: {model_name}")
    load_tokenizer(model_name)
    _ = load_base_model(model_name, device="cpu")
    DATA_MARKER.parent.mkdir(parents=True, exist_ok=True)
    DATA_MARKER.write_text("ok\n", encoding="utf-8")
    print("Done! Ready to train with: uv run train.py")


if __name__ == "__main__":
    verify_and_prepare()
