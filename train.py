"""
Autoresearch Qwen fine-tuning script. Single-GPU, single-file.
Usage: uv run train.py
"""

import gc
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.optim import AdamW

from prepare import (
    DEFAULT_MODEL,
    MAX_SEQ_LEN,
    dpo_batch_loss,
    evaluate_dpo_loss,
    evaluate_sft_loss,
    get_splits,
    load_base_model,
    load_tokenizer,
    make_dpo_dataloader,
    make_sft_dataloader,
    preview_val_generations,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

TRAINING_MODE = "dpo"  # "sft" | "dpo"

# LoRA
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training schedule (epoch-based, no wall-clock limit)
MAX_EPOCHS = 10
EVAL_EVERY_STEPS = 40       # ~1 epoch on 320 train samples with batch 8
EARLY_STOP_PATIENCE = 3     # stop after this many evals without val improvement

# Optimization
LEARNING_RATE = 2e-4
LEARNING_RATE_DPO = 1e-5
WEIGHT_DECAY = 0.01
ADAM_BETAS = (0.9, 0.999)
WARMUP_RATIO = 0.05
WARMDOWN_RATIO = 0.10
FINAL_LR_FRAC = 0.1
DEVICE_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4

# DPO
BETA_DPO = 0.1

# Checkpointing
SAVE_BEST_CHECKPOINT = True
CHECKPOINT_DIR = "checkpoints"
# Set to an existing LoRA folder (e.g. "checkpoints/sft/best") to continue DPO from SFT.
LORA_ADAPTER_PATH = "checkpoints/sft/best"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_ctx = torch.amp.autocast(
    device_type="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.bfloat16,
    enabled=torch.cuda.is_available(),
)

print(f"Training mode: {TRAINING_MODE}")
print(f"Model: {DEFAULT_MODEL}")
print(f"Device: {device}")

tokenizer = load_tokenizer()
train_records, val_records = get_splits()
print(f"Train samples: {len(train_records)}, Val samples: {len(val_records)}")

model = load_base_model(device=device)
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM",
)
if LORA_ADAPTER_PATH:
    print(f"Loading LoRA adapter: {LORA_ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH, is_trainable=True)
else:
    model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
model.train()

ref_model = None
if TRAINING_MODE == "dpo" and LORA_ADAPTER_PATH:
    print(f"DPO reference model: {LORA_ADAPTER_PATH}")
    ref_model = load_base_model(device=device)
    ref_model = PeftModel.from_pretrained(ref_model, LORA_ADAPTER_PATH)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

optimizer = AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=LEARNING_RATE_DPO if TRAINING_MODE == "dpo" else LEARNING_RATE,
    betas=ADAM_BETAS,
    weight_decay=WEIGHT_DECAY,
)

if TRAINING_MODE == "sft":
    train_loader = make_sft_dataloader(train_records, tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN)
else:
    train_loader = make_dpo_dataloader(train_records, tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN)

samples_per_step = DEVICE_BATCH_SIZE * GRAD_ACCUM_STEPS
steps_per_epoch = max(1, math.ceil(len(train_records) / samples_per_step))
total_steps = MAX_EPOCHS * steps_per_epoch
best_checkpoint_path = Path(CHECKPOINT_DIR) / TRAINING_MODE / "best"

print(f"Max epochs: {MAX_EPOCHS} ({total_steps} steps, {steps_per_epoch} steps/epoch)")
print(f"Eval every {EVAL_EVERY_STEPS} steps, early stop patience {EARLY_STOP_PATIENCE}")
print(f"Batch size: {DEVICE_BATCH_SIZE}, grad accum: {GRAD_ACCUM_STEPS}")

# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    if progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    cooldown = (1.0 - progress) / WARMDOWN_RATIO
    return cooldown * 1.0 + (1 - cooldown) * FINAL_LR_FRAC


def run_val_eval():
    if TRAINING_MODE == "sft":
        return evaluate_sft_loss(model, tokenizer, val_records, DEVICE_BATCH_SIZE)
    return evaluate_dpo_loss(
        model, tokenizer, val_records, DEVICE_BATCH_SIZE, beta=BETA_DPO, ref_model=ref_model
    )


def save_checkpoint(path):
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0.0
step = 0
best_val_metric = float("inf")
best_step = -1
evals_without_improvement = 0
final_val_metric = float("inf")

while step < total_steps:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    optimizer.zero_grad(set_to_none=True)
    train_loss = None

    for micro_step in range(GRAD_ACCUM_STEPS):
        batch = next(train_loader)
        batch = {k: v.to(device) for k, v in batch.items()}

        with autocast_ctx:
            if TRAINING_MODE == "sft":
                outputs = model(**batch)
                loss = outputs.loss / GRAD_ACCUM_STEPS
            else:
                loss = dpo_batch_loss(model, batch, BETA_DPO, ref_model=ref_model) / GRAD_ACCUM_STEPS

        train_loss = loss.detach() * GRAD_ACCUM_STEPS
        loss.backward()

    progress = min(step / max(total_steps - 1, 1), 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        base_lr = LEARNING_RATE_DPO if TRAINING_MODE == "dpo" else LEARNING_RATE
        group["lr"] = base_lr * lrm
    optimizer.step()

    train_loss_f = train_loss.item()
    if math.isnan(train_loss_f) or train_loss_f > 100:
        print("FAIL")
        raise SystemExit(1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dt = time.time() - t0

    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    epoch = step / steps_per_epoch

    print(
        f"\rstep {step + 1:05d}/{total_steps} | epoch {epoch:.1f} | "
        f"loss: {debiased_smooth_loss:.6f} | best_val: {best_val_metric:.6f} | "
        f"lrm: {lrm:.2f} | dt: {dt * 1000:.0f}ms    ",
        end="",
        flush=True,
    )

    if step == 0:
        gc.collect()
        if hasattr(gc, "freeze"):
            gc.freeze()
        gc.disable()
    elif (step + 1) % 500 == 0:
        gc.collect()

    step += 1
    should_eval = (step % EVAL_EVERY_STEPS == 0) or (step >= total_steps)
    if should_eval:
        val_metric = run_val_eval()
        final_val_metric = val_metric
        improved = val_metric < best_val_metric
        if improved:
            best_val_metric = val_metric
            best_step = step
            evals_without_improvement = 0
            if SAVE_BEST_CHECKPOINT:
                save_checkpoint(best_checkpoint_path)
        else:
            evals_without_improvement += 1

        print(
            f"\n  eval step {step}: val_metric={val_metric:.6f}"
            + (" *best*" if improved else "")
        )
        preview_val_generations(model, tokenizer, val_records, n_examples=1)

        if evals_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"Early stop at step {step} (no improvement for {EARLY_STOP_PATIENCE} evals)")
            break

print()

# ---------------------------------------------------------------------------
# Final eval on best checkpoint
# ---------------------------------------------------------------------------

if SAVE_BEST_CHECKPOINT and best_checkpoint_path.exists() and best_step >= 0:
    print(f"Loading best checkpoint from step {best_step}: {best_checkpoint_path}")
    base_model = load_base_model(device=device)
    model = PeftModel.from_pretrained(base_model, best_checkpoint_path)
    model.eval()
    val_metric = run_val_eval()
    preview_val_generations(model, tokenizer, val_records, n_examples=3)
else:
    val_metric = final_val_metric if best_step < 0 else best_val_metric
    preview_val_generations(model, tokenizer, val_records, n_examples=3)

total_training_time = time.time() - t_start_training
t_end = time.time()
peak_vram_mb = (
    torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0
)

print("---")
print(f"val_metric:       {best_val_metric:.6f}")
print(f"final_val_metric: {final_val_metric:.6f}")
print(f"best_step:        {best_step}")
print(f"training_mode:    {TRAINING_MODE}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"max_epochs:       {MAX_EPOCHS}")
print(f"lora_r:           {LORA_R}")
print(f"learning_rate:    {LEARNING_RATE}")
if SAVE_BEST_CHECKPOINT and best_checkpoint_path.exists():
    print(f"checkpoint:       {best_checkpoint_path}")
if LORA_ADAPTER_PATH:
    print(f"lora_adapter_in:  {LORA_ADAPTER_PATH}")