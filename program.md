# autoresearch (Qwen SFT/DPO)

This is an experiment to have the LLM do its own research on Qwen fine-tuning.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `may24`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data loading (400-sample split from `data/dpo.jsonl`), dataloaders, evaluation. Do not modify.
   - `train.py` — the file you modify. LoRA config, optimizer, training mode (SFT/DPO), hyperparameters.
4. **Verify data exists**: Check that `data/dpo.jsonl` exists and run `uv run prepare.py` if needed.
5. **Initialize results.tsv**: Create `results.tsv` with the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. Training is **epoch-based** (not wall-clock limited). Launch with: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. LoRA rank, learning rate, batch size, `MAX_EPOCHS`, `TRAINING_MODE` (`sft` or `dpo`), DPO beta, eval frequency, early-stop patience, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only (data split, dataloaders, eval harness).
- Install new packages or add dependencies. Use only what's in `pyproject.toml`.
- Modify the evaluation harness. `evaluate_sft_loss` / `evaluate_dpo_loss` in `prepare.py` are the ground truth.

**The goal: get the lowest `val_metric` within the same `training_mode`.** Lower is better. SFT and DPO metrics are not directly comparable — always compare within the same mode.

**Best checkpoint**: Each run saves the **best val checkpoint** to `checkpoints/sft/best/` or `checkpoints/dpo/best/`. The printed `val_metric` is the **best** val score, not the final step.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful gains.

**Simplicity criterion**: All else being equal, simpler is better.

**The first run**: Always establish the baseline with the script as-is (`TRAINING_MODE = "sft"`).

## Output format

```
---
val_metric:       0.215452
final_val_metric: 0.312000
best_step:        120
training_mode:    sft
training_seconds: 180.5
total_seconds:    195.2
peak_vram_mb:     5028.0
num_steps:        280
max_epochs:       10
lora_r:           16
learning_rate:    0.0002
checkpoint:       checkpoints/sft/best
---
```

Extract the key metric:

```
grep "^val_metric:\|^best_step:\|^checkpoint:" run.log
```

## Logging results

Log to `results.tsv` (tab-separated):

```
commit	val_metric	memory_gb	mode	status	description
```

1. git commit hash (short, 7 chars)
2. **best** val_metric achieved — use `9.999999` for crashes
3. peak memory in GB (peak_vram_mb / 1024)
4. mode: `sft` or `dpo`
5. status: `keep`, `discard`, or `crash`
6. short description

Example:

```
commit	val_metric	memory_gb	mode	status	description
a1b2c3d	0.215452	4.9	sft	keep	baseline SFT 10 epochs
b2c3d4e	0.198100	4.9	sft	keep	LR 3e-4
c3d4e5f	0.452100	6.1	dpo	keep	DPO from sft/best, beta=0.1
```

## The experiment loop

LOOP FOREVER:

1. Look at git state
2. Tune `train.py`
3. git commit
4. Run: `uv run train.py > run.log 2>&1`
5. Read: `grep "^val_metric:\|^best_step:\|^peak_vram_mb:\|^training_mode:" run.log`
6. If crashed, `tail -n 50 run.log` and fix or discard
7. Record in `results.tsv` (do not commit results.tsv)
8. If val_metric improved (lower, same mode), keep the commit; else git reset

**Timeout**: No fixed wall-clock limit. Typical SFT run (10 epochs, early stop) is ~3–8 min; DPO is slower (~10–20 min). Kill if a run exceeds **30 minutes** (likely stuck or misconfigured).

**NEVER STOP**: Continue autonomously until manually interrupted.

## SFT → DPO pipeline

1. Run SFT experiments until best `checkpoints/sft/best/` is satisfactory.
2. Set `TRAINING_MODE = "dpo"` and `LORA_ADAPTER_PATH = "checkpoints/sft/best"`.
3. Run DPO experiments; best lands in `checkpoints/dpo/best/`.
