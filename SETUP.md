# Setup

Environment notes for running this project on LIACS compute.

## Connecting

Two hops. From a local terminal:

```bash
ssh s4566319@ssh.liacs.nl      # lands on silver
ssh vibranium.liacs.nl          # GPU node, 2x RTX 3090 (24GB each)
```

Alternative GPU nodes if vibranium is busy: `tritanium.liacs.nl`,
`duranium.liacs.nl`. For L40S cards via SLURM: `fs.cc.liacs.nl`.

Check GPU availability before starting:

```bash
nvidia-smi
```

Pick a card whose memory usage is near zero and use it for everything:

```bash
export CUDA_VISIBLE_DEVICES=1     # or 0, whichever is free
```

## Storage

`/home` is capped at 7.5GB and `/data` is full, so all work lives in `/local`.
Note that `/local` is node-specific and may be cleared periodically. Code and
results are on GitHub; model weights and vectors are not backed up and would
need re-downloading (about 1 minute) and re-extracting (about 8 minutes).

```bash
export HF_HOME=/local/$USER/hf    # already in ~/.bashrc
```

## Per-session

```bash
cd /local/$USER/MasterThesis
source /local/$USER/venv/bin/activate
```

The prompt should show `(venv)`.

## First-time setup on a new node

```bash
mkdir -p /local/$USER && cd /local/$USER
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy transformers accelerate sentencepiece protobuf scikit-learn tqdm

export HF_HOME=/local/$USER/hf
mkdir -p $HF_HOME
hf auth login                     # needs a HuggingFace read token
hf download mistralai/Mistral-7B-Instruct-v0.3 --exclude "*.pth" "consolidated*"

git clone https://github.com/pranayram003/MasterThesis.git
cd MasterThesis
mkdir -p reports vectors          # scripts fail late if these are missing
```

## Running

Always pass `--no-4bit`. All current results are full-precision bfloat16;
4-bit quantisation depressed negation accuracy from 0.630 to 0.485 and is not
comparable.

```bash
# unsteered baseline, negated vs unnegated
CUDA_VISIBLE_DEVICES=1 python3 probe_negation.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 --n 200 --no-4bit \
  --out reports/negation_probe_bf16.json

# extract the CAA direction (about 8 minutes, 1992 pairs)
CUDA_VISIBLE_DEVICES=1 python3 extract_vectors.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --pairs data/caa_train.jsonl --no-4bit --out vectors/mistral_bf16.pt

# validate on the held-out split
CUDA_VISIBLE_DEVICES=1 python3 validate_vectors.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --vectors vectors/mistral_bf16.pt --no-4bit \
  --out reports/mistral_val_bf16.json

# steering sweep
CUDA_VISIBLE_DEVICES=1 python3 steer_pilot.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --vectors vectors/mistral_bf16.pt \
  --layers 18 20 22 --positions last --n 150 --no-4bit \
  --out reports/steer.json

# controls: run both before believing any steering result
CUDA_VISIBLE_DEVICES=1 python3 bias_control.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 --no-4bit \
  --out reports/bias_control_bf16.json

CUDA_VISIBLE_DEVICES=1 python3 steer_pilot.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --vectors vectors/mistral_bf16.pt --layers 22 --positions last \
  --n 150 --random-dir --seed 1 --no-4bit --out reports/random_s1.json
```

For long runs over SSH, use `screen -S name` so a dropped connection does not
kill the job. Detach with Ctrl+A then D, return with `screen -r name`. Note
that screen starts a fresh shell, so re-activate the venv inside it. Scrolling
inside screen is Ctrl+A then `[`, and `q` to exit scroll mode.

## Gotchas

- Two prompt formats exist. `prompts.py` is the current two-option format used
  by `extract_vectors.py`, `validate_vectors.py`, `steer_pilot.py` and
  `check_hook.py`. `data.py` holds the superseded three-option format used by
  `baseline_eval.py` and `build_pairs.py`; running `baseline_eval.py` will
  reproduce the abandoned format's failure, not the current baseline.
- `steer_pilot.py` prints a hardcoded "baseline for comparison: 0.485" line.
  That is the old quantised number. The real unsteered baseline on
  `caa_val.jsonl` at n=150 is 0.607.
- Newer `transformers` versions set `config.head_dim` to `None` rather than
  omitting it, which breaks `getattr(cfg, "head_dim", default)`. Already patched
  in `modeling.py`; watch for the same pattern elsewhere.
- Editing files through the GitHub web interface can silently truncate them.
  `data/caa_train.jsonl` arrived empty this way. Check file sizes after any
  web-based edit.
- Ctrl+V does not paste in a Linux terminal. Use right-click or Ctrl+Shift+V.

## Current baselines

All Mistral-7B-Instruct-v0.3, bfloat16, two-option format with demonstrations.

| Measure | Value |
| --- | --- |
| Negated accuracy, n=200 | 0.630 |
| Unnegated control, n=200 | 0.995 |
| Held-out AUROC, best layer (32) | 0.755 |
| Unsteered on `caa_val`, n=150 | 0.607 |
| Label-bias ceiling | 0.647 |
| Random direction, layer 22 | 0.607–0.613 |
| CAA steering, layer 22 | 0.693 |
| CAA steering, layer 18 | 0.780 (control unstable) |
