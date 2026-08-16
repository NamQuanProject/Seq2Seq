# Seq2Seq — Robust EN→VI Machine Translation under a Parameter Budget

Noisy English → clean Vietnamese translation. The source English text has
injected noise (missing spaces, garbage/gibberish words, elongated
characters, inverted grammar); the Vietnamese target is clean. The model
is a **GPT-small-style decoder-only Transformer** trained as a prefix
language model, sized to land around **~2.2–2.4M trainable parameters**
— comfortably under the assignment's 5,000,000 hard cap and inside the
2,500,000 bonus tier.

This README walks through the whole pipeline, start to end: setup, data
placement, running each script in order, and what to look at for the
report.

---

## 0. Repository layout

```
tokenizer.py    # denoising rules + BPE subword tokenizer
preprocess.py   # STEP 1 — writes a fully denoised copy of the corpus
data.py         # loads data (raw or preprocessed) into DataLoaders
model.py        # GPTTranslator: decoder-only Transformer, greedy/beam decoding
train.py        # STEP 2 — trains the model, saves checkpoint + loss curve
test.py         # STEP 3 — BLEU eval + qualitative/attention analysis
requirements.txt
exercise-machine-translation-student/
  machine_translation.ipynb   # original baseline notebook (vanilla RNN) + task spec
  report_requirements.md      # required report structure
```

`en-vi-translation-data/` (raw corpus) and `output/` (everything this
codebase generates: checkpoints, tokenizers, plots, cleaned data, stats,
attention maps) are both gitignored — neither is source code.

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

Place the raw parallel corpus under `./en-vi-translation-data/`:

```
en-vi-translation-data/
  train_noisy.en.txt   train.vi.txt
  val_noisy.en.txt      val.vi.txt
  test_noisy.en.txt    test.vi.txt
```

Each `*.en.txt` / `*.vi.txt` pair must have the same number of lines
(line *i* in the English file is the noisy source for line *i* in the
Vietnamese file).

---

## 2. Step-by-step run

### Step 1 — Denoise the corpus (`preprocess.py`)

```bash
python preprocess.py --data_dir ./en-vi-translation-data --output_dir ./output/clean_data
```

What it does, per English sentence:
1. **Self-correct** what can be corrected — collapse elongated characters
   (`"soooo"` → `"soo"`) and repeated punctuation (`"!!!"` → `"!"`), strip
   disallowed characters, and repair missing-space concatenations
   (`"ihavean apple"` → `"i have an apple"`) via dictionary-checked word
   segmentation.
2. **Delete** what's left — any token that still isn't a recognizable
   English word after step 1 (real injected gibberish, not a typo) is
   removed from the sentence entirely.
3. If a sentence becomes empty after deletion, the whole pair is dropped
   (an empty source carries no signal).

The Vietnamese target is only lightly normalized (already clean) — the
destructive deletion step never runs on it.

Outputs:
- `output/clean_data/{train,val,test}_clean.en.txt` and `.vi.txt`
- `output/preprocess_stats.json` — per-split counts (elongation/punct
  fixes, concatenation repairs, garbage tokens deleted, pairs dropped,
  source vocabulary size before/after) — use this in the report's
  "Denoising & Tokenization" section as quantitative evidence.

If you skip this step, `train.py`/`test.py` still work — `data.py` falls
back to a conservative on-the-fly clean (no deletion) and prints a
warning. Running `preprocess.py` first is recommended for the "super
clean" strategy.

### Step 2 — Train (`train.py`)

```bash
python train.py --data_dir ./en-vi-translation-data --epochs 20
```

What it does:
- Loads `output/clean_data/` if present (else raw + on-the-fly clean).
- Trains **one shared BPE vocabulary** (default 6000 tokens) over the
  concatenation of cleaned English + Vietnamese training text, and packs
  every pair into a single sequence `<sos> src_tokens <sep> trg_tokens
  <eos>` (`output/tokenizers/joint_bpe.json`).
- Builds the model (`build_model`), prints the trainable parameter count,
  and **raises an error if it exceeds 5,000,000** (hard budget) — it also
  reports whether it clears the 2,500,000 bonus threshold.
- Trains as a causal LM with the loss masked to the Vietnamese span only
  (`CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)`), with
  gradient clipping and `ReduceLROnPlateau`.
- Saves the best (lowest val-loss) checkpoint to `output/checkpoint.pt`
  and the train/val loss curve to `output/loss_curve.png` after every
  epoch that improves.

Useful flags (all optional, see `python train.py --help`):
| Flag | Default | Purpose |
|---|---|---|
| `--epochs` | 20 | training epochs |
| `--max_train_samples` | 30000 | cap on training pairs (speed knob) |
| `--vocab_size` | 6000 | shared BPE vocab size |
| `--d_model` / `--nhead` / `--n_layer` / `--d_ff` | 192 / 6 / 5 / 256 | model size — tune these to move the parameter count |
| `--batch_size` | 64 | |
| `--lr` | 3e-4 | |
| `--max_len` | 128 | max packed sequence length (`<sos>src<sep>trg<eos>`) |
| `--output_dir` | `./output` | where everything gets written |

**First thing to check after training starts:** the printed line
`Trainable parameters: N,NNN,NNN (budget: 5,000,000)` and the bonus-tier
message right after it. If you change `--d_model`/`--n_layer`/`--d_ff`/
`--vocab_size`, re-run and re-check this before doing a full training run
— it fails fast (raises before training starts) if you exceed 5M.

You can also just run `python model.py` for a quick parameter count
without touching any data.

### Step 3 — Evaluate (`test.py`)

```bash
python test.py --ckpt_path ./output/checkpoint.pt --data_dir ./en-vi-translation-data
```

What it does:
- Reloads the exact model config and cleaning mode used at training time
  (stored inside the checkpoint), so evaluation is reproducible.
- Computes corpus BLEU-4 (with smoothing) on the noisy test set for
  **greedy** decoding, then again for **beam search** (`--beam_width`,
  default 5) — printed side by side so you can compare in the report.
- Runs a qualitative pass on 3 hand-picked noisy sentences (including a
  clean vs. garbled apples example and a sentence full of slang/gibberish
  like `"vacx"`/`"lmao"`), printing greedy vs. beam predictions and saving
  a self-attention heatmap for each to
  `output/attention_example_{1,2,3}.png` — look at whether the model's
  attention on generated Vietnamese tokens is concentrated on the real
  source words rather than the noisy/garbage ones.

Useful flags: `--max_eval_samples N` to subsample the test set for a
quick sanity check before running full BLEU; `--beam_width`.

---

## 3. What lands in `./output/` after a full run

```
output/
  clean_data/                    # preprocess.py output
    train_clean.en.txt  train_clean.vi.txt
    val_clean.en.txt    val_clean.vi.txt
    test_clean.en.txt   test_clean.vi.txt
  preprocess_stats.json          # noise-fix/deletion counts per split
  tokenizers/joint_bpe.json      # shared BPE vocabulary
  denoise_stats.json             # clean_text-level stats (conservative pass)
  checkpoint.pt                  # best model weights + full config + loss history
  loss_curve.png                 # train/val loss vs. epoch
  attention_example_1.png ...3   # attention heatmaps for the qualitative analysis
```

Everything needed for the report's plots/tables comes from this folder.

---

## 4. Architecture summary (for the report)

`model.py`'s `GPTTranslator` is a decoder-only Transformer:
- **One** shared token embedding (joint EN/VI BPE vocab) instead of
  separate encoder/decoder vocabularies — a translation is generated by
  continuing the sequence `<sos> src <sep>` autoregressively, i.e.
  translation is framed as "continue this sequence" rather than
  "encode, then decode with cross-attention".
- 5 pre-LN Transformer blocks (causal self-attention + MLP), weight-tied
  output projection (reuses the token embedding matrix instead of a
  second `d_model × vocab` matrix).
- Trained with the loss masked to the target span only (prefix-LM /
  "GPT fine-tuned for translation" recipe).
- Both **greedy** and **beam search** generation are implemented
  (`greedy_generate`, `beam_search_generate`) to satisfy the
  "compare decoding strategies" requirement.
- Self-attention lets any generated Vietnamese token attend directly back
  to any English source position (including noisy ones) — the same
  attention-map analysis an encoder-decoder would give you.

Full parameter-budget math is in `model.py`'s module docstring.

---

## 5. Troubleshooting

- **`FileNotFoundError` on `train_noisy.en.txt`**: the raw corpus isn't
  under `--data_dir` — check step 1 of setup.
- **Parameter budget exceeded** (`train.py` raises `ValueError`): reduce
  `--d_model`, `--d_ff`, `--n_layer`, or `--vocab_size` and re-run;
  `python model.py` gives a fast parameter-count check without loading data.
- **Training loss not decreasing**: sanity-check with a small
  `--max_train_samples` and a handful of `--epochs` first; verify
  `output/clean_data/` looks reasonable (open a few lines by hand) before
  a full run.
- **`test.py` results look off vs. training**: make sure `--ckpt_path`
  points at the checkpoint you expect, and that `en-vi-translation-data/`
  and `output/clean_data/` haven't changed since training (the checkpoint
  pins the exact config but not the data itself).

---

See `exercise-machine-translation-student/machine_translation.ipynb` for
the original baseline (vanilla RNN) and full task description, and
`exercise-machine-translation-student/report_requirements.md` for the
required report structure.
