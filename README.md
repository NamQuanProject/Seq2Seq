# Seq2Seq — Robust EN→VI Machine Translation under a Parameter Budget

Noisy English → clean Vietnamese translation. The source English text has
injected noise (missing spaces, garbage/gibberish words, elongated
characters, inverted grammar); the Vietnamese target is clean. The model
is a **tiny Transformer encoder-decoder**, sized to land around
**~2.4M trainable parameters** — comfortably under the assignment's
5,000,000 hard cap and inside the 2,500,000 bonus tier:

| Component | Value |
|---|---|
| Encoder layers | 3 |
| Decoder layers | 3 |
| Model dimension | 128 |
| Attention heads | 4 |
| FFN dimension | 512 |
| Vocabulary | 8,000 joint BPE tokens (shared EN+VI) |
| Embeddings | shared across encoder input, decoder input, and output projection |
| Position encoding | sinusoidal (0 extra parameters) |
| Dropout | 0.1 |

This README walks through the whole pipeline, start to end: setup, data
placement, running each script in order, and what to look at for the
report.

---

## 0. Repository layout

```
tokenizer.py            # denoising rules, noise augmentation, BPE subword tokenizer
preprocess.py           # STEP 1 — writes a fully denoised copy of the corpus
data.py                 # loads data (raw or preprocessed) into DataLoaders
eval_utils.py           # shared fair-comparison eval subset (used by baseline.py AND test.py)
baseline.py             # OPTIONAL — faithful vanilla-RNN baseline reproduction, for BLEU comparison
model.py                # Seq2SeqTransformer: encoder-decoder + all decoding/scoring primitives
train.py                # STEP 2 — trains the model, saves checkpoint(s) + loss curve
average_checkpoints.py  # OPTIONAL — averages the last-K epoch checkpoints (SWA-style)
rerank.py                # generate-many -> filter -> rerank inference pipeline
test.py                 # STEP 3 — BLEU eval (Tier-1 fair / Tier-2 full, greedy/beam/rerank) + qualitative analysis
Seq2Seq_MachineTranslation.ipynb  # notebook version of the same pipeline, runnable end to end
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
2. **Delete** what's left — any token that still isn't even a *leniently*
   recognizable English word after step 1 (real injected gibberish, not a
   typo) is removed from the sentence entirely. This deletion threshold
   (`_MIN_DELETE_FREQ`) is deliberately much lower than the threshold used
   to accept a concatenation-repair split (`_MIN_WORD_FREQ`) — deletion is
   irreversible, so it only fires on tokens that are essentially
   unattested, not merely "less common than the strict 300k bar." Unlike
   the exercise notebook's `normalize_string` (which never deletes
   anything), this step *can* destroy real content if the threshold is
   too aggressive — worth spot-checking `output/clean_data/` against the
   raw file, and citing `preprocess_stats.json`'s vocab-reduction numbers
   in the report to show the tradeoff wasn't excessive.
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

### Step 1b — (optional but recommended) Train the baseline for comparison

```bash
python baseline.py --data_dir ./en-vi-translation-data --epochs 5
```

`baseline.py` is a **faithful, standalone reproduction** of the exercise
notebook's vanilla-RNN Seq2Seq (whitespace `Vocabulary`, no attention,
greedy-only decoding) — same `read_langs`/`normalize_string`/dataset
logic, same architecture, same unsmoothed first-200-sentences BLEU
methodology. It additionally evaluates on the exact same fixed subset
`test.py` uses (`eval_utils.build_fair_eval_subset`), so its BLEU number
is directly comparable to `test.py`'s "Tier 1" output — see §2's Step 3
and the note on sample-size fairness there. Saves
`output/baseline_checkpoint.pt`, `output/baseline_loss_curve.png`, and
`output/baseline_results.json`.

### Step 2 — Train (`train.py`)

```bash
python train.py --data_dir ./en-vi-translation-data --epochs 20
```

What it does:
- Loads `output/clean_data/` if present (else raw + on-the-fly clean).
- Trains **one shared BPE vocabulary** (default 8000 tokens) over the
  concatenation of cleaned English + Vietnamese training text
  (`output/tokenizers/joint_bpe.json`), used by both the encoder and
  decoder (see the architecture table above).
- Packs each pair as `encoder_ids = <sos> <toXX> src... <eos>` /
  `decoder_input = <sos> trg...` / `decoder_target = trg... <eos>` — a
  leading direction tag (`<tovi>`/`<toen>`) tells the shared
  encoder+decoder which way to translate (see bidirectional training below).
- Builds the model (`build_model`), prints the trainable parameter count,
  and **raises an error if it exceeds 5,000,000** (hard budget) — it also
  reports whether it clears the 2,500,000 bonus threshold.
- Trains with standard cross-entropy over the decoder target
  (`CrossEntropyLoss(ignore_index=<pad>, label_smoothing=0.1)`) and
  gradient clipping.
- **Xavier/Glorot weight init** (`model.py`'s `_init_weights`) for every
  Linear/attention projection, and a **Noam warmup + inverse-sqrt-decay LR
  schedule** by default (`--lr_schedule noam`, stepped every batch, not
  every epoch) — both are standard, well-established fixes for the
  otherwise-slow, occasionally-unstable early training a from-scratch
  small Transformer gets under a flat LR + default PyTorch init. Pass
  `--lr_schedule plateau` to fall back to the old flat-lr +
  `ReduceLROnPlateau` behavior.
- **Bidirectional training**: a configurable fraction (`--p_reverse`,
  default 0.3) of training examples are packed VI→EN instead of EN→VI, so
  the SAME encoder+decoder pair can later score how well a candidate
  translation reverses back to the noisy source — this is what powers the
  reverse-model term in `rerank.py`.
- **Source-noise augmentation**: by default (`--augment_noise`, on by
  default) the English source is further corrupted at train time
  (keyboard-adjacent typos, dropped characters, random casing) via
  `tokenizer.augment_noise`, resampled every epoch — usually matters more
  for robustness than changing the decoding algorithm.
- Saves the best (lowest val-loss) checkpoint to `output/checkpoint.pt`
  and the train/val loss curve to `output/loss_curve.png` after every
  epoch that improves. Also keeps a rolling window of the last
  `--save_last_k` (default 3) epoch checkpoints in `output/ckpts/` for
  weight averaging (see step 2b).

Useful flags (all optional, see `python train.py --help`):
| Flag | Default | Purpose |
|---|---|---|
| `--epochs` | 20 | training epochs |
| `--max_train_samples` | 30000 | cap on training pairs (speed knob) |
| `--vocab_size` | 8000 | shared BPE vocab size |
| `--d_model` / `--nhead` | 128 / 4 | model width / attention heads |
| `--num_encoder_layers` / `--num_decoder_layers` | 3 / 3 | depth — tune these (and `--d_model`/`--d_ff`) to move the parameter count |
| `--d_ff` | 512 | feedforward dimension |
| `--batch_size` | 64 | |
| `--lr_schedule` | `noam` | `noam` (warmup+decay, recommended) or `plateau` (old behavior) |
| `--lr` | 1.0 | Noam scale multiplier (peak lr ≈1.4e-3 at defaults); flat LR if `--lr_schedule plateau` |
| `--warmup_steps` | 4000 | Noam warmup length in optimizer steps |
| `--max_len` | 128 | max encoder/decoder sequence length |
| `--p_reverse` | 0.3 | fraction of examples trained VI→EN (bidirectional model) |
| `--augment_noise` / `--no_augment_noise` | on | train-time source noise augmentation |
| `--save_last_k` | 3 | epoch checkpoints kept in `output/ckpts/` for averaging |
| `--output_dir` | `./output` | where everything gets written |

**First thing to check after training starts:** the printed line
`Trainable parameters: N,NNN,NNN (budget: 5,000,000)` and the bonus-tier
message right after it. If you change `--d_model`/`--num_encoder_layers`/
`--num_decoder_layers`/`--d_ff`/`--vocab_size`, re-run and re-check this
before doing a full training run — it fails fast (raises before training
starts) if you exceed 5M.

You can also just run `python model.py` for a quick parameter count
without touching any data.

### Step 2b — (optional) Checkpoint averaging

```bash
python average_checkpoints.py --ckpts_dir ./output/ckpts --output_path ./output/checkpoint_avg.pt
```

Averages the weights of the last `--save_last_k` epoch checkpoints
(SWA-style) — a free way to often pick up a bit of BLEU without changing
the parameter count. Evaluate it exactly like any other checkpoint:
`python test.py --ckpt_path ./output/checkpoint_avg.pt`.

### Step 3 — Evaluate (`test.py`)

```bash
python test.py --ckpt_path ./output/checkpoint.pt --data_dir ./en-vi-translation-data
```

What it does:
- Reloads the exact model config and cleaning mode used at training time
  (stored inside the checkpoint), so evaluation is reproducible.
- **On sample-size fairness:** `baseline.py` (and the exercise notebook's
  own baseline BLEU) is computed on only its first 200 test sentences.
  Comparing that directly against a BLEU computed on the *entire* (much
  larger, harder, noisier) test set is not apples-to-apples — a bigger,
  unfiltered sample will almost always score lower regardless of model
  quality. So `test.py` reports BLEU in two tiers:
  - **Tier 1 (fair comparison)** — greedy / beam / rerank evaluated on
    the exact same fixed subset `baseline.py` uses
    (`eval_utils.build_fair_eval_subset`, `--eval_sample_size`, default
    200). **This is the number to cite against the baseline.**
  - **Tier 2 (full test set)** — evaluated on everything, for a more
    rigorous (but NOT baseline-comparable) estimate; skip with
    `--skip_full_test_set` for a faster run.
- Add `--rerank` to also evaluate the generate-many-then-rerank strategy
  (both tiers) — see §4 below. It's much slower per sentence, so Tier 2's
  rerank pass defaults to a 50-sentence subset (`--rerank_eval_samples`).
- Runs a qualitative pass on 3 hand-picked noisy sentences (including a
  clean vs. garbled apples example and a sentence full of slang/gibberish
  like `"vacx"`/`"lmao"`), printing greedy vs. beam **vs. rerank**
  predictions side by side, and saving a **cross-attention** heatmap for
  each to `output/attention_example_{1,2,3}.png` (decoder positions ×
  source positions) — look at whether the model's attention on generated
  Vietnamese tokens is concentrated on the real source words rather than
  the noisy/garbage ones.

Useful flags: `--eval_sample_size` (Tier 1 subset size, default 200,
matching `baseline.py`); `--max_eval_samples N` to subsample Tier 2;
`--skip_full_test_set` to skip Tier 2 entirely; `--beam_width`;
`--rerank`, `--rerank_eval_samples`, `--n_beam`, `--n_sample`,
`--temperature`, `--alpha`, `--lambda_cov`, `--lambda_rev`,
`--lambda_rep`, `--use_mbr` (all tune the rerank pipeline, see §4).

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
  ckpts/epoch_*.pt               # rolling checkpoints for averaging
  loss_curve.png                 # train/val loss vs. epoch
  attention_example_1.png ...3   # cross-attention heatmaps for the qualitative analysis
  baseline_checkpoint.pt         # baseline.py output (optional)
  baseline_loss_curve.png
  baseline_results.json          # baseline BLEU, both methodologies (own-first-200 and shared-fair-subset)
```

Everything needed for the report's plots/tables comes from this folder.

---

## 4. The rerank pipeline (`rerank.py`, `--rerank`)

Beyond plain greedy/beam decoding, `rerank.py` implements a
"generate many, then pick the best one" strategy:

1. **Generate candidates**: a k-best beam search pool
   (`model.beam_search_candidates`, `--n_beam`, default 15) plus several
   low-temperature top-k sampling draws
   (`model.sample_generate`, `--n_sample`, default 10, `--temperature`)
   — beam search alone tends toward near-duplicate high-probability
   hypotheses, sampling fills in genuinely different ones. The encoder
   runs ONCE per source sentence and its output is reused across every
   candidate.
2. **Filter malformed candidates** (`filter_malformed`): missing `<eos>`
   within the generation budget, leaked special tokens, degenerate/empty
   output, excessive repeated n-grams.
3. **No-repeat n-gram blocking** (`no_repeat_ngram_size=3` by default) and
   **min-length control** are applied *during* generation itself (both
   beam and sampling), not just as a post-hoc filter.
4. **Rerank the survivors** with:

   ```
   S(y|x) = logP(y|x)/|y|^alpha
             + lambda_cov * C(x,y)
             + lambda_rev * logP(x|y)
             - lambda_rep * R(y)
   ```

   | Term | Flag | Implementation |
   |---|---|---|
   | length-normalized `logP(y|x)` | `--alpha` | `model.score_sequence` (teacher-forced) |
   | coverage `C(x,y)` | `--lambda_cov` | GNMT-style penalty from the decoder's own cross-attention into the encoder (`model.coverage_vector`) — rewards attending to every source token at least once, useful when noisy input makes it easy to silently drop a word |
   | reverse `logP(x|y)` | `--lambda_rev` | the SAME encoder+decoder scores the candidate translated back to English, using the `<toen>` direction tag it was trained on (`--p_reverse` in `train.py`) — no second model needed |
   | repetition `R(y)` | `--lambda_rep` | fraction of repeated trigrams (`repetition_penalty`) |

5. **Optional MBR reranking** (`--use_mbr`): instead of taking the top
   `S(y|x)` candidate directly, pick among the top `mbr_pool` survivors
   whichever has the highest average **chrF** similarity to the others
   (`mbr_select`) — a self-consistency signal that avoids picking a
   one-off decoding artifact that happened to score well.

Two more BLEU-relevant strategies live outside `rerank.py` itself:
- **Source-noise augmentation** (`tokenizer.augment_noise`, `train.py`
  `--augment_noise`) — usually matters more than the decoding algorithm.
- **Checkpoint averaging** (`average_checkpoints.py`) — free BLEU, zero
  extra inference-time parameters.

---

## 5. Architecture summary (for the report)

`model.py`'s `Seq2SeqTransformer` is the tiny encoder-decoder from the
table at the top of this README:
- Bidirectional encoder self-attention over the (denoised) noisy source;
  causal decoder self-attention + cross-attention into the encoder output.
- **One** shared token embedding (joint EN/VI BPE vocab), used for the
  encoder input, the decoder input, AND (weight-tied) the output
  projection — instead of three separate tables.
- A leading `<toen>`/`<tovi>` direction tag on the encoder input makes
  the same encoder+decoder pair bidirectional (see §4) at zero extra
  parameters.
- Sinusoidal positional encoding (0 trainable parameters).
- Three decoding strategies are implemented and compared in `test.py`:
  **greedy** (`greedy_generate`), **beam search**
  (`beam_search_generate`/`beam_search_candidates`), and the
  **generate-many-then-rerank** pipeline (§4) — satisfying the "compare
  decoding strategies" requirement with more than a single alternative.
  The encoder is run ONCE per sentence and its output (`memory`) is
  reused across every decode step/candidate.
- Decoder cross-attention gives a direct per-step distribution over
  source positions — the natural fit for both the attention-map
  qualitative analysis and the coverage-penalty term in §4.

Full parameter-budget math is in `model.py`'s module docstring.

---

## 6. Troubleshooting

- **`FileNotFoundError` on `train_noisy.en.txt`**: the raw corpus isn't
  under `--data_dir` — check step 1 of setup.
- **Parameter budget exceeded** (`train.py` raises `ValueError`): reduce
  `--d_model`, `--d_ff`, `--num_encoder_layers`, `--num_decoder_layers`,
  or `--vocab_size` and re-run; `python model.py` gives a fast
  parameter-count check without loading data.
- **Training loss not decreasing**: sanity-check with a small
  `--max_train_samples` and a handful of `--epochs` first; verify
  `output/clean_data/` looks reasonable (open a few lines by hand) before
  a full run.
- **`test.py` results look off vs. training**: make sure `--ckpt_path`
  points at the checkpoint you expect, and that `en-vi-translation-data/`
  and `output/clean_data/` haven't changed since training (the checkpoint
  pins the exact config but not the data itself).
- **`--rerank` is slow**: expected — it runs ~25+ forward trajectories per
  sentence instead of 1-5. Use `--rerank_eval_samples` (default 50) for
  BLEU, and lower `--n_beam`/`--n_sample` if even that's too slow.
- **Reverse score (`lambda_rev`) looks meaningless / near-random**: it
  relies on the checkpoint having been trained with `--p_reverse > 0`
  (the default). A checkpoint trained with `--p_reverse 0` never learned
  the `<toen>` direction, so set `--lambda_rev 0` when evaluating it.
- **LR looks tiny / training seems slow at the very start**: expected
  under `--lr_schedule noam` — it linearly warms up over `--warmup_steps`
  (default 4000) batches before decaying; watch the printed `lr=` value
  per epoch. If your dataset is small enough that 4000 steps is most of
  training, lower `--warmup_steps` (e.g. to ~10% of total steps =
  `epochs * (train_pairs / batch_size)`).
- **Baseline and this model's Tier 1 numbers use different sentences**:
  shouldn't happen since both go through
  `eval_utils.build_fair_eval_subset` with the same `--eval_sample_size`,
  but if you changed `--data_dir` between the two runs, re-run both
  against the same corpus.

---

See `exercise-machine-translation-student/machine_translation.ipynb` for
the original baseline (vanilla RNN) and full task description, and
`exercise-machine-translation-student/report_requirements.md` for the
required report structure.
