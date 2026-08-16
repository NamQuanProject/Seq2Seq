# Seq2Seq

Noisy EN->VI machine translation: compact Transformer encoder-decoder
(<=5M trainable params) trained on a denoised + BPE-tokenized corpus.

## Setup
```
pip install -r requirements.txt
```
Place the raw corpus under `./en-vi-translation-data/` with files:
`train_noisy.en.txt`, `train.vi.txt`, `val_noisy.en.txt`, `val.vi.txt`,
`test_noisy.en.txt`, `test.vi.txt`. (This directory and `./output/` are
gitignored — they hold raw data / generated artifacts, not source.)

## Pipeline
- `tokenizer.py` — rule-based denoising (missing-space repair via
  dictionary-checked word segmentation, elongated-char / repeated-punctuation
  collapse, disallowed-character stripping) + a shared BPE subword
  vocabulary trained only on the cleaned training split. Also exposes
  `denoise_report(...)`, which quantifies how much noise was actually fixed
  (counts + vocab-size reduction) for the report's data-centric analysis.
- `data.py` — reads the parallel corpus, applies `tokenizer.py`, writes
  `output/denoise_stats.json`, and builds DataLoaders (`load_data(...)`).
- `model.py` — compact Transformer (self-attention encoder + custom decoder
  that exposes cross-attention weights for visualization), with weight-tied
  output projection to save parameters. Greedy and beam-search decoding
  (`build_model`, `count_parameters`).
- `train.py` — trains the model with label smoothing, checks the parameter
  budget, saves `output/checkpoint.pt` + `output/loss_curve.png`.
- `test.py` — loads a checkpoint, reports BLEU for greedy vs. beam search,
  and dumps attention-map visualizations (`output/attention_example_*.png`)
  for a few noisy sentences.

## Run
```
python train.py --data_dir ./en-vi-translation-data --epochs 20
python test.py --ckpt_path ./output/checkpoint.pt --data_dir ./en-vi-translation-data
```
All generated artifacts (checkpoints, tokenizers, plots, stats) land in
`./output/` by default (`--output_dir` to override).

## Why Transformer over the notebook's vanilla RNN
Self-attention gives every source position a direct (O(1)-hop) path to
every other position, so the encoder can learn to down-weight a garbage
token or a locally-inverted phrase when building context for the rest of
the sentence — instead of that signal having to survive being carried
through a chain of recurrent hidden states. It's also more
parameter-efficient per unit of capacity than stacked recurrent gates,
which matters directly under the 5M budget (see the parameter breakdown
in `model.py`'s docstring).

See `exercise-machine-translation-student/machine_translation.ipynb` for the
baseline vanilla-RNN reference and `report_requirements.md` for the report
structure.
