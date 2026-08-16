# Seq2Seq

Noisy EN->VI machine translation: a GPT-small-style decoder-only
Transformer (~2.2M trainable params, well under the 5M budget and the
2.5M bonus threshold) trained on a denoised + jointly BPE-tokenized
corpus, using a prefix-LM translation recipe.

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
  collapse, disallowed-character stripping) + a single BPE subword
  vocabulary shared across English and Vietnamese, trained only on the
  cleaned training split. Also exposes `denoise_report(...)`, which
  quantifies how much noise was actually fixed (counts + vocab-size
  reduction) for the report's data-centric analysis.
- `data.py` — reads the parallel corpus, applies `tokenizer.py`, writes
  `output/denoise_stats.json`, and packs each pair into one sequence
  `<sos> src_tokens <sep> trg_tokens <eos>` with loss-mask labels
  (`load_data(...)`, `encode_pair(...)`).
- `model.py` — `GPTTranslator`: a compact GPT-style decoder-only
  Transformer (causal self-attention blocks, weight-tied token
  embedding/output projection). Greedy and beam-search generation
  (`build_model`, `count_parameters`).
- `train.py` — trains the model as a prefix language model (loss computed
  only on the target span) with label smoothing, checks the parameter
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

## Why a GPT-style decoder-only model
Translation is framed as "continue this sequence": the model reads
`<sos> src_tokens <sep>` and autoregressively writes the Vietnamese
continuation, trained with the loss masked to the target span only. This
uses ONE shared token embedding and ONE Transformer stack for both
reading the noisy source and writing the clean target, instead of an
encoder-decoder's two separate towers plus cross-attention — which is
what lets a reasonably deep/wide model (4 layers, d_model=192) land at
~2.2M params instead of ~4.5M+ for an equivalent-capacity encoder-decoder.
Self-attention over the full prefix+target sequence still lets any
generated Vietnamese token look directly back at any English source
token (including noisy/gibberish ones), so the same attention-map
analysis (does the model ignore garbage source tokens?) still applies —
see `test.py`'s attention plots.

See `exercise-machine-translation-student/machine_translation.ipynb` for the
baseline vanilla-RNN reference and `report_requirements.md` for the report
structure.
