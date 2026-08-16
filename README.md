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
- `tokenizer.py` — rule-based denoising building blocks:
  - `clean_text(...)` — conservative pass: elongated-char / repeated-punctuation
    collapse, disallowed-character stripping, and dictionary-checked
    missing-space repair ("self-correct"). Leaves unrecoverable gibberish
    tokens in place for the model to learn to ignore.
  - `super_clean_text(...)` — aggressive pass used by `preprocess.py`: runs
    `clean_text` then DELETES any token that still isn't a recognizable
    word after repair ("self-correct, then delete what can't be corrected").
    English-source-only; never applied to the Vietnamese target.
  - `denoise_report(...)` — quantifies how much noise was fixed (counts +
    vocab-size reduction) for the report's data-centric analysis.
  - a single BPE subword vocabulary shared across English and Vietnamese,
    trained only on the cleaned training split.
- `preprocess.py` — **run this first.** Produces a fully denoised copy of
  the corpus at `output/clean_data/{train,val,test}_clean.{en,vi}.txt`
  using `super_clean_text` on the English source (drops any pair that
  becomes empty after deletion) and light normalization on the Vietnamese
  target, plus a stats report at `output/preprocess_stats.json`.
- `data.py` — `load_data(...)` prefers the `preprocess.py` output in
  `clean_dir` when present; otherwise it falls back to the raw noisy files
  with the conservative on-the-fly `clean_text` (and prints a warning
  suggesting you run `preprocess.py`). Writes `output/denoise_stats.json`
  and packs each pair into one sequence `<sos> src_tokens <sep>
  trg_tokens <eos>` with loss-mask labels (`encode_pair(...)`).
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
python preprocess.py --data_dir ./en-vi-translation-data --output_dir ./output/clean_data
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
