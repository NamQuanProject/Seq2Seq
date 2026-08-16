# Seq2Seq

Noisy EN->VI machine translation (BiGRU + Bahdanau attention, <=5M params).

## Setup
```
pip install -r requirements.txt
```
Place the raw corpus under `./en-vi-translation-data/` with files:
`train_noisy.en.txt`, `train.vi.txt`, `val_noisy.en.txt`, `val.vi.txt`,
`test_noisy.en.txt`, `test.vi.txt`.

## Pipeline
- `tokenizer.py` — rule-based denoising (concatenation repair, elongation/punct
  collapse, character filtering) + BPE subword tokenization, trained on the
  cleaned training split only.
- `data.py` — reads the parallel corpus, applies `tokenizer.py`, builds
  DataLoaders (`load_data(...)`).
- `model.py` — bidirectional-GRU encoder + Bahdanau-attention GRU decoder,
  greedy and beam-search decoding (`build_model`, `count_parameters`).
- `train.py` — trains the model, checks the parameter budget, saves
  `checkpoint.pt` + `loss_curve.png`.
- `test.py` — loads a checkpoint, reports BLEU for greedy vs. beam search,
  and dumps attention-map visualizations for a few noisy sentences.

## Run
```
python train.py --data_dir ./en-vi-translation-data --epochs 20
python test.py --ckpt_path ./checkpoint.pt --data_dir ./en-vi-translation-data
```

See `exercise-machine-translation-student/machine_translation.ipynb` for the
baseline vanilla-RNN reference and `report_requirements.md` for the report
structure.
