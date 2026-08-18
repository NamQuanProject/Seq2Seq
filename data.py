"""
data.py
-------
Data-centric pipeline for a tiny Transformer ENCODER-DECODER translation
model (see model.py).

We train ONE shared BPE vocabulary over the concatenation of cleaned
English + Vietnamese training text, and use it for both the encoder input
and the decoder input/output. A leading direction tag (`<tovi>`/`<toen>`)
on the ENCODER side tells the (single) shared-weight encoder+decoder pair
which way to translate, so the same model is bidirectional at zero extra
parameters -- used by rerank.py's reverse-model scoring.

Per example:
  encoder_ids  = <sos> <toXX> a_tokens... <eos>
  decoder_in   = <sos> b_tokens...              (teacher-forcing input)
  decoder_tgt  = b_tokens... <eos>              (shifted-by-one loss target)
where (a, b) = (src, trg) normally, or (trg, src) for the occasional
reverse-direction training example.
"""
import json
import os
import random
from collections import namedtuple

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import (
    PAD_ID, SOS_ID, EOS_ID, TOEN_ID, TOVI_ID,
    clean_text, build_or_load_tokenizer, denoise_report, augment_noise,
)

DataBundle = namedtuple(
    "DataBundle",
    [
        "train_loader", "val_loader", "test_loader",
        "tok",  # single shared tokenizer
        "train_pairs", "val_pairs", "test_pairs",  # raw (uncleaned) pairs, for display
        "train_pairs_clean", "val_pairs_clean", "test_pairs_clean",
    ],
)


def read_parallel(src_path, trg_path, max_samples=None):
    with open(src_path, "r", encoding="utf-8") as f:
        src_lines = [l.strip() for l in f.readlines()]
    with open(trg_path, "r", encoding="utf-8") as f:
        trg_lines = [l.strip() for l in f.readlines()]
    assert len(src_lines) == len(trg_lines), (
        f"Mismatched line counts: {src_path} has {len(src_lines)}, {trg_path} has {len(trg_lines)}"
    )
    pairs = list(zip(src_lines, trg_lines))
    if max_samples is not None:
        pairs = pairs[:max_samples]
    return pairs


def encode_pair(src_text, trg_text, tok, max_len, direction="en2vi"):
    """Returns (encoder_ids, decoder_input_ids, decoder_target_ids).
    direction="en2vi" is the real task (a=src, b=trg); "vi2en" packs the
    reverse direction instead (a=trg, b=src) -- trained alongside the
    forward direction so the SAME model can later score log P(src|trg)
    for reranking; see rerank.py."""
    tag_id = TOVI_ID if direction == "en2vi" else TOEN_ID
    a_text, b_text = (src_text, trg_text) if direction == "en2vi" else (trg_text, src_text)

    enc_budget = max_len - 3  # <sos>, <toXX>, <eos>
    dec_budget = max_len - 2  # <sos>, <eos>
    a_ids = tok.encode_ids(a_text)[:enc_budget]
    b_ids = tok.encode_ids(b_text)[:dec_budget]

    encoder_ids = [SOS_ID, tag_id] + a_ids + [EOS_ID]
    decoder_input = [SOS_ID] + b_ids
    decoder_target = b_ids + [EOS_ID]
    return encoder_ids, decoder_input, decoder_target


class TranslationDataset(Dataset):
    """Holds already-cleaned text pairs and packs them lazily into
    (encoder_ids, decoder_input, decoder_target) triples.

    If `p_reverse > 0`, a fraction of examples are packed in the reverse
    (VI->EN) direction instead, making the model bidirectional (used for
    reverse-model rescoring at inference). If `augment` is True, the
    English source is corrupted with `tokenizer.augment_noise` BEFORE
    packing (independently, each epoch, since re-sampled per __getitem__
    call) -- exposes the model to noise beyond what's fixed in the file.
    Both are meant for the TRAINING split only; leave both off for val/test
    so evaluation is deterministic and measures the real task."""

    def __init__(self, cleaned_pairs, tok, max_len=128, p_reverse=0.0, augment=False, seed=42):
        self.pairs = cleaned_pairs
        self.tok = tok
        self.max_len = max_len
        self.p_reverse = p_reverse
        self.augment = augment
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_s, trg_s = self.pairs[idx]
        if self.augment:
            src_s = augment_noise(src_s, self.rng)
        direction = "vi2en" if (self.p_reverse > 0 and self.rng.random() < self.p_reverse) else "en2vi"
        enc_ids, dec_in, dec_tgt = encode_pair(src_s, trg_s, self.tok, self.max_len, direction=direction)
        return (
            torch.tensor(enc_ids, dtype=torch.long),
            torch.tensor(dec_in, dtype=torch.long),
            torch.tensor(dec_tgt, dtype=torch.long),
        )


def collate_fn(batch):
    enc_list, dec_in_list, dec_tgt_list = zip(*batch)
    enc_padded = torch.nn.utils.rnn.pad_sequence(enc_list, batch_first=True, padding_value=PAD_ID)
    dec_in_padded = torch.nn.utils.rnn.pad_sequence(dec_in_list, batch_first=True, padding_value=PAD_ID)
    dec_tgt_padded = torch.nn.utils.rnn.pad_sequence(dec_tgt_list, batch_first=True, padding_value=PAD_ID)
    return enc_padded, dec_in_padded, dec_tgt_padded


def _resolve_split(data_dir, clean_dir, split, raw_src_name, raw_trg_name, max_samples=None):
    """Prefer the `preprocess.py`-produced "super clean" files
    (`<split>_clean.en.txt` / `.vi.txt` in `clean_dir`) when they exist;
    otherwise fall back to the raw noisy files + on-the-fly (conservative,
    non-deleting) `clean_text`. Returns (raw_pairs, cleaned_pairs, used_preprocessed)."""
    if clean_dir is not None:
        clean_src = os.path.join(clean_dir, f"{split}_clean.en.txt")
        clean_trg = os.path.join(clean_dir, f"{split}_clean.vi.txt")
        if os.path.exists(clean_src) and os.path.exists(clean_trg):
            cleaned_pairs = read_parallel(clean_src, clean_trg, max_samples=max_samples)
            raw_src = os.path.join(data_dir, raw_src_name)
            raw_trg = os.path.join(data_dir, raw_trg_name)
            raw_pairs = (
                read_parallel(raw_src, raw_trg, max_samples=max_samples)
                if os.path.exists(raw_src) and os.path.exists(raw_trg)
                else cleaned_pairs
            )
            return raw_pairs, cleaned_pairs, True

    raw_pairs = read_parallel(
        os.path.join(data_dir, raw_src_name), os.path.join(data_dir, raw_trg_name),
        max_samples=max_samples,
    )
    cleaned_pairs = [(clean_text(s), clean_text(t)) for s, t in raw_pairs]
    return raw_pairs, cleaned_pairs, False


def load_data(
    data_dir="./en-vi-translation-data",
    clean_dir="./output/clean_data",
    tok_dir="./output/tokenizers",
    max_train_samples=30000,
    max_len=128,
    batch_size=64,
    vocab_size=8000,
    num_workers=0,
    stats_path="./output/denoise_stats.json",
    p_reverse=0.3,
    augment_noise_p=True,
):
    train_pairs, train_pairs_clean, used_preprocessed = _resolve_split(
        data_dir, clean_dir, "train", "train_noisy.en.txt", "train.vi.txt", max_samples=max_train_samples,
    )
    val_pairs, val_pairs_clean, _ = _resolve_split(
        data_dir, clean_dir, "val", "val_noisy.en.txt", "val.vi.txt",
    )
    test_pairs, test_pairs_clean, _ = _resolve_split(
        data_dir, clean_dir, "test", "test_noisy.en.txt", "test.vi.txt",
    )
    if used_preprocessed:
        print(f"Using pre-cleaned corpus from {clean_dir} (run preprocess.py to regenerate).")
    else:
        print(
            f"No pre-cleaned corpus found in {clean_dir} -- falling back to on-the-fly "
            f"clean_text (conservative, no garbage deletion). Run "
            f"`python preprocess.py --data_dir {data_dir} --output_dir {clean_dir}` first "
            f"for the more aggressive self-correct/delete strategy."
        )

    if stats_path is not None and not os.path.exists(stats_path):
        # Measures how much noise the denoising pipeline actually fixed on
        # the (noisy) English source side of the training split -- report
        # material for the "Denoising & Tokenization" section.
        stats = denoise_report(s for s, _ in train_pairs)
        os.makedirs(os.path.dirname(stats_path) or ".", exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Denoising stats ({stats_path}): {stats}")

    os.makedirs(tok_dir, exist_ok=True)
    # ONE joint vocabulary shared across English and Vietnamese: this is what
    # lets the encoder and decoder use a single (tied) embedding/output
    # table instead of three separate ones (halving-plus embedding cost),
    # and lets the same encoder+decoder pair run in either direction.
    joint_lines = (s for pair in train_pairs_clean for s in pair)
    tok = build_or_load_tokenizer(
        joint_lines, os.path.join(tok_dir, "joint_bpe.json"), vocab_size=vocab_size,
    )

    # Bidirectional training + noise augmentation apply to TRAIN only --
    # val/test stay deterministic (forward direction, unaugmented) so BLEU
    # measures the real EN->VI task honestly.
    train_ds = TranslationDataset(
        train_pairs_clean, tok, max_len=max_len,
        p_reverse=p_reverse, augment=augment_noise_p,
    )
    val_ds = TranslationDataset(val_pairs_clean, tok, max_len=max_len)
    test_ds = TranslationDataset(test_pairs_clean, tok, max_len=max_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=num_workers,
    )

    return DataBundle(
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        tok=tok,
        train_pairs=train_pairs, val_pairs=val_pairs, test_pairs=test_pairs,
        train_pairs_clean=train_pairs_clean, val_pairs_clean=val_pairs_clean,
        test_pairs_clean=test_pairs_clean,
    )


if __name__ == "__main__":
    bundle = load_data(max_train_samples=2000)
    print(f"joint vocab: {bundle.tok.vocab_size_actual}")
    print(f"train batches: {len(bundle.train_loader)}, val batches: {len(bundle.val_loader)}")
    enc, dec_in, dec_tgt = next(iter(bundle.train_loader))
    print("encoder batch shape:", enc.shape, "decoder batch shape:", dec_in.shape)
