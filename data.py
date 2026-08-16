"""
data.py
-------
Data-centric pipeline: reads the raw noisy parallel corpus, applies the
rule-based denoising + subword tokenization from tokenizer.py, and wraps
everything into PyTorch Datasets/DataLoaders.

This replaces the notebook baseline's whitespace `Vocabulary` (one entry
per raw token -> vocabulary explosion on noisy text) with a shared-size
BPE vocabulary trained only on the (cleaned) training split.
"""
import json
import os
from collections import namedtuple

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import (
    PAD_ID, EOS_ID,
    clean_text, build_or_load_tokenizer, denoise_report,
)

DataBundle = namedtuple(
    "DataBundle",
    [
        "train_loader", "val_loader", "test_loader",
        "src_tok", "trg_tok",
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


class TranslationDataset(Dataset):
    """Holds already-cleaned text pairs and encodes them lazily with BPE tokenizers."""

    def __init__(self, cleaned_pairs, src_tok, trg_tok, max_len=60):
        self.pairs = cleaned_pairs
        self.src_tok = src_tok
        self.trg_tok = trg_tok
        self.max_len = max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_s, trg_s = self.pairs[idx]
        src_ids = self.src_tok.encode_with_specials(src_s)[: self.max_len]
        trg_ids = self.trg_tok.encode_with_specials(trg_s)[: self.max_len]
        if src_ids[-1] != EOS_ID:
            src_ids[-1] = EOS_ID
        if trg_ids[-1] != EOS_ID:
            trg_ids[-1] = EOS_ID
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(trg_ids, dtype=torch.long)


def collate_fn(batch):
    src_list, trg_list = zip(*batch)
    src_lens = torch.tensor([len(s) for s in src_list], dtype=torch.long)
    src_padded = torch.nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=PAD_ID)
    trg_padded = torch.nn.utils.rnn.pad_sequence(trg_list, batch_first=True, padding_value=PAD_ID)
    return src_padded, src_lens, trg_padded


def load_data(
    data_dir="./en-vi-translation-data",
    tok_dir="./output/tokenizers",
    max_train_samples=30000,
    max_len=60,
    batch_size=64,
    src_vocab_size=6000,
    trg_vocab_size=6000,
    num_workers=0,
    stats_path="./output/denoise_stats.json",
):
    train_pairs = read_parallel(
        os.path.join(data_dir, "train_noisy.en.txt"),
        os.path.join(data_dir, "train.vi.txt"),
        max_samples=max_train_samples,
    )
    val_pairs = read_parallel(
        os.path.join(data_dir, "val_noisy.en.txt"),
        os.path.join(data_dir, "val.vi.txt"),
    )
    test_pairs = read_parallel(
        os.path.join(data_dir, "test_noisy.en.txt"),
        os.path.join(data_dir, "test.vi.txt"),
    )

    def clean_pairs(pairs):
        # Source is noisy English -> rule-based denoising pass.
        # Target Vietnamese is already clean -> only normalize whitespace/case
        # consistently through the same cleaner (no English-specific rules fire
        # since word-segmentation only touches pure a-z ascii tokens).
        return [(clean_text(s), clean_text(t)) for s, t in pairs]

    train_pairs_clean = clean_pairs(train_pairs)
    val_pairs_clean = clean_pairs(val_pairs)
    test_pairs_clean = clean_pairs(test_pairs)

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
    src_tok = build_or_load_tokenizer(
        (s for s, _ in train_pairs_clean),
        os.path.join(tok_dir, "src_bpe.json"),
        vocab_size=src_vocab_size,
    )
    trg_tok = build_or_load_tokenizer(
        (t for _, t in train_pairs_clean),
        os.path.join(tok_dir, "trg_bpe.json"),
        vocab_size=trg_vocab_size,
    )

    train_ds = TranslationDataset(train_pairs_clean, src_tok, trg_tok, max_len=max_len)
    val_ds = TranslationDataset(val_pairs_clean, src_tok, trg_tok, max_len=max_len)
    test_ds = TranslationDataset(test_pairs_clean, src_tok, trg_tok, max_len=max_len)

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
        src_tok=src_tok, trg_tok=trg_tok,
        train_pairs=train_pairs, val_pairs=val_pairs, test_pairs=test_pairs,
        train_pairs_clean=train_pairs_clean, val_pairs_clean=val_pairs_clean,
        test_pairs_clean=test_pairs_clean,
    )


if __name__ == "__main__":
    bundle = load_data(max_train_samples=2000)
    print(f"src vocab: {bundle.src_tok.vocab_size_actual}, trg vocab: {bundle.trg_tok.vocab_size_actual}")
    print(f"train batches: {len(bundle.train_loader)}, val batches: {len(bundle.val_loader)}")
    src, src_lens, trg = next(iter(bundle.train_loader))
    print("src batch shape:", src.shape, "trg batch shape:", trg.shape)
