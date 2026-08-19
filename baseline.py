"""
baseline.py
-----------
Standalone, faithful reproduction of the exercise notebook's vanilla-RNN
Seq2Seq baseline (whitespace `Vocabulary`, no attention, greedy decoding
only) -- trains it and reports BLEU two ways:
  1. On its own first `--eval_sample_size` test sentences, UNSMOOTHED
     corpus_bleu -- exactly the exercise notebook's own methodology.
  2. On the SAME shared fair-comparison subset `test.py` uses (see
     eval_utils.py) -- so this number is directly comparable to
     `python test.py`'s Tier-1 output for the improved model.

Run:
    python baseline.py --data_dir ./en-vi-translation-data --epochs 5
    python test.py --ckpt_path ./output/checkpoint.pt   # compare Tier-1 BLEU
"""
import argparse
import json
import os
import random
import re
import time

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from eval_utils import build_fair_eval_subset, normalize_string, DEFAULT_EVAL_SAMPLE_SIZE


class Vocabulary:
    def __init__(self):
        self.word2index = {"<pad>": 0, "<unk>": 1, "<sos>": 2, "<eos>": 3}
        self.index2word = {0: "<pad>", 1: "<unk>", 2: "<sos>", 3: "<eos>"}
        self.num_words = 4

    def add_sentence(self, sentence):
        for word in sentence.split():
            if word not in self.word2index:
                self.word2index[word] = self.num_words
                self.index2word[self.num_words] = word
                self.num_words += 1


class BaselineDataset(Dataset):
    def __init__(self, pairs, src_vocab, trg_vocab, max_len=50):
        self.pairs, self.src_vocab, self.trg_vocab, self.max_len = pairs, src_vocab, trg_vocab, max_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_s, trg_s = self.pairs[idx]
        src_tokens = [self.src_vocab.word2index.get(w, 1) for w in src_s.split()][: self.max_len - 2]
        trg_tokens = [self.trg_vocab.word2index.get(w, 1) for w in trg_s.split()][: self.max_len - 2]
        return [2] + src_tokens + [3], [2] + trg_tokens + [3]


def collate_fn(batch):
    src_list = [torch.tensor(s) for s, _ in batch]
    trg_list = [torch.tensor(t) for _, t in batch]
    src_padded = nn.utils.rnn.pad_sequence(src_list, batch_first=True, padding_value=0)
    trg_padded = nn.utils.rnn.pad_sequence(trg_list, batch_first=True, padding_value=0)
    return src_padded, trg_padded


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_size, batch_first=True)

    def forward(self, src):
        outputs, hidden = self.rnn(self.embedding(src))
        return outputs, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_size):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.RNN(emb_dim, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, output_dim)

    def forward(self, input, hidden):
        embedded = self.embedding(input.unsqueeze(1))
        output, hidden = self.rnn(embedded, hidden)
        return self.fc_out(output.squeeze(1)), hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder, self.decoder, self.device = encoder, decoder, device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)
        _, hidden = self.encoder(src)
        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden)
            outputs[:, t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1)
        return outputs


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def read_langs(src_path, trg_path, max_samples=None):
    """Matches the exercise notebook's read_langs exactly, including the
    empty-after-normalization filter."""
    with open(src_path, "r", encoding="utf-8") as f:
        src_lines = f.readlines()
    with open(trg_path, "r", encoding="utf-8") as f:
        trg_lines = f.readlines()
    pairs = []
    for s, t in zip(src_lines, trg_lines):
        s_norm, t_norm = normalize_string(s), normalize_string(t)
        if s_norm and t_norm:
            pairs.append((s_norm, t_norm))
    return pairs[:max_samples] if max_samples else pairs


def translate_sentence(model, sentence, src_vocab, trg_vocab, device, max_len=50):
    model.eval()
    tokens = [src_vocab.word2index.get(t, 1) for t in sentence.split()]
    tokens = [2] + tokens + [3]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    with torch.no_grad():
        _, hidden = model.encoder(src_tensor)
    trg_tokens = [2]
    for _ in range(max_len):
        trg_tensor = torch.LongTensor([trg_tokens[-1]]).to(device)
        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden)
        best_token = output.argmax(1).item()
        trg_tokens.append(best_token)
        if best_token == 3:
            break
    return [trg_vocab.index2word.get(idx, "<unk>") for idx in trg_tokens[1:]]


def calculate_bleu(model, pairs, src_vocab, trg_vocab, device, max_samples=200, smoothing=False):
    targets, predictions = [], []
    for src_s, trg_s in pairs[:max_samples] if max_samples else pairs:
        pred_words = translate_sentence(model, src_s, src_vocab, trg_vocab, device)
        if pred_words and pred_words[-1] == "<eos>":
            pred_words = pred_words[:-1]
        targets.append([trg_s.split()])
        predictions.append(pred_words)
    smoothing_fn = SmoothingFunction().method4 if smoothing else None
    return corpus_bleu(targets, predictions, smoothing_function=smoothing_fn) * 100


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./en-vi-translation-data")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--max_train_samples", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=5,
                         help="The baseline is slow per-epoch (no batching tricks, greedy RNN); "
                              "keep this short unless you specifically need the fully-converged baseline.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_sample_size", type=int, default=DEFAULT_EVAL_SAMPLE_SIZE)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_src = os.path.join(args.data_dir, "train_noisy.en.txt")
    train_trg = os.path.join(args.data_dir, "train.vi.txt")
    val_src = os.path.join(args.data_dir, "val_noisy.en.txt")
    val_trg = os.path.join(args.data_dir, "val.vi.txt")
    test_src = os.path.join(args.data_dir, "test_noisy.en.txt")
    test_trg = os.path.join(args.data_dir, "test.vi.txt")

    train_pairs = read_langs(train_src, train_trg, max_samples=args.max_train_samples)
    val_pairs = read_langs(val_src, val_trg)
    test_pairs = read_langs(test_src, test_trg)

    src_vocab, trg_vocab = Vocabulary(), Vocabulary()
    for s, t in train_pairs:
        src_vocab.add_sentence(s)
        trg_vocab.add_sentence(t)
    print(f"Source vocab: {src_vocab.num_words:,}  Target vocab: {trg_vocab.num_words:,}")

    train_ds = BaselineDataset(train_pairs, src_vocab, trg_vocab, args.max_len)
    val_ds = BaselineDataset(val_pairs, src_vocab, trg_vocab, args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    enc = Encoder(src_vocab.num_words, args.emb_dim, args.hidden_size)
    dec = Decoder(trg_vocab.num_words, args.emb_dim, args.hidden_size)
    model = Seq2Seq(enc, dec, device).to(device)
    print(f"Baseline trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_losses, val_losses = [], []
    for epoch in range(args.epochs):
        start = time.time()
        model.train()
        epoch_loss = 0.0
        for src, trg in tqdm(train_loader, desc=f"epoch {epoch+1}", leave=False):
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg, 0.5)
            output_dim = output.shape[-1]
            loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, trg in val_loader:
                src, trg = src.to(device), trg.to(device)
                output = model(src, trg, 0)
                output_dim = output.shape[-1]
                val_loss += criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1)).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        elapsed = time.time() - start
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {avg_train_loss:.4f} "
              f"| Val Loss: {avg_val_loss:.4f} | {elapsed:.1f}s")

    ckpt_path = os.path.join(args.output_dir, "baseline_checkpoint.pt")
    torch.save({
        "model_state_dict": model.state_dict(), "args": vars(args),
        "src_vocab": src_vocab.__dict__, "trg_vocab": trg_vocab.__dict__,
    }, ckpt_path)

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="dodgerblue", lw=2)
    plt.plot(val_losses, label="Validation Loss", color="crimson", lw=2)
    plt.title("Baseline (vanilla RNN): Training and Validation Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
    plot_path = os.path.join(args.output_dir, "baseline_loss_curve.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved {ckpt_path} and {plot_path}")

    # --- Evaluation, matching test.py's two-tier methodology ---
    print(f"\n=== Tier 0: exact exercise-notebook methodology (own first "
          f"{args.eval_sample_size} test sentences, unsmoothed) ===")
    bleu_own = calculate_bleu(model, test_pairs, src_vocab, trg_vocab, device,
                                max_samples=args.eval_sample_size, smoothing=False)
    print(f"Baseline BLEU: {bleu_own:.2f}")

    print(f"\n=== Tier 1: shared fair-comparison subset (matches test.py's Tier 1) ===")
    fair_pairs = build_fair_eval_subset(test_src, test_trg, n=args.eval_sample_size)
    fair_pairs_normalized = [(normalize_string(s), normalize_string(t)) for s, t in fair_pairs]
    bleu_fair = calculate_bleu(model, fair_pairs_normalized, src_vocab, trg_vocab, device,
                                 max_samples=len(fair_pairs_normalized), smoothing=True)
    print(f"Baseline BLEU on shared {len(fair_pairs)}-sentence subset (smoothed): {bleu_fair:.2f}")
    print(f"Run `python test.py --ckpt_path {os.path.join(args.output_dir, 'checkpoint.pt')}` "
          f"and compare its Tier 1 numbers directly against {bleu_fair:.2f}.")

    results_path = os.path.join(args.output_dir, "baseline_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump({
            "trainable_parameters": count_parameters(model),
            "bleu_own_methodology": bleu_own,
            "bleu_fair_subset": bleu_fair,
            "eval_sample_size": args.eval_sample_size,
            "epochs": args.epochs,
        }, f, indent=2)
    print(f"Saved {results_path}")


if __name__ == "__main__":
    main()
