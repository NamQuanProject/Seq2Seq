"""
train.py
--------
Trains the tiny Transformer encoder-decoder translator (see model.py).
Loss is standard cross-entropy over the decoder's target sequence
(`<pad>` ignored) -- no prefix-masking needed since the encoder only ever
sees the source and the decoder only ever sees/predicts the target.
Checks the 5,000,000 parameter budget and saves a checkpoint + loss curve
plot into ./output/.

Run:
    python train.py --data_dir ./en-vi-translation-data --epochs 20
"""
import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data import load_data
from model import build_model, count_parameters
from tokenizer import PAD_ID


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for enc_ids, dec_in, dec_tgt in loader:
            enc_ids, dec_in, dec_tgt = enc_ids.to(device), dec_in.to(device), dec_tgt.to(device)
            if train:
                optimizer.zero_grad()

            logits, _ = model(enc_ids, dec_in)
            vocab_size = logits.shape[-1]
            loss = criterion(logits.reshape(-1, vocab_size), dec_tgt.reshape(-1))

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./en-vi-translation-data")
    parser.add_argument("--clean_dir", default="./output/clean_data")
    parser.add_argument("--tok_dir", default="./output/tokenizers")
    parser.add_argument("--max_train_samples", type=int, default=30000)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_encoder_layers", type=int, default=3)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--param_budget", type=int, default=5_000_000)
    parser.add_argument("--bonus_param_budget", type=int, default=2_500_000)
    parser.add_argument("--p_reverse", type=float, default=0.3,
                         help="Fraction of training examples packed VI->EN instead of EN->VI, "
                              "making the model bidirectional (enables reverse-model rescoring in rerank.py).")
    parser.add_argument("--augment_noise", action="store_true", default=True,
                         help="Apply tokenizer.augment_noise (typos/case/char-drop) to the English "
                              "source at train time, resampled every epoch.")
    parser.add_argument("--no_augment_noise", dest="augment_noise", action="store_false")
    parser.add_argument("--save_last_k", type=int, default=3,
                         help="Also keep a rolling window of the last K epoch checkpoints in "
                              "output/ckpts/ for weight averaging (see average_checkpoints.py). 0 disables.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, "checkpoint.pt")
    plot_path = os.path.join(args.output_dir, "loss_curve.png")

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bundle = load_data(
        data_dir=args.data_dir,
        clean_dir=args.clean_dir,
        tok_dir=args.tok_dir,
        max_train_samples=args.max_train_samples,
        max_len=args.max_len,
        batch_size=args.batch_size,
        vocab_size=args.vocab_size,
        stats_path=os.path.join(args.output_dir, "denoise_stats.json"),
        p_reverse=args.p_reverse,
        augment_noise_p=args.augment_noise,
    )

    model = build_model(
        bundle.tok.vocab_size_actual, device,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers, num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.d_ff, dropout=args.dropout, max_len=args.max_len,
    )

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params:,} (budget: {args.param_budget:,})")
    if n_params > args.param_budget:
        raise ValueError(
            f"Model has {n_params:,} params, exceeding the {args.param_budget:,} budget. "
            "Reduce --d_model / --d_ff / --num_encoder_layers / --num_decoder_layers / --vocab_size."
        )
    if n_params <= args.bonus_param_budget:
        margin = args.bonus_param_budget - n_params
        print(f"Clears the bonus threshold (<={args.bonus_param_budget:,} params) "
              f"with {margin:,} params ({100*margin/args.bonus_param_budget:.1f}%) to spare.")
    else:
        print(f"NOTE: {n_params:,} params exceeds the {args.bonus_param_budget:,} bonus threshold "
              f"(still within the {args.param_budget:,} hard budget).")

    # Label smoothing regularizes the output distribution -- useful here
    # because noisy source sentences make some target tokens genuinely
    # ambiguous/uncertain, and smoothing keeps the model from becoming
    # overconfident on those. ignore_index skips <pad> positions in the
    # (batched, variable-length) decoder target.
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=args.label_smoothing)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    ckpts_dir = os.path.join(args.output_dir, "ckpts")
    if args.save_last_k > 0:
        os.makedirs(ckpts_dir, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    saved_ckpts = []

    print("Starting training...")
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = run_epoch(model, bundle.train_loader, optimizer, criterion, device, train=True)
        val_loss = run_epoch(model, bundle.val_loader, optimizer, criterion, device, train=False)
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        elapsed = time.time() - start

        ckpt_payload = {
            "model_state_dict": model.state_dict(),
            "args": vars(args),
            "vocab_size": bundle.tok.vocab_size_actual,
            "train_losses": train_losses,
            "val_losses": val_losses,
        }

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(ckpt_payload, ckpt_path)

        # Rolling window of the last K epoch checkpoints, independent of
        # whether this epoch was the single best -- weight averaging (see
        # average_checkpoints.py) benefits from a handful of checkpoints
        # near convergence, not just the strict single best one.
        if args.save_last_k > 0:
            epoch_ckpt_path = os.path.join(ckpts_dir, f"epoch_{epoch+1:03d}.pt")
            torch.save(ckpt_payload, epoch_ckpt_path)
            saved_ckpts.append(epoch_ckpt_path)
            if len(saved_ckpts) > args.save_last_k:
                stale = saved_ckpts.pop(0)
                if os.path.exists(stale):
                    os.remove(stale)

        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} | {elapsed:.1f}s | best ckpt saved: {is_best}")

    # Plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="dodgerblue", lw=2)
    plt.plot(val_losses, label="Validation Loss", color="crimson", lw=2)
    plt.title("Training and Validation Cross-Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved loss curve to {plot_path}")
    print(f"Best checkpoint (val_loss={best_val_loss:.4f}) saved to {ckpt_path}")


if __name__ == "__main__":
    main()
