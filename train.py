"""
train.py
--------
Trains the attention Seq2Seq model, tracks train/val loss, checks the
5,000,000 parameter budget, and saves a checkpoint + loss curve plot.

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
from tqdm import tqdm
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


def run_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for src, src_lens, trg in loader:
            src, trg = src.to(device), trg.to(device)
            if train:
                optimizer.zero_grad()
            output = model(src, src_lens, trg, teacher_forcing_ratio if train else 0.0)

            vocab_size = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, vocab_size)
            trg_flat = trg[:, 1:].reshape(-1)
            loss = criterion(output_flat, trg_flat)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./en-vi-translation-data")
    parser.add_argument("--tok_dir", default="./tokenizers")
    parser.add_argument("--max_train_samples", type=int, default=30000)
    parser.add_argument("--src_vocab_size", type=int, default=6000)
    parser.add_argument("--trg_vocab_size", type=int, default=6000)
    parser.add_argument("--emb_dim", type=int, default=160)
    parser.add_argument("--hidden_dim", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5)
    parser.add_argument("--max_len", type=int, default=60)
    parser.add_argument("--ckpt_path", default="./checkpoint.pt")
    parser.add_argument("--plot_path", default="./loss_curve.png")
    parser.add_argument("--param_budget", type=int, default=5_000_000)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    bundle = load_data(
        data_dir=args.data_dir,
        tok_dir=args.tok_dir,
        max_train_samples=args.max_train_samples,
        max_len=args.max_len,
        batch_size=args.batch_size,
        src_vocab_size=args.src_vocab_size,
        trg_vocab_size=args.trg_vocab_size,
    )

    model = build_model(
        bundle.src_tok.vocab_size_actual,
        bundle.trg_tok.vocab_size_actual,
        device,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    n_params = count_parameters(model)
    print(f"Trainable parameters: {n_params:,} (budget: {args.param_budget:,})")
    if n_params > args.param_budget:
        raise ValueError(
            f"Model has {n_params:,} params, exceeding the {args.param_budget:,} budget. "
            "Reduce --emb_dim / --hidden_dim / vocab sizes."
        )

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    print("Starting training...")
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = run_epoch(
            model, bundle.train_loader, optimizer, criterion, device,
            args.teacher_forcing_ratio, train=True,
        )
        val_loss = run_epoch(
            model, bundle.val_loader, optimizer, criterion, device,
            0.0, train=False,
        )
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        elapsed = time.time() - start

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "args": vars(args),
                "src_vocab_size": bundle.src_tok.vocab_size_actual,
                "trg_vocab_size": bundle.trg_tok.vocab_size_actual,
                "train_losses": train_losses,
                "val_losses": val_losses,
            }, args.ckpt_path)

        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} | {elapsed:.1f}s | best ckpt saved: {val_loss <= best_val_loss}")

    # Plot loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss", color="dodgerblue", lw=2)
    plt.plot(val_losses, label="Validation Loss", color="crimson", lw=2)
    plt.title("Training and Validation Cross-Entropy Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(args.plot_path, dpi=150, bbox_inches="tight")
    print(f"Saved loss curve to {args.plot_path}")
    print(f"Best checkpoint (val_loss={best_val_loss:.4f}) saved to {args.ckpt_path}")


if __name__ == "__main__":
    main()