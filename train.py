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


def noam_lr_lambda(step, d_model, warmup_steps):
    """Transformer "Noam" schedule (Vaswani et al.): linear warmup for
    `warmup_steps`, then inverse-sqrt decay. Standard for training small
    Transformers from scratch -- a flat or plateau-triggered LR tends to
    either destabilize the randomly-initialized attention layers early on
    (too high before the model has found reasonable attention patterns)
    or decay too late/coarsely once loss has already plateaued. Returns a
    multiplier consumed by `LambdaLR` against a base optimizer lr of 1.0,
    so the number IS the effective learning rate."""
    step = max(step, 1)
    return d_model ** -0.5 * min(step ** -0.5, step * warmup_steps ** -1.5)


def run_epoch(model, loader, optimizer, criterion, device, scheduler=None, train=True):
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
                if scheduler is not None:
                    scheduler.step()
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
    parser.add_argument("--lr_schedule", choices=["noam", "plateau"], default="noam",
                         help="'noam' (default): linear warmup + inverse-sqrt decay, stepped every "
                              "batch -- standard for training small Transformers from scratch and "
                              "generally converges faster/more stably than a flat-then-plateau LR. "
                              "'plateau': the old flat-lr + ReduceLROnPlateau behavior.")
    parser.add_argument("--lr", type=float, default=1.0,
                         help="With --lr_schedule=noam, this is a scale multiplier on the Noam curve "
                              "(1.0 gives a sensible peak LR at these defaults, ~1.4e-3). "
                              "With --lr_schedule=plateau, this is the flat starting LR (e.g. 3e-4).")
    parser.add_argument("--warmup_steps", type=int, default=4000,
                         help="Noam schedule warmup length, in optimizer steps (only used with --lr_schedule=noam).")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--param_budget", type=int, default=5_000_000)
    parser.add_argument("--bonus_param_budget", type=int, default=2_500_000)
    parser.add_argument("--p_reverse", type=float, default=0.3,
                         help="Fraction of training examples packed VI->EN instead of EN->VI, "
                              "making the model bidirectional (enables reverse-model rescoring in rerank.py).")
    parser.add_argument("--augment_mode", choices=["none", "char", "word", "both"], default="char",
                         help="Train-time English-source noise augmentation: 'char' (keyboard typos/"
                              "case/char-drop), 'word' (join/swap/delete/garbage/unk -- mirrors the "
                              "assignment's actual noise types), 'both', or 'none'.")
    parser.add_argument("--subword_regularization", action="store_true",
                         help="Sample a random (still-valid) SentencePiece Unigram segmentation per "
                              "training access instead of the single deterministic best split -- an "
                              "extra axis of robustness on top of --augment_mode.")
    parser.add_argument("--max_tokens_per_batch", type=int, default=None,
                         help="If set, uses token-budget bucketed batching (LengthBucketBatchSampler) "
                              "instead of fixed-size batches -- less padding waste. --batch_size still "
                              "caps the example count per batch as a safety net.")
    parser.add_argument("--save_last_k", type=int, default=3,
                         help="Also keep a rolling window of the last K epoch checkpoints in "
                              "output/ckpts/ for weight averaging (see average_checkpoints.py). 0 disables.")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                         help="Stop if val loss hasn't improved by --early_stopping_min_delta for this "
                              "many epochs. 0 (default) disables early stopping.")
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
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
        augment_mode=args.augment_mode,
        subword_regularization=args.subword_regularization,
        max_tokens_per_batch=args.max_tokens_per_batch,
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

    if args.lr_schedule == "noam":
        optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
        train_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: args.lr * noam_lr_lambda(step, args.d_model, args.warmup_steps),
        )
        plateau_scheduler = None
        print(f"LR schedule: noam (warmup_steps={args.warmup_steps}, scale={args.lr}, "
              f"peak lr~{args.lr * noam_lr_lambda(args.warmup_steps, args.d_model, args.warmup_steps):.2e})")
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        train_scheduler = None
        plateau_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
        print(f"LR schedule: plateau (flat lr={args.lr}, halved on val-loss plateau)")

    ckpts_dir = os.path.join(args.output_dir, "ckpts")
    if args.save_last_k > 0:
        os.makedirs(ckpts_dir, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    saved_ckpts = []
    epochs_since_improvement = 0

    print("Starting training...")
    for epoch in range(args.epochs):
        start = time.time()
        train_loss = run_epoch(model, bundle.train_loader, optimizer, criterion, device,
                                 scheduler=train_scheduler, train=True)
        val_loss = run_epoch(model, bundle.val_loader, optimizer, criterion, device, train=False)
        if plateau_scheduler is not None:
            plateau_scheduler.step(val_loss)
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
        meaningfully_improved = val_loss < best_val_loss - args.early_stopping_min_delta
        if is_best:
            best_val_loss = val_loss
            torch.save(ckpt_payload, ckpt_path)
        epochs_since_improvement = 0 if meaningfully_improved else epochs_since_improvement + 1

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

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {train_loss:.4f} "
              f"| Val Loss: {val_loss:.4f} | lr={current_lr:.2e} | {elapsed:.1f}s | best ckpt saved: {is_best}")

        if args.early_stopping_patience > 0 and epochs_since_improvement >= args.early_stopping_patience:
            print(f"Early stopping: no val-loss improvement > {args.early_stopping_min_delta} for "
                  f"{args.early_stopping_patience} epochs.")
            break

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
