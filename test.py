"""
test.py
-------
Evaluation script: loads a trained checkpoint, reports BLEU on the noisy
test set for both greedy and beam-search decoding, and prints a small
qualitative analysis (with attention-map visualization) for a few
hand-picked noisy sentences.

Run:
    python test.py --ckpt_path ./checkpoint.pt --data_dir ./en-vi-translation-data
"""
import argparse

import torch
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

from data import load_data
from model import build_model, count_parameters
from tokenizer import clean_text


def ids_to_words(tok, ids):
    return tok.decode_ids(ids, skip_specials=True).split()


@torch.no_grad()
def evaluate_bleu(model, loader, trg_tok, device, decode_fn, max_samples=None, max_len=60):
    """decode_fn(model, src, src_lens) -> list[int] token ids for a single sentence."""
    model.eval()
    refs, hyps = [], []
    n_seen = 0
    for src, src_lens, trg in tqdm(loader, desc="Evaluating"):
        for i in range(src.shape[0]):
            if max_samples is not None and n_seen >= max_samples:
                break
            s = src[i : i + 1].to(device)
            s_len = src_lens[i : i + 1]
            pred_ids = decode_fn(model, s, s_len, max_len=max_len)
            hyp_words = ids_to_words(trg_tok, pred_ids)
            ref_words = ids_to_words(trg_tok, trg[i].tolist())
            hyps.append(hyp_words)
            refs.append([ref_words])
            n_seen += 1
        if max_samples is not None and n_seen >= max_samples:
            break
    smoothing = SmoothingFunction().method4
    bleu = corpus_bleu(refs, hyps, smoothing_function=smoothing) * 100
    return bleu, refs, hyps


def greedy_decode_fn(model, src, src_lens, max_len=60):
    seqs, _ = model.greedy_decode(src, src_lens, max_len=max_len)
    return seqs[0]


def beam_decode_fn(model, src, src_lens, max_len=60, beam_width=5):
    return model.beam_search_decode(src, src_lens, max_len=max_len, beam_width=beam_width)


def plot_attention(attn, src_tokens, trg_tokens, save_path):
    fig, ax = plt.subplots(figsize=(max(6, len(src_tokens) * 0.5), max(4, len(trg_tokens) * 0.4)))
    im = ax.imshow(attn, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=90)
    ax.set_yticks(range(len(trg_tokens)))
    ax.set_yticklabels(trg_tokens)
    ax.set_xlabel("Source (noisy English)")
    ax.set_ylabel("Predicted (Vietnamese)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved attention map to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./en-vi-translation-data")
    parser.add_argument("--tok_dir", default="./tokenizers")
    parser.add_argument("--ckpt_path", default="./checkpoint.pt")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--beam_width", type=int, default=5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    train_args = ckpt["args"]

    bundle = load_data(
        data_dir=args.data_dir,
        tok_dir=args.tok_dir,
        max_train_samples=train_args["max_train_samples"],
        max_len=train_args["max_len"],
        batch_size=train_args["batch_size"],
        src_vocab_size=train_args["src_vocab_size"],
        trg_vocab_size=train_args["trg_vocab_size"],
    )

    model = build_model(
        ckpt["src_vocab_size"], ckpt["trg_vocab_size"], device,
        emb_dim=train_args["emb_dim"], hidden_dim=train_args["hidden_dim"],
        dropout=train_args["dropout"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    n_params = count_parameters(model)
    print(f"Loaded model with {n_params:,} trainable parameters "
          f"(budget: {train_args.get('param_budget', 5_000_000):,})")

    # --- Quantitative: BLEU, greedy vs beam search ---
    max_len = train_args["max_len"]
    print("\n=== Greedy decoding ===")
    bleu_greedy, _, _ = evaluate_bleu(
        model, bundle.test_loader, bundle.trg_tok, device,
        greedy_decode_fn, max_samples=args.max_eval_samples, max_len=max_len,
    )
    print(f"Greedy BLEU on noisy test set: {bleu_greedy:.2f}")

    print("\n=== Beam search decoding ===")
    bleu_beam, _, _ = evaluate_bleu(
        model, bundle.test_loader, bundle.trg_tok, device,
        lambda m, s, sl, max_len: beam_decode_fn(m, s, sl, max_len=max_len, beam_width=args.beam_width),
        max_samples=args.max_eval_samples, max_len=max_len,
    )
    print(f"Beam search (width={args.beam_width}) BLEU on noisy test set: {bleu_beam:.2f}")

    # --- Qualitative: hand-picked noisy sentences + attention map ---
    print("\n=== Qualitative analysis ===")
    qual_sentences = [
        "i  havean   apple!!!  soooo goood lol vacx blah",
        "apples i like .",
        "the science bruh lmao vacx behind a climate headline",
    ]
    for i, raw in enumerate(qual_sentences):
        cleaned = clean_text(raw)
        src_ids = torch.tensor([bundle.src_tok.encode_with_specials(cleaned)], dtype=torch.long).to(device)
        src_lens = torch.tensor([src_ids.shape[1]])

        greedy_ids = greedy_decode_fn(model, src_ids, src_lens, max_len=max_len)
        beam_ids = beam_decode_fn(model, src_ids, src_lens, max_len=max_len, beam_width=args.beam_width)

        print(f"\n--- Example {i+1} ---")
        print(f"Raw source     : {raw}")
        print(f"Cleaned source : {cleaned}")
        print(f"Greedy pred    : {' '.join(ids_to_words(bundle.trg_tok, greedy_ids))}")
        print(f"Beam pred      : {' '.join(ids_to_words(bundle.trg_tok, beam_ids))}")

        seqs, attn = model.greedy_decode(src_ids, src_lens, max_len=max_len)
        pred_ids = seqs[0]
        n_steps = len(pred_ids) - 1  # skip leading <sos>
        src_pieces = bundle.src_tok.tk.encode(cleaned).tokens
        trg_pieces = [bundle.trg_tok.decode_ids([t], skip_specials=False) or "?" for t in pred_ids[1 : n_steps + 1]]
        attn_matrix = attn[0, :n_steps, : len(src_pieces)].numpy()
        plot_attention(attn_matrix, src_pieces, trg_pieces, f"attention_example_{i+1}.png")


if __name__ == "__main__":
    main()
