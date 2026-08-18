"""
test.py
-------
Evaluation script for the tiny Transformer encoder-decoder translator:
loads a checkpoint, reports BLEU on the noisy test set for greedy, beam
search, and (optionally) the generate-many-then-rerank pipeline, and
prints a qualitative analysis (with cross-attention heatmaps) for a few
hand-picked noisy sentences.

Run:
    python test.py --ckpt_path ./output/checkpoint.pt --data_dir ./en-vi-translation-data
"""
import argparse
import os

import torch
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from tqdm import tqdm

from data import load_data, encode_pair
from model import build_model, count_parameters
from tokenizer import clean_text, EOS_ID
from rerank import generate_and_rerank


def ids_to_words(tok, ids):
    return tok.decode_ids(ids, skip_specials=True).split()


@torch.no_grad()
def evaluate_bleu(model, loader, tok, device, generate_fn, max_samples=None, max_new_tokens=60):
    """generate_fn(model, enc_ids) -> list[int] decoder sequence (starting with <sos>)."""
    model.eval()
    refs, hyps = [], []
    n_seen = 0
    for enc_ids, _dec_in, dec_tgt in tqdm(loader, desc="Evaluating"):
        for i in range(enc_ids.shape[0]):
            if max_samples is not None and n_seen >= max_samples:
                break
            enc = enc_ids[i : i + 1].to(device)
            dec_seq = generate_fn(model, enc, max_new_tokens=max_new_tokens)
            gen_ids = dec_seq[1:]
            if EOS_ID in gen_ids:
                gen_ids = gen_ids[: gen_ids.index(EOS_ID)]

            ref_ids = [t for t in dec_tgt[i].tolist() if t != 0]  # 0 = <pad>
            if ref_ids and ref_ids[-1] == EOS_ID:
                ref_ids = ref_ids[:-1]

            hyps.append(ids_to_words(tok, gen_ids))
            refs.append([ids_to_words(tok, ref_ids)])
            n_seen += 1
        if max_samples is not None and n_seen >= max_samples:
            break
    smoothing = SmoothingFunction().method4
    bleu = corpus_bleu(refs, hyps, smoothing_function=smoothing) * 100
    return bleu, refs, hyps


def greedy_generate_fn(model, enc_ids, max_new_tokens=60):
    seq, _ = model.greedy_generate(enc_ids, max_new_tokens=max_new_tokens)
    return seq


def beam_generate_fn(model, enc_ids, max_new_tokens=60, beam_width=5):
    return model.beam_search_generate(enc_ids, max_new_tokens=max_new_tokens, beam_width=beam_width)


@torch.no_grad()
def evaluate_bleu_rerank(model, tok, pairs, device, max_len, max_samples=None, **rerank_kwargs):
    """pairs: list of (cleaned_src, cleaned_trg). Runs the full
    generate-candidates -> filter -> rerank pipeline (rerank.py) per
    sentence -- much more expensive than greedy/beam (10-30 forward
    trajectories per sentence instead of 1-5), so `max_samples` matters
    more here than for the other two evaluate_bleu calls."""
    model.eval()
    refs, hyps = [], []
    subset = pairs[:max_samples] if max_samples is not None else pairs
    for src, trg in tqdm(subset, desc="Reranked eval"):
        best_text, _, _, _ = generate_and_rerank(model, tok, src, max_len, device, **rerank_kwargs)
        hyps.append(best_text.split())
        refs.append([trg.split()])
    smoothing = SmoothingFunction().method4
    bleu = corpus_bleu(refs, hyps, smoothing_function=smoothing) * 100
    return bleu


def plot_attention(attn, src_tokens, trg_tokens, save_path):
    """attn: cross-attention matrix [trg_len, src_len] -- for each
    generated Vietnamese token (row), how much weight the decoder placed
    on each English source token (column)."""
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
    parser.add_argument("--clean_dir", default=None,
                         help="Defaults to the clean_dir used at training time (falls back to ./output/clean_data).")
    parser.add_argument("--tok_dir", default="./output/tokenizers")
    parser.add_argument("--ckpt_path", default="./output/checkpoint.pt")
    parser.add_argument("--output_dir", default="./output")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--rerank", action="store_true",
                         help="Also evaluate the generate-many-then-rerank pipeline (rerank.py). "
                              "Much slower per sentence than greedy/beam -- consider --rerank_eval_samples.")
    parser.add_argument("--rerank_eval_samples", type=int, default=50,
                         help="Subset size for the (expensive) reranked BLEU eval; None = full test set.")
    parser.add_argument("--n_beam", type=int, default=15)
    parser.add_argument("--n_sample", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--alpha", type=float, default=0.9, help="Length-normalization exponent.")
    parser.add_argument("--lambda_cov", type=float, default=0.3, help="Coverage-penalty weight.")
    parser.add_argument("--lambda_rev", type=float, default=0.2, help="Reverse log P(x|y) weight.")
    parser.add_argument("--lambda_rep", type=float, default=0.5, help="Repetition-penalty weight.")
    parser.add_argument("--use_mbr", action="store_true",
                         help="After S(y|x) reranking, pick among the top candidates by MBR (chrF self-consistency) instead of the top score directly.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    train_args = ckpt["args"]

    clean_dir = args.clean_dir or train_args.get("clean_dir", "./output/clean_data")
    bundle = load_data(
        data_dir=args.data_dir,
        clean_dir=clean_dir,
        tok_dir=args.tok_dir,
        max_train_samples=train_args["max_train_samples"],
        max_len=train_args["max_len"],
        batch_size=train_args["batch_size"],
        vocab_size=train_args["vocab_size"],
    )

    model = build_model(
        ckpt["vocab_size"], device,
        d_model=train_args["d_model"], nhead=train_args["nhead"],
        num_encoder_layers=train_args["num_encoder_layers"],
        num_decoder_layers=train_args["num_decoder_layers"],
        dim_feedforward=train_args["d_ff"],
        dropout=train_args["dropout"], max_len=train_args["max_len"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    n_params = count_parameters(model)
    print(f"Loaded model with {n_params:,} trainable parameters "
          f"(budget: {train_args.get('param_budget', 5_000_000):,})")

    max_new_tokens = train_args["max_len"] // 2

    # --- Quantitative: BLEU, greedy vs beam search ---
    print("\n=== Greedy decoding ===")
    bleu_greedy, _, _ = evaluate_bleu(
        model, bundle.test_loader, bundle.tok, device,
        greedy_generate_fn, max_samples=args.max_eval_samples, max_new_tokens=max_new_tokens,
    )
    print(f"Greedy BLEU on noisy test set: {bleu_greedy:.2f}")

    print("\n=== Beam search decoding ===")
    bleu_beam, _, _ = evaluate_bleu(
        model, bundle.test_loader, bundle.tok, device,
        lambda m, e, max_new_tokens: beam_generate_fn(m, e, max_new_tokens=max_new_tokens, beam_width=args.beam_width),
        max_samples=args.max_eval_samples, max_new_tokens=max_new_tokens,
    )
    print(f"Beam search (width={args.beam_width}) BLEU on noisy test set: {bleu_beam:.2f}")

    rerank_kwargs = dict(
        n_beam=args.n_beam, n_sample=args.n_sample, beam_width=args.beam_width,
        temperature=args.temperature, max_new_tokens=max_new_tokens,
        alpha=args.alpha, lambda_cov=args.lambda_cov, lambda_rev=args.lambda_rev,
        lambda_rep=args.lambda_rep, use_mbr=args.use_mbr,
    )
    if args.rerank:
        print("\n=== Generate-candidates + rerank decoding ===")
        bleu_rerank = evaluate_bleu_rerank(
            model, bundle.tok, bundle.test_pairs_clean, device, train_args["max_len"],
            max_samples=args.rerank_eval_samples, **rerank_kwargs,
        )
        n_eval = args.rerank_eval_samples or len(bundle.test_pairs_clean)
        print(f"Rerank BLEU on noisy test set (n={n_eval}): {bleu_rerank:.2f}")

    # --- Qualitative: hand-picked noisy sentences + attention map ---
    print("\n=== Qualitative analysis ===")
    qual_sentences = [
        "i  havean   apple!!!  soooo goood lol vacx blah",
        "apples i like .",
        "the science bruh lmao vacx behind a climate headline",
    ]
    for i, raw in enumerate(qual_sentences):
        cleaned = clean_text(raw)
        encoder_ids, _, _ = encode_pair(cleaned, "", bundle.tok, train_args["max_len"])
        enc_tensor = torch.tensor([encoder_ids], dtype=torch.long, device=device)

        greedy_seq, attn = model.greedy_generate(enc_tensor, max_new_tokens=max_new_tokens)
        beam_seq = model.beam_search_generate(enc_tensor, max_new_tokens=max_new_tokens, beam_width=args.beam_width)

        def extract_gen(seq):
            gen = seq[1:]
            if EOS_ID in gen:
                gen = gen[: gen.index(EOS_ID)]
            return gen

        rerank_text, rerank_best, _, dropped = generate_and_rerank(
            model, bundle.tok, cleaned, train_args["max_len"], device, **rerank_kwargs,
        )

        print(f"\n--- Example {i+1} ---")
        print(f"Raw source     : {raw}")
        print(f"Cleaned source : {cleaned}")
        print(f"Greedy pred    : {' '.join(ids_to_words(bundle.tok, extract_gen(greedy_seq)))}")
        print(f"Beam pred      : {' '.join(ids_to_words(bundle.tok, extract_gen(beam_seq)))}")
        print(f"Rerank pred    : {rerank_text}  "
              f"(score={rerank_best['score']:.3f}, rep={rerank_best['repetition']:.2f}, "
              f"{len(dropped)} malformed candidates dropped)")

        if attn is not None:
            # `attn` is the LAST decode step's cross-attention, i.e. it
            # covers decoder positions [0, L) where L is one shorter than
            # the final greedy sequence (the final token is appended right
            # after that pass) -- trim the target-side pieces to match.
            trg_len = attn.shape[1]
            src_pieces = [bundle.tok.decode_ids([t], skip_specials=False) or "?" for t in encoder_ids[2:-1]]
            trg_pieces = [bundle.tok.decode_ids([t], skip_specials=False) or "?" for t in greedy_seq[1 : trg_len + 1]]
            attn_matrix = attn[0, :, 2:-1].cpu().numpy()  # drop <sos>/<toXX>/<eos> source columns
            plot_attention(
                attn_matrix, src_pieces, trg_pieces,
                os.path.join(args.output_dir, f"attention_example_{i+1}.png"),
            )


if __name__ == "__main__":
    main()
