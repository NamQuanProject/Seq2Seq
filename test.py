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
from eval_utils import build_fair_eval_subset, DEFAULT_EVAL_SAMPLE_SIZE


def ids_to_words(tok, ids):
    return tok.decode_ids(ids, skip_specials=True).split()


@torch.no_grad()
def evaluate_bleu_on_raw_pairs(model, tok, raw_pairs, device, max_len, generate_fn, max_new_tokens=60):
    """Translates each (raw_src, raw_trg) pair directly through our own
    clean_text pipeline (not the DataLoader) -- used for the fixed
    fair-comparison subset (see eval_utils.py) so the sentence set is
    guaranteed identical to baseline.py's."""
    model.eval()
    refs, hyps = [], []
    for s_raw, t_raw in raw_pairs:
        cleaned_src, cleaned_trg = clean_text(s_raw), clean_text(t_raw)
        encoder_ids, _, _ = encode_pair(cleaned_src, "", tok, max_len)
        enc = torch.tensor([encoder_ids], dtype=torch.long, device=device)
        dec_seq = generate_fn(model, enc, max_new_tokens=max_new_tokens)
        gen_ids = dec_seq[1:]
        if EOS_ID in gen_ids:
            gen_ids = gen_ids[: gen_ids.index(EOS_ID)]
        hyps.append(ids_to_words(tok, gen_ids))
        refs.append([cleaned_trg.split()])
    smoothing = SmoothingFunction().method4
    return corpus_bleu(refs, hyps, smoothing_function=smoothing) * 100


@torch.no_grad()
def evaluate_bleu(model, loader, tok, device, generate_fn, max_samples=None, max_new_tokens=60):
    """generate_fn(model, enc_ids) -> list[int] decoder sequence (starting with <sos>).
    Full-test-set evaluation via the DataLoader (fast, batched iteration)."""
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


@torch.no_grad()
def evaluate_bleu_rerank_on_raw_pairs(model, tok, raw_pairs, device, max_len, **rerank_kwargs):
    model.eval()
    refs, hyps = [], []
    for s_raw, t_raw in raw_pairs:
        cleaned_src, cleaned_trg = clean_text(s_raw), clean_text(t_raw)
        best_text, _, _, _ = generate_and_rerank(model, tok, cleaned_src, max_len, device, **rerank_kwargs)
        hyps.append(best_text.split())
        refs.append([cleaned_trg.split()])
    smoothing = SmoothingFunction().method4
    return corpus_bleu(refs, hyps, smoothing_function=smoothing) * 100


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
    parser.add_argument("--eval_sample_size", type=int, default=DEFAULT_EVAL_SAMPLE_SIZE,
                         help="Size of the fixed, baseline.py-comparable Tier-1 evaluation subset.")
    parser.add_argument("--skip_full_test_set", action="store_true",
                         help="Skip the (larger, slower) Tier-2 full-test-set evaluation and only run Tier 1.")
    parser.add_argument("--max_eval_samples", type=int, default=None,
                         help="Optional cap on the Tier-2 full-test-set evaluation (speed knob); None = every test sentence.")
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

    # Must match training's decoder budget (data.py's encode_pair:
    # dec_budget = max_len - 2), NOT an arbitrary fraction of it -- capping
    # generation lower than what training taught the model to produce
    # would silently truncate any translation that legitimately needs more
    # tokens than the cap, regardless of what the model actually learned.
    max_new_tokens = train_args["max_len"] - 2

    rerank_kwargs = dict(
        n_beam=args.n_beam, n_sample=args.n_sample, beam_width=args.beam_width,
        temperature=args.temperature, max_new_tokens=max_new_tokens,
        alpha=args.alpha, lambda_cov=args.lambda_cov, lambda_rev=args.lambda_rev,
        lambda_rep=args.lambda_rep, use_mbr=args.use_mbr,
    )

    # --- Tier 1: fair comparison against baseline.py ---------------------
    # baseline.py's BLEU (and the exercise notebook's own methodology) is
    # computed on a fixed, small sentence subset. Comparing that directly
    # against a BLEU computed over the full (much larger, harder) test set
    # is not apples-to-apples -- so evaluate this model on the EXACT SAME
    # subset baseline.py uses, for a number that's actually comparable.
    test_src = os.path.join(args.data_dir, "test_noisy.en.txt")
    test_trg = os.path.join(args.data_dir, "test.vi.txt")
    fair_eval_pairs = build_fair_eval_subset(test_src, test_trg, n=args.eval_sample_size)
    print(f"=== Tier 1: fair comparison on the shared {len(fair_eval_pairs)}-sentence subset "
          f"(run baseline.py for the comparable baseline number) ===")

    bleu_greedy_fair = evaluate_bleu_on_raw_pairs(
        model, bundle.tok, fair_eval_pairs, device, train_args["max_len"],
        greedy_generate_fn, max_new_tokens=max_new_tokens,
    )
    print(f"Ours -- greedy      : {bleu_greedy_fair:.2f}")

    bleu_beam_fair = evaluate_bleu_on_raw_pairs(
        model, bundle.tok, fair_eval_pairs, device, train_args["max_len"],
        lambda m, e, max_new_tokens: beam_generate_fn(m, e, max_new_tokens=max_new_tokens, beam_width=args.beam_width),
        max_new_tokens=max_new_tokens,
    )
    print(f"Ours -- beam search  : {bleu_beam_fair:.2f}")

    if args.rerank:
        bleu_rerank_fair = evaluate_bleu_rerank_on_raw_pairs(
            model, bundle.tok, fair_eval_pairs, device, train_args["max_len"], **rerank_kwargs,
        )
        print(f"Ours -- rerank       : {bleu_rerank_fair:.2f}")

    # --- Tier 2: full test set (this model only, NOT baseline-comparable) --
    if not args.skip_full_test_set:
        print(f"\n=== Tier 2: full test set (n={len(bundle.test_pairs_clean)}, larger/harder -- "
              f"NOT directly comparable to Tier 1 above) ===")
        bleu_greedy, _, _ = evaluate_bleu(
            model, bundle.test_loader, bundle.tok, device,
            greedy_generate_fn, max_samples=args.max_eval_samples, max_new_tokens=max_new_tokens,
        )
        print(f"Ours -- greedy      : {bleu_greedy:.2f}")

        bleu_beam, _, _ = evaluate_bleu(
            model, bundle.test_loader, bundle.tok, device,
            lambda m, e, max_new_tokens: beam_generate_fn(m, e, max_new_tokens=max_new_tokens, beam_width=args.beam_width),
            max_samples=args.max_eval_samples, max_new_tokens=max_new_tokens,
        )
        print(f"Ours -- beam search  : {bleu_beam:.2f}")

        if args.rerank:
            bleu_rerank = evaluate_bleu_rerank(
                model, bundle.tok, bundle.test_pairs_clean, device, train_args["max_len"],
                max_samples=args.rerank_eval_samples, **rerank_kwargs,
            )
            n_eval = args.rerank_eval_samples or len(bundle.test_pairs_clean)
            print(f"Ours -- rerank (n={n_eval}) : {bleu_rerank:.2f}")

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
