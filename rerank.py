"""
rerank.py
---------
"Generate many, then pick the best one" inference pipeline for the tiny
Transformer encoder-decoder (model.py):

  1. Generate 10-30 candidates: a k-best beam search pool (diverse-ish by
     construction once combined with no-repeat-n-gram blocking) PLUS a
     handful of low-temperature top-k sampling draws (beam search alone
     tends to produce near-duplicate high-probability hypotheses; sampling
     fills in genuinely different candidates). The encoder runs ONCE per
     source sentence and its output is reused across every candidate.
  2. Filter malformed candidates: no <eos> within the length budget,
     leaked special tokens, degenerate/near-empty output, excessive
     repeated n-grams.
  3. Rerank the survivors with a translation-oriented score:

         S(y|x) = logP(y|x)/|y|^alpha
                   + lambda_cov * C(x,y)
                   + lambda_rev * logP(x|y)
                   - lambda_rep * R(y)

     - length normalization (`alpha`) stops the score from favoring
       short/empty translations.
     - C(x,y): GNMT-style coverage penalty from the decoder's own
       cross-attention into the encoder -- rewards attending to (not
       silently dropping) every source token, which matters most when the
       source is noisy and easy to under-translate.
     - log P(x|y): reverse-direction score from the SAME model (trained
       bidirectionally, see data.py's `p_reverse`) -- does translating the
       candidate back reconstruct the noisy source reasonably well?
     - R(y): repeated-n-gram fraction, penalizing loops beam/no-repeat
       blocking didn't fully prevent.
  4. Optionally, instead of taking the top-scoring candidate directly, do
     Minimum-Bayes-Risk (MBR) selection among the top few S(y|x) survivors:
     pick whichever candidate has the highest average chrF similarity to
     the others -- a self-consistency signal that tends to avoid picking
     one-off decoding artifacts that happened to score well under S(y|x).

This module is architecture-agnostic beyond assuming `model` exposes
`beam_search_candidates`, `sample_generate`, `score_sequence`,
`coverage_vector` (see model.py) and that `tok` is a
`tokenizer.SubwordTokenizer`.
"""
import math
from collections import Counter

import torch

from data import encode_pair
from tokenizer import SOS_ID, EOS_ID, TOEN_ID, SPECIAL_TOKENS


# ---------------------------------------------------------------------------
# chrF (character n-gram F-score) -- self-contained, no extra dependency.
# Used for MBR reranking (candidate-vs-candidate similarity).
# ---------------------------------------------------------------------------
def _char_ngrams(s, n):
    s = s.replace(" ", "")
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def chrf_score(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    if not hyp or not ref:
        return 0.0
    precisions, recalls = [], []
    for n in range(1, max_n + 1):
        h_ngrams = Counter(_char_ngrams(hyp, n))
        r_ngrams = Counter(_char_ngrams(ref, n))
        match = sum((h_ngrams & r_ngrams).values())
        h_total, r_total = sum(h_ngrams.values()), sum(r_ngrams.values())
        precisions.append(match / h_total if h_total else 0.0)
        recalls.append(match / r_total if r_total else 0.0)
    P, R = sum(precisions) / max_n, sum(recalls) / max_n
    if P + R == 0:
        return 0.0
    return (1 + beta ** 2) * P * R / (beta ** 2 * P + R)


def mbr_select(texts):
    """Return the index of the candidate with highest average chrF
    similarity to all the others (Minimum Bayes Risk under a chrF utility)."""
    n = len(texts)
    if n == 0:
        return None, []
    if n == 1:
        return 0, [1.0]
    scores = []
    for i in range(n):
        total = sum(chrf_score(texts[i], texts[j]) for j in range(n) if j != i)
        scores.append(total / (n - 1))
    best_idx = max(range(n), key=lambda i: scores[i])
    return best_idx, scores


# ---------------------------------------------------------------------------
# Repetition penalty + malformed-output filtering
# ---------------------------------------------------------------------------
def repetition_penalty(tokens, n=3):
    """Fraction of n-grams in `tokens` that are repeats of an earlier one --
    0 for no repetition, approaching 1 for a fully looping sequence."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    return repeats / max(1, len(ngrams))


def filter_malformed(candidates, max_new_tokens, special_ids, min_gen_len=1, max_rep=0.3):
    """candidates: list of {"seq": [sos, ...decoder tokens...]}. Every
    candidate seq starts with a single leading <sos> (the decoder always
    starts fresh, unlike the old prefix-LM packing). Returns (kept,
    dropped_reasons). A candidate's generated span is trimmed to its first
    <eos>; candidates that never produce <eos> within the generation
    budget, are degenerate (empty/too short), leak a special token into
    the body, or loop excessively are dropped."""
    kept, reasons = [], []
    for cand in candidates:
        gen = cand["seq"][1:]  # strip the leading <sos>
        if EOS_ID not in gen:
            reasons.append("missing_eos")
            continue
        body = gen[: gen.index(EOS_ID)]
        if len(body) < min_gen_len:
            reasons.append("too_short")
            continue
        if len(gen) > max_new_tokens:
            reasons.append("excessive_length")
            continue
        if any(t in special_ids for t in body):
            reasons.append("leaked_special_token")
            continue
        if repetition_penalty(body, n=3) > max_rep:
            reasons.append("excessive_repetition")
            continue
        cand = dict(cand)
        cand["body"] = body  # generated tokens, <eos>-trimmed, specials-free
        kept.append(cand)
    return kept, reasons


# ---------------------------------------------------------------------------
# S(y|x) reranking score
# ---------------------------------------------------------------------------
def score_candidates(
    model, enc_ids_tensor, src_ids, src_span, candidates,
    alpha=0.9, lambda_cov=0.3, lambda_rev=0.2, lambda_rep=0.5,
):
    """candidates: list of {"body": [...]} (post-`filter_malformed`).
    Returns the same list, sorted best-first, each with an added
    "score"/"logprob_fwd"/"logprob_rev_norm"/"coverage"/"repetition" breakdown."""
    device = next(model.parameters()).device

    out = []
    for cand in candidates:
        body = cand["body"]
        dec_full = torch.tensor([[SOS_ID] + body + [EOS_ID]], dtype=torch.long, device=device)

        fwd_logprob, _ = model.score_sequence(enc_ids_tensor, dec_full)
        length_norm = fwd_logprob / (len(body) + 1) ** alpha

        cov_term = 0.0
        coverage = None
        if lambda_cov and src_span is not None:
            full_cov = model.coverage_vector(enc_ids_tensor, dec_full)
            if full_cov:
                coverage = full_cov[src_span[0]:src_span[1]]
                cov_term = sum(math.log(min(c, 1.0) + 1e-6) for c in coverage)

        rev_logprob = 0.0
        if lambda_rev and src_ids:
            rev_enc = torch.tensor([[SOS_ID, TOEN_ID] + body + [EOS_ID]], dtype=torch.long, device=device)
            rev_dec = torch.tensor([[SOS_ID] + src_ids + [EOS_ID]], dtype=torch.long, device=device)
            if rev_enc.shape[1] <= model.max_len and rev_dec.shape[1] <= model.max_len:
                raw_rev_logprob, _ = model.score_sequence(rev_enc, rev_dec)
                rev_logprob = raw_rev_logprob / (len(src_ids) + 1)

        rep = repetition_penalty(body, n=3)
        score = length_norm + lambda_cov * cov_term + lambda_rev * rev_logprob - lambda_rep * rep

        out.append({
            **cand, "score": score, "logprob_fwd": fwd_logprob,
            "logprob_fwd_norm": length_norm, "logprob_rev_norm": rev_logprob,
            "coverage": coverage, "repetition": rep,
        })

    out.sort(key=lambda c: c["score"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# End-to-end: generate candidates -> filter -> rerank (-> optional MBR)
# ---------------------------------------------------------------------------
def generate_and_rerank(
    model, tok, src_text, max_len, device,
    n_beam=15, n_sample=10, beam_width=10, temperature=0.7, top_k=20,
    max_new_tokens=60, no_repeat_ngram_size=3, min_length=1,
    alpha=0.9, lambda_cov=0.3, lambda_rev=0.2, lambda_rep=0.5,
    use_mbr=False, mbr_pool=5,
):
    """Returns (best_text, best_candidate_dict, all_scored_candidates,
    dropped_reasons) for one source sentence."""
    encoder_ids, _, _ = encode_pair(src_text, "", tok, max_len, direction="en2vi")
    enc_ids_tensor = torch.tensor([encoder_ids], dtype=torch.long, device=device)
    src_ids = encoder_ids[2:-1]  # strip <sos>, <tovi>, and the trailing <eos>
    src_span = (2, len(encoder_ids) - 1)

    raw_candidates = []
    beam_cands = model.beam_search_candidates(
        enc_ids_tensor, max_new_tokens=max_new_tokens, beam_width=beam_width, num_return=n_beam,
        no_repeat_ngram_size=no_repeat_ngram_size, min_length=min_length,
    )
    raw_candidates.extend({"seq": c["seq"]} for c in beam_cands)
    for _ in range(n_sample):
        s = model.sample_generate(
            enc_ids_tensor, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k,
            no_repeat_ngram_size=no_repeat_ngram_size, min_length=min_length,
        )
        raw_candidates.append({"seq": s["seq"]})

    special_ids = set(range(len(SPECIAL_TOKENS)))
    kept, dropped_reasons = filter_malformed(
        raw_candidates, max_new_tokens=max_new_tokens, special_ids=special_ids,
    )
    if not kept:
        # Degenerate fallback: nothing survived filtering (e.g. very short
        # max_new_tokens) -- fall back to plain greedy so we always return
        # SOMETHING rather than raising.
        greedy_seq, _ = model.greedy_generate(
            enc_ids_tensor, max_new_tokens=max_new_tokens,
            no_repeat_ngram_size=no_repeat_ngram_size, min_length=min_length,
        )
        gen = greedy_seq[1:]
        body = gen[: gen.index(EOS_ID)] if EOS_ID in gen else gen
        kept = [{"seq": greedy_seq, "body": body}]

    # dedupe identical generated bodies (keep first / highest beam rank)
    seen, deduped = set(), []
    for c in kept:
        key = tuple(c["body"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)

    scored = score_candidates(
        model, enc_ids_tensor, src_ids, src_span, deduped,
        alpha=alpha, lambda_cov=lambda_cov, lambda_rev=lambda_rev, lambda_rep=lambda_rep,
    )

    if use_mbr and len(scored) > 1:
        pool = scored[:mbr_pool]
        texts = [tok.decode_ids(c["body"], skip_specials=True) for c in pool]
        idx, mbr_scores = mbr_select(texts)
        for c, s in zip(pool, mbr_scores):
            c["mbr_score"] = s
        best = pool[idx]
    else:
        best = scored[0]

    best_text = tok.decode_ids(best["body"], skip_specials=True)
    return best_text, best, scored, dropped_reasons
