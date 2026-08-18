"""
model.py
--------
GPT-small-style decoder-only Transformer for noisy EN->VI translation,
trained as a prefix language model:

    <sos> src_tokens... <sep> trg_tokens... <eos>

with loss masked to 0 on the <sos>/src/<sep> prefix (see data.py). At
inference time, the model is given the prompt `<sos> src_tokens <sep>`
and autoregressively continues it -- this IS translation, just phrased as
"continue this sequence" instead of "encode then decode two towers".

Why this beats a separate encoder+decoder under a tight parameter budget:
  - A single shared token embedding (one joint EN/VI BPE vocabulary) and a
    single Transformer stack do the job that an encoder-decoder spends on
    TWO embedding tables and TWO stacks (with cross-attention on top).
    Every parameter is reused for both "reading" the noisy source and
    "writing" the clean target, instead of being split across towers.
  - Self-attention over the whole prefix+target sequence still lets any
    generated Vietnamese token attend directly back to any English source
    token (including noisy/gibberish ones) -- we recover the same
    attention-map analysis by reading the row for a generated token and
    looking at the columns over the source span.

Parameter budget accounting (see build_model defaults, joint vocab=6000):
  - token embedding (tied with output projection): 6000 * 192      ~= 1.152M
  - learned positional embedding:                    128 * 192      ~= 0.025M
  - 5 pre-LN transformer blocks (self-attn + FF, d_ff=256):         ~= 1.239M
  - output bias + final layernorm:                                   ~= 6.4K
  Total                                                              ~= 2.42M
(exact math per-layer: attn = 4*d_model^2 + 4*d_model; ff =
385*d_ff + 192; block = attn + ff + 2*2*d_model for the two layernorms).
This targets the 2,500,000 bonus threshold with a deliberate ~3% safety
margin (about 78K params) below it, and clears the 5,000,000 hard budget
with enormous margin. Always confirm with `count_parameters(model)` /
`python model.py` before training -- the exact number also depends on the
tokenizer's actual trained vocab size, which can come in under the
requested `vocab_size` if the training corpus is small.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import PAD_ID, SOS_ID, EOS_ID, SEP_ID


def _banned_next_tokens(seq, n):
    """No-repeat-n-gram blocking (Paulus et al. / fairseq-style): if the
    last (n-1) tokens of `seq` have appeared as an n-gram prefix before,
    ban whatever token followed it previously -- prevents the classic
    decoding failure mode of looping on the same short phrase."""
    if n is None or n <= 0 or len(seq) < n - 1:
        return ()
    prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
    banned = set()
    for i in range(len(seq) - n + 1):
        if tuple(seq[i:i + n - 1]) == prefix:
            banned.add(seq[i + n - 1])
    return banned


class GPTBlock(nn.Module):
    """Pre-LN Transformer decoder block: causal self-attention + MLP.
    Custom (rather than nn.TransformerEncoderLayer) so we can pull out the
    self-attention weights for the qualitative attention-map analysis --
    the built-in module doesn't expose them layer-by-layer conveniently."""

    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask, key_padding_mask):
        h = self.ln1(x)
        attn_out, attn_w = self.attn(
            h, h, h, attn_mask=causal_mask, key_padding_mask=key_padding_mask,
            need_weights=True, average_attn_weights=True,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x, attn_w  # attn_w: [batch, seq_len, seq_len]


class GPTTranslator(nn.Module):
    def __init__(
        self, vocab_size, device,
        d_model=192, nhead=6, n_layer=5, d_ff=256, dropout=0.1, max_len=128,
        pad_id=PAD_ID, sos_id=SOS_ID, eos_id=EOS_ID, sep_id=SEP_ID,
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id, self.sos_id, self.eos_id, self.sep_id = pad_id, sos_id, eos_id, sep_id

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([GPTBlock(d_model, nhead, d_ff, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        # Output projection weight is tied to tok_emb (standard GPT weight
        # tying): only an extra per-token bias is learned on top of it.
        self.lm_bias = nn.Parameter(torch.zeros(vocab_size))

    def _logits(self, x):
        return F.linear(x, self.tok_emb.weight, self.lm_bias)

    @staticmethod
    def _causal_mask(sz, device):
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def forward(self, ids):
        _, seq_len = ids.shape
        pos = torch.arange(seq_len, device=ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        causal_mask = self._causal_mask(seq_len, ids.device)
        key_padding_mask = ids == self.pad_id

        attn_w = None
        for block in self.blocks:
            x, attn_w = block(x, causal_mask, key_padding_mask)
        x = self.ln_f(x)
        logits = self._logits(x)
        return logits, attn_w

    @torch.no_grad()
    def greedy_generate(self, prompt_ids, max_new_tokens=60, no_repeat_ngram_size=3, min_length=1):
        """prompt_ids: [1, prompt_len] = <sos> <toXX> src... <sep>. Returns
        the full sequence (prompt + generated continuation up to <eos>) and
        the last self-attention map."""
        self.eval()
        ids = prompt_ids.clone()
        prompt_len = ids.size(1)
        last_attn = None
        for _ in range(max_new_tokens):
            if ids.size(1) >= self.max_len:
                break
            logits, attn = self.forward(ids)
            last_attn = attn
            step_logits = logits[0, -1].clone()
            if ids.size(1) - prompt_len < min_length:
                step_logits[self.eos_id] = float("-inf")
            for banned in _banned_next_tokens(ids[0].tolist(), no_repeat_ngram_size):
                step_logits[banned] = float("-inf")
            next_tok = step_logits.argmax().view(1, 1)
            ids = torch.cat([ids, next_tok], dim=1)
            if next_tok.item() == self.eos_id:
                break
        return ids[0].tolist(), last_attn

    @torch.no_grad()
    def beam_search_generate(
        self, prompt_ids, max_new_tokens=60, beam_width=5, length_penalty=0.7,
        no_repeat_ngram_size=3, min_length=1,
    ):
        """Single-best beam search (thin wrapper over `beam_search_candidates`,
        kept for backward compatibility / simple call sites)."""
        candidates = self.beam_search_candidates(
            prompt_ids, max_new_tokens=max_new_tokens, beam_width=beam_width,
            num_return=1, length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size, min_length=min_length,
        )
        return candidates[0]["seq"]

    @torch.no_grad()
    def beam_search_candidates(
        self, prompt_ids, max_new_tokens=60, beam_width=10, num_return=10,
        length_penalty=0.7, no_repeat_ngram_size=3, min_length=1, src_span=None,
    ):
        """k-best beam search: returns up to `num_return` candidates as
        dicts {"seq": [...], "logprob": float, "coverage": [float]|None},
        sorted best-first by length-normalized log-probability. This is the
        primary candidate source for `rerank.py`'s "generate N candidates,
        then rerank" pipeline (paired with `sample_generate` for extra
        diversity via low-temperature sampling).

        If `src_span=(start, end)` (the source-token column range in the
        packed sequence) is given, per-step self-attention over that span
        is accumulated into a per-candidate `coverage` vector -- this is
        the GNMT-style coverage signal `rerank.py` turns into a coverage
        penalty (rewards attending to ALL source tokens at least once,
        useful when noisy/garbled inputs make it easy to silently drop a
        source word from the translation).
        """
        assert prompt_ids.shape[0] == 1, "beam_search_candidates expects batch_size=1"
        self.eval()
        prompt = prompt_ids[0].tolist()
        src_len = (src_span[1] - src_span[0]) if src_span else 0

        def norm_score(seq, score):
            return score / max(1, len(seq) - len(prompt)) ** length_penalty

        # beam entry: (seq, logprob, finished, coverage)
        beams = [(prompt, 0.0, False, [0.0] * src_len if src_span else None)]
        finished_hyps = []

        for _ in range(max_new_tokens):
            candidates = []
            any_active = False
            for seq, score, finished, cov in beams:
                if finished or len(seq) >= self.max_len:
                    if finished:
                        finished_hyps.append((seq, score, cov))
                    continue
                any_active = True
                ids = torch.tensor([seq], dtype=torch.long, device=self.device)
                logits, attn = self.forward(ids)
                step_attn = attn[0, -1, src_span[0]:src_span[1]].tolist() if src_span else None

                log_probs = F.log_softmax(logits[0, -1], dim=-1).clone()
                if len(seq) - len(prompt) < min_length:
                    log_probs[self.eos_id] = float("-inf")
                for banned in _banned_next_tokens(seq, no_repeat_ngram_size):
                    log_probs[banned] = float("-inf")

                topk = min(beam_width, log_probs.size(-1))
                topk_log_probs, topk_ids = log_probs.topk(topk)
                for k in range(topk):
                    tok = topk_ids[k].item()
                    new_cov = [c + a for c, a in zip(cov, step_attn)] if cov is not None else None
                    candidates.append((
                        seq + [tok], score + topk_log_probs[k].item(),
                        tok == self.eos_id, new_cov,
                    ))
            if not any_active:
                break

            candidates.sort(key=lambda c: norm_score(c[0], c[1]), reverse=True)
            beams = candidates[:beam_width]
            if len(finished_hyps) >= num_return and all(b[2] for b in beams):
                break

        # Flush whatever's left in `beams` (both properly-finished-but-not-
        # yet-flushed hypotheses from an early `num_return`-triggered break,
        # and unfinished ones that simply ran out of budget) -- dedupe
        # against what's already in finished_hyps by sequence identity.
        already = {tuple(s) for s, _, _ in finished_hyps}
        for seq, score, finished, cov in beams:
            if tuple(seq) not in already:
                finished_hyps.append((seq, score, cov))

        finished_hyps.sort(key=lambda c: norm_score(c[0], c[1]), reverse=True)
        return [
            {"seq": seq, "logprob": score, "coverage": cov}
            for seq, score, cov in finished_hyps[:num_return]
        ]

    @torch.no_grad()
    def sample_generate(
        self, prompt_ids, max_new_tokens=60, temperature=0.7, top_k=20,
        no_repeat_ngram_size=3, min_length=1,
    ):
        """Low-temperature top-k sampling: cheap source of DIVERSE candidates
        that beam search (which tends toward near-duplicate high-probability
        hypotheses) under-represents. Returns {"seq": [...], "logprob": float}."""
        assert prompt_ids.shape[0] == 1
        self.eval()
        ids = prompt_ids.clone()
        prompt_len = ids.size(1)
        total_logprob = 0.0
        for _ in range(max_new_tokens):
            if ids.size(1) >= self.max_len:
                break
            logits, _ = self.forward(ids)
            step_logits = logits[0, -1] / max(temperature, 1e-5)
            if ids.size(1) - prompt_len < min_length:
                step_logits[self.eos_id] = float("-inf")
            for banned in _banned_next_tokens(ids[0].tolist(), no_repeat_ngram_size):
                step_logits[banned] = float("-inf")
            if top_k and top_k > 0:
                v, _ = step_logits.topk(min(top_k, step_logits.size(-1)))
                step_logits = step_logits.clone()
                step_logits[step_logits < v[-1]] = float("-inf")
            probs = F.softmax(step_logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            total_logprob += torch.log(probs[next_tok] + 1e-12).item()
            ids = torch.cat([ids, next_tok.view(1, 1)], dim=1)
            if next_tok.item() == self.eos_id:
                break
        return {"seq": ids[0].tolist(), "logprob": total_logprob}

    @torch.no_grad()
    def coverage_vector(self, ids, src_span, gen_start):
        """One forward pass over a full (prompt+candidate) sequence; sums
        the last layer's self-attention weights that the generated
        positions [gen_start:] place on source columns `src_span=(start,
        end)`. Works uniformly for ANY candidate (beam or sampled) since it
        recomputes from the finished sequence, rather than needing
        incremental bookkeeping during decoding. Used by rerank.py's
        GNMT-style coverage penalty: a source token that's never attended
        to across the whole generation is a token that was likely silently
        dropped from the translation."""
        ids = ids if ids.dim() == 2 else ids.unsqueeze(0)
        _, attn = self.forward(ids)
        if attn is None:
            return None
        sub = attn[0, gen_start:, src_span[0]:src_span[1]]
        return sub.sum(dim=0).tolist()

    @torch.no_grad()
    def score_sequence(self, ids):
        """Teacher-forced total log P(ids[1:] | ids[0]) under the model, and
        the per-token log-probs. Used by `rerank.py` to score a candidate's
        forward probability log P(y|x) and (via a reverse-tagged sequence)
        its reverse probability log P(x|y), without re-running generation."""
        ids = ids if ids.dim() == 2 else ids.unsqueeze(0)
        logits, _ = self.forward(ids[:, :-1])
        log_probs = F.log_softmax(logits, dim=-1)
        targets = ids[:, 1:]
        token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return token_logprobs.sum().item(), token_logprobs[0].tolist()


def build_model(
    vocab_size, device,
    d_model=192, nhead=6, n_layer=5, d_ff=256, dropout=0.1, max_len=128,
):
    model = GPTTranslator(
        vocab_size, device,
        d_model=d_model, nhead=nhead, n_layer=n_layer, d_ff=d_ff,
        dropout=dropout, max_len=max_len,
    ).to(device)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cpu")
    m = build_model(6000, device)
    print(f"Trainable parameters: {count_parameters(m):,}")
