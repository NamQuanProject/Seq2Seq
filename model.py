"""
model.py
--------
Tiny Transformer ENCODER-DECODER for noisy EN<->VI translation.

    Component           Value
    Encoder layers       3
    Decoder layers       3
    Model dimension       128
    Attention heads        4
    FFN dimension         512
    Vocabulary          8,000 joint BPE tokens (shared EN+VI, see tokenizer.py)
    Embeddings          shared across encoder input, decoder input, AND
                         the output projection (weight tying)
    Position encoding   sinusoidal (0 extra parameters)
    Dropout              0.1

This is the standard architecture for the task (bidirectional encoder
self-attention over the full noisy source, causal decoder self-attention
+ cross-attention into the encoder) rather than the single-stack
decoder-only "prefix-LM" trick used in an earlier version of this file --
worth it here because:
  - Cross-attention gives a direct, per-decoder-step distribution over
    source positions -- exactly what the assignment's attention-map /
    coverage-penalty analysis wants, without any workaround.
  - Encoding the source ONCE and reusing that fixed `memory` across every
    decoding step (greedy/beam/sampling) is cheaper at inference than the
    decoder-only version's need to re-run a full forward pass over the
    ever-growing concatenated [prompt + generation-so-far] sequence.
  - A shared-vocabulary encoder-decoder is still small: the vocabulary
    dominates parameter count, and it's paid for once (see below).

A leading direction tag on the encoder side (<tovi>/<toen>, see
tokenizer.py) tells this SAME encoder+decoder pair which way to
translate, so it's bidirectional (used by rerank.py's reverse-model
scoring) without any extra parameters.

Parameter budget accounting (see build_model defaults, joint vocab=8000):
  - shared token embedding (tied with output projection): 8000*128  ~= 1.024M
  - 3 encoder layers (self-attn + FF, d_ff=512):                     ~= 0.595M
  - 3 decoder layers (self-attn + cross-attn + FF, d_ff=512):        ~= 0.794M
  - output bias + final layernorm:                                    ~= 8.3K
  Total                                                               ~= 2.42M
(exact math per layer: self/cross-attn = 4*d_model^2 + 4*d_model each;
ff = 2*d_model*d_ff + d_ff + d_model; encoder layer has 1 attn + 2
layernorms, decoder layer has 2 attn (self+cross) + 3 layernorms).
This clears the 2,500,000 bonus threshold with a deliberate ~3% safety
margin, and clears the 5,000,000 hard budget with enormous margin.
Always confirm with `count_parameters(model)` / `python model.py` --
the exact number also depends on the tokenizer's actual trained vocab
size, which can come in under the requested `vocab_size` if the
training corpus is small.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import PAD_ID, SOS_ID, EOS_ID


def _banned_next_tokens(seq, n):
    """No-repeat-n-gram blocking: if the last (n-1) tokens of `seq` have
    appeared as an n-gram prefix before, ban whatever token followed it
    previously -- prevents the classic decoding failure mode of looping
    on the same short phrase."""
    if n is None or n <= 0 or len(seq) < n - 1:
        return ()
    prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
    banned = set()
    for i in range(len(seq) - n + 1):
        if tuple(seq[i:i + n - 1]) == prefix:
            banned.add(seq[i + n - 1])
    return banned


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding -- adds no trainable parameters."""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class DecoderLayer(nn.Module):
    """Pre-LN: causal self-attention -> cross-attention into the encoder
    memory -> MLP. Custom (rather than nn.TransformerDecoderLayer) so we
    can pull out the cross-attention weights for the coverage-penalty and
    attention-map qualitative analysis -- the built-in module doesn't
    expose them layer-by-layer conveniently."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, causal_mask, tgt_key_padding_mask, memory_key_padding_mask):
        h = self.ln1(x)
        sa_out, _ = self.self_attn(
            h, h, h, attn_mask=causal_mask, key_padding_mask=tgt_key_padding_mask, need_weights=False,
        )
        x = x + self.dropout(sa_out)

        h2 = self.ln2(x)
        ca_out, ca_w = self.cross_attn(
            h2, memory, memory, key_padding_mask=memory_key_padding_mask,
            need_weights=True, average_attn_weights=True,
        )
        x = x + self.dropout(ca_out)

        x = x + self.dropout(self.ff(self.ln3(x)))
        return x, ca_w  # ca_w: [batch, tgt_len, src_len]


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self, vocab_size, device,
        d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=512, dropout=0.1, max_len=128,
        pad_id=PAD_ID, sos_id=SOS_ID, eos_id=EOS_ID,
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id, self.sos_id, self.eos_id = pad_id, sos_id, eos_id

        # ONE embedding table, shared by the encoder input, the decoder
        # input, AND (weight-tied) the output projection.
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.emb_dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot init for every weight matrix with >1 dim (Linear
        and MultiheadAttention projections), matching the original
        "Attention Is All You Need" recipe. Left to PyTorch's default
        (small normal), a from-scratch Transformer this small tends to
        start with noisy attention logits and takes noticeably longer to
        leave a poor loss plateau; Xavier init keeps the variance of
        activations roughly constant through the stack from step 1,
        which measurably speeds up and stabilizes early convergence.
        Biases are left at zero (PyTorch default) and LayerNorm/embedding
        parameters are left alone (embeddings already get a principled
        scale from the sqrt(d_model) multiply in `_embed`)."""
        for name, p in self.named_parameters():
            if p.dim() > 1 and "tok_emb" not in name:
                nn.init.xavier_uniform_(p)

    def _embed(self, ids):
        return self.emb_dropout(self.pos_enc(self.tok_emb(ids) * math.sqrt(self.d_model)))

    def _output_proj(self, x):
        return F.linear(x, self.tok_emb.weight, self.output_bias)

    @staticmethod
    def _causal_mask(sz, device):
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def encode(self, enc_ids):
        memory_kpm = enc_ids == self.pad_id
        x = self._embed(enc_ids)
        memory = self.encoder(x, src_key_padding_mask=memory_kpm)
        return memory, memory_kpm

    def decode(self, dec_ids, memory, memory_kpm):
        tgt_kpm = dec_ids == self.pad_id
        causal_mask = self._causal_mask(dec_ids.size(1), dec_ids.device)
        x = self._embed(dec_ids)
        cross_attn = None
        for layer in self.decoder_layers:
            x, cross_attn = layer(x, memory, causal_mask, tgt_kpm, memory_kpm)
        x = self.ln_f(x)
        logits = self._output_proj(x)
        return logits, cross_attn

    def forward(self, enc_ids, dec_ids):
        memory, memory_kpm = self.encode(enc_ids)
        logits, cross_attn = self.decode(dec_ids, memory, memory_kpm)
        return logits, cross_attn

    # -- greedy / beam / sampling generation (batch_size=1) -----------------
    @torch.no_grad()
    def greedy_generate(self, enc_ids, max_new_tokens=60, no_repeat_ngram_size=3, min_length=1):
        """enc_ids: [1, src_len] = <sos> <toXX> src... <eos>. Returns the
        decoder token sequence (starting with <sos>) and the last
        cross-attention map [1, tgt_len, src_len]."""
        assert enc_ids.shape[0] == 1
        self.eval()
        memory, memory_kpm = self.encode(enc_ids)
        dec_ids = torch.tensor([[self.sos_id]], dtype=torch.long, device=enc_ids.device)
        last_attn = None
        for _ in range(max_new_tokens):
            if dec_ids.size(1) >= self.max_len:
                break
            logits, attn = self.decode(dec_ids, memory, memory_kpm)
            last_attn = attn
            step_logits = logits[0, -1].clone()
            if dec_ids.size(1) - 1 < min_length:
                step_logits[self.eos_id] = float("-inf")
            for banned in _banned_next_tokens(dec_ids[0].tolist(), no_repeat_ngram_size):
                step_logits[banned] = float("-inf")
            next_tok = step_logits.argmax().view(1, 1)
            dec_ids = torch.cat([dec_ids, next_tok], dim=1)
            if next_tok.item() == self.eos_id:
                break
        return dec_ids[0].tolist(), last_attn

    @torch.no_grad()
    def beam_search_generate(
        self, enc_ids, max_new_tokens=60, beam_width=5, length_penalty=0.7,
        no_repeat_ngram_size=3, min_length=1,
    ):
        """Single-best beam search (thin wrapper over `beam_search_candidates`)."""
        candidates = self.beam_search_candidates(
            enc_ids, max_new_tokens=max_new_tokens, beam_width=beam_width,
            num_return=1, length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size, min_length=min_length,
        )
        return candidates[0]["seq"]

    @torch.no_grad()
    def beam_search_candidates(
        self, enc_ids, max_new_tokens=60, beam_width=10, num_return=10,
        length_penalty=0.7, no_repeat_ngram_size=3, min_length=1,
    ):
        """Batched beam search: all `beam_width` beams are stacked into one
        batch dimension and decoded with a SINGLE forward pass per step,
        instead of a Python loop calling `decode()` once per beam. The
        encoder still runs ONCE and `memory` is expanded (not recomputed)
        across the beam dimension. This is the practical win for a small
        model like this one: on top of the encoder-once optimization, the
        per-step decoder forward pass is what dominates beam search cost,
        and batching it keeps the GPU (or even CPU vectorized ops) doing
        one wide matmul instead of `beam_width` narrow sequential ones.

        Finished beams are frozen in place (kept alive by forcing their
        next token to <eos> at zero additional log-prob) rather than
        removed, which is what keeps every step a fixed-shape batched op --
        the standard trick (e.g. fairseq's SequenceGenerator) for
        batched beam search without variable-width tensors. One
        consequence: at most `beam_width` distinct trajectories are ever
        tracked, so if `num_return` is requested larger than `beam_width`,
        the search is silently run with a wider beam instead of quietly
        returning fewer candidates than asked for.

        Returns up to `num_return` candidates as dicts {"seq": [...],
        "logprob": float, "coverage": None} -- `coverage` is no longer
        computed here (rerank.py always recomputes it per final candidate
        via `coverage_vector`, so carrying it through the beam search was
        dead weight)."""
        assert enc_ids.shape[0] == 1
        self.eval()
        device = enc_ids.device
        beam_width = max(beam_width, num_return)

        memory, memory_kpm = self.encode(enc_ids)
        memory = memory.expand(beam_width, -1, -1)
        memory_kpm = memory_kpm.expand(beam_width, -1)

        beams_seq = [[self.sos_id] for _ in range(beam_width)]
        beam_scores = torch.zeros(beam_width, device=device)
        beam_scores[1:] = float("-inf")  # only beam 0 is "real" at step 0 -- avoids beam_width identical seeds
        finished = torch.zeros(beam_width, dtype=torch.bool, device=device)

        def norm_score(seq_len, score):
            return score / max(1, seq_len - 1) ** length_penalty

        for _ in range(max_new_tokens):
            if bool(finished.all()) or len(beams_seq[0]) >= self.max_len:
                break
            dec_ids = torch.tensor(beams_seq, dtype=torch.long, device=device)  # [beam_width, cur_len]
            logits, _ = self.decode(dec_ids, memory, memory_kpm)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)  # [beam_width, vocab]

            for b in range(beam_width):
                if finished[b].item():
                    log_probs[b] = float("-inf")
                    log_probs[b, self.eos_id] = 0.0  # zero-cost "stay finished" continuation
                    continue
                if len(beams_seq[b]) - 1 < min_length:
                    log_probs[b, self.eos_id] = float("-inf")
                for banned in _banned_next_tokens(beams_seq[b], no_repeat_ngram_size):
                    log_probs[b, banned] = float("-inf")

            candidate_scores = (beam_scores.unsqueeze(1) + log_probs).reshape(-1)  # [beam_width * vocab]
            vocab_size = log_probs.size(-1)
            topk_scores, topk_flat = candidate_scores.topk(beam_width)
            parent_beams = torch.div(topk_flat, vocab_size, rounding_mode="floor")
            next_tokens = topk_flat % vocab_size

            new_beams_seq, new_finished = [], []
            for i in range(beam_width):
                parent = parent_beams[i].item()
                tok = next_tokens[i].item()
                new_beams_seq.append(beams_seq[parent] + [tok])
                new_finished.append(bool(finished[parent].item()) or tok == self.eos_id)
            beams_seq = new_beams_seq
            beam_scores = topk_scores
            finished = torch.tensor(new_finished, dtype=torch.bool, device=device)

        results = [
            {"seq": beams_seq[b], "logprob": beam_scores[b].item(), "coverage": None}
            for b in range(beam_width)
        ]
        results.sort(key=lambda c: norm_score(len(c["seq"]), c["logprob"]), reverse=True)
        return results[:num_return]

    @torch.no_grad()
    def sample_generate(
        self, enc_ids, max_new_tokens=60, temperature=0.7, top_k=20,
        no_repeat_ngram_size=3, min_length=1,
    ):
        """Low-temperature top-k sampling: cheap source of DIVERSE candidates
        that beam search under-represents. Returns {"seq": [...], "logprob": float}."""
        assert enc_ids.shape[0] == 1
        self.eval()
        memory, memory_kpm = self.encode(enc_ids)
        dec_ids = torch.tensor([[self.sos_id]], dtype=torch.long, device=enc_ids.device)
        total_logprob = 0.0
        for _ in range(max_new_tokens):
            if dec_ids.size(1) >= self.max_len:
                break
            logits, _ = self.decode(dec_ids, memory, memory_kpm)
            step_logits = logits[0, -1] / max(temperature, 1e-5)
            if dec_ids.size(1) - 1 < min_length:
                step_logits[self.eos_id] = float("-inf")
            for banned in _banned_next_tokens(dec_ids[0].tolist(), no_repeat_ngram_size):
                step_logits[banned] = float("-inf")
            if top_k and top_k > 0:
                v, _ = step_logits.topk(min(top_k, step_logits.size(-1)))
                step_logits = step_logits.clone()
                step_logits[step_logits < v[-1]] = float("-inf")
            probs = F.softmax(step_logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            total_logprob += torch.log(probs[next_tok] + 1e-12).item()
            dec_ids = torch.cat([dec_ids, next_tok.view(1, 1)], dim=1)
            if next_tok.item() == self.eos_id:
                break
        return {"seq": dec_ids[0].tolist(), "logprob": total_logprob}

    # -- scoring (for rerank.py) --------------------------------------------
    @torch.no_grad()
    def score_sequence(self, enc_ids, dec_full_ids):
        """Teacher-forced total log P(dec_full_ids[1:] | enc_ids) and the
        per-token log-probs. `dec_full_ids` = <sos> ... <eos>."""
        dec_full_ids = dec_full_ids if dec_full_ids.dim() == 2 else dec_full_ids.unsqueeze(0)
        memory, memory_kpm = self.encode(enc_ids)
        logits, _ = self.decode(dec_full_ids[:, :-1], memory, memory_kpm)
        log_probs = F.log_softmax(logits, dim=-1)
        targets = dec_full_ids[:, 1:]
        token_logprobs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return token_logprobs.sum().item(), token_logprobs[0].tolist()

    @torch.no_grad()
    def coverage_vector(self, enc_ids, dec_full_ids):
        """One teacher-forced decode pass; sums the last decoder layer's
        cross-attention weights over the FULL encoder sequence across every
        decoder position. rerank.py slices the result down to the real
        source-token span (excluding <sos>/<toXX>/<eos>). A source position
        that's never attended to across the whole generation is a source
        token that was likely silently dropped from the translation."""
        dec_full_ids = dec_full_ids if dec_full_ids.dim() == 2 else dec_full_ids.unsqueeze(0)
        memory, memory_kpm = self.encode(enc_ids)
        _, cross_attn = self.decode(dec_full_ids[:, :-1], memory, memory_kpm)
        if cross_attn is None:
            return None
        return cross_attn[0].sum(dim=0).tolist()


def build_model(
    vocab_size, device,
    d_model=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
    dim_feedforward=512, dropout=0.1, max_len=128,
):
    model = Seq2SeqTransformer(
        vocab_size, device,
        d_model=d_model, nhead=nhead,
        num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward, dropout=dropout, max_len=max_len,
    ).to(device)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cpu")
    m = build_model(8000, device)
    print(f"Trainable parameters: {count_parameters(m):,}")
