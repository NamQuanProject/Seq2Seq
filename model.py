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
    def greedy_generate(self, prompt_ids, max_new_tokens=60):
        """prompt_ids: [1, prompt_len] = <sos> src... <sep>. Returns the
        full sequence (prompt + generated continuation up to <eos>) and the
        last self-attention map."""
        self.eval()
        ids = prompt_ids.clone()
        last_attn = None
        for _ in range(max_new_tokens):
            if ids.size(1) >= self.max_len:
                break
            logits, attn = self.forward(ids)
            last_attn = attn
            next_tok = logits[:, -1].argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_tok], dim=1)
            if next_tok.item() == self.eos_id:
                break
        return ids[0].tolist(), last_attn

    @torch.no_grad()
    def beam_search_generate(self, prompt_ids, max_new_tokens=60, beam_width=5, length_penalty=0.7):
        """Beam search over the continuation only (prompt is fixed context).
        Greedy decoding commits irrevocably to the argmax token at every
        step; for a noisy prompt that often locks the model into a locally
        plausible but globally repetitive/wrong continuation. Beam search
        keeps the top-k partial continuations at each step instead."""
        assert prompt_ids.shape[0] == 1, "beam_search_generate expects batch_size=1"
        self.eval()
        prompt = prompt_ids[0].tolist()

        beams = [(prompt, 0.0, False)]
        for _ in range(max_new_tokens):
            candidates = []
            any_active = False
            for seq, score, finished in beams:
                if finished or len(seq) >= self.max_len:
                    candidates.append((seq, score, True))
                    continue
                any_active = True
                ids = torch.tensor([seq], dtype=torch.long, device=self.device)
                logits, _ = self.forward(ids)
                log_probs = F.log_softmax(logits[0, -1], dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_width)
                for k in range(beam_width):
                    tok = topk_ids[k].item()
                    candidates.append((seq + [tok], score + topk_log_probs[k].item(), tok == self.eos_id))
            if not any_active:
                break

            gen_len = lambda seq: len(seq) - len(prompt)
            candidates.sort(key=lambda c: c[1] / max(1, gen_len(c[0])) ** length_penalty, reverse=True)
            beams = candidates[:beam_width]
            if all(b[2] for b in beams):
                break

        gen_len = lambda seq: len(seq) - len(prompt)
        best_seq = max(beams, key=lambda c: c[1] / max(1, gen_len(c[0])) ** length_penalty)[0]
        return best_seq


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
