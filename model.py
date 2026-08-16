"""
model.py
--------
Compact Transformer encoder-decoder for noisy EN->VI translation.

Why Transformer over the recurrent (BiGRU+attention) baseline:
  - Self-attention lets every source position attend to every other
    position in O(1) path length, so a single garbage/gibberish token or
    a locally-inverted word order doesn't have to survive being carried
    through a chain of recurrent hidden states to reach the words it
    should influence -- the encoder can directly learn to down-weight
    noisy tokens when building each position's representation.
  - For a fixed parameter budget, attention layers are more
    parameter-efficient per unit of modeling power than stacked
    recurrent gates (no separate reset/update/candidate weight
    matrices), so more of the 5M budget goes toward representational
    capacity (heads/layers) instead of gating overhead.
  - Fully parallel training over the sequence length (no sequential
    per-timestep recurrence) is also just faster to iterate on.

Parameter budget accounting (see build_model defaults, vocab=6000/side):
  - token embeddings (src + trg):      2 * 6000 * 192   ~= 2.30M
  - encoder stack (3 layers):                            ~= 0.81M
  - decoder stack (3 layers):                             ~= 1.26M
  - output projection bias:                                ~= 6.0K
  Total                                                   ~= 4.38M  (< 5.0M budget)
The decoder's output projection WEIGHT is tied to the target token
embedding (standard weight tying), so it costs 0 extra parameters
instead of another vocab*d_model matrix.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Custom (rather than nn.TransformerDecoderLayer) so we can pull out
    the cross-attention weights of every layer for the qualitative
    attention-map analysis -- the built-in module discards them."""

    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, memory, trg_mask, trg_key_padding_mask, memory_key_padding_mask):
        sa_out, _ = self.self_attn(
            trg, trg, trg, attn_mask=trg_mask, key_padding_mask=trg_key_padding_mask, need_weights=False,
        )
        trg = self.norm1(trg + self.dropout(sa_out))

        ca_out, ca_weights = self.cross_attn(
            trg, memory, memory, key_padding_mask=memory_key_padding_mask,
            need_weights=True, average_attn_weights=True,
        )
        trg = self.norm2(trg + self.dropout(ca_out))

        ff_out = self.ff(trg)
        trg = self.norm3(trg + self.dropout(ff_out))
        return trg, ca_weights  # ca_weights: [batch, trg_len, src_len]


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self, src_vocab_size, trg_vocab_size, device,
        d_model=192, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=320, dropout=0.1, max_len=64,
        pad_id=0, sos_id=2, eos_id=3,
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.trg_tok_emb = nn.Embedding(trg_vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.emb_dropout = nn.Dropout(dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True, activation="relu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_decoder_layers)
        ])

        # Weight-tied output projection: reuses trg_tok_emb's weight matrix
        # instead of learning a second (d_model x vocab) matrix from scratch.
        self.output_bias = nn.Parameter(torch.zeros(trg_vocab_size))

    def _output_proj(self, x):
        return F.linear(x, self.trg_tok_emb.weight, self.output_bias)

    @staticmethod
    def _causal_mask(sz, device):
        return torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)

    def encode(self, src):
        src_key_padding_mask = src == self.pad_id  # [batch, src_len]
        x = self.emb_dropout(self.pos_enc(self.src_tok_emb(src) * math.sqrt(self.d_model)))
        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory, src_key_padding_mask

    def decode_step(self, trg, memory, memory_key_padding_mask):
        trg_key_padding_mask = trg == self.pad_id
        trg_mask = self._causal_mask(trg.size(1), trg.device)
        x = self.emb_dropout(self.pos_enc(self.trg_tok_emb(trg) * math.sqrt(self.d_model)))
        attn = None
        for layer in self.decoder_layers:
            x, attn = layer(x, memory, trg_mask, trg_key_padding_mask, memory_key_padding_mask)
        logits = self._output_proj(x)
        return logits, attn

    def forward(self, src, src_lens, trg, teacher_forcing_ratio=None):
        # Transformer decoding is trained with full teacher forcing via a
        # causal mask (standard practice: the whole target sequence is
        # scored in one parallel pass), so teacher_forcing_ratio is
        # accepted for interface compatibility with train.py but unused.
        memory, memory_kpm = self.encode(src)
        logits, _ = self.decode_step(trg[:, :-1], memory, memory_kpm)
        # Left-pad the first output slot so downstream indexing / loss
        # (which does output[:, 1:]) lines up with the RNN-baseline convention.
        batch_size, _, vocab_size = logits.shape
        outputs = torch.zeros(batch_size, trg.size(1), vocab_size, device=self.device)
        outputs[:, 1:] = logits
        return outputs

    @torch.no_grad()
    def greedy_decode(self, src, src_lens, max_len=60):
        self.eval()
        batch_size = src.size(0)
        memory, memory_kpm = self.encode(src)
        trg = torch.full((batch_size, 1), self.sos_id, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        last_attn = None

        for _ in range(max_len):
            logits, attn = self.decode_step(trg, memory, memory_kpm)
            last_attn = attn  # [batch, trg_len_so_far, src_len]
            next_tok = logits[:, -1].argmax(-1)
            next_tok = torch.where(finished, torch.full_like(next_tok, self.pad_id), next_tok)
            trg = torch.cat([trg, next_tok.unsqueeze(1)], dim=1)
            finished = finished | (next_tok == self.eos_id)
            if finished.all():
                break

        sequences = [trg[i].tolist() for i in range(batch_size)]
        return sequences, last_attn

    @torch.no_grad()
    def beam_search_decode(self, src, src_lens, max_len=60, beam_width=5, length_penalty=0.7):
        assert src.shape[0] == 1, "beam_search_decode expects batch_size=1"
        self.eval()
        memory, memory_kpm = self.encode(src)

        beams = [([self.sos_id], 0.0, False)]
        for _ in range(max_len):
            candidates = []
            any_active = False
            for seq, score, finished in beams:
                if finished:
                    candidates.append((seq, score, finished))
                    continue
                any_active = True
                trg = torch.tensor([seq], dtype=torch.long, device=self.device)
                logits, _ = self.decode_step(trg, memory, memory_kpm)
                log_probs = F.log_softmax(logits[0, -1], dim=-1)
                topk_log_probs, topk_ids = log_probs.topk(beam_width)
                for k in range(beam_width):
                    tok = topk_ids[k].item()
                    candidates.append((seq + [tok], score + topk_log_probs[k].item(), tok == self.eos_id))
            if not any_active:
                break

            candidates.sort(key=lambda c: c[1] / (len(c[0]) ** length_penalty), reverse=True)
            beams = candidates[:beam_width]
            if all(b[2] for b in beams):
                break

        best_seq = max(beams, key=lambda c: c[1] / (len(c[0]) ** length_penalty))[0]
        return best_seq


def build_model(
    src_vocab_size, trg_vocab_size, device,
    d_model=192, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
    dim_feedforward=320, dropout=0.1, max_len=64,
):
    model = Seq2SeqTransformer(
        src_vocab_size, trg_vocab_size, device,
        d_model=d_model, nhead=nhead,
        num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward, dropout=dropout, max_len=max_len,
    ).to(device)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cpu")
    m = build_model(6000, 6000, device)
    print(f"Trainable parameters: {count_parameters(m):,}")
