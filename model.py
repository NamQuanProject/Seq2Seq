"""
model.py
--------
Improved Seq2Seq architecture for noisy EN->VI translation:
  - Bidirectional GRU encoder (captures context from both directions,
    important since noise/garbage tokens can appear anywhere in the
    sentence and the decoder needs to learn to "look past" them).
  - Bahdanau (additive) attention decoder: at every decode step the
    decoder computes a weighted sum over ALL encoder states instead of
    relying on a single fixed context vector. This directly helps with
    noise robustness (the attention weights can learn to down-weight
    garbage/gibberish source tokens) and gives us something to visualize
    for the qualitative analysis.
  - GRU instead of vanilla RNN: mitigates vanishing gradients on longer
    noisy sentences with 3x fewer gate parameters than an LSTM.

Everything is sized to respect the 5,000,000 trainable parameter budget;
`count_parameters` should be used to verify this after construction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        # Project concatenated final fwd/bwd hidden state down to decoder size.
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, src_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # hidden: [2, batch, hidden_dim] (fwd, bwd) -> combine into decoder init state
        hidden_cat = torch.cat((hidden[0], hidden[1]), dim=1)  # [batch, 2*hidden_dim]
        hidden = torch.tanh(self.fc(hidden_cat))  # [batch, hidden_dim]
        return outputs, hidden  # outputs: [batch, src_len, 2*hidden_dim]


class Attention(nn.Module):
    """Bahdanau additive attention over bidirectional encoder outputs."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, mask):
        # decoder_hidden: [batch, hidden_dim], encoder_outputs: [batch, src_len, 2*hidden_dim]
        src_len = encoder_outputs.shape[1]
        hidden_rep = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden_rep, encoder_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)  # [batch, src_len]
        scores = scores.masked_fill(mask == 0, -1e10)
        return F.softmax(scores, dim=1)  # attention weights


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=0)
        self.attention = Attention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        # Bottleneck before the vocab projection: projecting the raw
        # concatenation (emb+hidden+context) straight to vocab_size would
        # dominate the parameter budget (concat_dim * vocab_size params).
        # Routing through a hidden_dim-sized bottleneck first cuts that
        # cost roughly by (concat_dim / hidden_dim)x with negligible loss
        # in expressiveness.
        self.pre_out = nn.Linear(emb_dim + hidden_dim * 3, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, encoder_outputs, mask):
        # input: [batch] (single decoding step)
        input = input.unsqueeze(1)  # [batch, 1]
        embedded = self.dropout(self.embedding(input))  # [batch, 1, emb_dim]

        attn_weights = self.attention(hidden, encoder_outputs, mask)  # [batch, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, 2*hidden_dim]

        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)

        embedded = embedded.squeeze(1)
        output = output.squeeze(1)
        context = context.squeeze(1)
        pre_out = torch.tanh(self.pre_out(torch.cat((output, context, embedded), dim=1)))
        prediction = self.fc_out(pre_out)
        return prediction, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, src_pad_id=0, sos_id=2, eos_id=3):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.src_pad_id = src_pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id

    def create_mask(self, src):
        return (src != self.src_pad_id).to(self.device)

    def forward(self, src, src_lens, trg, teacher_forcing_ratio=0.5):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)
        encoder_outputs, hidden = self.encoder(src, src_lens)
        mask = self.create_mask(src)

        input = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, _ = self.decoder(input, hidden, encoder_outputs, mask)
            outputs[:, t] = output
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
        return outputs

    @torch.no_grad()
    def greedy_decode(self, src, src_lens, max_len=60):
        self.eval()
        encoder_outputs, hidden = self.encoder(src, src_lens)
        mask = self.create_mask(src)
        batch_size = src.shape[0]

        input = torch.full((batch_size,), self.sos_id, dtype=torch.long, device=self.device)
        sequences = [[self.sos_id] for _ in range(batch_size)]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)
        attentions = []

        for _ in range(max_len):
            output, hidden, attn_weights = self.decoder(input, hidden, encoder_outputs, mask)
            attentions.append(attn_weights.detach().cpu())
            top1 = output.argmax(1)
            for i in range(batch_size):
                if not finished[i]:
                    sequences[i].append(top1[i].item())
            finished = finished | (top1 == self.eos_id)
            input = top1
            if finished.all():
                break
        return sequences, torch.stack(attentions, dim=1)  # [batch, trg_len, src_len]

    @torch.no_grad()
    def beam_search_decode(self, src, src_lens, max_len=60, beam_width=5, length_penalty=0.7):
        """Batch-size-1 beam search decoding (used at inference/eval time).

        Greedy decoding commits irrevocably to the argmax token at every
        step, which for a noisy source often locks the decoder into a
        locally-plausible but globally repetitive/wrong path. Beam search
        keeps the top-k partial hypotheses at each step so a slightly
        lower-probability early choice that leads to a much better full
        sentence is not discarded prematurely.
        """
        assert src.shape[0] == 1, "beam_search_decode expects batch_size=1"
        self.eval()
        encoder_outputs, hidden = self.encoder(src, src_lens)
        mask = self.create_mask(src)

        # Each beam: (token_seq, hidden_state, cumulative_log_prob, finished)
        beams = [([self.sos_id], hidden.squeeze(0) if hidden.dim() == 3 else hidden, 0.0, False)]

        for _ in range(max_len):
            all_candidates = []
            any_active = False
            for seq, h, score, finished in beams:
                if finished:
                    all_candidates.append((seq, h, score, finished))
                    continue
                any_active = True
                input = torch.tensor([seq[-1]], device=self.device)
                output, new_hidden, _ = self.decoder(input, h.unsqueeze(0) if h.dim() == 1 else h, encoder_outputs, mask)
                log_probs = F.log_softmax(output, dim=1).squeeze(0)  # [vocab]
                topk_log_probs, topk_ids = log_probs.topk(beam_width)
                for k in range(beam_width):
                    tok = topk_ids[k].item()
                    new_seq = seq + [tok]
                    new_score = score + topk_log_probs[k].item()
                    new_finished = tok == self.eos_id
                    all_candidates.append((new_seq, new_hidden, new_score, new_finished))
            if not any_active:
                break

            def norm_score(c):
                seq, _, score, _ = c
                # length-normalized log-prob to avoid biasing towards short sequences
                return score / (len(seq) ** length_penalty)

            all_candidates.sort(key=norm_score, reverse=True)
            beams = all_candidates[:beam_width]
            if all(b[3] for b in beams):
                break

        best_seq = max(beams, key=lambda c: c[2] / (len(c[0]) ** length_penalty))[0]
        return best_seq


def build_model(src_vocab_size, trg_vocab_size, device, emb_dim=144, hidden_dim=192, dropout=0.1):
    encoder = Encoder(src_vocab_size, emb_dim, hidden_dim, dropout=dropout)
    decoder = Decoder(trg_vocab_size, emb_dim, hidden_dim, dropout=dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    device = torch.device("cpu")
    m = build_model(6000, 6000, device)
    print(f"Trainable parameters: {count_parameters(m):,}")
