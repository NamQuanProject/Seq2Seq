"""
eval_utils.py
-------------
Shared utilities for a FAIR comparison between `baseline.py` (vanilla RNN,
reproducing the exercise notebook) and our model (`train.py`/`test.py`).

The exercise notebook's own baseline BLEU is computed on only its first
200 test sentences (`calculate_bleu(..., max_samples=200)`). Comparing
that directly against a BLEU computed over the FULL (much larger, harder,
noisier) test set is not an apples-to-apples comparison -- a bigger,
unfiltered sample will almost always score lower regardless of model
quality. `build_fair_eval_subset` builds ONE fixed subset of raw test
sentences that survive both pipelines' own preprocessing, so
`baseline.py` and `test.py` can each translate the exact same sentences
with their own decoding and report directly comparable numbers.
"""
import re

from tokenizer import clean_text

_ALLOWED_RE = re.compile(
    r"[^a-zA-Z0-9.!?áàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s]+"
)

DEFAULT_EVAL_SAMPLE_SIZE = 200


def normalize_string(s: str) -> str:
    """Matches the exercise notebook's `normalize_string` exactly
    (word-level, no BPE/denoising) -- this IS the baseline's own
    preprocessing, kept deliberately separate from `tokenizer.clean_text`
    so `baseline.py` is a faithful reproduction, not a re-interpretation."""
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = _ALLOWED_RE.sub("", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def build_fair_eval_subset(src_path, trg_path, n=DEFAULT_EVAL_SAMPLE_SIZE):
    """Returns the first `n` raw (untouched) (src, trg) line pairs that
    survive BOTH `normalize_string` (baseline's own empty-line filter) and
    `clean_text` (our pipeline's) being non-empty -- the shared ground
    truth for a fair baseline-vs-ours BLEU comparison. Each caller then
    applies its OWN preprocessing (normalize_string vs. clean_text) to
    these same raw lines before translating."""
    with open(src_path, "r", encoding="utf-8") as f:
        src_lines = f.readlines()
    with open(trg_path, "r", encoding="utf-8") as f:
        trg_lines = f.readlines()
    pairs = []
    for s, t in zip(src_lines, trg_lines):
        if normalize_string(s) and normalize_string(t) and clean_text(s) and clean_text(t):
            pairs.append((s.strip(), t.strip()))
        if len(pairs) >= n:
            break
    return pairs
