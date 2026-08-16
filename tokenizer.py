"""
tokenizer.py
------------
Denoising text cleanup + subword (BPE) tokenization.

Why BPE instead of whitespace-split word vocab (like the baseline)?
  - The baseline's `Vocabulary` treats every distinct whitespace-separated
    token as a new word. Noisy data (missing spaces, garbage strings,
    typos) explodes the vocabulary with one-off "words" that are seen once
    and never learned well -> huge embedding tables + terrible generalization.
  - A subword vocabulary (Byte-Pair Encoding) caps vocab size, represents
    unseen/garbled tokens as combinations of known sub-word pieces instead
    of a single <unk>, and is inherently more robust to spelling noise
    ("apple" vs "aple" share subword pieces).
  - Smaller, shared-size vocab also directly reduces trainable parameters
    (embedding tables + output projection), which helps the 5M param budget.

This module is intentionally the ONLY place that knows about the raw text
denoising rules AND the subword model, so data.py can stay simple.
"""
import html
import os
import re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC, Lowercase, Sequence as NormSequence
from wordsegment import load as _ws_load, segment as _ws_segment, UNIGRAMS as _UNIGRAMS

_ws_load()  # loads bundled unigram/bigram frequency tables (no network needed)

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
SPECIAL_TOKENS = [PAD, UNK, SOS, EOS]
PAD_ID, UNK_ID, SOS_ID, EOS_ID = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# 1) Rule-based denoising (data-centric pass, applied BEFORE tokenization)
# ---------------------------------------------------------------------------
# These are heuristics for the noise types described in the notebook:
# missing spaces, garbage/gibberish words, inverted grammar.
# We deliberately do NOT try to "fix" grammar (word order) here -- that is
# exactly the kind of noise the *model* should learn to be robust to via
# training data exposure, not something a rule-based cleaner should silently
# undo (otherwise the model never sees the noise it must be tested on).

_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")           # "sooooo" -> "soo"
_MULTI_PUNCT_RE = re.compile(r"([!?.,])\1+")          # "!!!" -> "!"
_NON_ALLOWED_RE = re.compile(
    r"[^a-zA-Z0-9.!?'áàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ\s]+"
)
_SPACE_RE = re.compile(r"\s+")
_PUNCT_SPACE_RE = re.compile(r"([.!?])")
_ALPHA_RE = re.compile(r"^[a-z]+$")

# NOTE: wordsegment's UNIGRAMS table is raw Google-Ngrams derived and is
# itself noisy -- it contains OCR fragments and even some accidentally
# merged word pairs with nonzero counts (e.g. "weare": 191,043,
# "onthe": 214,991, "gtu": 107,726 -- all far below real-word frequencies
# like "coworker": 343,002 or "envy": 2,464,445). A bare membership check
# (`tok in UNIGRAMS`) therefore both (a) wrongly treats noisy merges as
# "already a known word" and skips fixing them, and (b) wrongly accepts
# gibberish fragments as valid split pieces. A minimum-frequency cutoff
# fixes both failure modes.
_MIN_WORD_FREQ = 300_000


def _is_known_word(tok: str) -> bool:
    return _UNIGRAMS.get(tok, 0) >= _MIN_WORD_FREQ


def _fix_concatenation(token: str) -> list:
    """Repair a suspected 'missing space' merge like 'Ithought' -> ['i','thought'].

    Only fires when:
      - the token is pure lowercase alphabetic (numbers/punct untouched),
      - it is NOT already a recognized (high-frequency) English word, so
        real words like 'myself', 'singing', or 'coworker' are left alone, and
      - word-segmentation finds >=2 pieces that are ALL themselves
        recognized (high-frequency) words.
    This last condition is what protects gibberish noise like 'gtuql' or
    'pqlma': segmentation of random letters produces low-frequency
    fragments, so the filter rejects the split and the token is left
    untouched for the subword tokenizer (and the model's attention) to
    handle as noise, rather than being silently rewritten into something
    that looks clean but is fabricated."""
    if not _ALPHA_RE.match(token) or len(token) < 4 or _is_known_word(token):
        return [token]
    parts = _ws_segment(token)
    if len(parts) >= 2 and all(_is_known_word(p) for p in parts):
        return parts
    return [token]


def clean_text(s: str) -> str:
    """Rule-based denoising pipeline, applied in order:
      1. Decode HTML entities (&quot; &apos; &amp; ...) introduced by the
         noise injector, so downstream steps see real characters.
      2. Lowercase + collapse elongated chars / repeated punctuation.
      3. Split punctuation off words, strip disallowed characters.
      4. Repair missing-space concatenation ('Ithought' -> 'i thought')
         via dictionary-checked word segmentation, without touching
         genuine unrecognizable gibberish tokens.
      5. Collapse whitespace.
    """
    s = html.unescape(s)
    s = s.lower().strip()
    s = _REPEAT_CHAR_RE.sub(r"\1\1", s)          # collapse elongated chars
    s = _MULTI_PUNCT_RE.sub(r"\1", s)            # "!!!" -> "!" (before spacing)
    s = _PUNCT_SPACE_RE.sub(r" \1", s)           # split off punctuation
    s = _NON_ALLOWED_RE.sub("", s)
    s = _SPACE_RE.sub(" ", s).strip()

    fixed_tokens = []
    for tok in s.split(" "):
        fixed_tokens.extend(_fix_concatenation(tok))
    return " ".join(fixed_tokens)


# ---------------------------------------------------------------------------
# 2) BPE subword tokenizer (trained on the cleaned training corpus only)
# ---------------------------------------------------------------------------
class SubwordTokenizer:
    def __init__(self, vocab_size: int = 6000):
        self.vocab_size = vocab_size
        self.tk = Tokenizer(BPE(unk_token=UNK))
        self.tk.normalizer = NormSequence([NFKC(), Lowercase()])
        self.tk.pre_tokenizer = Whitespace()

    def train(self, lines):
        """lines: iterable of already-cleaned strings."""
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=2,
        )
        self.tk.train_from_iterator(lines, trainer=trainer)

    # -- encode / decode -----------------------------------------------
    def encode_ids(self, text: str):
        return self.tk.encode(text).ids

    def encode_with_specials(self, text: str):
        return [SOS_ID] + self.encode_ids(text) + [EOS_ID]

    def decode_ids(self, ids, skip_specials=True):
        if skip_specials:
            ids = [i for i in ids if i not in (PAD_ID, SOS_ID, EOS_ID)]
        return self.tk.decode(ids)

    @property
    def vocab_size_actual(self):
        return self.tk.get_vocab_size()

    # -- persistence ------------------------------------------------------
    def save(self, path):
        self.tk.save(path)

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.tk = Tokenizer.from_file(path)
        obj.vocab_size = obj.tk.get_vocab_size()
        return obj


def build_or_load_tokenizer(cleaned_lines, save_path, vocab_size=6000):
    """Train a tokenizer if save_path doesn't exist yet, else load it.
    Always train tokenizers ONLY on the training split to avoid leaking
    val/test vocabulary/statistics into preprocessing."""
    if os.path.exists(save_path):
        return SubwordTokenizer.load(save_path)
    tok = SubwordTokenizer(vocab_size=vocab_size)
    tok.train(cleaned_lines)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    tok.save(save_path)
    return tok


if __name__ == "__main__":
    # quick smoke test
    sample = "I  havean   apple!!!  soooo goood lol vacx blah"
    print("raw   :", sample)
    print("clean :", clean_text(sample))