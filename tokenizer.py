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

PAD, UNK, SOS, EOS, SEP = "<pad>", "<unk>", "<sos>", "<eos>", "<sep>"
SPECIAL_TOKENS = [PAD, UNK, SOS, EOS, SEP]
PAD_ID, UNK_ID, SOS_ID, EOS_ID, SEP_ID = 0, 1, 2, 3, 4


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


def clean_text(s: str, stats: dict = None) -> str:
    """Rule-based denoising pipeline, applied in order:
      1. Decode HTML entities (&quot; &apos; &amp; ...) introduced by the
         noise injector, so downstream steps see real characters.
      2. Lowercase + collapse elongated chars / repeated punctuation.
      3. Split punctuation off words, strip disallowed characters.
      4. Repair missing-space concatenation ('Ithought' -> 'i thought')
         via dictionary-checked word segmentation, without touching
         genuine unrecognizable gibberish tokens.
      5. Collapse whitespace.

    If `stats` (a dict) is passed, per-line counters are accumulated into
    it in place (see `denoise_report` below) so the effect of each rule
    can be measured over a whole corpus for the report.
    """
    orig_len = len(s)
    s = html.unescape(s)
    s = s.lower().strip()

    if stats is not None:
        stats["elongation_collapses"] += len(_REPEAT_CHAR_RE.findall(s))
        stats["punct_run_collapses"] += len(_MULTI_PUNCT_RE.findall(s))

    s = _REPEAT_CHAR_RE.sub(r"\1\1", s)          # collapse elongated chars
    s = _MULTI_PUNCT_RE.sub(r"\1", s)            # "!!!" -> "!" (before spacing)
    s = _PUNCT_SPACE_RE.sub(r" \1", s)           # split off punctuation

    if stats is not None:
        stats["disallowed_chars_stripped"] += len(_NON_ALLOWED_RE.findall(s))

    s = _NON_ALLOWED_RE.sub("", s)
    s = _SPACE_RE.sub(" ", s).strip()

    fixed_tokens = []
    n_concat_fixes = 0
    for tok in s.split(" "):
        pieces = _fix_concatenation(tok)
        if len(pieces) > 1:
            n_concat_fixes += 1
        fixed_tokens.extend(pieces)

    if stats is not None:
        stats["concatenation_fixes"] += n_concat_fixes
        stats["lines"] += 1
        stats["chars_before"] += orig_len
        stats["chars_after"] += len(" ".join(fixed_tokens))

    return " ".join(fixed_tokens)


def _is_unrecoverable_garbage(tok: str) -> bool:
    """True for a token that survived `clean_text`'s concatenation-repair
    pass (see `_fix_concatenation`) and is STILL not a recognizable word --
    i.e. it isn't a number, isn't punctuation, and isn't a short/common
    token, so there is nothing left to "self-correct": it's noise, not a
    typo. Short tokens (len<=2, e.g. 'i', 'a', 'ok', 'no') are exempted
    since real short words are common and mostly below the frequency
    cutoff `_is_known_word` uses for longer tokens."""
    if not _ALPHA_RE.match(tok):
        return False  # numbers / punctuation are never "garbage" here
    if len(tok) <= 2:
        return False
    return not _is_known_word(tok)


def super_clean_text(s: str, stats: dict = None) -> str:
    """Aggressive denoising for the noisy English SOURCE side only:
    1. Run the standard `clean_text` pipeline (elongation/punctuation
       collapse, char filtering, dictionary-checked concatenation repair
       -- i.e. "self-correct what can be self-corrected").
    2. DELETE any token that still isn't a recognizable word after that
       repair pass -- i.e. remove what can't be corrected, instead of
       leaving raw gibberish in the sentence for the model to absorb.

    This is intentionally more destructive than `clean_text` (which keeps
    unrecovered noise for the BPE tokenizer/attention to learn to ignore).
    Use this for a dedicated denoised copy of the corpus via
    `preprocess.py`; do NOT apply it to the Vietnamese target side (already
    clean, and Vietnamese words wouldn't pass an English frequency table).
    """
    cleaned = clean_text(s, stats=stats)
    kept = []
    n_deleted = 0
    for tok in cleaned.split(" "):
        if not tok:
            continue
        if _is_unrecoverable_garbage(tok):
            n_deleted += 1
            continue
        kept.append(tok)
    if stats is not None:
        stats["garbage_tokens_deleted"] = stats.get("garbage_tokens_deleted", 0) + n_deleted
    return _SPACE_RE.sub(" ", " ".join(kept)).strip()


def denoise_report(raw_lines) -> dict:
    """Aggregate before/after statistics of the denoising pipeline over a
    corpus. Useful evidence for the report's data-centric analysis:
    how much noise (elongation, punctuation runs, disallowed characters,
    missing-space concatenations) was actually present and fixed, and how
    much the whitespace-token vocabulary shrinks as a result (the
    baseline `Vocabulary` in the notebook builds one entry per raw
    whitespace token, so this directly explains part of its param blowup).
    """
    stats = {
        "lines": 0,
        "elongation_collapses": 0,
        "punct_run_collapses": 0,
        "disallowed_chars_stripped": 0,
        "concatenation_fixes": 0,
        "chars_before": 0,
        "chars_after": 0,
    }
    raw_vocab, cleaned_vocab = set(), set()
    for line in raw_lines:
        raw_vocab.update(line.lower().split())
        cleaned = clean_text(line, stats=stats)
        cleaned_vocab.update(cleaned.split())

    stats["raw_whitespace_vocab_size"] = len(raw_vocab)
    stats["cleaned_whitespace_vocab_size"] = len(cleaned_vocab)
    stats["vocab_reduction_pct"] = (
        100.0 * (1 - len(cleaned_vocab) / max(1, len(raw_vocab)))
    )
    return stats


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