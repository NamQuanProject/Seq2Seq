"""
tokenizer.py
------------
Denoising text cleanup + subword (SentencePiece Unigram) tokenization.

Why a subword vocabulary instead of the baseline's whitespace `Vocabulary`?
  - The baseline treats every distinct whitespace-separated token as a new
    word. Noisy data (missing spaces, garbage strings, typos) explodes the
    vocabulary with one-off "words" that are seen once and never learned
    well -> huge embedding tables + terrible generalization.
  - A subword vocabulary caps vocab size, represents unseen/garbled tokens
    as combinations of known sub-word pieces instead of a single <unk>,
    and is inherently more robust to spelling noise ("apple" vs "aple"
    share subword pieces).
  - Smaller, shared-size vocab also directly reduces trainable parameters
    (embedding tables + output projection), which helps the 5M param budget.

Why Unigram (SentencePiece) rather than BPE specifically: Unigram is a
probabilistic segmentation model (keeps a large candidate piece inventory
and prunes to the vocab that maximizes corpus likelihood) rather than
BPE's greedy pairwise-merge heuristic, and it natively supports **subword
regularization** -- sampling a different, still-valid segmentation of the
same sentence on each training pass instead of always the single
deterministic split. For a from-scratch model trained on already-noisy
text, exposing it to segmentation variance is another axis of robustness
on top of the source-noise augmentation below (see `encode_ids(sample=True)`).

This module is intentionally the ONLY place that knows about the raw text
denoising rules AND the subword model, so data.py can stay simple.
"""
import html
import os
import random
import re

import sentencepiece as spm
from wordsegment import load as _ws_load, segment as _ws_segment, UNIGRAMS as _UNIGRAMS

_ws_load()  # loads bundled unigram/bigram frequency tables (no network needed)

PAD, UNK, SOS, EOS = "<pad>", "<unk>", "<sos>", "<eos>"
# Direction tags: training on BOTH <toVI> (EN->VI, the real task) and
# <toEN> (VI->EN) examples with the same shared vocab/stack turns the one
# model bidirectional at zero extra inference-time parameters. This is
# what lets `rerank.py` score a candidate translation's reverse
# probability log P(x|y) with the SAME model instead of needing a second
# trained reverse model.
TOEN, TOVI = "<toen>", "<tovi>"
SPECIAL_TOKENS = [PAD, UNK, SOS, EOS, TOEN, TOVI]
PAD_ID, UNK_ID, SOS_ID, EOS_ID, TOEN_ID, TOVI_ID = 0, 1, 2, 3, 4, 5


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
#
# Two DIFFERENT thresholds, deliberately not shared, because they guard
# opposite kinds of mistakes:
#   - _MIN_WORD_FREQ (strict): used to accept a word-segmentation SPLIT
#     PIECE as real (`_is_known_word`) and to decide whether a token is
#     "common enough" that concatenation-repair shouldn't touch it. Being
#     strict here is safe -- worst case a genuine-but-rare word is left
#     alone (or a rare compound doesn't get split), which is a no-op, not
#     data loss.
#   - _MIN_DELETE_FREQ (lenient): used ONLY to decide whether
#     `super_clean_text` DELETES a token outright. Reusing the strict
#     300k threshold here (as an earlier version of this file did) is
#     dangerous: plenty of legitimate, moderately-common English words
#     never reach 300k in this table, so strict-threshold deletion
#     silently destroys real source content the model should be
#     translating -- exactly the kind of information loss the baseline's
#     `normalize_string` (which deletes nothing) never risks. A low bar
#     here means "delete only if this looks essentially unattested",
#     erring toward keeping content rather than losing it.
_MIN_WORD_FREQ = 300_000
_MIN_DELETE_FREQ = 1_000


def _is_known_word(tok: str) -> bool:
    return _UNIGRAMS.get(tok, 0) >= _MIN_WORD_FREQ


def _is_attested_word(tok: str) -> bool:
    return _UNIGRAMS.get(tok, 0) >= _MIN_DELETE_FREQ


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
    pass (see `_fix_concatenation`) and is STILL not even a LENIENTLY
    recognizable word -- i.e. it isn't a number, isn't punctuation, isn't
    a short/common token, and doesn't appear at all meaningfully in the
    reference frequency table, so there is nothing left to "self-correct":
    it's noise, not a real (if uncommon) word. Uses `_is_attested_word`
    (low bar, `_MIN_DELETE_FREQ`), NOT the strict `_is_known_word` bar --
    deletion is destructive and irreversible, so it should only fire on
    tokens that are essentially unattested, not merely "less common than
    300k in a noisy frequency table." Short tokens (len<=2, e.g. 'i', 'a',
    'ok', 'no') are exempted since real short words are common and mostly
    below any frequency cutoff anyway."""
    if not _ALPHA_RE.match(tok):
        return False  # numbers / punctuation are never "garbage" here
    if len(tok) <= 2:
        return False
    return not _is_attested_word(tok)


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
# 1b) Source-noise augmentation (data-centric, applied at TRAIN time only)
# ---------------------------------------------------------------------------
# The noise injector's own perturbations (missing spaces, garbage words,
# elongation) are already covered by train_noisy.en.txt itself. What's
# usually MORE valuable than a fancier decoding algorithm is simply
# exposing the model to more of the noise distribution it'll face at test
# time: keyboard-adjacent typos, dropped diacritics/casing, and random
# character deletions. This is applied on top of the (already-cleaned)
# training source at data-loading time, per epoch, so the model never
# memorizes one fixed noisy surface form per sentence.
_QWERTY_NEIGHBORS = {
    "q": "wa", "w": "qes", "e": "wrd", "r": "etf", "t": "ryg", "y": "tuh",
    "u": "yij", "i": "uok", "o": "ipl", "p": "ol", "a": "qsz", "s": "awd",
    "d": "sfe", "f": "dgr", "g": "fht", "h": "gjy", "j": "hku", "k": "jli",
    "l": "ko", "z": "ax", "x": "zc", "c": "xv", "v": "cb", "b": "vn", "n": "bm", "m": "n",
}


def augment_noise(s: str, rng, p_char: float = 0.03, p_delete: float = 0.01, p_case: float = 0.05) -> str:
    """Randomly corrupt an already-cleaned sentence to simulate additional
    realistic noise beyond what the injector already produced:
      - keyboard-adjacent character swaps (typos), probability `p_char` per char
      - random character deletion, probability `p_delete` per char
      - random uppercasing of a character, probability `p_case` per char
    `rng` is a `random.Random` instance (caller-owned, for reproducibility).
    Only ever call this on the noisy ENGLISH source, and only at train time
    -- val/test must stay exactly as given so BLEU is measured honestly.
    """
    out = []
    for ch in s:
        if ch.isalpha() and rng.random() < p_delete:
            continue  # drop the character
        if ch.lower() in _QWERTY_NEIGHBORS and rng.random() < p_char:
            ch = rng.choice(_QWERTY_NEIGHBORS[ch.lower()])
        elif ch.isalpha() and rng.random() < p_case:
            ch = ch.upper() if ch.islower() else ch.lower()
        out.append(ch)
    return "".join(out)


# ---------------------------------------------------------------------------
# 2) SentencePiece Unigram subword tokenizer (trained on the cleaned
#    training corpus only), with byte-fallback so no input character can
#    ever produce a hard failure, and optional subword-regularization
#    sampling for training-time segmentation variance.
# ---------------------------------------------------------------------------
class SubwordTokenizer:
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        self.sp = None  # spm.SentencePieceProcessor, set by train()/load()

    def train(self, lines, model_prefix):
        """lines: iterable of already-cleaned strings (write once to a temp
        corpus file, since SentencePieceTrainer trains from a file path).
        `model_prefix` (no extension) is where the `.model`/`.vocab` files
        land -- same base path `save()`/`load()` use."""
        os.makedirs(os.path.dirname(model_prefix) or ".", exist_ok=True)
        corpus_path = model_prefix + "_corpus.tmp.txt"
        with open(corpus_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
        try:
            spm.SentencePieceTrainer.train(
                input=corpus_path, model_prefix=model_prefix, model_type="unigram",
                vocab_size=self.vocab_size, character_coverage=1.0, shuffle_input_sentence=True,
                pad_id=PAD_ID, unk_id=UNK_ID, bos_id=SOS_ID, eos_id=EOS_ID,
                pad_piece=PAD, unk_piece=UNK, bos_piece=SOS, eos_piece=EOS,
                user_defined_symbols=[TOEN, TOVI],  # appended right after the 4 control ids -> TOEN_ID, TOVI_ID
                byte_fallback=True, hard_vocab_limit=True,
            )
        finally:
            if os.path.exists(corpus_path):
                os.remove(corpus_path)
        self.sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
        self._verify_special_ids()

    def _verify_special_ids(self):
        """Fail loudly at load time (not silently mid-training) if the
        trained/loaded model's special-token ids ever drift from what the
        rest of the codebase assumes."""
        expected = {PAD_ID: PAD, UNK_ID: UNK, SOS_ID: SOS, EOS_ID: EOS, TOEN_ID: TOEN, TOVI_ID: TOVI}
        for expected_id, piece in expected.items():
            actual_id = self.sp.piece_to_id(piece)
            assert actual_id == expected_id, (
                f"Tokenizer special-token id drift: {piece!r} is id {actual_id}, expected {expected_id}. "
                "Delete the stale tokenizer files under output/tokenizers/ and retrain."
            )

    # -- encode / decode -----------------------------------------------
    def encode_ids(self, text: str, sample: bool = False):
        """sample=True draws a random (still-valid) Unigram segmentation
        instead of the single best one -- subword regularization, meant
        for the TRAINING split only (val/test must stay deterministic)."""
        if sample:
            return self.sp.encode(text, out_type=int, enable_sampling=True,
                                    nbest_size=-1, alpha=0.1)
        return self.sp.encode(text, out_type=int)

    def encode_with_specials(self, text: str):
        return [SOS_ID] + self.encode_ids(text) + [EOS_ID]

    def decode_ids(self, ids, skip_specials=True):
        if skip_specials:
            special_ids = (PAD_ID, SOS_ID, EOS_ID, TOEN_ID, TOVI_ID)
            ids = [i for i in ids if i not in special_ids]
        return self.sp.decode(ids)

    @property
    def vocab_size_actual(self):
        return self.sp.get_piece_size()

    # -- persistence ------------------------------------------------------
    def save(self, path):
        pass  # SentencePieceTrainer already wrote <prefix>.model/.vocab in train()

    @classmethod
    def load(cls, model_prefix):
        obj = cls()
        obj.sp = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
        obj.vocab_size = obj.sp.get_piece_size()
        obj._verify_special_ids()
        return obj


def build_or_load_tokenizer(cleaned_lines, save_path, vocab_size=8000):
    """Train a tokenizer if save_path (a `<prefix>.model` path) doesn't
    exist yet, else load it. Always train tokenizers ONLY on the training
    split to avoid leaking val/test vocabulary/statistics into preprocessing."""
    model_prefix = save_path[:-6] if save_path.endswith(".model") else save_path
    if os.path.exists(model_prefix + ".model"):
        return SubwordTokenizer.load(model_prefix)
    tok = SubwordTokenizer(vocab_size=vocab_size)
    tok.train(cleaned_lines, model_prefix)
    return tok


# ---------------------------------------------------------------------------
# 1c) Word-level noise augmentation (alternative/complementary to the
#     character-level `augment_noise` above; applied at TRAIN time only)
# ---------------------------------------------------------------------------
# Directly mimics the noise TYPES the assignment describes at the word
# level: "join" simulates a missing-space concatenation, "garbage"
# simulates an injected nonsense token, "swap"/"delete" simulate local
# word-order/dropout noise, and "unk" forces exposure to the <unk> id so
# the model doesn't treat it as a rare, ignorable event at test time.
_WORD_AUGMENT_PROBS = {"join": 0.45, "garbage": 0.25, "swap": 0.15, "delete": 0.10, "unk": 0.05}


def augment_noise_word_level(s: str, rng, sentence_prob: float = 0.2, probs: dict = None):
    """Returns (augmented_sentence, force_unk). `force_unk` signals the
    caller to overwrite one encoded id with UNK_ID post-tokenization (the
    "unk" operation can't be expressed as a text edit). `sentence_prob` is
    the chance any augmentation fires at all for this sentence (matches
    most sentences passing through unaltered, like `augment_noise`)."""
    if rng.random() >= sentence_prob:
        return s, False
    words = s.split()
    if len(words) < 2:
        return s, False
    probs = probs or _WORD_AUGMENT_PROBS
    draw, total, operation = rng.random(), 0.0, "unk"
    for name, probability in probs.items():
        total += probability
        if draw < total:
            operation = name
            break
    index = rng.randrange(len(words) - 1) if operation in ("join", "swap") else rng.randrange(len(words))
    if operation == "join":
        words[index:index + 2] = [words[index] + words[index + 1]]
    elif operation == "swap":
        words[index], words[index + 1] = words[index + 1], words[index]
    elif operation == "delete":
        del words[index]
    elif operation == "garbage":
        words.insert(index, "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 8))))
    return " ".join(words), operation == "unk"


if __name__ == "__main__":
    # quick smoke test
    sample = "I  havean   apple!!!  soooo goood lol vacx blah"
    print("raw   :", sample)
    print("clean :", clean_text(sample))