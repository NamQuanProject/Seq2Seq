"""
preprocess.py
--------------
Standalone data-cleaning step: produces a "super clean" copy of the noisy
English source for the train/val/test splits, applied BEFORE
tokenization/training. Two-stage strategy per token:

  1. SELF-CORRECT what can be corrected: elongated-char / repeated-punct
     collapse, disallowed-character stripping, and dictionary-checked
     missing-space repair ("ihavean" -> "i have an") -- see
     `tokenizer.clean_text`.
  2. DELETE what's left over: any token that still isn't a recognizable
     word after step 1 (garbage/gibberish injected by the noise process)
     is removed completely from the sentence, rather than being carried
     forward for the model to absorb -- see `tokenizer.super_clean_text`.

The Vietnamese target side is only lightly normalized (it's already
clean) -- the destructive deletion step is English-only, since it is
driven by an English word-frequency table and running it on Vietnamese
would delete real Vietnamese words.

Output: `<output_dir>/{train,val,test}_clean.en.txt` /
`.vi.txt`, plus a JSON report of how much noise was fixed/deleted per
split at `<output_dir>/../preprocess_stats.json`.

Run:
    python preprocess.py --data_dir ./en-vi-translation-data --output_dir ./output/clean_data
"""
import argparse
import json
import os

from tokenizer import clean_text, super_clean_text

SPLITS = [
    ("train_noisy.en.txt", "train.vi.txt", "train"),
    ("val_noisy.en.txt", "val.vi.txt", "val"),
    ("test_noisy.en.txt", "test.vi.txt", "test"),
]

EMPTY_STATS = {
    "lines": 0,
    "elongation_collapses": 0,
    "punct_run_collapses": 0,
    "disallowed_chars_stripped": 0,
    "concatenation_fixes": 0,
    "garbage_tokens_deleted": 0,
    "chars_before": 0,
    "chars_after": 0,
}


def process_split(src_path, trg_path, out_src_path, out_trg_path):
    with open(src_path, "r", encoding="utf-8") as f:
        src_lines = [l.strip() for l in f.readlines()]
    with open(trg_path, "r", encoding="utf-8") as f:
        trg_lines = [l.strip() for l in f.readlines()]
    assert len(src_lines) == len(trg_lines), (
        f"Mismatched line counts: {src_path} has {len(src_lines)}, {trg_path} has {len(trg_lines)}"
    )

    stats = dict(EMPTY_STATS)
    src_vocab_before, src_vocab_after = set(), set()

    cleaned_src, cleaned_trg = [], []
    dropped_empty = 0
    for s, t in zip(src_lines, trg_lines):
        src_vocab_before.update(s.lower().split())
        cs = super_clean_text(s, stats=stats)
        src_vocab_after.update(cs.split())
        ct = clean_text(t)

        # A sentence that becomes empty after garbage deletion carries no
        # signal (and would break BPE encoding downstream) -- drop the
        # pair entirely rather than train on an empty source.
        if not cs.strip() or not ct.strip():
            dropped_empty += 1
            continue
        cleaned_src.append(cs)
        cleaned_trg.append(ct)

    stats["pairs_in"] = len(src_lines)
    stats["pairs_out"] = len(cleaned_src)
    stats["pairs_dropped_empty"] = dropped_empty
    stats["src_raw_whitespace_vocab_size"] = len(src_vocab_before)
    stats["src_cleaned_whitespace_vocab_size"] = len(src_vocab_after)
    stats["src_vocab_reduction_pct"] = (
        100.0 * (1 - len(src_vocab_after) / max(1, len(src_vocab_before)))
    )

    os.makedirs(os.path.dirname(out_src_path) or ".", exist_ok=True)
    with open(out_src_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_src) + "\n")
    with open(out_trg_path, "w", encoding="utf-8") as f:
        f.write("\n".join(cleaned_trg) + "\n")

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./en-vi-translation-data")
    parser.add_argument("--output_dir", default="./output/clean_data")
    parser.add_argument("--stats_path", default="./output/preprocess_stats.json")
    args = parser.parse_args()

    all_stats = {}
    for src_name, trg_name, split in SPLITS:
        src_path = os.path.join(args.data_dir, src_name)
        trg_path = os.path.join(args.data_dir, trg_name)
        if not os.path.exists(src_path) or not os.path.exists(trg_path):
            print(f"[{split}] skipped: {src_path} or {trg_path} not found")
            continue

        out_src = os.path.join(args.output_dir, f"{split}_clean.en.txt")
        out_trg = os.path.join(args.output_dir, f"{split}_clean.vi.txt")
        stats = process_split(src_path, trg_path, out_src, out_trg)
        all_stats[split] = stats
        print(f"[{split}] pairs {stats['pairs_in']} -> {stats['pairs_out']} "
              f"(dropped {stats['pairs_dropped_empty']} now-empty), "
              f"garbage tokens deleted: {stats['garbage_tokens_deleted']}, "
              f"src vocab reduction: {stats['src_vocab_reduction_pct']:.1f}%")
        print(f"  wrote {out_src}")
        print(f"  wrote {out_trg}")

    os.makedirs(os.path.dirname(args.stats_path) or ".", exist_ok=True)
    with open(args.stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nSaved stats to {args.stats_path}")


if __name__ == "__main__":
    main()
