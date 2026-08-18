"""
average_checkpoints.py
-----------------------
Checkpoint (weight) averaging: averages the state_dicts of the last K
epoch checkpoints train.py saved to `output/ckpts/` and writes the result
as a new checkpoint. This is a free (zero inference-time parameter cost,
zero extra training) way to often improve BLEU a bit by landing in a
flatter region of the loss surface than any single epoch's weights --
standard practice (SWA / Transformer "checkpoint averaging").

Only averages `model_state_dict` tensors; everything else (args,
vocab_size) is copied from the most recent checkpoint in the group.

Run:
    python average_checkpoints.py --ckpts_dir ./output/ckpts --output_path ./output/checkpoint_avg.pt
    python test.py --ckpt_path ./output/checkpoint_avg.pt   # evaluate the averaged model
"""
import argparse
import glob
import os

import torch


def average_checkpoints(ckpt_paths):
    if not ckpt_paths:
        raise ValueError("No checkpoints given to average.")
    payloads = [torch.load(p, map_location="cpu") for p in ckpt_paths]
    state_dicts = [p["model_state_dict"] for p in payloads]

    avg_state = {}
    for key in state_dicts[0].keys():
        tensors = [sd[key].float() for sd in state_dicts]
        avg_state[key] = torch.stack(tensors, dim=0).mean(dim=0).to(state_dicts[0][key].dtype)

    merged = dict(payloads[-1])  # keep args/vocab_size/etc. from the most recent checkpoint
    merged["model_state_dict"] = avg_state
    merged["averaged_from"] = [os.path.basename(p) for p in ckpt_paths]
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts_dir", default="./output/ckpts")
    parser.add_argument("--output_path", default="./output/checkpoint_avg.pt")
    parser.add_argument("--k", type=int, default=None,
                         help="Average only the K most recent checkpoints (default: all found).")
    args = parser.parse_args()

    ckpt_paths = sorted(glob.glob(os.path.join(args.ckpts_dir, "epoch_*.pt")))
    if args.k is not None:
        ckpt_paths = ckpt_paths[-args.k:]

    if len(ckpt_paths) < 2:
        raise ValueError(
            f"Found only {len(ckpt_paths)} checkpoint(s) in {args.ckpts_dir} -- need at least 2 to "
            "average. Train with --save_last_k >= 2 (train.py's default is 3)."
        )

    print(f"Averaging {len(ckpt_paths)} checkpoints:")
    for p in ckpt_paths:
        print(f"  {p}")

    merged = average_checkpoints(ckpt_paths)
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(merged, args.output_path)
    print(f"Saved averaged checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()
