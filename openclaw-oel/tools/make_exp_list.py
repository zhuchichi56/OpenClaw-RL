#!/usr/bin/env python3
"""Build an experience_list.txt from extracted experience files.

Scans the OEL extract output directory and creates a list of experience file
paths for the consolidate phase to consume.

Usage:
    python make_exp_list.py \\
        --exp-name oel-openclaw-q3-4b-extract-round1 \\
        --ckpt-start 50 --ckpt-end 500 --ckpt-step 50 \\
        --val-samples-limit 100 --val-samples-use 50 \\
        [--base-dir /tmp]

Output:
    /tmp/{exp_name}/experience_list.txt
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Build experience_list.txt for OEL consolidation")
    parser.add_argument("--exp-name", required=True, help="Experiment name (directory prefix)")
    parser.add_argument("--ckpt-start", type=int, required=True, help="Start checkpoint/seed")
    parser.add_argument("--ckpt-end", type=int, required=True, help="End checkpoint/seed (inclusive)")
    parser.add_argument("--ckpt-step", type=int, required=True, help="Step between checkpoints")
    parser.add_argument("--val-samples-limit", type=int, required=True,
                        help="Number of samples extracted per seed (directory naming)")
    parser.add_argument("--val-samples-use", type=int, required=True,
                        help="Number of samples to use (experience file naming)")
    parser.add_argument("--base-dir", default="/tmp",
                        help="Base directory containing experiment outputs")
    parser.add_argument("--output", default="",
                        help="Output file path (default: {base_dir}/{exp_name}/experience_list.txt)")
    args = parser.parse_args()

    exp_dir = os.path.join(args.base_dir, args.exp_name)
    output_path = args.output or os.path.join(exp_dir, "experience_list.txt")

    paths = []
    for step in range(args.ckpt_start, args.ckpt_end + 1, args.ckpt_step):
        exp_file = os.path.join(
            exp_dir,
            f"global_step_{step}",
            f"extract_{args.val_samples_limit}_samples",
            "experiences",
            f"experience_{args.val_samples_use}.txt",
        )
        if os.path.exists(exp_file):
            paths.append(exp_file)
        else:
            print(f"[WARNING] Experience file not found: {exp_file}", file=sys.stderr)

    if not paths:
        print("[ERROR] No experience files found!", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        for p in paths:
            f.write(p + "\n")

    print(f"[OK] Wrote {len(paths)} paths to {output_path}")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
