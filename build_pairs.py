"""
Build the CAA contrast pair files.

    python build_pairs.py --train train.jsonl --val val.jsonl --out data/

Writes:
    data/caa_train.jsonl   diagnostic items only, used to build the vector
    data/caa_val.jsonl     diagnostic items only, used to validate it
    data/eval_train.jsonl  all items including neutral, for accuracy and controls
    data/eval_val.jsonl    all items including neutral, for accuracy and controls
"""

import argparse
import collections
import os

from data import prepare, read_jsonl, write_jsonl


def report(name, rows):
    gold = collections.Counter(r["gold_label"] for r in rows)
    diag = sum(r["diagnostic"] for r in rows)
    letters = collections.Counter(r["gold_letter"] for r in rows)
    print("%-10s n=%-5d diagnostic=%-5d gold=%s gold_letter=%s"
          % (name, len(rows), diag, dict(gold), dict(letters)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--val", default="val.jsonl")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for split, path in (("train", args.train), ("val", args.val)):
        rows = prepare(read_jsonl(path), seed=args.seed)
        report(split, rows)
        write_jsonl(os.path.join(args.out, "eval_%s.jsonl" % split), rows)
        pairs = [r for r in rows if r["diagnostic"]]
        write_jsonl(os.path.join(args.out, "caa_%s.jsonl" % split), pairs)

    print("\nwrote to", os.path.abspath(args.out))
    print(prepare(read_jsonl(args.val), seed=args.seed)[0]["prompt"])


if __name__ == "__main__":
    main()
