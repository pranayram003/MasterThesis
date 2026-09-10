"""
Which heads are necessary for the negation reading?

rank_heads_band.py ranks heads correlationally, by how much their output
differs between the correct and shortcut branches. That produces candidates,
not results: a head can carry the distinction without the model relying on it,
and the global ranking is dominated by layers 29-31 where the answer is already
decided. This script asks the causal question instead, by disabling one head at
a time and measuring what happens to accuracy.

Ablation is by mean substitution rather than zeroing. The mean of each head's
output is measured once over a calibration pass and then written in place of
the live value at every position. Zeroing removes the head's average
contribution as well as its item-specific one, so a drop can reflect the model
being pushed off-distribution rather than the head's role in this task. Mean
substitution removes only the part that varies with the item, which is the
necessity claim worth making.

Heads are addressed at the input to o_proj, matching HeadOutputCapture, so head
h occupies columns h*d_head to (h+1)*d_head. Layer indices here are decoder
block indices, 0 to n_layers-1. Note this differs from the residual stream
convention used by steer_pilot.py and validate_vectors.py, where index i is the
hidden state after block i-1: block 17 here is residual layer 18 there.

    python ablate_heads.py --model mistralai/Mistral-7B-Instruct-v0.3 \
        --layers 17 22 --n 150
"""

import argparse
import collections
import json
import math
import os
import time

import torch

from modeling import head_geometry, letter_token_ids, load_model
from prompts import option_order, read_jsonl, render


def wilson(hits, n):
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


class HeadAblator:
    """Replaces one head's o_proj input with a fixed mean, at every position."""

    def __init__(self, model, n_heads, d_head):
        self.n_heads = n_heads
        self.d_head = d_head
        self.target = None          # (layer, head) or None
        self.means = {}             # (layer, head) -> tensor[d_head]
        self.collect = None         # layer index while calibrating
        self.sums = collections.defaultdict(float)
        self.counts = collections.defaultdict(int)
        self.handles = []
        from modeling import decoder_layers
        for i, layer in enumerate(decoder_layers(model)):
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._make_hook(i))
            )

    def _make_hook(self, index):
        def hook(_module, args):
            z = args[0]
            shape = z.shape
            view = z.reshape(shape[0], shape[1], self.n_heads, self.d_head)
            if self.collect is not None:
                # accumulate per-head means over all positions
                flat = view.reshape(-1, self.n_heads, self.d_head).float()
                self.sums[index] = self.sums[index] + flat.sum(dim=0).cpu()
                self.counts[index] += flat.shape[0]
                return None
            if self.target is not None and self.target[0] == index:
                head = self.target[1]
                mean = self.means[self.target].to(z.dtype).to(z.device)
                view = view.clone()
                view[:, :, head, :] = mean
                return (view.reshape(shape),) + args[1:]
            return None
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def evaluate(model, tokenizer, items, demos_path, ids):
    hits = 0
    predicted = collections.Counter()
    by_gold = collections.defaultdict(lambda: [0, 0])
    for item in items:
        text = render(item, demos_path, tokenizer)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits[0, -1]
        scores = {ltr: logits[tid].item() for ltr, tid in ids.items()}
        letter = max(scores, key=scores.get)
        label = option_order(item)["AB".index(letter)]
        predicted[label] += 1
        ok = label == item["gold_label"]
        hits += ok
        by_gold[item["gold_label"]][0] += ok
        by_gold[item["gold_label"]][1] += 1
    return hits, predicted, by_gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default="data/para_orig.jsonl")
    ap.add_argument("--demos", default="data/eval_train.jsonl")
    ap.add_argument("--layers", type=int, nargs=2, default=[17, 22],
                    metavar=("LO", "HI"), help="inclusive band of decoder blocks")
    ap.add_argument("--heads", type=int, nargs="+", default=None,
                    help="specific heads only; default is all heads in the band")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--calib", type=int, default=32,
                    help="items used to estimate the per-head means")
    ap.add_argument("--demo-seed", type=int, default=7)
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--out", default="reports/ablate_heads.json")
    args = ap.parse_args()

    from prompts import DEMO_SEED
    DEMO_SEED["value"] = args.demo_seed

    items = read_jsonl(args.items)[: args.n]
    model, tokenizer = load_model(args.model, load_in_4bit=args.quantize)
    ids = letter_token_ids(tokenizer, letters=("A", "B"))
    n_heads, d_head = head_geometry(model)

    ablator = HeadAblator(model, n_heads, d_head)

    # calibration: per-head means over a subset, all positions
    ablator.collect = True
    evaluate(model, tokenizer, items[: args.calib], args.demos, ids)
    ablator.collect = None
    for layer in list(ablator.sums):
        mean = ablator.sums[layer] / max(1, ablator.counts[layer])
        for h in range(n_heads):
            ablator.means[(layer, h)] = mean[h]
    print("calibrated means over %d items" % args.calib, flush=True)

    # baseline, nothing ablated
    ablator.target = None
    base_hits, base_pred, base_gold = evaluate(model, tokenizer, items, args.demos, ids)
    base_acc = base_hits / len(items)
    low, high = wilson(base_hits, len(items))
    print("baseline acc=%.3f [%.3f, %.3f]  %s\n"
          % (base_acc, low, high, dict(base_pred)), flush=True)

    lo, hi = args.layers
    heads = args.heads if args.heads is not None else list(range(n_heads))
    rows = []
    start = time.time()
    total = (hi - lo + 1) * len(heads)
    done = 0
    for layer in range(lo, hi + 1):
        for head in heads:
            ablator.target = (layer, head)
            hits, pred, gold = evaluate(model, tokenizer, items, args.demos, ids)
            acc = hits / len(items)
            split = {k: v[0] / v[1] for k, v in gold.items()}
            rows.append({"layer": layer, "head": head, "accuracy": acc,
                         "delta": acc - base_acc, "predicted": dict(pred),
                         "by_gold": split})
            done += 1
            rate = done / max(1e-9, time.time() - start)
            print("  L%-2d H%-2d  acc=%.3f  delta=%+.3f   (%d/%d, %.1f/min, eta %.0f min)"
                  % (layer, head, acc, acc - base_acc, done, total, rate * 60,
                     (total - done) / max(1e-9, rate) / 60), flush=True)
    ablator.target = None
    ablator.remove()

    rows.sort(key=lambda r: r["delta"])
    print("\nlargest drops:")
    for r in rows[:15]:
        print("  L%-2d H%-2d  %+.3f  (acc %.3f)" % (r["layer"], r["head"], r["delta"], r["accuracy"]))
    print("\nlargest gains:")
    for r in rows[-10:][::-1]:
        print("  L%-2d H%-2d  %+.3f  (acc %.3f)" % (r["layer"], r["head"], r["delta"], r["accuracy"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"model": args.model, "items": args.items, "n": len(items),
                   "band": [lo, hi], "baseline": base_acc,
                   "demo_seed": args.demo_seed, "rows": rows}, fh, indent=2)
    print("\nwrote", args.out)
    print("note: layer indices here are decoder blocks; block L is residual layer L+1")


if __name__ == "__main__":
    main()
