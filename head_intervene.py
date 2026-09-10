"""
Directional ablation and head-targeted steering on the shortcut heads.

Mean substitution disables a head entirely, which answers whether the head
matters but not what about it matters. L19H10 does many things; only some of
them are the negation shortcut. This script intervenes on the direction rather
than the head, in two ways:

    project    remove the component along a chosen direction from what the head
               writes, leaving everything orthogonal to it intact. The surgical
               necessity test: if accuracy still rises, the shortcut lives in
               that direction and not merely in that head.

    add        inject the direction into the head's output, scaled by a
               fraction of the head's own norm. The sufficiency test: if adding
               it back hurts, the direction is doing the damage.

Direction defaults to head_diff for the chosen head, the correct-minus-shortcut
contrast measured at that head during extraction. --direction resid uses the
residual stream vector at the corresponding layer instead, projected into the
head subspace, which asks whether the head carries the same distinction the
residual stream does.

Layer indices are decoder blocks, matching ablate_heads.py: block L here is
residual layer L+1 in steer_pilot.py.

    python head_intervene.py --model mistralai/Mistral-7B-Instruct-v0.3 \
        --vectors vectors/mistral_bf16.pt --head 19 10 --mode project --n 400
"""

import argparse
import collections
import json
import math
import os

import torch

from modeling import decoder_layers, head_geometry, letter_token_ids, load_model
from prompts import DEMO_SEED, option_order, read_jsonl, render

COEFFS = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]


def wilson(hits, n):
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


class HeadDirection:
    """Projects out of, or adds to, one head's contribution at o_proj input."""

    def __init__(self, model, layer, head, unit, n_heads, d_head):
        self.layer = layer
        self.head = head
        self.unit = unit                # [d_head], unit norm
        self.n_heads = n_heads
        self.d_head = d_head
        self.mode = "off"               # off | project | add
        self.coeff = 0.0
        self.norms = []
        module = decoder_layers(model)[layer].self_attn.o_proj
        self.handle = module.register_forward_pre_hook(self._hook)

    def _hook(self, _module, args):
        z = args[0]
        shape = z.shape
        view = z.reshape(shape[0], shape[1], self.n_heads, self.d_head)
        slab = view[:, :, self.head, :]
        self.norms.append(slab[0, -1, :].float().norm().item())
        if self.mode == "off":
            return None
        unit = self.unit.to(z.dtype).to(z.device)
        view = view.clone()
        slab = view[:, :, self.head, :]
        if self.mode == "project":
            # remove the component along unit
            proj = (slab.float() @ unit.float()).unsqueeze(-1) * unit.float()
            view[:, :, self.head, :] = (slab.float() - proj).to(z.dtype)
        else:
            view[:, :, self.head, :] = (slab.float()
                                        + self.coeff * unit.float()).to(z.dtype)
        return (view.reshape(shape),) + args[1:]

    def remove(self):
        self.handle.remove()


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
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--head", type=int, nargs=2, required=True,
                    metavar=("LAYER", "HEAD"), help="decoder block and head index")
    ap.add_argument("--mode", choices=["project", "add", "both"], default="both")
    ap.add_argument("--direction", choices=["head", "resid", "random"], default="head")
    ap.add_argument("--items", default="data/para_orig.jsonl")
    ap.add_argument("--demos", default="data/eval_train.jsonl")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--demo-seed", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--out", default="reports/head_intervene.json")
    args = ap.parse_args()

    DEMO_SEED["value"] = args.demo_seed
    layer, head = args.head

    payload = torch.load(args.vectors, map_location="cpu")
    items = read_jsonl(args.items)[: args.n]
    model, tokenizer = load_model(args.model, load_in_4bit=args.quantize)
    ids = letter_token_ids(tokenizer, letters=("A", "B"))
    n_heads, d_head = head_geometry(model)

    if args.direction == "head":
        raw = payload["head_diff"][layer, head].float()
    elif args.direction == "resid":
        # residual layer index is block + 1; slice this head's subspace
        vec = payload["resid_diff"][layer + 1].float()
        raw = vec[head * d_head:(head + 1) * d_head]
    else:
        g = torch.Generator().manual_seed(args.seed)
        raw = torch.randn(d_head, generator=g)
    unit = raw / raw.norm().clamp_min(1e-8)

    hook = HeadDirection(model, layer, head, unit, n_heads, d_head)

    hook.mode = "off"
    base_hits, base_pred, base_gold = evaluate(model, tokenizer, items, args.demos, ids)
    base_acc = base_hits / len(items)
    low, high = wilson(base_hits, len(items))
    scale = sum(hook.norms) / max(1, len(hook.norms))
    print("L%d H%d, direction=%s, mean head norm %.2f" % (layer, head, args.direction, scale))
    print("baseline acc=%.3f [%.3f, %.3f]  %s\n" % (base_acc, low, high, dict(base_pred)),
          flush=True)

    results = {"baseline": base_acc, "head_norm": scale}

    if args.mode in ("project", "both"):
        hook.mode = "project"
        hits, pred, gold = evaluate(model, tokenizer, items, args.demos, ids)
        acc = hits / len(items)
        lo, hi = wilson(hits, len(items))
        split = {k: v[0] / v[1] for k, v in gold.items()}
        results["project"] = {"accuracy": acc, "delta": acc - base_acc,
                              "ci95": [lo, hi], "predicted": dict(pred),
                              "by_gold": split}
        print("project   acc=%.3f  delta=%+.3f  [%.3f, %.3f]  %s"
              % (acc, acc - base_acc, lo, hi, split), flush=True)

    if args.mode in ("add", "both"):
        rows = []
        print()
        for fraction in COEFFS:
            hook.mode = "off" if fraction == 0.0 else "add"
            hook.coeff = fraction * scale
            hits, pred, gold = evaluate(model, tokenizer, items, args.demos, ids)
            acc = hits / len(items)
            lo, hi = wilson(hits, len(items))
            split = {k: v[0] / v[1] for k, v in gold.items()}
            rows.append({"fraction": fraction, "accuracy": acc,
                         "delta": acc - base_acc, "ci95": [lo, hi],
                         "predicted": dict(pred), "by_gold": split})
            print("add %+.2f   acc=%.3f  delta=%+.3f  [%.3f, %.3f]  %s"
                  % (fraction, acc, acc - base_acc, lo, hi, split), flush=True)
        results["add"] = rows

    hook.remove()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"model": args.model, "head": [layer, head],
                   "direction": args.direction, "items": args.items,
                   "n": len(items), "demo_seed": args.demo_seed,
                   "results": results}, fh, indent=2)
    print("\nwrote", args.out)
    print("compare against mean substitution of this head: +0.110 at n=400")


if __name__ == "__main__":
    main()
