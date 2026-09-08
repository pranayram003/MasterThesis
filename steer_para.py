"""
Does steering still work when the negation is worded differently?

Phase 1 built the CAA direction on "didn't visit" items and showed a causal
effect at layers 17 to 22. The paraphrase study then showed that detection
degrades on reworded items, and that "never visited" fails asymmetrically:
accuracy collapses when the gold label is contradiction but holds up when it is
entailment. That asymmetry is the shortcut signature.

Detection is not intervention. This script closes that gap by running the same
steering sweep on the reworded eval sets, and reporting accuracy split by gold
label so the asymmetry is visible before and after steering.

What to look for:

    steering lifts contradiction accuracy toward entailment accuracy
        the direction generalises past the surface word "not"

    steering lifts both, or neither, or only entailment
        the direction is tied to the original wording, which is the case for
        rebuilding it from mixed phrasings

Prompts are built with the demo-matched builder, so demonstrations are phrased
the same way as the test items. Without that, part of any drop is the mismatch
artifact rather than the wording itself.

    python steer_para.py --model mistralai/Mistral-7B-Instruct-v0.3 \
        --vectors vectors/mistral7b.pt --layers 18 22 --n 200
"""

import argparse
import collections
import json
import os

import torch

from modeling import letter_token_ids, load_model
from probe_negation_demomatch import (DEMO_VARIANT, build, read_jsonl, render,
                                      wilson)
from steer_pilot import Steerer

COEFFS = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]


def evaluate(model, tokenizer, items, train_items, ids, steerer, coeff):
    steerer.coeff = coeff
    hits = 0
    predicted = collections.Counter()
    by_gold = collections.defaultdict(lambda: [0, 0])
    for item in items:
        user_turn, order, correct = build(item, "negated", train_items)
        enc = tokenizer(render(user_turn, tokenizer), return_tensors="pt",
                        add_special_tokens=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits[0, -1]
        scores = {ltr: logits[tid].item() for ltr, tid in ids.items()}
        letter = max(scores, key=scores.get)
        label = order["AB".index(letter)]
        predicted[label] += 1
        ok = label == correct
        hits += ok
        by_gold[correct][0] += ok
        by_gold[correct][1] += 1
    return hits, predicted, by_gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--items", nargs="+",
                    default=["data/para_never.jsonl", "data/para_not_been.jsonl"])
    ap.add_argument("--train", default="data/eval_train.jsonl")
    ap.add_argument("--layers", type=int, nargs="+", default=[18, 22])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--positions", choices=["all", "last"], default="all")
    ap.add_argument("--quantize", action="store_true",
                    help="load in 4-bit; off by default so numbers match the "
                         "full precision Phase 1 results")
    ap.add_argument("--random-dir", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="reports/steer_para.json")
    args = ap.parse_args()

    payload = torch.load(args.vectors, map_location="cpu")
    vectors = payload["resid_diff"]
    train_items = read_jsonl(args.train)

    model, tokenizer = load_model(args.model, load_in_4bit=args.quantize)
    ids = letter_token_ids(tokenizer, letters=("A", "B"))

    results = {}
    for path in args.items:
        items = [i for i in read_jsonl(path) if i.get("diagnostic", True)][: args.n]
        if not items:
            print("no items in", path)
            continue
        DEMO_VARIANT["name"] = items[0].get("variant", "orig")
        print("\n%s   n=%d   demos rephrased to: %s"
              % (path, len(items), DEMO_VARIANT["name"]), flush=True)
        results[path] = {"variant": DEMO_VARIANT["name"], "n": len(items),
                         "layers": {}}

        for layer in args.layers:
            if not 1 <= layer <= len(model.model.layers):
                print("skipping layer %d, outside the steerable range" % layer)
                continue
            if args.random_dir:
                g = torch.Generator().manual_seed(args.seed * 1000 + layer)
                raw = torch.randn(vectors[layer].shape, generator=g)
                unit = raw / raw.norm().clamp_min(1e-8)
            else:
                unit = vectors[layer] / vectors[layer].norm().clamp_min(1e-8)
            steerer = Steerer(model, layer, unit, positions=args.positions)

            steerer.coeff = 0.0
            steerer.norms = []
            evaluate(model, tokenizer, items[:10], train_items, ids, steerer, 0.0)
            scale = sum(steerer.norms) / len(steerer.norms)
            print("  layer %d, mean residual norm %.1f" % (layer, scale), flush=True)

            rows = []
            for fraction in COEFFS:
                hits, predicted, by_gold = evaluate(
                    model, tokenizer, items, train_items, ids, steerer,
                    fraction * scale)
                low, high = wilson(hits, len(items))
                split = {k: {"accuracy": v[0] / v[1], "n": v[1]}
                         for k, v in sorted(by_gold.items())}
                rows.append({"fraction": fraction, "coeff": fraction * scale,
                             "accuracy": hits / len(items), "ci95": [low, high],
                             "predicted": dict(predicted), "by_gold": split})
                ent = split.get("entailment", {}).get("accuracy", float("nan"))
                con = split.get("contradiction", {}).get("accuracy", float("nan"))
                print("    coeff %+.2f  acc=%.3f [%.3f, %.3f]   ent=%.3f  con=%.3f"
                      % (fraction, hits / len(items), low, high, ent, con),
                      flush=True)
            steerer.remove()
            results[path]["layers"][str(layer)] = {"scale": scale, "sweep": rows}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"model": args.model, "vectors": args.vectors,
                   "positions": args.positions,
                   "quantized": args.quantize, "results": results}, fh, indent=2)
    print("\nwrote", args.out)
    print("compare against the unsteered coeff 0.00 row in each block")


if __name__ == "__main__":
    main()
