"""
Does the Phase 1 vector actually move behaviour?

Phase 1 established that a linear direction separates the correct reading from
the negation-ignoring reading on held-out data at AUROC 0.745, while the model's
own answers sit at chance. That is a correlational result. This script asks the
causal question by adding the vector to the residual stream during the forward
pass and measuring accuracy.

The sweep is over layer and coefficient. Coefficients are expressed as a
fraction of the mean residual norm at the target layer, measured on the fly, so
the numbers mean the same thing across layers and models. Both signs are tried:
positive should help if the vector points from the shortcut reading toward the
correct one, and negative should hurt. A vector with real causal purchase
produces a curve, not a flat line. Symmetry matters as much as peak accuracy,
since an effect in one direction only is more likely to be a generic disruption
than a handle on the concept.

Layer indices match the validation table, where index i is the hidden state
after decoder layer i-1. Index 0 is the embedding output and index 32 sits after
the final norm, so neither can be steered; the usable range is 1 to 31.

    python steer_pilot.py --model mistralai/Mistral-7B-Instruct-v0.3 \
        --vectors vectors/mistral7b.pt --layers 22 25 31 --n 150
"""

import argparse
import collections
import json
import math
import os

import torch

from modeling import letter_token_ids, load_model
from prompts import option_order, read_jsonl, render

COEFFS = [-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4]


def wilson(hits, n):
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


class Steerer:
    """Adds a fixed vector to the output of one decoder layer."""

    def __init__(self, model, layer_index, direction, positions="all"):
        # hidden_states index i is the output of decoder layer i-1
        self.module = model.model.layers[layer_index - 1]
        self.direction = direction
        self.positions = positions
        self.coeff = 0.0
        self.norms = []
        self.handle = self.module.register_forward_hook(self._hook)

    def _hook(self, _module, _args, output):
        is_tuple = isinstance(output, tuple)
        hidden = output[0] if is_tuple else output
        self.norms.append(hidden[0, -1, :].float().norm().item())
        if self.coeff != 0.0:
            delta = (self.coeff * self.direction).to(hidden.dtype).to(hidden.device)
            if self.positions == "last":
                hidden = hidden.clone()
                hidden[:, -1, :] += delta
            else:
                hidden = hidden + delta
        return (hidden,) + output[1:] if is_tuple else hidden

    def remove(self):
        self.handle.remove()


def evaluate(model, tokenizer, items, demos_path, ids, steerer=None, coeff=0.0):
    if steerer is not None:
        steerer.coeff = coeff
    hits = 0
    predicted = collections.Counter()
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
        hits += label == item["gold_label"]
    return hits, predicted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--items", default="data/caa_val.jsonl")
    ap.add_argument("--demos", default="data/eval_train.jsonl")
    ap.add_argument("--layers", type=int, nargs="+", default=[22, 25, 31])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--positions", choices=["all", "last"], default="all")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--out", default="reports/steer_pilot.json")
    args = ap.parse_args()

    payload = torch.load(args.vectors, map_location="cpu")
    vectors = payload["resid_diff"]

    items = read_jsonl(args.items)[: args.n]
    model, tokenizer = load_model(args.model, load_in_4bit=not args.no_4bit)
    ids = letter_token_ids(tokenizer, letters=("A", "B"))

    results = {}
    for layer in args.layers:
        if not 1 <= layer <= len(model.model.layers):
            print("skipping layer %d, outside the steerable range" % layer)
            continue
        unit = vectors[layer] / vectors[layer].norm().clamp_min(1e-8)
        steerer = Steerer(model, layer, unit, positions=args.positions)

        # calibrate: mean residual norm at this layer, from a short unsteered pass
        steerer.coeff = 0.0
        steerer.norms = []
        evaluate(model, tokenizer, items[:10], args.demos, ids, steerer, 0.0)
        scale = sum(steerer.norms) / len(steerer.norms)
        print("\nlayer %d, mean residual norm %.1f" % (layer, scale), flush=True)

        rows = []
        for fraction in COEFFS:
            hits, predicted = evaluate(
                model, tokenizer, items, args.demos, ids, steerer, fraction * scale
            )
            low, high = wilson(hits, len(items))
            rows.append({"fraction": fraction, "coeff": fraction * scale,
                         "accuracy": hits / len(items), "ci95": [low, high],
                         "predicted": dict(predicted)})
            print("  coeff %+.2f x norm   acc=%.3f  [%.3f, %.3f]  %s"
                  % (fraction, hits / len(items), low, high, dict(predicted)), flush=True)
        steerer.remove()
        results[str(layer)] = {"scale": scale, "sweep": rows}

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"model": args.model, "vectors": args.vectors,
                   "n": len(items), "positions": args.positions,
                   "results": results}, fh, indent=2)
    print("\nwrote", args.out)
    print("baseline for comparison: 0.485 unsteered, chance is 0.500")


if __name__ == "__main__":
    main()
