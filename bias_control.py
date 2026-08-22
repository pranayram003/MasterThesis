"""
Control for the steering pilot: how much of the accuracy gain is available from
a pure label-bias term? Adds a constant to the entailment option's logit and
sweeps it. If this reaches the same accuracy as steering, the CAA vector is not
contributing anything beyond a bias correction.
One forward pass per item; the sweep is done arithmetically on cached logits.
"""
import argparse, collections, json, math
import torch
from modeling import letter_token_ids, load_model
from prompts import option_order, read_jsonl, render

def wilson(hits, n):
    if n == 0: return (0.0, 0.0)
    p, z = hits / n, 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default="data/caa_val.jsonl")
    ap.add_argument("--demos", default="data/eval_train.jsonl")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--out", default="reports/bias_control.json")
    args = ap.parse_args()

    items = read_jsonl(args.items)[: args.n]
    model, tokenizer = load_model(args.model, load_in_4bit=not args.no_4bit)
    ids = letter_token_ids(tokenizer)

    cached = []
    for item in items:
        text = render(item, args.demos, tokenizer)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits[0, -1]
        order = option_order(item)
        scores = {}
        for ltr, tid in ids.items():
            if ltr not in "AB":
                continue
            idx = "AB".index(ltr)
            if idx < len(order):
                scores[order[idx]] = logits[tid].item()
        cached.append((scores, item["gold_label"]))

    print("gold:", collections.Counter(g for _, g in cached))
    sweep = []
    for b in [x / 2 for x in range(-12, 13)]:
        hits, predicted = 0, collections.Counter()
        for scores, gold in cached:
            adj = dict(scores)
            adj["entailment"] = adj.get("entailment", 0.0) + b
            label = max(adj, key=adj.get)
            predicted[label] += 1
            hits += label == gold
        acc = hits / len(cached)
        lo, hi = wilson(hits, len(cached))
        sweep.append({"bias": b, "accuracy": acc, "ci95": [lo, hi],
                      "predicted": dict(predicted)})
        print("  bias %+5.1f  acc=%.3f  [%.3f, %.3f]  %s"
              % (b, acc, lo, hi, dict(predicted)))

    best = max(sweep, key=lambda r: r["accuracy"])
    print("\nbest bias %+.1f -> %.3f" % (best["bias"], best["accuracy"]))
    print("unsteered (bias 0) -> %.3f"
          % next(r["accuracy"] for r in sweep if r["bias"] == 0.0))
    with open(args.out, "w") as fh:
        json.dump({"model": args.model, "n": len(cached), "sweep": sweep,
                   "best": best}, fh, indent=2)
    print("wrote", args.out)

if __name__ == "__main__":
    main()
