"""
Unsteered baseline, which is the number the whole project rests on.

At the answer position the logits of the three letter tokens are compared and
mapped back through this item's option order to a label. Reported:

    accuracy overall and per gold label
    shortcut rate, the fraction of diagnostic items where the model picks the
        label it would give if it ignored the negation
    neutral items are negation invariant and act as a within-dataset control

If the shortcut rate is not clearly above chance there is nothing for Phase 3 to
repair, so this is worth running before any intervention work.

    python baseline_eval.py \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --items data/eval_val.jsonl \
        --out reports/baseline_val.json
"""

import argparse
import collections
import json
import os

import torch

from data import read_jsonl, render
from modeling import letter_token_ids, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--items", default="data/eval_val.jsonl")
    ap.add_argument("--out", default="reports/baseline.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--no-chat-template", action="store_true")
    args = ap.parse_args()

    items = read_jsonl(args.items)
    if args.limit:
        items = items[: args.limit]

    model, tokenizer = load_model(args.model, load_in_4bit=not args.no_4bit)
    letter_ids = letter_token_ids(tokenizer)

    correct = collections.Counter()
    total = collections.Counter()
    shortcut_hits = 0
    diagnostic = 0
    confusion = collections.Counter()
    records = []

    for step, item in enumerate(items):
        prefix = render(item, tokenizer, use_chat_template=not args.no_chat_template)
        enc = tokenizer(prefix, return_tensors="pt", add_special_tokens=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits[0, -1]
        scores = {ltr: logits[tid].item() for ltr, tid in letter_ids.items()}
        picked_letter = max(scores, key=scores.get)
        predicted = item["option_order"]["ABC".index(picked_letter)]

        total[item["gold_label"]] += 1
        if predicted == item["gold_label"]:
            correct[item["gold_label"]] += 1
        confusion[(item["gold_label"], predicted)] += 1
        if item["diagnostic"]:
            diagnostic += 1
            if predicted == item["shortcut_label"]:
                shortcut_hits += 1
        records.append({"idx": item["idx"], "gold": item["gold_label"],
                        "shortcut": item["shortcut_label"], "pred": predicted,
                        "letter_logits": scores})
        if (step + 1) % 200 == 0:
            print("%d/%d" % (step + 1, len(items)), flush=True)

    n = sum(total.values())
    summary = {
        "model": args.model,
        "items": args.items,
        "n": n,
        "accuracy": sum(correct.values()) / n,
        "accuracy_by_label": {k: correct[k] / v for k, v in total.items()},
        "n_by_label": dict(total),
        "diagnostic_n": diagnostic,
        "shortcut_rate": shortcut_hits / diagnostic if diagnostic else None,
        "confusion": {"%s->%s" % k: v for k, v in confusion.items()},
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "confusion"}, indent=2))
    print("confusion (gold -> predicted):")
    for k, v in sorted(confusion.items()):
        print("  %-13s -> %-13s %d" % (k[0], k[1], v))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"summary": summary, "records": records}, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
