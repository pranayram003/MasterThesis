"""
Is the failure about negation, or about finding the right person in the list?

Every item requires locating one name among up to twelve premise clauses before
the negation matters at all. Chance accuracy on the negated items is consistent
with two very different stories, and Phase 1 only makes sense under one of them.

This script runs the same items twice:

    negated     "Dana didn't visit Bristol"   correct answer = gold label
    unnegated   "Dana visited Bristol"        correct answer = shortcut label

Same premises, same retrieval load, negation present or absent. Results are also
broken down by premise length, since retrieval difficulty should scale with the
number of clauses while negation difficulty should not.

How to read the outcome:

    unnegated high, negated at chance
        the lookup works and negation breaks it, which is the premise of the
        project and makes the steering vector worth building

    both at chance
        the model cannot do the lookup, so there is no correct negation
        computation to recover and no target for Phase 3, and the benchmark
        section or the model needs to change before going further

    both high
        no failure to repair on this section

    python probe_negation.py --model mistralai/Mistral-7B-Instruct-v0.3 --n 200
"""

import argparse
import collections
import json
import math
import random
import re

HYPOTHESIS = re.compile(r"^(.+?) didn't visit (.+)$")
WORDS = {"entailment": "True", "contradiction": "False"}
LABELS = ["entailment", "contradiction"]
ASSISTANT_PREFIX = "Answer: ("

INSTRUCTION = (
    "Decide whether the hypothesis is true or false given the premise. The "
    "premise lists every place each person has visited. Answer with a single "
    "letter."
)


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def unnegate(hypothesis):
    match = HYPOTHESIS.match(hypothesis)
    return "%s visited %s" % (match.group(1), match.group(2))


def variant_of(item, condition):
    """Return (hypothesis_text, correct_label) for this condition."""
    if condition == "negated":
        return item["hypothesis"], item["gold_label"]
    return unnegate(item.get("orig_hypothesis", item["hypothesis"])), item["shortcut_label"]


def block(premise, hypothesis, order, answer_label=None):
    lines = ["Premise: " + premise, "Hypothesis: " + hypothesis, "Choices:"]
    for letter, label in zip("AB", order):
        lines.append("(%s) %s" % (letter, WORDS[label]))
    if answer_label is None:
        lines.append(ASSISTANT_PREFIX)
    else:
        letter = "AB"[order.index(answer_label)]
        lines.append("Answer: (%s) %s" % (letter, WORDS[answer_label]))
    return "\n".join(lines)


def pick_demos(train_items, condition, seed=7):
    """Two demonstrations, one per label, answers on different letters."""
    rng = random.Random(seed)
    pool = collections.defaultdict(list)
    for item in train_items:
        if item["diagnostic"]:
            pool[item["gold_label"]].append(item)
    demos = []
    for position, gold in enumerate(LABELS):
        item = rng.choice(pool[gold])
        hypothesis, correct = variant_of(item, condition)
        other = [l for l in LABELS if l != correct][0]
        order = [correct, other] if position == 0 else [other, correct]
        demos.append((item["premise"], hypothesis, order, correct))
    return demos


def build(item, condition, train_items):
    hypothesis, correct = variant_of(item, condition)
    rng = random.Random(item["idx"])
    order = LABELS[:]
    rng.shuffle(order)
    parts = [INSTRUCTION, ""]
    for premise, demo_hyp, demo_order, demo_answer in pick_demos(train_items, condition):
        parts.append(block(premise, demo_hyp, demo_order, answer_label=demo_answer))
        parts.append("")
    parts.append(block(item["premise"], hypothesis, order))
    return "\n".join(parts), order, correct


def render(user_turn, tokenizer):
    if tokenizer is None or tokenizer.chat_template is None:
        return user_turn
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_turn.rsplit(ASSISTANT_PREFIX, 1)[0].rstrip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return text + ASSISTANT_PREFIX


def wilson(hits, n):
    """95 percent interval, honest about small samples."""
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    z = 1.96
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def bucket(premise):
    n = len(premise.split(", "))
    if n <= 4:
        return "short 2-4"
    if n <= 8:
        return "mid 5-8"
    return "long 9-12"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--items", default="data/eval_val.jsonl")
    ap.add_argument("--train", default="data/eval_train.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--out", default="reports/negation_probe.json")
    args = ap.parse_args()

    items = [i for i in read_jsonl(args.items) if i["diagnostic"]][: args.n]
    train_items = read_jsonl(args.train)

    if args.dry_run:
        for condition in ("negated", "unnegated"):
            user_turn, order, correct = build(items[0], condition, train_items)
            print("=" * 70)
            print(condition, " correct:", correct, " order:", order)
            print("=" * 70)
            print(user_turn)
            print()
        return

    import torch
    from modeling import letter_token_ids, load_model

    model, tokenizer = load_model(args.model, load_in_4bit=not args.no_4bit)
    ids = letter_token_ids(tokenizer, letters=("A", "B"))

    results = {}
    per_item = []
    for condition in ("negated", "unnegated"):
        hits = 0
        by_bucket = collections.defaultdict(lambda: [0, 0])
        predicted = collections.Counter()
        for step, item in enumerate(items):
            user_turn, order, correct = build(item, condition, train_items)
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
            key = bucket(item["premise"])
            by_bucket[key][0] += ok
            by_bucket[key][1] += 1
            per_item.append({"idx": item["idx"], "condition": condition,
                             "correct": correct, "pred": label,
                             "clauses": len(item["premise"].split(", "))})
            if (step + 1) % 50 == 0:
                print("  %s %d/%d" % (condition, step + 1, len(items)), flush=True)

        low, high = wilson(hits, len(items))
        results[condition] = {
            "n": len(items),
            "accuracy": hits / len(items),
            "ci95": [low, high],
            "predicted": dict(predicted),
            "by_premise_length": {
                k: {"accuracy": v[0] / v[1], "n": v[1]} for k, v in sorted(by_bucket.items())
            },
        }
        print("%-10s acc=%.3f  95%% CI [%.3f, %.3f]  preds=%s"
              % (condition, hits / len(items), low, high, dict(predicted)), flush=True)
        for k, v in sorted(by_bucket.items()):
            print("    %-10s %.3f  (n=%d)" % (k, v[0] / v[1], v[1]))

    gap = results["unnegated"]["accuracy"] - results["negated"]["accuracy"]
    print("\nunnegated minus negated: %+.3f" % gap)
    print("chance on this binary task is 0.500")

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"summary": results, "gap": gap, "per_item": per_item}, fh, indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
