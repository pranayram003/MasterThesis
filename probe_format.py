"""
Prompt format probe.

The zero-shot three-option prompt makes Mistral answer "Undetermined" almost
every time, which hides whatever it does with negation. This script runs several
formats over the same items and reports, for each, how the answers are
distributed and how often the model lands on the reading that ignores the
negation.

Formats compared:

    zeroshot_3opt        current baseline, True / False / Undetermined
    fewshot_3opt         same, with three worked examples drawn from train
    fewshot_3opt_nli     same, labels worded as entailment / contradiction / neutral
    zeroshot_2opt        True / False only, diagnostic items only
    fewshot_2opt         True / False only with demonstrations

The demonstrations come from train.jsonl, which uses country names, while the
evaluation runs on val.jsonl, which uses US place names. The two vocabularies
barely overlap, so the demonstrations cannot leak the answers.

    python probe_format.py --dry-run
    python probe_format.py --model mistralai/Mistral-7B-Instruct-v0.3 --n 60
"""

import argparse
import collections
import json
import random

WORDS_PLAIN = {
    "entailment": "True",
    "contradiction": "False",
    "neutral": "Undetermined",
}
WORDS_NLI = {
    "entailment": "Entailment",
    "contradiction": "Contradiction",
    "neutral": "Neutral",
}

INSTRUCTION_3 = (
    "Decide whether the hypothesis is true, false, or undetermined given the "
    "premise. The premise lists every place each person has visited, so a person "
    "who is not mentioned cannot be judged. Answer with a single letter."
)
INSTRUCTION_3_NLI = (
    "Decide the inference relation between the premise and the hypothesis. The "
    "premise lists every place each person has visited, so a person who is not "
    "mentioned cannot be judged. Answer with a single letter."
)
INSTRUCTION_2 = (
    "Decide whether the hypothesis is true or false given the premise. The "
    "premise lists every place each person has visited. Answer with a single "
    "letter."
)
ASSISTANT_PREFIX = "Answer: ("


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def block(item, order, words, answer_label=None):
    lines = [
        "Premise: " + item["premise"],
        "Hypothesis: " + item["hypothesis"],
        "Choices:",
    ]
    for letter, label in zip("ABC", order):
        lines.append("(%s) %s" % (letter, words[label]))
    if answer_label is None:
        lines.append(ASSISTANT_PREFIX)
    else:
        letter = "ABC"[order.index(answer_label)]
        lines.append("Answer: (%s) %s" % (letter, words[answer_label]))
    return "\n".join(lines)


def pick_demos(train_items, labels_wanted, seed=7):
    """One demo per wanted label, with the answer on a different letter each time."""
    rng = random.Random(seed)
    pool = collections.defaultdict(list)
    for item in train_items:
        pool[item["gold_label"]].append(item)
    demos = []
    for position, label in enumerate(labels_wanted):
        item = rng.choice(pool[label])
        others = [l for l in labels_wanted if l != label]
        rng.shuffle(others)
        order = others[:]
        order.insert(position, label)   # answer sits on A, then B, then C
        demos.append((item, order, label))
    return demos


def build(item, variant, train_items):
    """Return (user_turn, option_order, words)."""
    two_option = variant.endswith("2opt")
    words = WORDS_NLI if "nli" in variant else WORDS_PLAIN
    labels = ["entailment", "contradiction"] if two_option else [
        "entailment", "contradiction", "neutral"]
    instruction = INSTRUCTION_2 if two_option else (
        INSTRUCTION_3_NLI if "nli" in variant else INSTRUCTION_3)

    rng = random.Random(item["idx"])
    order = list(labels)
    rng.shuffle(order)

    parts = [instruction, ""]
    if variant.startswith("fewshot"):
        for demo, demo_order, demo_label in pick_demos(train_items, labels):
            parts.append(block(demo, demo_order, words, answer_label=demo_label))
            parts.append("")
    parts.append(block(item, order, words))
    return "\n".join(parts), order, words


def render(user_turn, tokenizer, use_chat_template=True):
    if tokenizer is None:
        return user_turn
    if not use_chat_template or tokenizer.chat_template is None:
        return user_turn
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": user_turn.rsplit(ASSISTANT_PREFIX, 1)[0].rstrip()}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return text + ASSISTANT_PREFIX


VARIANTS = [
    "zeroshot_3opt",
    "fewshot_3opt",
    "fewshot_3opt_nli",
    "zeroshot_2opt",
    "fewshot_2opt",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--items", default="data/eval_val.jsonl")
    ap.add_argument("--train", default="data/eval_train.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--dry-run", action="store_true",
                    help="print one prompt per variant and exit, no model needed")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--out", default="reports/format_probe.json")
    args = ap.parse_args()

    items = read_jsonl(args.items)[: args.n]
    train_items = read_jsonl(args.train)

    if args.dry_run:
        for variant in VARIANTS:
            source = items if not variant.endswith("2opt") else [
                i for i in items if i["diagnostic"]]
            user_turn, order, _ = build(source[0], variant, train_items)
            print("=" * 70)
            print(variant, " gold:", source[0]["gold_label"], " order:", order)
            print("=" * 70)
            print(user_turn)
            print()
        return

    import torch
    from modeling import letter_token_ids, load_model

    model, tokenizer = load_model(args.model, load_in_4bit=not args.no_4bit)
    all_ids = letter_token_ids(tokenizer)

    results = {}
    for variant in VARIANTS:
        source = items if not variant.endswith("2opt") else [
            i for i in items if i["diagnostic"]]
        letters = "AB" if variant.endswith("2opt") else "ABC"
        ids = {k: v for k, v in all_ids.items() if k in letters}

        hits = 0
        shortcut = 0
        diagnostic = 0
        predicted = collections.Counter()
        margins = []

        for item in source:
            user_turn, order, _ = build(item, variant, train_items)
            text = render(user_turn, tokenizer)
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                logits = model(**enc, use_cache=False).logits[0, -1]
            scores = {ltr: logits[tid].item() for ltr, tid in ids.items()}
            ranked = sorted(scores.values(), reverse=True)
            margins.append(ranked[0] - ranked[1])
            letter = max(scores, key=scores.get)
            label = order["ABC".index(letter)]
            predicted[label] += 1
            if label == item["gold_label"]:
                hits += 1
            if item["diagnostic"]:
                diagnostic += 1
                if label == item["shortcut_label"]:
                    shortcut += 1

        results[variant] = {
            "n": len(source),
            "accuracy": hits / len(source),
            "shortcut_rate": shortcut / diagnostic if diagnostic else None,
            "predicted": dict(predicted),
            "mean_margin": sum(margins) / len(margins),
        }
        print("%-18s n=%-4d acc=%.3f shortcut=%s margin=%.2f  %s"
              % (variant, len(source), results[variant]["accuracy"],
                 "%.3f" % results[variant]["shortcut_rate"]
                 if results[variant]["shortcut_rate"] is not None else "n/a",
                 results[variant]["mean_margin"], dict(predicted)), flush=True)

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
