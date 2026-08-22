"""
The prompt format used for everything from here on.

Chosen empirically, not by preference. The three-option zero-shot format made
Mistral answer "Undetermined" on 51 of 60 items, and few-shot did not shift it,
so no negation behaviour was observable through it. The two-option format below
makes the model commit, and the unnegated control run through this same format
reaches 0.940 accuracy, which shows the format itself is not the bottleneck.

Vector construction and evaluation both use this format, so the steering vector
is never asked to transfer across formats.

Neutral items cannot be expressed with two options and are excluded here. They
were already excluded from the contrast pairs as negation-invariant. Side effect
measurement moves to the non-negation control sections of CURRICULUM, which is
where the proposal already puts it.
"""

import json
import random

LABELS = ["entailment", "contradiction"]
WORDS = {"entailment": "True", "contradiction": "False"}
ASSISTANT_PREFIX = "Answer: ("

INSTRUCTION = (
    "Decide whether the hypothesis is true or false given the premise. The "
    "premise lists every place each person has visited. Answer with a single "
    "letter."
)

_DEMO_CACHE = {}


def read_jsonl(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _block(premise, hypothesis, order, answer_label=None):
    lines = ["Premise: " + premise, "Hypothesis: " + hypothesis, "Choices:"]
    for letter, label in zip("AB", order):
        lines.append("(%s) %s" % (letter, WORDS[label]))
    if answer_label is None:
        lines.append(ASSISTANT_PREFIX)
    else:
        letter = "AB"[order.index(answer_label)]
        lines.append("Answer: (%s) %s" % (letter, WORDS[answer_label]))
    return "\n".join(lines)


def demos(train_path, seed=7):
    """
    Two demonstrations drawn from train.jsonl, one per label, with the answer on
    a different letter in each. Train uses country names and evaluation uses US
    place names with almost no overlap, so the demonstrations cannot leak.
    """
    key = (train_path, seed)
    if key in _DEMO_CACHE:
        return _DEMO_CACHE[key]
    rng = random.Random(seed)
    pool = {label: [] for label in LABELS}
    for item in read_jsonl(train_path):
        if item["diagnostic"]:
            pool[item["gold_label"]].append(item)
    built = []
    for position, label in enumerate(LABELS):
        item = rng.choice(pool[label])
        other = [l for l in LABELS if l != label][0]
        order = [label, other] if position == 0 else [other, label]
        built.append(_block(item["premise"], item["hypothesis"], order, answer_label=label))
    _DEMO_CACHE[key] = built
    return built


def option_order(item):
    """Deterministic per item, seeded by idx so every script agrees."""
    rng = random.Random(item["idx"])
    order = LABELS[:]
    rng.shuffle(order)
    return order


def letters_for(item):
    """(gold_letter, shortcut_letter) under this item's option order."""
    order = option_order(item)
    return "AB"[order.index(item["gold_label"])], "AB"[order.index(item["shortcut_label"])]


def user_turn(item, train_path):
    parts = [INSTRUCTION, ""]
    for demo in demos(train_path):
        parts.append(demo)
        parts.append("")
    parts.append(_block(item["premise"], item["hypothesis"], option_order(item)))
    return "\n".join(parts)


def render(item, train_path, tokenizer=None, use_chat_template=True):
    """Text up to and including 'Answer: (', so the next token is the letter."""
    text = user_turn(item, train_path)
    if tokenizer is None or not use_chat_template or tokenizer.chat_template is None:
        return text
    question = text.rsplit(ASSISTANT_PREFIX, 1)[0].rstrip()
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered + ASSISTANT_PREFIX
