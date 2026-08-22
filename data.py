"""
Parsing, shortcut-label derivation and prompt construction for the
CURRICULUM negation ("has only visited" / "didn't visit") fragment.

Every item has the form:
    premise:    "<Name> has only visited <Place>, <Name> has only visited <Place>, ..."
    hypothesis: "<Name> didn't visit <Place>"

Gold label (verified to reproduce the released labels exactly, 4000/4000):
    name not in premise                  -> neutral
    name in premise, place == its place  -> contradiction
    name in premise, place != its place  -> entailment

Shortcut label = the label the model would give if it dropped the negation and
read the hypothesis as "<Name> visited <Place>":
    name not in premise                  -> neutral
    name in premise, place == its place  -> entailment
    name in premise, place != its place  -> contradiction

So on every entailment/contradiction item the shortcut is the exact inversion of
the gold label, and on every neutral item the shortcut and the gold agree.
Neutral items are therefore negation-invariant and are used as a control, not as
contrast pairs.
"""

import json
import random
import re

PREMISE_CLAUSE = re.compile(r"^(.+?) has only visited (.+)$")
HYPOTHESIS = re.compile(r"^(.+?) didn't visit (.+)$")

LABELS = ("entailment", "contradiction", "neutral")

VERBALISER = {
    "entailment": "True",
    "contradiction": "False",
    "neutral": "Undetermined",
}

QUESTION = (
    "Given the premise, is the hypothesis true, false, or undetermined?"
)


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def parse_item(row):
    """Return (visits, hypothesis_name, hypothesis_place)."""
    visits = {}
    for clause in row["premise"].split(", "):
        match = PREMISE_CLAUSE.match(clause)
        if match is None:
            raise ValueError("unparsable premise clause: %r" % clause)
        visits[match.group(1)] = match.group(2)
    match = HYPOTHESIS.match(row["hypothesis"])
    if match is None:
        raise ValueError("unparsable hypothesis: %r" % row["hypothesis"])
    return visits, match.group(1), match.group(2)


def derive_labels(row):
    """Return (gold_label, shortcut_label) derived from the surface form."""
    visits, name, place = parse_item(row)
    if name not in visits:
        return "neutral", "neutral"
    if visits[name] == place:
        return "contradiction", "entailment"
    return "entailment", "contradiction"


ASSISTANT_PREFIX = "Answer: ("


def build_user_turn(row, option_order):
    """
    Three-option multiple choice question. option_order is a tuple of the three
    labels in the order they are presented, so option_order[0] is choice (A).

    The same format is used for vector construction and for evaluation, so the
    steering vector is never asked to transfer across prompt formats.
    """
    lines = [
        "Premise: " + row["premise"],
        "Hypothesis: " + row["hypothesis"],
        "Question: " + QUESTION,
        "",
        "Choices:",
    ]
    for letter, label in zip("ABC", option_order):
        lines.append("(%s) %s" % (letter, VERBALISER[label]))
    return "\n".join(lines)


def build_prompt(row, option_order):
    """Plain concatenation, for base models or for inspection."""
    return build_user_turn(row, option_order) + "\n\n" + ASSISTANT_PREFIX


def render(item, tokenizer=None, use_chat_template=True):
    """
    Turn a prepared item into the text that precedes the answer letter.

    With an instruct model the question goes in a user turn and the assistant
    turn is opened with "Answer: (" so the next token is the letter itself.
    """
    if tokenizer is None or not use_chat_template or tokenizer.chat_template is None:
        return item["prompt"]
    messages = [{"role": "user", "content": item["user_turn"]}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text + ASSISTANT_PREFIX


def letter_of(label, option_order):
    return "ABC"[option_order.index(label)]


def prepare(rows, seed=0):
    """
    Attach derived labels, a seeded option order and the prompt to every item.

    The option order is shuffled per item. Because the positive and negative
    branch of a contrast pair differ only in which letter is appended, and the
    letters are counterbalanced across the dataset, the mean difference cancels
    out any "letter A vs letter B" component and leaves the difference between
    committing to the correct reading and committing to the shortcut reading.
    """
    rng = random.Random(seed)
    out = []
    for row in rows:
        gold, shortcut = derive_labels(row)
        if gold != row["gold_label"]:
            raise ValueError("derived label disagrees with gold at idx %s" % row["idx"])
        order = list(LABELS)
        rng.shuffle(order)
        order = tuple(order)
        out.append(
            {
                "idx": row["idx"],
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "gold_label": gold,
                "shortcut_label": shortcut,
                "diagnostic": gold != shortcut,
                "option_order": list(order),
                "gold_letter": letter_of(gold, order),
                "shortcut_letter": letter_of(shortcut, order),
                "user_turn": build_user_turn(row, order),
                "prompt": build_prompt(row, order),
            }
        )
    return out
