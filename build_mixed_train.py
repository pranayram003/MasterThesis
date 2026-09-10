"""
Mixed-wording training set for CAA extraction.

The direction in vectors/mistral_bf16.pt was extracted from caa_train.jsonl,
where every negation is "NAME didn't visit PLACE". Three results suggest it
encodes that template rather than negation itself: steering lifts entailment
accuracy while leaving contradiction near chance, projecting it out of L19H10
recovers only 0.022 of the 0.110 that disabling the head gives, and adding it
into that head lands inside the random-direction null.

This assigns the six surface forms round-robin across the training items, so a
direction extracted here has to average over phrasings and cannot lean on the
word "not". Names, places, premises and gold labels are untouched; only the
negation form varies.

    python build_mixed_train.py
"""

import collections
import json
import re

import prompts

SRC = "data/caa_train.jsonl"
OUT = "data/caa_train_mixed.jsonl"
PAT = re.compile(r"^(?P<name>.+?) didn'?t visit (?P<place>.+?)$")

VARIANTS = [
    ("orig",       "{name} didn't visit {place}"),
    ("did_not",    "{name} did not visit {place}"),
    ("never",      "{name} never visited {place}"),
    ("not_been",   "{name} has not been to {place}"),
    ("cleft",      "It is not true that {name} visited {place}"),
    ("never_been", "{name} has never been to {place}"),
]

items = prompts.read_jsonl(SRC)
out, counts, skipped = [], collections.Counter(), 0
for i, it in enumerate(items):
    m = PAT.match(it["hypothesis"])
    if not m:
        skipped += 1
        continue
    name, template = VARIANTS[i % len(VARIANTS)]
    new = dict(it)
    new["orig_hypothesis"] = it["hypothesis"]
    new["hypothesis"] = template.format(**m.groupdict())
    new["variant"] = name
    # stale fields from the abandoned three-option format
    new.pop("user_turn", None)
    new.pop("prompt", None)
    out.append(new)
    counts[name] += 1

with open(OUT, "w", encoding="utf-8") as fh:
    for row in out:
        fh.write(json.dumps(row) + "\n")

print("wrote %s  n=%d  skipped=%d" % (OUT, len(out), skipped))
for name, _ in VARIANTS:
    print("  %-11s %d" % (name, counts[name]))
