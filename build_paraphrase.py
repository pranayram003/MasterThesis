"""
Paraphrase generalisation set.

Every negation in eval_val is "NAME didn't visit PLACE", so a direction
extracted there may encode that template rather than negation. These variants
hold names, places, premise and gold labels fixed and vary only the negation
surface form, so any drop in AUROC is attributable to phrasing.

Stored user_turn/prompt fields are OVERWRITTEN: the originals are from the
abandoned three-option format and are not used by the live pipeline.
"""

import json, re
import prompts

TRAIN_PATH = "data/eval_train.jsonl"
SRC = "data/caa_val.jsonl"
PAT = re.compile(r"^(?P<name>.+?) didn'?t visit (?P<place>.+?)$")

VARIANTS = {
    "orig":       "{name} didn't visit {place}",      # control: rebuilt, not copied
    "did_not":    "{name} did not visit {place}",
    "never":      "{name} never visited {place}",
    "not_been":   "{name} has not been to {place}",
    "cleft":      "It is not true that {name} visited {place}",
    "never_been": "{name} has never been to {place}",
}

items = prompts.read_jsonl(SRC)
skipped = sum(1 for it in items if not PAT.match(it["hypothesis"]))
if skipped:
    print(f"note: {skipped} hypotheses did not match the template, skipped")

for name, template in VARIANTS.items():
    out = []
    for it in items:
        m = PAT.match(it["hypothesis"])
        if not m:
            continue
        new = dict(it)
        new["hypothesis"] = template.format(**m.groupdict())
        new["variant"] = name
        new["orig_hypothesis"] = it["hypothesis"]
        new["user_turn"] = prompts.user_turn(new, TRAIN_PATH)
        new["prompt"] = prompts.render(new, TRAIN_PATH)
        out.append(new)
    path = f"data/para_{name}.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for o in out:
            fh.write(json.dumps(o) + "\n")
    print(f"{path:<26} {len(out):>5} items   e.g. {out[0]['hypothesis']}")
