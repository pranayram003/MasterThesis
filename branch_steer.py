"""
One direction per branch of the negation.

The clustering check showed the model codes "the hypothesis place matches the
premise" perfectly, the same with or without negation, and the error is in
applying the negation to that signal. Steering with the CAA direction repairs
one branch (no match -> True) and leaves the other (match -> False) at noise.
This script builds a separate direction for each branch and steers with them.

Why not just take the CAA difference over false items only. On every false
item, correct minus shortcut is "answer False" minus "answer True", so that
direction mostly learns to say False. It would lift false items and sink true
ones, which is a label bias (the bias control already puts its ceiling at
0.647). It is kept here as false_naive, as a check on that confound, not as a
fix.

The branch directions cancel the answer instead. Each training item is run
negated and unnegated, and at every layer we take correct minus wrong:

    D(neg, match)     = h(False) - h(True)   in the negated prompt
    D(unneg, match)   = h(True)  - h(False)  in the unnegated prompt

    false_branch = D(neg, match) + D(unneg, match)
                 = [h(False) - h(True)]_neg - [h(False) - h(True)]_unneg

which is how the negation changes the False-vs-True contrast on matching
items, with the bare "say False" part cancelled. true_branch is the same on
non-matching items, and both_branch is their sum. caa_neg is the plain CAA
direction rebuilt from these prompts, as a sanity check against
vectors/mistral_bf16.pt (the cosine is printed).

Steering adds a unit direction times a fraction of the residual norm at the
layer, at every position, exactly as steer_para.py does. No gold labels are
used at test time, so every row is a deployable intervention. Each row reports
accuracy split by gold label and how many answers it fixed and broke relative
to no steering.

What to look for:

    false_branch lifts false items without breaking true ones
        the second branch has its own direction, and the asymmetry is two
        separate failures rather than one

    both_branch beats caa_neg on false items
        a single added vector can repair both branches if built this way

    false_naive lifts false and sinks true by similar amounts
        the expected label-bias confound, nothing more

    python branch_steer.py --model mistralai/Mistral-7B-Instruct-v0.3

    python branch_steer.py --skip-extract      # reuse vectors/mistral_branches.pt

Follow-up after the layer check (false_branch at layer 18, +0.2, was the first
intervention to move false items). --random N adds N random unit directions
as a null at the same layers and fractions. --combine runs extra rows with
several steerers at once, each given as direction@layer:fraction, so the two
branches can each get their own fix in the same forward pass.

    python branch_steer.py --skip-extract --layers 16 17 18 19 20 \
        --directions false_branch --fracs 0.1 0.15 0.2 0.25 0.3 --random 5 \
        --combine "false_branch@18:0.2,caa_neg@22:-0.4" "false_branch@18:0.15,caa_neg@22:-0.4"
"""

import argparse
import collections
import json
import os
import time

import torch

from modeling import letter_token_ids, load_model
from probe_negation_demomatch import DEMO_VARIANT, build, read_jsonl, render
from steer_para import evaluate
from steer_pilot import Steerer

DIRECTIONS = ["caa_neg", "false_naive", "false_branch", "true_branch", "both_branch"]
FRACS = [-0.4, -0.2, 0.2, 0.4]


def extract(model, tokenizer, items, train_items, out_path):
    DEMO_VARIANT["name"] = "orig"
    sums = {}
    counts = collections.Counter()
    skipped = 0
    start = time.time()
    for step, item in enumerate(items):
        match = item["gold_label"] == "contradiction"
        for cond in ("negated", "unnegated"):
            user_turn, order, correct = build(item, cond, train_items)
            prefix = render(user_turn, tokenizer)
            right = "AB"[list(order).index(correct)]
            wrong = "AB"[1 - list(order).index(correct)]
            enc = tokenizer([prefix + right, prefix + wrong], return_tensors="pt",
                            add_special_tokens=True)
            ids = enc["input_ids"]
            if ids[0].shape != ids[1].shape or not torch.equal(ids[0, :-1], ids[1, :-1]):
                skipped += 1
                continue
            enc = {k: v.to(model.device) for k, v in enc.items()}
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, use_cache=False)
            resid = torch.stack([h[:, -1, :] for h in out.hidden_states], 0).float().cpu()
            diff = resid[:, 0, :] - resid[:, 1, :]                 # [L+1, d]
            key = (cond, match)
            sums[key] = diff if key not in sums else sums[key] + diff
            counts[key] += 1
        if (step + 1) % 200 == 0:
            print("  extract %d/%d  %.1f items/s" % (step + 1, len(items),
                  (step + 1) / (time.time() - start)), flush=True)

    D = {k: sums[k] / counts[k] for k in sums}
    n_neg = counts[("negated", True)] + counts[("negated", False)]
    vectors = {
        "caa_neg": (sums[("negated", True)] + sums[("negated", False)]) / n_neg,
        "false_naive": D[("negated", True)],
        "false_branch": D[("negated", True)] + D[("unnegated", True)],
        "true_branch": D[("negated", False)] + D[("unnegated", False)],
    }
    vectors["both_branch"] = vectors["false_branch"] + vectors["true_branch"]
    torch.save({"directions": vectors,
                "counts": {"%s_%s" % (c, "match" if m else "differs"): n
                           for (c, m), n in counts.items()},
                "skipped": skipped}, out_path)
    print("saved %s   counts %s   skipped %d" % (out_path, dict(counts), skipped), flush=True)
    return vectors


def unit(v):
    return v / v.norm().clamp_min(1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--pairs", default="data/caa_train.jsonl")
    ap.add_argument("--train", default="data/eval_train.jsonl")
    ap.add_argument("--limit", type=int, default=None, help="training items for extraction")
    ap.add_argument("--vectors-out", default="vectors/mistral_branches.pt")
    ap.add_argument("--reference", default="vectors/mistral_bf16.pt")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--items", nargs="+", default=["data/para_orig.jsonl",
                    "data/para_never.jsonl", "data/para_cleft.jsonl"])
    ap.add_argument("--layers", type=int, nargs="*", default=[22])
    ap.add_argument("--directions", nargs="+", default=DIRECTIONS)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--fracs", type=float, nargs="+", default=FRACS)
    ap.add_argument("--random", type=int, default=0,
                    help="add this many random unit directions as a null")
    ap.add_argument("--combine", nargs="*", default=[],
                    help='e.g. "false_branch@18:0.2,caa_neg@22:-0.4"')
    ap.add_argument("--out", default="reports/branch_steer.json")
    args = ap.parse_args()

    train_items = read_jsonl(args.train)
    model, tokenizer = load_model(args.model, load_in_4bit=False)
    ids = letter_token_ids(tokenizer, letters=("A", "B"))

    if args.skip_extract:
        vectors = torch.load(args.vectors_out, map_location="cpu")["directions"]
    else:
        pairs = [i for i in read_jsonl(args.pairs) if i.get("diagnostic", True)]
        if args.limit:
            pairs = pairs[: args.limit]
        os.makedirs(os.path.dirname(args.vectors_out) or ".", exist_ok=True)
        vectors = extract(model, tokenizer, pairs, train_items, args.vectors_out)

    d_model = vectors["caa_neg"].shape[-1]
    n_layers = vectors["caa_neg"].shape[0]
    for k in range(args.random):
        g = torch.Generator().manual_seed(1000 + k)
        vectors["random_s%d" % k] = torch.randn(n_layers, d_model, generator=g)
        args.directions = list(args.directions) + ["random_s%d" % k]

    # how the directions relate, at each steering layer
    ref = None
    if os.path.exists(args.reference):
        ref = torch.load(args.reference, map_location="cpu")["resid_diff"]
    for layer in args.layers:
        print("\ncosines at layer %d" % layer)
        names = [n for n in vectors if not n.startswith("random_")]
        print("%13s " % "" + " ".join("%12s" % n for n in names))
        for a in names:
            row = [torch.nn.functional.cosine_similarity(vectors[a][layer], vectors[b][layer], 0).item()
                   for b in names]
            print("%13s " % a + " ".join("%+12.2f" % c for c in row))
        if ref is not None:
            print("  cos(caa_neg, mistral_bf16) = %+.3f   (should be high)"
                  % torch.nn.functional.cosine_similarity(vectors["caa_neg"][layer], ref[layer], 0).item())

    results = {}
    for path in args.items:
        items = [i for i in read_jsonl(path) if i.get("diagnostic", True)][args.offset: args.offset + args.n]
        DEMO_VARIANT["name"] = items[0].get("variant", "orig")
        results[path] = {}
        print("\n==== %s  n=%d ====" % (path, len(items)), flush=True)

        for layer in args.layers:
            # baseline, and the residual norm used to scale every direction
            steerer = Steerer(model, layer, unit(vectors["caa_neg"][layer]), positions="all")
            steerer.norms = []
            evaluate(model, tokenizer, items[:10], train_items, ids, steerer, 0.0)
            scale = sum(steerer.norms) / len(steerer.norms)
            _, _, base_by, base_items = evaluate(model, tokenizer, items, train_items, ids, steerer, 0.0)
            steerer.remove()
            base_ok = [r["correct"] for r in base_items]
            gold = [r["gold"] for r in base_items]
            bt = base_by["entailment"][0] / base_by["entailment"][1]
            bf = base_by["contradiction"][0] / base_by["contradiction"][1]
            print("layer %d  norm %.1f   no steering   acc %.3f   true %.3f   false %.3f"
                  % (layer, scale, sum(base_ok) / len(base_ok), bt, bf), flush=True)
            print("  %-13s %5s   acc    true   false   true fixed/broke   false fixed/broke"
                  % ("direction", "frac"))
            rows = []
            for name in args.directions:
                steerer = Steerer(model, layer, unit(vectors[name][layer]), positions="all")
                for frac in args.fracs:
                    hits, predicted, by_gold, per_item = evaluate(
                        model, tokenizer, items, train_items, ids, steerer, frac * scale)
                    ok = [r["correct"] for r in per_item]
                    tally = {}
                    for g in ("entailment", "contradiction"):
                        fixed = sum(1 for b, s, gg in zip(base_ok, ok, gold) if gg == g and not b and s)
                        broke = sum(1 for b, s, gg in zip(base_ok, ok, gold) if gg == g and b and not s)
                        tally[g] = (fixed, broke)
                    t = by_gold["entailment"][0] / by_gold["entailment"][1]
                    f = by_gold["contradiction"][0] / by_gold["contradiction"][1]
                    print("  %-13s %+5.1f   %.3f  %.3f  %.3f        %3d/%-3d            %3d/%-3d"
                          % (name, frac, hits / len(items), t, f,
                             tally["entailment"][0], tally["entailment"][1],
                             tally["contradiction"][0], tally["contradiction"][1]), flush=True)
                    rows.append({"direction": name, "fraction": frac, "accuracy": hits / len(items),
                                 "true_acc": t, "false_acc": f, "predicted": dict(predicted),
                                 "true_fixed_broke": tally["entailment"],
                                 "false_fixed_broke": tally["contradiction"],
                                 "per_item": per_item})
                steerer.remove()
            if args.random:
                summary = collections.defaultdict(list)
                for r in rows:
                    if r["direction"].startswith("random_"):
                        summary[r["fraction"]].append(r)
                for frac, rs in sorted(summary.items()):
                    fa = [r["false_acc"] for r in rs]
                    ta = [r["true_acc"] for r in rs]
                    print("  random null  %+5.2f   false %.3f-%.3f  true %.3f-%.3f  (n=%d)"
                          % (frac, min(fa), max(fa), min(ta), max(ta), len(rs)), flush=True)
            results[path][str(layer)] = {"scale": scale, "base_acc": sum(base_ok) / len(base_ok),
                                         "base_true": bt, "base_false": bf,
                                         "base_per_item": base_items, "rows": rows}

        if args.combine:
            # baseline once more without any steering, then each combination
            probe = Steerer(model, 22, unit(vectors["caa_neg"][22]))
            _, _, _, base_items = evaluate(model, tokenizer, items, train_items, ids, probe, 0.0)
            probe.remove()
            base_ok = [r["correct"] for r in base_items]
            gold = [r["gold"] for r in base_items]
            print("  combined steering")
            combo_rows = []
            for spec in args.combine:
                steerers = []
                for part in spec.split(","):
                    name, rest = part.split("@")
                    layer, frac = rest.split(":")
                    layer, frac = int(layer), float(frac)
                    st = Steerer(model, layer, unit(vectors[name][layer]), positions="all")
                    st.norms = []
                    evaluate(model, tokenizer, items[:10], train_items, ids, st, 0.0)
                    st.coeff_target = frac * sum(st.norms) / len(st.norms)
                    steerers.append(st)
                for st in steerers:
                    st.coeff = st.coeff_target
                hits = 0
                by = collections.defaultdict(lambda: [0, 0])
                ok = []
                for k, item in enumerate(items):
                    # evaluate() sets coeff on the steerer it is given; reuse the
                    # first one and leave the others at their target
                    h, _, b, per = evaluate(model, tokenizer, [item], train_items, ids,
                                            steerers[0], steerers[0].coeff_target)
                    hits += h
                    for g_, v in b.items():
                        by[g_][0] += v[0]
                        by[g_][1] += v[1]
                    ok.append(per[0]["correct"])
                tally = {}
                for g_ in ("entailment", "contradiction"):
                    tally[g_] = (sum(1 for b_, s_, gg in zip(base_ok, ok, gold) if gg == g_ and not b_ and s_),
                                 sum(1 for b_, s_, gg in zip(base_ok, ok, gold) if gg == g_ and b_ and not s_))
                t = by["entailment"][0] / by["entailment"][1]
                f = by["contradiction"][0] / by["contradiction"][1]
                print("  %-44s acc %.3f  true %.3f  false %.3f   true %d/%d  false %d/%d"
                      % (spec, hits / len(items), t, f, tally["entailment"][0], tally["entailment"][1],
                         tally["contradiction"][0], tally["contradiction"][1]), flush=True)
                combo_rows.append({"spec": spec, "accuracy": hits / len(items), "true_acc": t,
                                   "false_acc": f, "true_fixed_broke": tally["entailment"],
                                   "false_fixed_broke": tally["contradiction"]})
                for st in steerers:
                    st.remove()
            results[path]["combined"] = combo_rows

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"model": args.model, "vectors": args.vectors_out, "results": results}, fh)
    print("\nwrote", args.out)


if __name__ == "__main__":
    main()
