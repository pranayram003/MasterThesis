"""
Are the false items clustered together inside the model? (Gijs, 2 Oct)

Almost every intervention so far helps the true (entailment) items and leaves
the false (contradiction) items close to where they were. Gijs's suggestion is
that the false items may sit together in one tight region of activation space,
so a single added direction moves them as a block without carrying them across
the decision line. This script checks that directly, on the same prompts the
steering runs use.

Stage 1 (GPU). For each wording, run every item twice, unsteered and steered
at the best setting from the sweeps (layer 22, all tokens, -0.4 of the residual
norm), and keep the residual at the answer position ("Answer: (") for a set of
layers, plus the logit margin and the model's answer.

Stage 2 (CPU, numpy only). Per wording, layer and condition:

    spread        how far items of each label sit from their own centre,
                  relative to the spread of all items. Below 1 means tighter
                  than average. "Clustered false items" predicts the false
                  spread is clearly smaller than the true spread.
    pair cos      mean cosine between items of the same label (after removing
                  the overall mean). Higher means more alike.
    sep           distance between the two label centres divided by the
                  typical within-label spread. Near 0 means the labels overlap.
    probe         cross-validated accuracy of a linear probe for the gold
                  label. Tells us whether the model internally knows true from
                  false at all, even when it answers wrongly.
    caa auroc     how well the projection on the CAA direction separates the
                  gold labels, and, inside the false items, the right answers
                  from the wrong ones.
    wrong->other  for items the model gets wrong, the share that sit closer to
                  the other label's centre than to their own. High means the
                  wrong items are represented like the opposite label.
    kmeans ARI    split the items into two clusters with no labels and ask
                  what the split lines up with: gold label, right/wrong, the
                  model's answer, which letter is correct (A or B), or premise
                  length. 1 = perfect match, 0 = no relation. If the letter or
                  the premise length wins, the main structure is not about
                  true versus false at all.
    shift         (steered vs unsteered, layers at or after the steering
                  layer) how far steering moves each label along the line
                  from the false centre to the true centre. If it moves false
                  items as much as true items but they still fail, they start
                  too far away; if it barely moves them, they are being pushed
                  somewhere else.

Stage 2 also draws figures/cluster-pca.{pdf,png}: the items in the first two
principal components of the unsteered run, for each wording, with the steered
run projected onto the same axes so the movement is visible.

    python cluster_check.py --model mistralai/Mistral-7B-Instruct-v0.3 \
        --vectors vectors/mistral_bf16.pt --n 200

    python cluster_check.py --analyse-only      # rerun stage 2 from acts/

Unnegated follow-up (--condition unnegated). On negated items "the place
matches the premise" and "the answer is false" always go together, so a clean
true/false split could just be the model coding the match. Removing the
negation flips that: matching items become true. Stage 2 takes the line between
the two clusters from the negated run and scores the unnegated items on it.

    match AUROC near 1   unnegated items land on the same side as before:
                         the clusters code "place matches", not the answer
    match AUROC near 0   they swap sides: the clusters code the answer, so
                         the negation has already been applied by that layer
    near 0.5             the layout is reorganised

The same run also shows whether steering follows the gold label or the match,
from the margin shift per group printed in stage 1.

    python cluster_check.py --condition unnegated --items data/para_orig.jsonl
"""

import argparse
import json
import os

import numpy as np

ACTS_DIR = "acts"
DEFAULT_ITEMS = ["data/para_orig.jsonl", "data/para_never.jsonl",
                 "data/para_cleft.jsonl"]
DEFAULT_CAPTURE = [14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
DEFAULT_ANALYSE = [18, 22, 26, 32]
COND = ("base", "steered")
COLOURS = {"entailment": "#2a78d6", "contradiction": "#eb6834"}
NAMES = {"entailment": "true", "contradiction": "false"}


# --------------------------------------------------------------------------
# Stage 1: activations
# --------------------------------------------------------------------------

class ResidCapture:
    """
    Keeps the last-position residual after decoder layer i-1 for every
    requested hidden-state index i (same indexing as validate_vectors and
    steer_pilot). Index 32 here is the output of the last decoder layer before
    the final norm. Must be created after the Steerer so that on the steering
    layer it sees the steered output.
    """

    def __init__(self, model, layers):
        self.store = {}
        self.handles = []
        for index in layers:
            module = model.model.layers[index - 1]
            self.handles.append(module.register_forward_hook(self._make(index)))

    def _make(self, index):
        def hook(_module, _args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.store[index] = hidden[0, -1, :].float().cpu().numpy()
        return hook

    def remove(self):
        for handle in self.handles:
            handle.remove()


def n_clauses(premise):
    return premise.count(", ") + 1


def target_position(item):
    """Position of the hypothesis person among the premise clauses, from 0."""
    names = [c.split(" has only visited ")[0] for c in item["premise"].split(", ")]
    for i, name in enumerate(names):
        if item["hypothesis"].startswith(name + " ") or (" " + name + " ") in item["hypothesis"]:
            return i
    return -1


def collect(args):
    import torch

    from modeling import letter_token_ids, load_model
    from probe_negation_demomatch import DEMO_VARIANT, build, read_jsonl, render
    from steer_pilot import Steerer

    payload = torch.load(args.vectors, map_location="cpu")
    vectors = payload["resid_diff"]
    train_items = read_jsonl(args.train)

    model, tokenizer = load_model(args.model, load_in_4bit=args.quantize)
    ids = letter_token_ids(tokenizer, letters=("A", "B"))

    unit_steer = vectors[args.steer_layer] / vectors[args.steer_layer].norm().clamp_min(1e-8)
    steerer = Steerer(model, args.steer_layer, unit_steer, positions="all")
    capture = ResidCapture(model, args.capture)

    def run(item):
        user_turn, order, correct = build(item, args.condition, train_items)
        enc = tokenizer(render(user_turn, tokenizer), return_tensors="pt",
                        add_special_tokens=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        capture.store = {}
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits[0, -1]
        scores = {ltr: logits[tid].item() for ltr, tid in ids.items()}
        letter = max(scores, key=scores.get)
        ent = "AB"[list(order).index("entailment")]
        con = "AB"[list(order).index("contradiction")]
        acts = np.stack([capture.store[l] for l in args.capture])
        return acts, order["AB".index(letter)], correct, scores[ent] - scores[con], \
            "AB"[list(order).index(correct)]

    os.makedirs(ACTS_DIR, exist_ok=True)
    for path in args.items:
        items = [i for i in read_jsonl(path) if i.get("diagnostic", True)][: args.n]
        variant = items[0].get("variant", "orig")
        DEMO_VARIANT["name"] = variant

        steerer.coeff = 0.0
        steerer.norms = []
        for item in items[:10]:
            run(item)
        scale = float(np.mean(steerer.norms))
        coeff = args.steer_frac * scale
        print("\n%s  n=%d  variant=%s  layer %d norm %.1f  steer coeff %+.1f"
              % (path, len(items), variant, args.steer_layer, scale, coeff), flush=True)

        out = {"layers": np.array(args.capture),
               # the label the model should give in this condition
               "gold": np.array([build(i, args.condition, train_items)[2] for i in items]),
               # does the hypothesis place match the person's premise place
               "place_match": np.array([i["gold_label"] == "contradiction" for i in items]),
               "n_clauses": np.array([n_clauses(i["premise"]) for i in items]),
               "target_pos": np.array([target_position(i) for i in items]),
               "direction": np.stack([
                   (vectors[l] / vectors[l].norm().clamp_min(1e-8)).float().numpy()
                   for l in args.capture])}
        for cond in COND:
            steerer.coeff = 0.0 if cond == "base" else coeff
            acts, pred, margin, gold_letter = [], [], [], []
            for k, item in enumerate(items):
                a, p, _correct, m, gl = run(item)
                acts.append(a.astype(np.float16))
                pred.append(p)
                margin.append(m)
                gold_letter.append(gl)
                if (k + 1) % 50 == 0:
                    print("  %s %d/%d" % (cond, k + 1, len(items)), flush=True)
            out[cond + "_acts"] = np.stack(acts)                 # [n, L, d]
            out[cond + "_pred"] = np.array(pred)
            out[cond + "_margin"] = np.array(margin)
            out["gold_letter"] = np.array(gold_letter)
            acc = float(np.mean(out[cond + "_pred"] == out["gold"]))
            by = {g: float(np.mean(out[cond + "_pred"][out["gold"] == g] == g))
                  for g in ("entailment", "contradiction")}
            print("  %-8s acc %.3f   true %.3f   false %.3f"
                  % (cond, acc, by["entailment"], by["contradiction"]), flush=True)
        pm = out["place_match"]
        d_margin = out["steered_margin"] - out["base_margin"]
        print("  steering shift of the True-minus-False logit:  place matches %+.2f   "
              "place differs %+.2f" % (d_margin[pm].mean(), d_margin[~pm].mean()), flush=True)
        out["meta"] = np.array(json.dumps({
            "path": path, "variant": variant, "model": args.model,
            "vectors": args.vectors, "steer_layer": args.steer_layer,
            "condition": args.condition,
            "steer_frac": args.steer_frac, "scale": scale}))
        suffix = "" if args.condition == "negated" else "_unneg"
        dest = os.path.join(ACTS_DIR, "cluster_%s%s.npz" % (variant, suffix))
        np.savez_compressed(dest, **out)
        print("  saved", dest, flush=True)

    capture.remove()
    steerer.remove()


# --------------------------------------------------------------------------
# Stage 2: analysis, numpy only
# --------------------------------------------------------------------------

def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):
        tie = allv == v
        if tie.sum() > 1:
            ranks[tie] = ranks[tie].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def ari(a, b):
    """Adjusted Rand index between two labelings."""
    a = np.unique(np.asarray(a), return_inverse=True)[1]
    b = np.unique(np.asarray(b), return_inverse=True)[1]
    table = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(table, (a, b), 1)
    comb = lambda x: x * (x - 1) / 2
    n = len(a)
    s_ij = comb(table).sum()
    s_a = comb(table.sum(1)).sum()
    s_b = comb(table.sum(0)).sum()
    expected = s_a * s_b / comb(n)
    top = 0.5 * (s_a + s_b)
    return float((s_ij - expected) / (top - expected)) if top != expected else 0.0


def kmeans2(Z, seed=0, restarts=10, iters=100):
    rng = np.random.default_rng(seed)
    best, best_inertia = None, np.inf
    for _ in range(restarts):
        centres = Z[rng.choice(len(Z), 2, replace=False)]
        for _ in range(iters):
            assign = ((Z[:, None, :] - centres[None]) ** 2).sum(-1).argmin(1)
            new = np.stack([Z[assign == k].mean(0) if (assign == k).any() else centres[k]
                            for k in range(2)])
            if np.allclose(new, centres):
                break
            centres = new
        inertia = ((Z - centres[assign]) ** 2).sum()
        if inertia < best_inertia:
            best, best_inertia = assign, inertia
    return best


def probe_cv(Z, y, folds=5, seed=0, l2=1e-2, steps=400, lr=0.1):
    """Logistic regression on standardised PCA features, k-fold accuracy."""
    idx = np.random.default_rng(seed).permutation(len(y))
    hits = 0
    for f in range(folds):
        test = idx[f::folds]
        train = np.setdiff1d(idx, test)
        mu, sd = Z[train].mean(0), Z[train].std(0) + 1e-6
        Xtr, Xte = (Z[train] - mu) / sd, (Z[test] - mu) / sd
        w, b = np.zeros(Z.shape[1]), 0.0
        for _ in range(steps):
            p = 1 / (1 + np.exp(-(Xtr @ w + b)))
            g = p - y[train]
            w -= lr * (Xtr.T @ g / len(g) + l2 * w)
            b -= lr * g.mean()
        hits += ((Xte @ w + b > 0) == y[test]).sum()
    return hits / len(y)


def pca(X, k):
    centred = X - X.mean(0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:k].T, vt[:k], X.mean(0), s ** 2 / (s ** 2).sum()


def geometry(X, gold, pred, gold_letter, n_cl, direction):
    """All stage 2 numbers for one activation matrix [n, d]."""
    X = X.astype(np.float64)
    is_ent = gold == "entailment"
    correct = pred == gold
    centred = X - X.mean(0)
    unit = centred / np.linalg.norm(centred, axis=1, keepdims=True).clip(1e-8)
    overall = np.linalg.norm(centred, axis=1).mean()

    res = {}
    centres = {}
    for g, mask in (("entailment", is_ent), ("contradiction", ~is_ent)):
        Xg = X[mask]
        centres[g] = Xg.mean(0)
        spread = np.linalg.norm(Xg - centres[g], axis=1).mean()
        U = unit[mask]
        cos = U @ U.T
        m = len(U)
        res[g] = {"n": int(m),
                  "spread_rel": float(spread / overall),
                  "pair_cos": float((cos.sum() - m) / (m * (m - 1))),
                  "accuracy": float(correct[mask].mean())}
    within = 0.5 * (np.linalg.norm(X[is_ent] - centres["entailment"], axis=1).mean()
                    + np.linalg.norm(X[~is_ent] - centres["contradiction"], axis=1).mean())
    res["sep"] = float(np.linalg.norm(centres["entailment"] - centres["contradiction"]) / within)

    Z, _, _, var = pca(X, min(50, len(X) - 1))
    res["pca_var_top2"] = [float(v) for v in var[:2]]
    res["probe_acc"] = float(probe_cv(Z, is_ent.astype(float)))

    proj = X @ direction
    res["caa_auroc_gold"] = auroc(proj[is_ent], proj[~is_ent])
    con = ~is_ent
    res["caa_auroc_false_right_vs_wrong"] = auroc(proj[con & correct], proj[con & ~correct])
    res["caa_auroc_true_right_vs_wrong"] = auroc(proj[is_ent & correct], proj[is_ent & ~correct])

    # wrong items: closer to the other label's centre? leave-one-out centres
    placement = {}
    for g, mask in (("entailment", is_ent), ("contradiction", ~is_ent)):
        other = ~mask
        c_other = X[other].mean(0)
        for kind, sel in (("wrong", mask & ~correct), ("right", mask & correct)):
            rows = np.where(sel)[0]
            if len(rows) == 0:
                placement["%s_%s" % (g, kind)] = float("nan")
                continue
            closer = 0
            for r in rows:
                own = (X[mask].sum(0) - X[r]) / (mask.sum() - 1)
                closer += np.linalg.norm(X[r] - c_other) < np.linalg.norm(X[r] - own)
            placement["%s_%s" % (g, kind)] = float(closer / len(rows))
    res["closer_to_other"] = placement

    clusters = kmeans2(Z[:, :min(50, Z.shape[1])])
    long_premise = n_cl > np.median(n_cl)
    res["kmeans_ari"] = {"gold": ari(clusters, gold), "right_wrong": ari(clusters, correct),
                         "answer": ari(clusters, pred), "correct_letter": ari(clusters, gold_letter),
                         "long_premise": ari(clusters, long_premise)}
    return res, centres


def analyse(args):
    every = sorted(f for f in os.listdir(ACTS_DIR) if f.startswith("cluster_") and f.endswith(".npz"))
    files = [f for f in every if not f.endswith("_unneg.npz")]
    unneg = [f for f in every if f.endswith("_unneg.npz")]
    if not files:
        raise SystemExit("no activations in %s/, run stage 1 first" % ACTS_DIR)
    report = {}
    loaded = {}
    for fname in files:
        d = np.load(os.path.join(ACTS_DIR, fname))
        meta = json.loads(str(d["meta"]))
        variant = meta["variant"]
        loaded[variant] = d
        layers = list(d["layers"])
        gold, n_cl, gl = d["gold"], d["n_clauses"], d["gold_letter"]
        report[variant] = {"meta": meta, "layers": {}}
        print("\n==== %s  (n=%d, steered at L%d x %+.2f) ===="
              % (variant, len(gold), meta["steer_layer"], meta["steer_frac"]))
        print("layer cond     acc  T/F       spread T/F   pairCos T/F   sep   probe  "
              "caaAUC  F r/w   wrong->other T/F   ARI gold r/w ans letter long")
        for layer in args.analyse:
            if layer not in layers:
                continue
            li = layers.index(layer)
            direction = d["direction"][li].astype(np.float64)
            block = {}
            centres = {}
            for cond in COND:
                X = d[cond + "_acts"][:, li, :]
                pred = d[cond + "_pred"]
                r, centres[cond] = geometry(X, gold, pred, gl, n_cl, direction)
                block[cond] = r
                e, c, k = r["entailment"], r["contradiction"], r["kmeans_ari"]
                print("%5d %-8s %.2f %.2f/%.2f  %.2f/%.2f    %.3f/%.3f   %.2f  %.2f   "
                      "%.2f   %.2f      %.2f/%.2f        %+.2f %+.2f %+.2f %+.2f %+.2f"
                      % (layer, cond, np.mean(pred == gold), e["accuracy"], c["accuracy"],
                         e["spread_rel"], c["spread_rel"], e["pair_cos"], c["pair_cos"],
                         r["sep"], r["probe_acc"], r["caa_auroc_gold"],
                         r["caa_auroc_false_right_vs_wrong"],
                         r["closer_to_other"]["entailment_wrong"],
                         r["closer_to_other"]["contradiction_wrong"],
                         k["gold"], k["right_wrong"], k["answer"], k["correct_letter"],
                         k["long_premise"]))
            if layer >= meta["steer_layer"]:
                axis = centres["base"]["entailment"] - centres["base"]["contradiction"]
                gap = np.linalg.norm(axis)
                axis = axis / gap
                delta = (d["steered_acts"][:, li, :].astype(np.float64)
                         - d["base_acts"][:, li, :].astype(np.float64))
                Xb = d["base_acts"][:, li, :].astype(np.float64)
                mid = 0.5 * (centres["base"]["entailment"] + centres["base"]["contradiction"])
                shift = {}
                for g in ("entailment", "contradiction"):
                    m = gold == g
                    along = delta[m] @ axis
                    start = (Xb[m] - mid) @ axis
                    shift[g] = {"mean_shift_along_axis": float(along.mean() / gap),
                                "mean_start_from_mid": float(start.mean() / gap),
                                "mean_shift_norm": float(np.linalg.norm(delta[m], axis=1).mean() / gap),
                                "share_along_axis": float(np.abs(along).mean()
                                                          / np.linalg.norm(delta[m], axis=1).mean())}
                cos_ec = float(np.dot(delta[gold == "entailment"].mean(0),
                                      delta[gold == "contradiction"].mean(0))
                               / (np.linalg.norm(delta[gold == "entailment"].mean(0))
                                  * np.linalg.norm(delta[gold == "contradiction"].mean(0)) + 1e-12))
                block["shift"] = {"by_gold": shift, "cos_mean_shift_true_vs_false": cos_ec}
                print("      shift along false->true axis (units of centre gap):  "
                      "true %+.2f (starts %+.2f)   false %+.2f (starts %+.2f)   "
                      "on-axis share T/F %.2f/%.2f   cos(shift T, shift F) %.2f"
                      % (shift["entailment"]["mean_shift_along_axis"],
                         shift["entailment"]["mean_start_from_mid"],
                         shift["contradiction"]["mean_shift_along_axis"],
                         shift["contradiction"]["mean_start_from_mid"],
                         shift["entailment"]["share_along_axis"],
                         shift["contradiction"]["share_along_axis"], cos_ec))
            report[variant]["layers"][str(layer)] = block

    for fname in unneg:
        u = np.load(os.path.join(ACTS_DIR, fname))
        variant = json.loads(str(u["meta"]))["variant"]
        if variant not in loaded:
            print("\nno negated run for %s, skipping %s" % (variant, fname))
            continue
        d = loaded[variant]
        print("\n==== %s unnegated vs negated clusters ====" % variant)
        print("unnegated accuracy: base %.3f   steered %.3f"
              % (np.mean(u["base_pred"] == u["gold"]), np.mean(u["steered_pred"] == u["gold"])))
        print("layer   match AUROC on negated axis   (1 = codes the match, 0 = codes the answer)")
        transfer = {}
        layers = list(d["layers"])
        for layer in args.analyse:
            if layer not in layers or layer not in list(u["layers"]):
                continue
            li, lu = layers.index(layer), list(u["layers"]).index(layer)
            Xn = d["base_acts"][:, li, :].astype(np.float64)
            m = d["gold"] == "contradiction"          # negated: false = place matches
            axis = Xn[m].mean(0) - Xn[~m].mean(0)
            Xu = u["base_acts"][:, lu, :].astype(np.float64)
            score = Xu @ axis
            a = auroc(score[u["place_match"]], score[~u["place_match"]])
            transfer[str(layer)] = a
            print("%5d   %.3f" % (layer, a))
        report[variant + "_unneg"] = {"match_auroc_on_negated_axis": transfer}

    os.makedirs("reports", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)
    print("\nwrote", args.out)
    figure(loaded, args.fig_layers)


def figure(loaded, fig_layers):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False})
    order = [v for v in ("orig", "did_not", "never", "never_been", "not_been", "cleft")
             if v in loaded]
    cols = [(fig_layers[0], "base"), (fig_layers[1], "base"), (fig_layers[1], "steered")]
    fig, axes = plt.subplots(len(order), 3, figsize=(9, 2.8 * len(order)), squeeze=False)
    for row, variant in enumerate(order):
        d = loaded[variant]
        layers = list(d["layers"])
        gold = d["gold"]
        for col, (layer, cond) in enumerate(cols):
            ax = axes[row, col]
            if layer not in layers:
                ax.set_visible(False)
                continue
            li = layers.index(layer)
            Xb = d["base_acts"][:, li, :].astype(np.float64)
            _, vt, mean, var = pca(Xb, 2)
            X = d[cond + "_acts"][:, li, :].astype(np.float64)
            Z = (X - mean) @ vt.T
            ok = d[cond + "_pred"] == gold
            for g in ("entailment", "contradiction"):
                m = gold == g
                ax.scatter(Z[m & ok, 0], Z[m & ok, 1], s=14, color=COLOURS[g],
                           edgecolor="white", linewidth=0.5, zorder=3)
                ax.scatter(Z[m & ~ok, 0], Z[m & ~ok, 1], s=14, facecolor="none",
                           edgecolor=COLOURS[g], linewidth=1.0, zorder=3)
            if cond == "steered":
                Zb = (Xb - mean) @ vt.T
                ax.scatter(Zb[:, 0], Zb[:, 1], s=6, color="#c3c2b7", zorder=1)
            ax.grid(color="#e8e7e2", linewidth=0.5)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7)
            ax.set_xlabel("PC1 (%.0f%%)" % (100 * var[0]), fontsize=8)
            ax.set_ylabel("PC2 (%.0f%%)" % (100 * var[1]), fontsize=8)
            title = "layer %d, %s" % (layer, "unsteered" if cond == "base" else "steered")
            acc_t = np.mean(ok[gold == "entailment"])
            acc_f = np.mean(ok[gold == "contradiction"])
            ax.set_title("%s: %s\ntrue %.2f, false %.2f correct"
                         % (variant.replace("_", " "), title, acc_t, acc_f), fontsize=8)
    handles = [
        Line2D([], [], marker="o", ls="", color=COLOURS["entailment"], label="true item, answered right"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor=COLOURS["entailment"], label="true item, answered wrong"),
        Line2D([], [], marker="o", ls="", color=COLOURS["contradiction"], label="false item, answered right"),
        Line2D([], [], marker="o", ls="", markerfacecolor="none",
               markeredgecolor=COLOURS["contradiction"], label="false item, answered wrong"),
        Line2D([], [], marker="o", ls="", color="#c3c2b7", markersize=4,
               label="same items before steering"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    os.makedirs("figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig("figures/cluster-pca.%s" % ext, dpi=200)
    print("wrote figures/cluster-pca.pdf and .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.3")
    ap.add_argument("--vectors", default="vectors/mistral_bf16.pt")
    ap.add_argument("--items", nargs="+", default=DEFAULT_ITEMS)
    ap.add_argument("--train", default="data/eval_train.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--capture", type=int, nargs="+", default=DEFAULT_CAPTURE)
    ap.add_argument("--analyse", type=int, nargs="+", default=DEFAULT_ANALYSE)
    ap.add_argument("--fig-layers", type=int, nargs=2, default=[22, 32])
    ap.add_argument("--steer-layer", type=int, default=22)
    ap.add_argument("--steer-frac", type=float, default=-0.4,
                    help="best setting from the layer 22 sweeps (sign is inverted, "
                         "see the sign inversion section)")
    ap.add_argument("--quantize", action="store_true")
    ap.add_argument("--condition", choices=["negated", "unnegated"], default="negated")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--out", default="reports/cluster_check.json")
    args = ap.parse_args()

    if not args.analyse_only:
        collect(args)
    analyse(args)


if __name__ == "__main__":
    main()
