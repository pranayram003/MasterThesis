"""
Does the Phase 1 vector actually carry the concept, on data it was not built on?

Held-out check, run on data/caa_val.jsonl:

    for each layer l, project both branches of every val pair onto v_l and ask
    how well that single number separates the correct branch from the shortcut
    branch (AUROC, and Cohen's d on the paired difference)

    cosine similarity between the vector built on train and the vector the val
    set would have produced, which says whether the direction is a property of
    the concept or of the training lexicon

The val split uses US place names where train uses country names, with almost no
lexical overlap, so agreement here is a real generalisation result rather than a
restatement of the training data.

    python validate_vectors.py \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --vectors vectors/mistral7b.pt \
        --pairs data/caa_val.jsonl \
        --out reports/mistral7b_val.json
"""

import argparse
import json
import os

import torch

from modeling import load_model
from prompts import letters_for, read_jsonl, render


def auroc(pos, neg):
    """Rank based AUROC, no sklearn dependency."""
    scores = torch.cat([pos, neg])
    labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    order = scores.argsort()
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float64)
    # average ranks over ties
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    n_pos = len(pos)
    n_neg = len(neg)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--pairs", default="data/caa_val.jsonl")
    ap.add_argument("--demos", default="data/eval_train.jsonl")
    ap.add_argument("--out", default="reports/val.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--no-chat-template", action="store_true")
    args = ap.parse_args()

    payload = torch.load(args.vectors, map_location="cpu")
    vectors = payload["resid_diff"]                      # [L+1, d]
    unit = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    items = read_jsonl(args.pairs)
    if args.limit:
        items = items[: args.limit]

    model, tokenizer = load_model(args.model, load_in_4bit=not args.no_4bit)

    proj_correct = []
    proj_shortcut = []
    val_sum = None
    n_used = 0

    for step, item in enumerate(items):
        prefix = render(item, args.demos, tokenizer,
                        use_chat_template=not args.no_chat_template)
        gold_letter, shortcut_letter = letters_for(item)
        texts = [prefix + gold_letter, prefix + shortcut_letter]
        enc = tokenizer(texts, return_tensors="pt", add_special_tokens=True)
        if enc["input_ids"][0].shape != enc["input_ids"][1].shape:
            continue
        enc = {k: v.to(model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        resid = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=0).float().cpu()
        proj_correct.append((resid[:, 0, :] * unit).sum(-1))
        proj_shortcut.append((resid[:, 1, :] * unit).sum(-1))
        diff = resid[:, 0, :] - resid[:, 1, :]
        val_sum = diff if val_sum is None else val_sum + diff
        n_used += 1
        if (step + 1) % 100 == 0:
            print("%d/%d" % (step + 1, len(items)), flush=True)

    proj_correct = torch.stack(proj_correct)             # [n, L+1]
    proj_shortcut = torch.stack(proj_shortcut)
    val_vectors = val_sum / n_used

    per_layer = []
    for layer in range(proj_correct.shape[1]):
        a = proj_correct[:, layer].double()
        b = proj_shortcut[:, layer].double()
        paired = a - b
        cos = torch.nn.functional.cosine_similarity(
            vectors[layer], val_vectors[layer], dim=0
        ).item()
        per_layer.append(
            {
                "layer": layer,
                "auroc": auroc(a, b),
                "cohens_d": float(paired.mean() / paired.std().clamp_min(1e-8)),
                "mean_gap": float(paired.mean()),
                "sign_agreement": float((paired > 0).double().mean()),
                "train_val_cosine": cos,
                "vector_norm": float(vectors[layer].norm()),
            }
        )

    best = max(per_layer, key=lambda r: r["auroc"])
    print("\nlayer  auroc   d      sign%   cos(train,val)")
    for r in per_layer:
        print("%5d  %.3f  %6.2f  %.3f   %+.3f"
              % (r["layer"], r["auroc"], r["cohens_d"], r["sign_agreement"],
                 r["train_val_cosine"]))
    print("\nbest layer: %d (auroc %.3f)" % (best["layer"], best["auroc"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(
            {
                "model": args.model,
                "vectors": args.vectors,
                "pairs": args.pairs,
                "n_items": n_used,
                "best_layer": best["layer"],
                "per_layer": per_layer,
            },
            fh,
            indent=2,
        )
    print("wrote", args.out)


if __name__ == "__main__":
    main()
