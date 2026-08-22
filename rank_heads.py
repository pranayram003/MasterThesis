"""
Rank attention heads by how much their output differs between the two branches.

This is not a causal result and it does not replace Phase 2. It is a cheap
correlational shortlist from activations already cached in Phase 1, useful as a
cross-check: if the heads Zhou et al.'s causal method surfaces are also near the
top of this list, that is corroboration, and if they are not, that is a finding
worth reporting.

    python rank_heads.py --vectors vectors/mistral7b.pt --top 25
"""

import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    payload = torch.load(args.vectors, map_location="cpu")
    if "head_diff" not in payload:
        raise SystemExit("this vector file was built with --no-heads")

    diff = payload["head_diff"]                    # [n_layers, n_heads, d_head]
    norms = diff.norm(dim=-1)                      # [n_layers, n_heads]
    n_layers, n_heads = norms.shape

    flat = norms.flatten()
    order = flat.argsort(descending=True)[: args.top]
    print("model:", payload.get("model"), " items:", payload.get("n_items"))
    print("rank  layer  head   ||diff||   z")
    mean, std = flat.mean(), flat.std().clamp_min(1e-8)
    for rank, position in enumerate(order.tolist(), start=1):
        layer, head = divmod(position, n_heads)
        value = flat[position]
        print("%4d  %5d  %4d   %8.4f  %+.2f"
              % (rank, layer, head, value, (value - mean) / std))

    print("\nlayerwise total")
    totals = norms.sum(dim=-1)
    for layer in range(n_layers):
        bar = "#" * int(40 * totals[layer] / totals.max())
        print("%3d  %8.3f  %s" % (layer, totals[layer], bar))


if __name__ == "__main__":
    main()
