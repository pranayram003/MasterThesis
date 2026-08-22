"""
Rank heads within a chosen band of layers, scored against that band only.
The global ranking in rank_heads.py is dominated by layers 29-31, where the
answer is already largely decided, so mid-network heads never surface. This
restricts the comparison so a shortlist can be drawn from the layers where
steering actually works.
Correlational, like rank_heads.py. It produces candidates to test causally,
not a result.
    python rank_heads_band.py --vectors vectors/mistral_bf16.pt --layers 20 23
"""
import argparse
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--layers", type=int, nargs=2, default=[20, 23],
                    metavar=("LO", "HI"), help="inclusive band of layer indices")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    payload = torch.load(args.vectors, map_location="cpu")
    if "head_diff" not in payload:
        raise SystemExit("no head_diff in this file; re-run extract_vectors.py without --no-heads")

    head_diff = payload["head_diff"]          # [n_layers, H, d_head]
    n_layers, n_heads = head_diff.shape[0], head_diff.shape[1]
    lo, hi = args.layers
    lo, hi = max(0, lo), min(n_layers - 1, hi)
    print("model:", payload.get("model"), " items:", payload.get("n_items"))
    print("band: layers %d-%d of 0-%d, %d heads per layer\n" % (lo, hi, n_layers - 1, n_heads))

    norms = head_diff.norm(dim=-1)            # [n_layers, H]
    band = norms[lo:hi + 1]
    flat = band.flatten()
    mu, sd = flat.mean().item(), flat.std().item()

    rows = []
    for i in range(band.shape[0]):
        for h in range(n_heads):
            v = band[i, h].item()
            rows.append((lo + i, h, v, (v - mu) / sd if sd > 0 else 0.0))
    rows.sort(key=lambda r: -r[2])

    print("rank  layer  head   ||diff||   z (within band)")
    for rank, (layer, head, v, z) in enumerate(rows[: args.top], 1):
        print("%4d  %5d  %4d   %8.4f  %+.2f" % (rank, layer, head, v, z))

    print("\nper-layer totals within band")
    for i in range(band.shape[0]):
        total = band[i].sum().item()
        print("  %2d   %6.3f  %s" % (lo + i, total, "#" * int(total * 4)))

if __name__ == "__main__":
    main()
