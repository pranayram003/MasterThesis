"""
Thesis figures from results already on disk. No GPU needed.

    layer-sweep   change in negated-item accuracy from residual steering, by layer,
                  last-token and all-token steering drawn as separate series
    head-map      change in accuracy from mean-ablating each head, layers 17 to 22
    wording-auroc detection AUROC by layer for six phrasings, single vs mixed direction

Writes PDF (for LaTeX) and PNG (for quick viewing) into figures/, and prints the
numbers behind each figure so they can be checked against the thesis text.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

R = "reports"
OUT = "figures"
FRACS = [-0.4, -0.2, -0.1, 0.1, 0.2, 0.4]
VARIANTS = [("orig", "orig"), ("did_not", "did not"), ("never", "never"),
            ("never_been", "never been"), ("not_been", "not been"), ("cleft", "cleft")]
POS_LABEL = {"last": "last token", "all": "all tokens"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def load(name):
    with open(os.path.join(R, name + ".json")) as fh:
        return json.load(fh)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)),
                    bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("wrote %s/%s.pdf and .png" % (OUT, name))


def row_at(sweep, frac):
    for r in sweep:
        if abs(r["fraction"] - frac) < 1e-9:
            return r
    raise KeyError(frac)


def delta(sweep, frac):
    return row_at(sweep, frac)["accuracy"] - row_at(sweep, 0.0)["accuracy"]


def layer_sweep():
    series = {"last": {}, "all": {}}
    for fname in ("steer_early_bf16", "steer_band_bf16", "steer_all_sweep_bf16", "steer_pilot_bf16"):
        d = load(fname)
        for k, v in d["results"].items():
            series[d["positions"]][int(k)] = v["sweep"]

    for pos in ("last", "all"):
        print("\nLAYER SWEEP  %s  change in accuracy vs unsteered, n=150" % POS_LABEL[pos])
        print("layer   base  " + "  ".join("%+5.1f" % f for f in FRACS))
        for l in sorted(series[pos]):
            sw = series[pos][l]
            print("%5d  %.3f  " % (l, row_at(sw, 0.0)["accuracy"]) +
                  "  ".join("%+.3f" % delta(sw, f) for f in FRACS))

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    for pos, ls, mk in (("last", "-", "o"), ("all", "--", "s")):
        xs = sorted(series[pos])
        for frac, col in ((-0.4, "#1f5fa8"), (0.4, "#c0392b")):
            ax.plot(xs, [delta(series[pos][l], frac) for l in xs], color=col, ls=ls,
                    marker=mk, ms=4, lw=1.5,
                    label="%s, %+.1f of residual norm" % (POS_LABEL[pos], frac))

    print("\nRANDOM DIRECTIONS  change in accuracy at -0.4 and +0.4")
    first = True
    for layer, names in ((22, ["random_s1", "random_s2", "random_s3"]),
                         (18, ["random_L18_s1", "random_L18_s2", "random_L18_s3"])):
        for i, nm in enumerate(names):
            if not os.path.exists(os.path.join(R, nm + ".json")):
                continue
            d = load(nm)
            sw = d["results"][str(layer)]["sweep"]
            print("  %-14s layer %d  %s   %+.3f  %+.3f"
                  % (nm, layer, POS_LABEL[d["positions"]], delta(sw, -0.4), delta(sw, 0.4)))
            for frac in (-0.4, 0.4):
                ax.scatter(layer + (i - 1) * 0.18, delta(sw, frac), marker="x",
                           color="grey", s=22, zorder=3,
                           label="random direction, last token, \u00b10.4" if first else None)
                first = False

    first = True
    for i, nm in enumerate(["random_all_L22_s1", "random_all_L22_s2", "random_all_L22_s3"]):
        if not os.path.exists(os.path.join(R, nm + ".json")):
            continue
        sw = load(nm)["results"]["22"]["sweep"]
        print("  %-18s layer 22  all tokens   %+.3f  %+.3f" % (nm, delta(sw, -0.4), delta(sw, 0.4)))
        for frac in (-0.4, 0.4):
            ax.scatter(22.5 + (i - 1) * 0.18, delta(sw, frac), marker="+", color="black", s=30, zorder=3,
                       label="random direction, all tokens, \u00b10.4" if first else None)
            first = False

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Change in accuracy, negated items")
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    save(fig, "layer-sweep")


def head_map():
    d = load("ablate_mistral_L17-22")
    lo, hi = d["band"]
    nh = max(r["head"] for r in d["rows"]) + 1
    M = np.full((hi - lo + 1, nh), np.nan)
    for r in d["rows"]:
        M[r["layer"] - lo, r["head"]] = r["delta"]
    lim = np.nanmax(np.abs(M))

    top = sorted(d["rows"], key=lambda r: -r["delta"])[:6]
    print("\nHEAD MAP  n=%d  baseline %.3f  top heads by gain:" % (d["n"], d["baseline"]))
    for r in top:
        print("  L%dH%d  %+.3f   by gold %s" % (r["layer"], r["head"], r["delta"],
                                             {k: round(v, 3) for k, v in r["by_gold"].items()}))

    fig, ax = plt.subplots(figsize=(9, 2.8))
    im = ax.imshow(M, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_yticks(range(hi - lo + 1))
    ax.set_yticklabels([str(l) for l in range(lo, hi + 1)])
    ax.set_xticks(range(0, nh, 4))
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    for r in top[:4]:
        ax.add_patch(Rectangle((r["head"] - 0.5, r["layer"] - lo - 0.5), 1, 1,
                               fill=False, lw=1.4, ec="black"))
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Change in accuracy")
    for s in ("top", "right"):
        ax.spines[s].set_visible(True)
    save(fig, "head-map")


def wording():
    titles = {"bf16": "Single wording direction", "mixed": "Mixed wording direction"}
    data = {}
    best = {}
    for kind in ("bf16", "mixed"):
        data[kind] = {}
        best[kind] = {}
        for key, _ in VARIANTS:
            d = load("val_%s_%s" % (kind, key))
            pl = sorted(d["per_layer"], key=lambda e: e["layer"])
            data[kind][key] = (np.array([e["layer"] for e in pl]),
                               np.array([e["auroc"] for e in pl]))
            best[kind][key] = d["best_layer"]
        print("\n%s uses %s" % (kind, load("val_%s_orig" % kind)["vectors"]))

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, kind in zip(axes[:2], ("bf16", "mixed")):
        for key, lab in VARIANTS:
            x, y = data[kind][key]
            ax.plot(x, y, lw=1.3, label=lab)
        ax.axhline(0.5, color="grey", lw=0.8, ls="--")
        ax.set_title(titles[kind], fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("AUROC")
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")

    spreads = {}
    for kind, col in (("bf16", "#c0392b"), ("mixed", "#1f5fa8")):
        x = data[kind]["orig"][0]
        Y = np.vstack([data[kind][k][1] for k, _ in VARIANTS])
        spreads[kind] = Y.max(0) - Y.min(0)
        axes[2].plot(x, spreads[kind], color=col, lw=1.5, label=titles[kind])
    axes[2].set_title("Spread across phrasings", fontsize=10)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Max minus min AUROC")
    axes[2].legend(fontsize=8, frameon=False)

    x = data["bf16"]["orig"][0]
    print("\nWORDING  AUROC spread across six phrasings, same layer")
    print("layer  single  mixed")
    for i, l in enumerate(x):
        if l >= 12:
            print("%5d  %.3f   %.3f" % (l, spreads["bf16"][i], spreads["mixed"][i]))

    print("\nWORDING  alternative ways the 0.124 / 0.072 could have been computed")
    for kind in ("bf16", "mixed"):
        peak = {k: data[kind][k][1].max() for k, _ in VARIANTS}
        at_best = {k: data[kind][k][1][list(data[kind][k][0]).index(best[kind][k])]
                   for k, _ in VARIANTS}
        late = {k: data[kind][k][1][data[kind][k][0] >= 17].mean() for k, _ in VARIANTS}
        print("  %-6s best_layer per phrasing %s" % (kind, best[kind]))
        print("         spread of each phrasing's peak AUROC      %.3f" % (max(peak.values()) - min(peak.values())))
        print("         spread at each file's best_layer          %.3f" % (max(at_best.values()) - min(at_best.values())))
        print("         spread of mean AUROC over layers 17 to 32 %.3f" % (max(late.values()) - min(late.values())))
    save(fig, "wording-auroc")


if __name__ == "__main__":
    layer_sweep()
    head_map()
    wording()
