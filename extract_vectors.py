"""
Phase 1, concept isolation.

For every diagnostic item the same prompt is run twice, once with the correct
answer letter appended and once with the shortcut answer letter appended. The
two forward passes differ in exactly one token. Activations are read at that
token, and the CAA vector at each layer is

    v_l = mean_over_items( h_l(correct) - h_l(shortcut) )

Because the option order is counterbalanced across the dataset, the "which
letter is this" component of the difference averages out and what remains is the
direction that separates committing to the correct reading of the negation from
committing to the reading that ignores it.

    python extract_vectors.py \
        --model mistralai/Mistral-7B-Instruct-v0.3 \
        --pairs data/caa_train.jsonl \
        --out vectors/mistral7b.pt

Running means are accumulated, so memory does not grow with dataset size.
"""

import argparse
import json
import time

import torch

from modeling import HeadOutputCapture, load_model
from prompts import letters_for, read_jsonl, render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pairs", default="data/caa_train.jsonl")
    ap.add_argument("--demos", default="data/eval_train.jsonl",
                    help="source of the few-shot demonstrations")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--no-chat-template", action="store_true")
    ap.add_argument("--no-heads", action="store_true",
                    help="skip per-head capture if you only need residual vectors")
    args = ap.parse_args()

    items = read_jsonl(args.pairs)
    if args.limit:
        items = items[: args.limit]

    model, tokenizer = load_model(args.model, load_in_4bit=not args.no_4bit)
    capture = None if args.no_heads else HeadOutputCapture(model)

    resid_sum = None
    head_sum = None
    n_used = 0
    skipped = []
    start = time.time()

    for step, item in enumerate(items):
        prefix = render(item, args.demos, tokenizer,
                        use_chat_template=not args.no_chat_template)
        gold_letter, shortcut_letter = letters_for(item)
        texts = [prefix + gold_letter, prefix + shortcut_letter]
        enc = tokenizer(texts, return_tensors="pt", add_special_tokens=True)
        if enc["input_ids"].shape[1] == 0 or (
            enc["input_ids"][0].shape != enc["input_ids"][1].shape
        ):
            skipped.append(item["idx"])
            continue
        # the two branches must differ only in the final token
        if not torch.equal(enc["input_ids"][0, :-1], enc["input_ids"][1, :-1]):
            skipped.append(item["idx"])
            continue

        enc = {k: v.to(model.device) for k, v in enc.items()}
        if capture is not None:
            capture.clear()
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)

        # [n_layers+1, 2, d_model] at the answer-letter position,
        # row 0 is the correct branch and row 1 the shortcut branch
        resid = torch.stack([h[:, -1, :] for h in out.hidden_states], dim=0).float().cpu()
        if resid_sum is None:
            resid_sum = torch.zeros_like(resid)
        resid_sum += resid

        if capture is not None:
            heads = capture.stacked().cpu()  # [n_layers, 2, H, d_head]
            if head_sum is None:
                head_sum = torch.zeros_like(heads)
            head_sum += heads

        n_used += 1
        del out, resid
        if (step + 1) % 100 == 0:
            rate = (step + 1) / (time.time() - start)
            print("%d/%d  %.1f items/s  used=%d skipped=%d"
                  % (step + 1, len(items), rate, n_used, len(skipped)), flush=True)

    if capture is not None:
        capture.remove()
    if n_used == 0:
        raise SystemExit("no usable items")

    resid_mean = resid_sum / n_used                 # [L+1, 2, d]
    payload = {
        "model": args.model,
        "pairs_file": args.pairs,
        "n_items": n_used,
        "skipped_idx": skipped,
        "chat_template": not args.no_chat_template,
        "four_bit": not args.no_4bit,
        "resid_correct": resid_mean[:, 0, :],
        "resid_shortcut": resid_mean[:, 1, :],
        "resid_diff": resid_mean[:, 0, :] - resid_mean[:, 1, :],
    }
    if head_sum is not None:
        head_mean = head_sum / n_used               # [L, 2, H, d_head]
        payload["head_correct"] = head_mean[:, 0]
        payload["head_shortcut"] = head_mean[:, 1]
        payload["head_diff"] = head_mean[:, 0] - head_mean[:, 1]

    torch.save(payload, args.out)
    print("\nsaved", args.out)
    print("items used:", n_used, "skipped:", len(skipped))
    norms = payload["resid_diff"].norm(dim=-1)
    for layer, value in enumerate(norms.tolist()):
        print("layer %2d  ||v|| = %.3f" % (layer, value))
    print(json.dumps({"model": args.model, "n_items": n_used}))


if __name__ == "__main__":
    main()
