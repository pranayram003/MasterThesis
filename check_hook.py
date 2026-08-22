"""
Does the steering hook actually change the forward pass?

The pilot sweep produced almost no movement even at a perturbation of 0.4 times
the residual norm, which is large enough that a random direction should visibly
damage the model. That is more consistent with the hook failing to replace the
layer output than with the vector being inert, so this checks directly.

Three forward passes on a single item at one layer:

    unsteered
    steered along the CAA direction
    steered along a random direction of the same magnitude

Reading the result:

    all three identical
        the hook is not modifying anything and the pilot measured nothing.
        The decoder layer return convention in this transformers version does
        not match what the hook assumes.

    random moves the logits, CAA does not
        the hook works and the direction really is behaviourally inert, which
        is a genuine and reportable result.

    both move the logits
        the hook works and the pilot sweep was measuring something real, so the
        flat accuracy means the effect does not favour either answer.

    python check_hook.py --model mistralai/Mistral-7B-Instruct-v0.3 \
        --vectors vectors/mistral7b.pt --layer 31
"""

import argparse

import torch

from modeling import letter_token_ids, load_model
from prompts import read_jsonl, render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--items", default="data/caa_val.jsonl")
    ap.add_argument("--demos", default="data/eval_train.jsonl")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--fraction", type=float, default=0.4)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    payload = torch.load(args.vectors, map_location="cpu")
    vector = payload["resid_diff"][args.layer]
    unit = vector / vector.norm().clamp_min(1e-8)

    torch.manual_seed(0)
    random_dir = torch.randn_like(unit)
    random_dir = random_dir / random_dir.norm()

    item = read_jsonl(args.items)[0]
    model, tokenizer = load_model(args.model, load_in_4bit=not args.no_4bit)
    ids = letter_token_ids(tokenizer, letters=("A", "B"))

    text = render(item, args.demos, tokenizer)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}

    module = model.model.layers[args.layer - 1]
    state = {"delta": None, "output_type": None, "norm": None}

    def hook(_module, _args, output):
        is_tuple = isinstance(output, tuple)
        state["output_type"] = "tuple(len=%d)" % len(output) if is_tuple else "tensor"
        hidden = output[0] if is_tuple else output
        state["norm"] = hidden[0, -1, :].float().norm().item()
        if state["delta"] is None:
            return None
        delta = state["delta"].to(hidden.dtype).to(hidden.device)
        hidden = hidden + delta
        if is_tuple:
            return (hidden,) + output[1:]
        return hidden

    handle = module.register_forward_hook(hook)

    def run(delta):
        state["delta"] = delta
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits[0, -1]
        return {ltr: round(logits[tid].item(), 4) for ltr, tid in ids.items()}

    base = run(None)
    scale = args.fraction * state["norm"]
    caa = run(scale * unit)
    rand = run(scale * random_dir)
    handle.remove()

    print("layer %d, decoder layer output is a %s" % (args.layer, state["output_type"]))
    print("residual norm at answer position: %.2f" % state["norm"])
    print("perturbation norm: %.2f  (%.2f x residual)" % (scale, args.fraction))
    print()
    print("unsteered        ", base)
    print("CAA direction    ", caa)
    print("random direction ", rand)
    print()

    caa_moved = any(abs(caa[k] - base[k]) > 1e-3 for k in base)
    rand_moved = any(abs(rand[k] - base[k]) > 1e-3 for k in base)
    print("CAA changed the logits:   ", caa_moved)
    print("random changed the logits:", rand_moved)
    if not caa_moved and not rand_moved:
        print("\nThe hook is not modifying the forward pass. The pilot sweep is void.")
    elif rand_moved and not caa_moved:
        print("\nThe hook works and the CAA direction is behaviourally inert.")
    else:
        print("\nThe hook works and the pilot sweep was measuring a real intervention.")


if __name__ == "__main__":
    main()
