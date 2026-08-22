"""
Model loading and activation capture, sized for a single 8GB card.

Two kinds of activation are captured at the same token position:

    residual stream   hidden_states[l] for l in 0..L, shape [d_model]
                      l = 0 is the embedding output, l = i is the output of
                      decoder layer i. This is what CAA steering is added to.

    per-head output   the input to o_proj at layer i, reshaped to [H, d_head].
                      This is the z of each head before it is written into the
                      residual stream, and it is the object Phase 3 ablates or
                      projects a direction out of, so it is worth caching now
                      rather than running the model twice.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_id, load_in_4bit=True, device="cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0} if device == "cuda" else "cpu"}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    return model, tokenizer


def decoder_layers(model):
    return model.model.layers


def head_geometry(model):
    cfg = model.config
    n_heads = cfg.num_attention_heads
    d_head = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
    return n_heads, d_head


class HeadOutputCapture:
    """
    Forward pre-hooks on every o_proj, storing the last-position input of each
    layer as [batch, n_heads, d_head]. Call clear() before each forward.
    """

    def __init__(self, model):
        self.model = model
        self.n_heads, self.d_head = head_geometry(model)
        self.store = {}
        self.handles = []
        for i, layer in enumerate(decoder_layers(model)):
            self.handles.append(
                layer.self_attn.o_proj.register_forward_pre_hook(self._make_hook(i))
            )

    def _make_hook(self, index):
        def hook(_module, args):
            z = args[0][:, -1, :]
            self.store[index] = z.reshape(z.shape[0], self.n_heads, self.d_head).float()
        return hook

    def clear(self):
        self.store = {}

    def stacked(self):
        """[n_layers, batch, n_heads, d_head]"""
        return torch.stack([self.store[i] for i in sorted(self.store)], dim=0)

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []


def letter_token_ids(tokenizer, letters=("A", "B", "C")):
    """
    Token id of each answer letter as it appears directly after "(".
    Encoding the letter in context avoids the leading-space tokenisation trap.
    """
    ids = {}
    for letter in letters:
        with_paren = tokenizer.encode("(" + letter, add_special_tokens=False)
        only_paren = tokenizer.encode("(", add_special_tokens=False)
        tail = with_paren[len(only_paren):]
        if len(tail) != 1:
            raise ValueError(
                "letter %r does not tokenise to a single token after '(' : %r"
                % (letter, tail)
            )
        ids[letter] = tail[0]
    return ids
