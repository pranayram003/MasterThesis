# Phase 1, concept isolation

Builds the CAA vector for correct negation reasoning on the CURRICULUM negation
fragment, and checks on held-out data that the vector carries the concept rather
than the training lexicon.

## What the data is

Both files are the same template.

```
premise:    "<Name> has only visited <Place>, <Name> has only visited <Place>, ..."
hypothesis: "<Name> didn't visit <Place>"
```

The gold label follows from the surface form with no ambiguity, and the rule
below reproduces all 4000 released labels exactly.

| case | gold | label if the negation is ignored |
| --- | --- | --- |
| name absent from premise | neutral | neutral |
| name present, place matches | contradiction | entailment |
| name present, place differs | entailment | contradiction |

Two consequences shape everything else here.

**The shortcut label is computable.** On every entailment or contradiction item,
the reading that ignores the negation gives exactly the inverted label. That is
what makes a clean contrast pair possible without first running the model to see
where it fails. 1992 of 3000 train items and 662 of 1000 val items are
diagnostic in this sense, and they are balanced across the two directions.

**Neutral items are negation-invariant.** Dropping the negation does not change
their label, so they cannot form a contrast pair. They are kept out of vector
construction and used as a within-dataset control in evaluation: an intervention
that is really acting on negation should leave them alone.

One more property worth knowing. Train uses country names, val uses US place
names, and the two vocabularies barely overlap (5 shared places out of 196 and
1680, 9 shared names). The val split is therefore a lexical shift, not a
resample, which makes agreement across it a real generalisation result.

## How the contrast is defined

Standard CAA. One prompt, two continuations that differ in a single token.

```
Premise: Sharon has only visited San Pedro, Dana has only visited Bristol, ...
Hypothesis: Dana didn't visit Bristol
Question: Given the premise, is the hypothesis true, false, or undetermined?

Choices:
(A) True
(B) Undetermined
(C) False

Answer: (
```

The positive branch appends the gold letter, the negative branch appends the
shortcut letter. Activations are read at that letter position, and

```
v_l = mean over items of ( h_l(correct branch) - h_l(shortcut branch) )
```

The option order is shuffled per item with a fixed seed, so the "which letter is
this" component of the difference averages out and what survives is the
direction that separates committing to the correct reading of the negation from
committing to the reading that ignores it.

Note that the proposal describes Phase 1 as contrasting prompts the model gets
right against prompts where it falls for the shortcut. That is a different
construction: the two sets would be different items, so the resulting direction
would also pick up whatever makes an item easy or hard. The construction above
holds the item fixed and varies only the answer being committed to, which is
what Rimsky et al. do and what makes the direction interpretable. The behavioural
split is still worth running as a robustness variant once the baseline is in, and
`baseline_eval.py` writes the per-item predictions needed to build it.

The same three-option format is used for vector building and for evaluation, so
the vector is never asked to transfer across prompt formats.

## Files

| file | what it does |
| --- | --- |
| `data.py` | parsing, label derivation, prompt construction |
| `build_pairs.py` | writes the contrast pair and evaluation files |
| `modeling.py` | 4-bit loading, per-head output capture, letter token ids |
| `extract_vectors.py` | the Phase 1 vector itself |
| `validate_vectors.py` | held-out separation and train/val direction agreement |
| `baseline_eval.py` | unsteered accuracy and shortcut rate |
| `rank_heads.py` | correlational head shortlist, a cross-check for Phase 2 |

## Running it

```bash
pip install torch transformers accelerate bitsandbytes

python build_pairs.py --train train.jsonl --val val.jsonl --out data/

python baseline_eval.py --model mistralai/Mistral-7B-Instruct-v0.3 \
    --items data/eval_val.jsonl --out reports/mistral_baseline_val.json

python extract_vectors.py --model mistralai/Mistral-7B-Instruct-v0.3 \
    --pairs data/caa_train.jsonl --out vectors/mistral7b.pt

python validate_vectors.py --model mistralai/Mistral-7B-Instruct-v0.3 \
    --vectors vectors/mistral7b.pt --pairs data/caa_val.jsonl \
    --out reports/mistral_val.json

python rank_heads.py --vectors vectors/mistral7b.pt --top 25
```

Then repeat with `meta-llama/Llama-3.1-8B-Instruct`.

Run the baseline first. If the shortcut rate is not clearly above chance, there
is no failure for Phase 3 to repair and the experiment design needs revisiting
before any more compute goes into it.

Use `--limit 200` on the first pass of each script to confirm it runs end to end
before committing to the full set.

## Notes on the 8GB card

Batch size is two, which is one contrast pair, and running means are accumulated
rather than stored, so memory does not grow with the dataset. NF4 with double
quantisation and bfloat16 compute puts a 7B model around 4.5GB and leaves room
for activations at these sequence lengths. The full train pass is roughly 2000
forward passes of batch 2.

Per-head outputs are captured in the same pass as the residual stream and saved
alongside the vectors. Phase 3 needs head-level objects for directional ablation
and targeted steering, and capturing them now avoids a second full pass later.

## What to read off the output

`extract_vectors.py` prints the vector norm by layer. A shallow rise then a
plateau in the late layers is the expected shape.

`validate_vectors.py` is the result that matters. Per layer it reports AUROC for
separating the two branches by projection onto the vector, Cohen's d on the
paired difference, the fraction of pairs with the correct sign, and the cosine
between the train vector and the vector val would have produced. A late layer
with AUROC near 1, sign agreement near 1, and a high train/val cosine is the
Phase 1 result: the direction exists, it is linear, and it survives a change of
vocabulary. If the best layer is early or middle rather than late, that is
interesting and belongs in the writeup, since Zhou et al. place the shortcut
late.

`rank_heads.py` is correlational only. Treat it as a hypothesis to check against
the causal method in Phase 2, not as a substitute for it.

## Open decisions

- Whether the final Phase 3 numbers are reported on this val set or on a held
  back section of the CURRICULUM benchmark. Choosing the layer and the steering
  multiplier on val means val stops being a clean test set, so a third split is
  needed for the headline result.
- Whether to build one vector per model or a shared one. Different tokenisers and
  hidden sizes mean separate vectors, but the direction can still be compared
  across models after normalisation.
- Whether the conditional section gets the same treatment, since its shortcut
  label is not derivable from the surface form the way this one is.
