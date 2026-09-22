#!/usr/bin/env python3
"""Extract matched cycle/tree activations for a persistent-homology audit.

The two conditions use the same arbitrary tokens, entity-to-node permutation,
neutral query templates, model, normalized layer, and final pre-generation
readout.  They differ only in the declared relation: a cycle or a full binary
tree.  Raw per-template activations are retained for query-aware nulls and
bootstrap uncertainty in ``ph_matched_analyze.py``.

Example:
  python -u ph_matched_extract.py google/gemma-4-31B-it --sizes 7,15
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


TOKENS = [
    "Apple", "River", "Chair", "Cloud", "Tiger", "Bridge", "Lamp",
    "Anchor", "Violin", "Pepper", "Saddle", "Mirror", "Candle",
    "Garden", "Pencil",
]
TEMPLATES = [
    "Consider the item {e}.",
    "Take note of the item {e}.",
    "The item under discussion is {e}.",
    "Focus on this item: {e}.",
    "Here is an item from the set: {e}.",
    "Item of interest: {e}.",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--sizes", default="7,15")
    p.add_argument("--nscr", type=int, default=10)
    p.add_argument("--layer-frac", type=float, default=0.75)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--outdir", default="results/geometry/ph_matched")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def relation_lines(condition, perm):
    n = len(perm)
    if condition == "cycle":
        return [f"- {perm[i]} is immediately followed by {perm[(i + 1) % n]}." for i in range(n)]
    if condition == "tree":
        return [f"- {perm[(i - 1) // 2]} is the parent of {perm[i]}." for i in range(1, n)]
    raise ValueError(condition)


def make_definition(condition, perm, seed):
    lines = relation_lines(condition, perm)
    rng = np.random.default_rng(seed)
    lines = [lines[i] for i in rng.permutation(len(lines))]
    if condition == "cycle":
        intro = (
            "You are operating under a defined cyclic order. The ONLY valid immediate-following "
            "relations are listed below; the ordinary meanings of these words do not apply."
        )
        label = "Cyclic order"
    else:
        intro = (
            "You are operating under a defined hierarchy. The ONLY valid parent-child "
            "relations are listed below; the ordinary meanings of these words do not apply."
        )
        label = "Hierarchy"
    return f"{intro}\n{label}:\n" + "\n".join(lines)


def render(tokenizer, content):
    kwargs = dict(add_generation_prompt=True, tokenize=False)
    try:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": content}], enable_thinking=False, **kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template([{"role": "user", "content": content}], **kwargs)


def load_model(model_name):
    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map={"": 0}
        ).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map={"": 0}
        ).eval()


def build_design(n, nscr, seed):
    if n not in (7, 15):
        raise ValueError("Matched audit is pre-specified for N=7 and N=15")
    entities = TOKENS[:n]
    order_rng = np.random.default_rng(seed + n)
    perms = [list(order_rng.permutation(entities)) for _ in range(nscr)]
    prompts = {}
    for si, perm in enumerate(perms):
        for ci, condition in enumerate(("cycle", "tree")):
            definition = make_definition(condition, perm, seed + 10_000 * n + 100 * si + ci)
            prompts[(si, condition)] = [
                definition + "\n\n" + template.format(e=entity)
                for entity in entities
                for template in TEMPLATES
            ]
    return entities, perms, prompts


def main():
    args = parse_args()
    sizes = [int(x) for x in args.sizes.split(",") if x]
    designs = {n: build_design(n, args.nscr, args.seed) for n in sizes}
    for n, (entities, _, prompts) in designs.items():
        lengths = [len(p) for p in prompts.values()]
        print(
            f"DESIGN N={n}: 2 conditions x {args.nscr} scrambles x "
            f"{n} entities x {len(TEMPLATES)} templates = {sum(lengths)} prompts",
            flush=True,
        )
        assert set(lengths) == {n * len(TEMPLATES)}
        assert all(set(perm) == set(entities) for perm in designs[n][1])
    if args.dry_run:
        for n, (_, perms, prompts) in designs.items():
            print(f"N={n} scramble 1 permutation: {perms[0]}")
            print("CYCLE SAMPLE:\n" + prompts[(0, "cycle")][0])
            print("TREE SAMPLE:\n" + prompts[(0, "tree")][0])
        return

    cfg = AutoConfig.from_pretrained(args.model)
    text_cfg = getattr(cfg, "text_config", None)
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(text_cfg, "num_hidden_layers", None)
    layer = int(round(args.layer_frac * n_layers))
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.model)
    tag = args.model.split("/")[-1]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"MODEL {args.model}; layer {layer}/{n_layers}; device {model.device}", flush=True)

    for n in sizes:
        entities, perms, prompt_map = designs[n]
        hidden = int(getattr(model.config, "hidden_size", 0) or getattr(model.config.text_config, "hidden_size"))
        raw = np.empty((2, args.nscr, n, len(TEMPLATES), hidden), dtype=np.float16)
        conditions = ("cycle", "tree")
        for ci, condition in enumerate(conditions):
            for si in range(args.nscr):
                plain_prompts = prompt_map[(si, condition)]
                rendered = [render(tokenizer, p) for p in plain_prompts]
                rows = []
                for bi in range(0, len(rendered), args.batch_size):
                    batch = rendered[bi : bi + args.batch_size]
                    enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
                    with torch.inference_mode():
                        output = model(**enc, output_hidden_states=True)
                    rows.append(output.hidden_states[layer][:, -1, :].float().cpu().numpy())
                    del output
                values = np.concatenate(rows, axis=0).reshape(n, len(TEMPLATES), hidden)
                raw[ci, si] = values.astype(np.float16)
                print(f"  N={n:2d} {condition:5s} scramble {si + 1:2d}/{args.nscr}", flush=True)

        centroids = raw.astype(np.float32).mean(axis=3)
        npz_path = outdir / f"phmatched_{tag}_n{n}.npz"
        np.savez_compressed(
            npz_path,
            raw=raw,
            centroids=centroids,
            conditions=np.array(conditions),
            entities=np.array(entities),
            templates=np.array(TEMPLATES),
            permutations=np.array(perms),
            layer=np.array(layer),
            n_layers=np.array(n_layers),
            layer_frac=np.array(args.layer_frac),
            seed=np.array(args.seed),
        )
        metadata = {
            "model": args.model,
            "tag": tag,
            "N": n,
            "conditions": list(conditions),
            "nscr": args.nscr,
            "n_templates": len(TEMPLATES),
            "layer": layer,
            "n_layers": n_layers,
            "layer_frac": args.layer_frac,
            "readout": "last prompt token before generation",
            "distance_planned": "mean-centered cosine RDM",
            "seed": args.seed,
            "npz": str(npz_path.resolve()),
        }
        with open(npz_path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"SAVED {npz_path.resolve()} raw={raw.shape}", flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
