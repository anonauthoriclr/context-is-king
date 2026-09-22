#!/usr/bin/env python3
"""Cross-order entity-slot intervention.

The established causal-use experiment patches a donor entity from the same
context into the queried entity site.  This experiment holds the patch site
fixed and crosses donor identity with donor order:

  1. same entity, same order       (sham)
  2. different entity, same order (positive intervention)
  3. same entity, different order
  4. different entity, different order

For a cross-order patch into recipient order A from donor order B, an answer of
succ_A(donor) indicates identity transfer followed by recipient-context lookup;
succ_B(donor) indicates that the transplanted activation carries the donor's
context-bound relation.  The primary analysis uses only trials for which these
answers and the unpatched source answer are all distinct.

Example:
  python -u causal_cross_order.py google/gemma-4-31B-it \
    --npairs 20 --layers 12,27,36 \
    --out results/causal/crossorder_gemma-4-31B-it_n20.json
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import t as student_t
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


ENTITIES = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]
TEMPLATE = "What is 1 step after {entity}?"
PATCH = {"map": None}


def make_definition(order):
    numbered = "\n".join(f"{i + 1}. {entity}" for i, entity in enumerate(order))
    convention = (
        "\nConvention: start=position 0; 'k steps after X' = item reached "
        "by moving forward k steps."
    )
    prefix = (
        "You are operating in a REDEFINED calendar. The ONLY valid order "
        "applies below; the normal order does NOT apply."
    )
    return f"{prefix}\nOrder:\n{numbered}{convention}"


def render(tokenizer, content):
    message = content + "\n\nAnswer with ONLY the name, nothing else."
    conversation = [{"role": "user", "content": message}]
    try:
        return tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )


def imposed_prompt(tokenizer, order, entity):
    return render(
        tokenizer,
        make_definition(order) + "\n\n" + TEMPLATE.format(entity=entity),
    )


def parse_answer(text):
    best = None
    best_position = 10**9
    for entity in ENTITIES:
        match = re.search(r"\b" + re.escape(entity) + r"\b", text, re.I)
        if match and match.start() < best_position:
            best = entity
            best_position = match.start()
    return best


def successors(order):
    return {
        entity: order[(order.index(entity) + 1) % len(order)]
        for entity in ENTITIES
    }


def successor_differences(order_a, order_b):
    succ_a = successors(order_a)
    succ_b = successors(order_b)
    return sum(succ_a[entity] != succ_b[entity] for entity in ENTITIES)


def make_order_pairs(npairs, seed, min_differences):
    rng = np.random.default_rng(seed)
    recipient_orders = [list(rng.permutation(ENTITIES)) for _ in range(npairs)]
    pairs = []
    for pair_index, order_a in enumerate(recipient_orders):
        attempts = 0
        while True:
            attempts += 1
            order_b = list(rng.permutation(ENTITIES))
            differences = successor_differences(order_a, order_b)
            if differences >= min_differences:
                break
        pairs.append({
            "pair_index": pair_index,
            "order_a": order_a,
            "order_b": order_b,
            "successor_differences": differences,
            "donor_draw_attempts": attempts,
        })
    return pairs


def find_layers(model, number_of_layers):
    candidates = [
        module
        for _, module in model.named_modules()
        if isinstance(module, nn.ModuleList) and len(module) == number_of_layers
    ]
    for module in candidates:
        if any(
            hasattr(module[0], attribute)
            for attribute in ("self_attn", "attn", "self_attention", "attention", "mlp")
        ):
            return module
    if candidates:
        return candidates[0]
    raise RuntimeError("decoder layer list not found")


def make_hook():
    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] <= 1 or PATCH["map"] is None:
            return output
        for batch_index, item in enumerate(PATCH["map"]):
            if item is None:
                continue
            positions, vector = item
            donor_length = vector.shape[0]
            hidden[batch_index, positions[-donor_length:], :] = vector.to(
                dtype=hidden.dtype,
                device=hidden.device,
            )
        return output

    return hook


def tokenize_entity_positions(tokenizer, prompts, entities, device):
    encodings = [
        tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=False)
        for prompt in prompts
    ]
    token_ids = [encoding["input_ids"] for encoding in encodings]
    lengths = [len(ids) for ids in token_ids]
    max_length = max(lengths)
    entity_positions = []
    for prompt, entity, encoding in zip(prompts, entities, encodings):
        char_start = prompt.rfind(entity)
        if char_start < 0:
            raise ValueError(f"query entity {entity!r} not found in prompt")
        char_end = char_start + len(entity)
        positions = [
            index
            for index, (start, end) in enumerate(encoding["offset_mapping"])
            if end > start and start < char_end and end > char_start
        ]
        if not positions:
            raise ValueError(f"no token positions found for query entity {entity!r}")
        entity_positions.append(positions)

    padded_ids = []
    attention = []
    shifted_positions = []
    for ids, length, positions in zip(token_ids, lengths, entity_positions):
        shift = max_length - length
        padded_ids.append([tokenizer.pad_token_id] * shift + ids)
        attention.append([0] * shift + [1] * length)
        shifted_positions.append([position + shift for position in positions])
    return (
        torch.tensor(padded_ids, device=device),
        torch.tensor(attention, device=device),
        shifted_positions,
    )


def mean(values):
    return float(sum(values) / len(values)) if values else None


def interval(values):
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"n": 0, "mean": None, "ci95": None}
    center = mean(clean)
    if len(clean) == 1:
        return {"n": 1, "mean": center, "ci95": None}
    standard_error = float(np.std(clean, ddof=1) / math.sqrt(len(clean)))
    half_width = float(student_t.ppf(0.975, len(clean) - 1) * standard_error)
    return {
        "n": len(clean),
        "mean": center,
        "ci95": [center - half_width, center + half_width],
    }


def rate(records, flag):
    return mean([1.0 if record[flag] else 0.0 for record in records])


def summarize(results):
    pair_summaries = []
    for pair in results["pairs"]:
        for layer in results["layers"]:
            layer_records = [
                record
                for direction in pair["directions"]
                for record in direction["layers"][str(layer)]
            ]
            sham = [
                record for record in layer_records
                if record["condition"] == "same_entity_same_order" and record["clean"]
            ]
            positive = [
                record for record in layer_records
                if record["condition"] == "different_entity_same_order" and record["clean"]
            ]
            same_cross = [
                record for record in layer_records
                if record["condition"] == "same_entity_cross_order"
                and record["clean"] and record["order_diagnostic"]
            ]
            primary = [
                record for record in layer_records
                if record["condition"] == "different_entity_cross_order"
                and record["clean"] and record["fully_discriminating"]
            ]
            donor_rate = rate(primary, "hit_donor_map_donor")
            recipient_rate = rate(primary, "hit_recipient_map_donor")
            pair_summaries.append({
                "pair_index": pair["pair_index"],
                "layer": layer,
                "n_sham": len(sham),
                "sham_success": rate(sham, "hit_recipient_map_source"),
                "n_positive": len(positive),
                "positive_success": rate(positive, "hit_recipient_map_donor"),
                "n_same_entity_cross_order": len(same_cross),
                "same_entity_donor_map_rate": rate(same_cross, "hit_donor_map_donor"),
                "same_entity_recipient_map_rate": rate(same_cross, "hit_recipient_map_donor"),
                "n_primary": len(primary),
                "donor_map_donor_rate": donor_rate,
                "recipient_map_donor_rate": recipient_rate,
                "stuck_source_rate": rate(primary, "hit_recipient_map_source"),
                "other_rate": mean([
                    1.0
                    if not (
                        record["hit_donor_map_donor"]
                        or record["hit_recipient_map_donor"]
                        or record["hit_recipient_map_source"]
                    )
                    else 0.0
                    for record in primary
                ]),
                "binding_index": (
                    donor_rate - recipient_rate
                    if donor_rate is not None and recipient_rate is not None
                    else None
                ),
            })

    aggregate = {}
    metric_names = [
        "sham_success",
        "positive_success",
        "same_entity_donor_map_rate",
        "same_entity_recipient_map_rate",
        "donor_map_donor_rate",
        "recipient_map_donor_rate",
        "stuck_source_rate",
        "other_rate",
        "binding_index",
    ]
    for layer in results["layers"]:
        rows = [row for row in pair_summaries if row["layer"] == layer]
        aggregate[str(layer)] = {
            metric: interval([row[metric] for row in rows])
            for metric in metric_names
        }
        aggregate[str(layer)]["total_primary_trials"] = sum(
            row["n_primary"] for row in rows
        )
    return {"per_pair": pair_summaries, "aggregate_by_layer": aggregate}


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--npairs", type=int, default=20)
    parser.add_argument("--layers", default="12,27,36")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-differences", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=14)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    selected_layers = sorted({int(layer) for layer in args.layers.split(",")})
    tag = args.model.split("/")[-1]
    output_path = args.out or Path(
        f"results/causal/crossorder_{tag}_n{args.npairs}.json"
    )
    partial_path = output_path.with_name(output_path.stem + "_partial.json")
    order_pairs = make_order_pairs(args.npairs, args.seed, args.min_differences)

    config = AutoConfig.from_pretrained(args.model)
    number_of_layers = getattr(config, "num_hidden_layers", None) or getattr(
        getattr(config, "text_config", None), "num_hidden_layers", None
    )
    if number_of_layers is None:
        raise ValueError("could not determine model depth")
    invalid_layers = [
        layer for layer in selected_layers
        if layer < 0 or layer >= number_of_layers
    ]
    if invalid_layers:
        raise ValueError(f"invalid layers for depth {number_of_layers}: {invalid_layers}")

    print(
        f"CROSS-ORDER model={args.model} pairs={args.npairs} "
        f"layers={selected_layers} min_differences={args.min_differences}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map={"": 0},
        ).eval()
    except Exception:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            dtype=torch.bfloat16,
            device_map={"": 0},
        ).eval()
    decoder_layers = find_layers(model, number_of_layers)
    device = model.device

    def generate(input_ids, attention, patch_map):
        PATCH["map"] = patch_map
        try:
            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        finally:
            PATCH["map"] = None
        texts = tokenizer.batch_decode(
            generated[:, input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        return [(text, parse_answer(text)) for text in texts]

    def prepare_context(order):
        prompts = [imposed_prompt(tokenizer, order, entity) for entity in ENTITIES]
        input_ids, attention, entity_positions = tokenize_entity_positions(
            tokenizer, prompts, ENTITIES, device
        )
        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention,
                output_hidden_states=True,
            )
        cache = {}
        for entity_index, entity in enumerate(ENTITIES):
            positions = entity_positions[entity_index]
            cache[entity] = {
                str(layer): output.hidden_states[layer + 1][
                    entity_index, positions, :
                ].detach().to(torch.float16).cpu()
                for layer in selected_layers
            }
        del output
        baseline_outputs = generate(input_ids, attention, [None] * len(ENTITIES))
        successor = successors(order)
        baseline = {
            entity: {
                "answer": answer,
                "text": text,
                "correct": answer == successor[entity],
            }
            for entity, (text, answer) in zip(ENTITIES, baseline_outputs)
        }
        return {
            "order": order,
            "successor": successor,
            "prompts": prompts,
            "cache": cache,
            "baseline": baseline,
        }

    condition_specs = [
        ("same_entity_same_order", True, False),
        ("different_entity_same_order", False, False),
        ("same_entity_cross_order", True, True),
        ("different_entity_cross_order", False, True),
    ]

    def run_direction(recipient, donor, direction_name):
        direction = {
            "direction": direction_name,
            "recipient_order": recipient["order"],
            "donor_order": donor["order"],
            "recipient_baseline": recipient["baseline"],
            "donor_baseline": donor["baseline"],
            "layers": {},
        }
        for layer in selected_layers:
            layer_records = []
            hook = decoder_layers[layer].register_forward_hook(make_hook())
            try:
                for condition, same_entity, cross_order in condition_specs:
                    trials = [
                        (source, destination)
                        for source in range(len(ENTITIES))
                        for destination in range(len(ENTITIES))
                        if (source == destination) == same_entity
                    ]
                    for batch_start in range(0, len(trials), args.batch_size):
                        batch_trials = trials[
                            batch_start: batch_start + args.batch_size
                        ]
                        source_entities = [
                            ENTITIES[source] for source, _ in batch_trials
                        ]
                        prompts = [
                            recipient["prompts"][source]
                            for source, _ in batch_trials
                        ]
                        input_ids, attention, positions = tokenize_entity_positions(
                            tokenizer, prompts, source_entities, device
                        )
                        patch_map = []
                        donor_context = donor if cross_order else recipient
                        for batch_index, (_source, destination) in enumerate(batch_trials):
                            donor_entity = ENTITIES[destination]
                            vector = donor_context["cache"][donor_entity][str(layer)].to(device)
                            length = min(vector.shape[0], len(positions[batch_index]))
                            patch_map.append((positions[batch_index], vector[-length:]))
                        answers = generate(input_ids, attention, patch_map)
                        for (source, destination), (text, answer) in zip(
                            batch_trials, answers
                        ):
                            source_entity = ENTITIES[source]
                            donor_entity = ENTITIES[destination]
                            recipient_source = recipient["successor"][source_entity]
                            recipient_donor = recipient["successor"][donor_entity]
                            donor_source = donor["successor"][source_entity]
                            donor_donor = donor["successor"][donor_entity]
                            clean = (
                                recipient["baseline"][source_entity]["correct"]
                                and recipient["baseline"][donor_entity]["correct"]
                                and (
                                    not cross_order
                                    or donor["baseline"][donor_entity]["correct"]
                                )
                            )
                            layer_records.append({
                                "condition": condition,
                                "source": source_entity,
                                "donor": donor_entity,
                                "answer": answer,
                                "text": text,
                                "clean": bool(clean),
                                "order_diagnostic": recipient_donor != donor_donor,
                                "fully_discriminating": len({
                                    recipient_source,
                                    recipient_donor,
                                    donor_donor,
                                }) == 3,
                                "recipient_map_source": recipient_source,
                                "recipient_map_donor": recipient_donor,
                                "donor_map_source": donor_source,
                                "donor_map_donor": donor_donor,
                                "hit_recipient_map_source": answer == recipient_source,
                                "hit_recipient_map_donor": answer == recipient_donor,
                                "hit_donor_map_source": answer == donor_source,
                                "hit_donor_map_donor": answer == donor_donor,
                                "na": answer is None,
                            })
            finally:
                hook.remove()
            direction["layers"][str(layer)] = layer_records
            primary = [
                record for record in layer_records
                if record["condition"] == "different_entity_cross_order"
                and record["clean"] and record["fully_discriminating"]
            ]
            print(
                f"  pair direction={direction_name} L{layer}: "
                f"primary_n={len(primary)} "
                f"donor_map={rate(primary, 'hit_donor_map_donor'):.3f} "
                f"recipient_map={rate(primary, 'hit_recipient_map_donor'):.3f} "
                f"stuck={rate(primary, 'hit_recipient_map_source'):.3f}",
                flush=True,
            )
        return direction

    results = {
        "experiment": "cross-order entity-slot intervention",
        "model": args.model,
        "concept": "days",
        "number_of_layers": int(number_of_layers),
        "layers": selected_layers,
        "npairs": args.npairs,
        "seed": args.seed,
        "min_successor_differences": args.min_differences,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "prompt_template": TEMPLATE,
        "decisions": {
            "site": "query entity span",
            "operation": "overwrite",
            "subtokens": "all right-aligned donor subtokens",
            "directions": "both within each order pair",
            "independent_unit": "order pair",
            "primary_subset": (
                "clean trials with recipient-source, recipient-donor, and "
                "donor-donor successors all distinct"
            ),
        },
        "order_pair_design": order_pairs,
        "pairs": [],
    }

    for pair_design in order_pairs:
        pair_number = pair_design["pair_index"] + 1
        print(
            f"PAIR {pair_number}/{args.npairs}: "
            f"successor_differences={pair_design['successor_differences']}",
            flush=True,
        )
        context_a = prepare_context(pair_design["order_a"])
        context_b = prepare_context(pair_design["order_b"])
        pair_result = {
            **pair_design,
            "baselines": {
                "a": context_a["baseline"],
                "b": context_b["baseline"],
            },
            "directions": [
                run_direction(context_a, context_b, "B_to_A"),
                run_direction(context_b, context_a, "A_to_B"),
            ],
        }
        results["pairs"].append(pair_result)
        results["summary"] = summarize(results)
        save_json(partial_path, results)
        del context_a, context_b
        torch.cuda.empty_cache()

    results["summary"] = summarize(results)
    save_json(output_path, results)
    print(f"SAVED {output_path}", flush=True)


if __name__ == "__main__":
    main()
