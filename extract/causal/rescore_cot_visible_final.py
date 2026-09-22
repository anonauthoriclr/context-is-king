#!/usr/bin/env python3
"""Rescore cached Qwen CoT traces from the visible post-reasoning response.

The extraction runner retains full traces.  Its legacy online metric takes the
last entity mention anywhere in the trace, including unfinished reasoning.  For
the paper's Qwen CoT result, a committed answer instead requires a visible
response after the final ``</think>`` delimiter.  This script applies that rule
deterministically, preserves per-order results, and reports the diagnostic
subset whose imposed and pretrained successors differ.

Example:
  python rescore_cot_visible_final.py \
    --raw ../../data/causal/causalcot_Qwen3.5-27B.json \
          ../../data/causal/causalcot_Qwen3.5-27B_scr3_imposed_raw.json \
    --out ../../data/causal/causalcot_Qwen3.5-27B_imposed3_visible_final.json
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


ENTITIES = {
    "days": [
        "Monday", "Tuesday", "Wednesday", "Thursday",
        "Friday", "Saturday", "Sunday",
    ]
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def imposed_scrambles(payload):
    if "contexts" in payload:
        return payload["contexts"]["imposed"]["per_scr"]
    return [payload[k] for k in sorted(payload) if k.startswith("imposed|")]


def visible_answer(trace, entities):
    """Return the last entity in the visible response, or None if none exists."""
    if not trace or "</think>" not in trace:
        return None
    visible = trace.rsplit("</think>", 1)[1]
    pattern = re.compile(r"\b(" + "|".join(map(re.escape, entities)) + r")\b", re.I)
    hits = pattern.findall(visible)
    if not hits:
        return None
    canonical = {entity.lower(): entity for entity in entities}
    return canonical[hits[-1].lower()]


def mean(values):
    return sum(values) / len(values) if values else 0.0


def score_scramble(scramble, entities):
    n = len(entities)
    natural_index = {entity: i for i, entity in enumerate(entities)}
    imposed_index = {entity: i for i, entity in enumerate(scramble["order"])}
    successor_natural = {
        entity: entities[(natural_index[entity] + 1) % n] for entity in entities
    }
    successor_imposed = {
        entity: scramble["order"][(imposed_index[entity] + 1) % n]
        for entity in entities
    }

    baseline_answers = {
        entity: visible_answer(scramble["base_traces"][entity], entities)
        for entity in entities
    }
    baseline_ok = {
        entity: baseline_answers[entity] == successor_imposed[entity]
        for entity in entities
    }

    layers = {}
    for layer, layer_payload in scramble["layers"].items():
        scored = []
        for record in layer_payload["recs"]:
            answer = visible_answer(record.get("trace"), entities)
            clean = baseline_ok[record["src"]] and baseline_ok[record["dst"]]
            diagnostic = (
                successor_imposed[record["dst"]]
                != successor_natural[record["dst"]]
            )
            scored.append({
                "answer": answer,
                "clean": clean,
                "diagnostic": diagnostic,
                "own": answer == successor_imposed[record["dst"]],
                "pretrained_alternative": answer == successor_natural[record["dst"]],
                "stuck_source": answer == successor_imposed[record["src"]],
            })

        clean_records = [record for record in scored if record["clean"]]
        diagnostic_records = [
            record for record in clean_records if record["diagnostic"]
        ]
        layers[layer] = {
            "n_clean": len(clean_records),
            "patch_success_clean": mean([record["own"] for record in clean_records]),
            "visible_final_rate": mean([
                record["answer"] is not None for record in clean_records
            ]),
            "stuck_source_clean": mean([
                record["stuck_source"] for record in clean_records
            ]),
            "n_diagnostic": len(diagnostic_records),
            "diagnostic_own_count": sum(
                record["own"] for record in diagnostic_records
            ),
            "diagnostic_pretrained_alternative_count": sum(
                record["pretrained_alternative"] for record in diagnostic_records
            ),
        }

    return {
        "order": scramble["order"],
        "baseline_correct": sum(baseline_ok.values()),
        "baseline_total": len(entities),
        "layers": layers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    payloads = [json.loads(path.read_text()) for path in args.raw]
    concept = payloads[0]["concept"] if "concept" in payloads[0] else "days"
    entities = ENTITIES[concept]

    scrambles = []
    seen_orders = set()
    for payload in payloads:
        for scramble in imposed_scrambles(payload):
            order = tuple(scramble["order"])
            if order not in seen_orders:
                seen_orders.add(order)
                scrambles.append(scramble)

    scored = [score_scramble(scramble, entities) for scramble in scrambles]
    layer_names = sorted(scored[0]["layers"], key=int)
    aggregate = {}
    for layer in layer_names:
        rows = [scramble["layers"][layer] for scramble in scored]
        aggregate[layer] = {
            "mean_patch_success_across_orders": mean([
                row["patch_success_clean"] for row in rows
            ]),
            "mean_visible_final_rate_across_orders": mean([
                row["visible_final_rate"] for row in rows
            ]),
            "total_clean": sum(row["n_clean"] for row in rows),
            "total_diagnostic": sum(row["n_diagnostic"] for row in rows),
            "diagnostic_own_count": sum(
                row["diagnostic_own_count"] for row in rows
            ),
            "diagnostic_pretrained_alternative_count": sum(
                row["diagnostic_pretrained_alternative_count"] for row in rows
            ),
        }

    output = {
        "model": "Qwen/Qwen3.5-27B",
        "concept": concept,
        "context": "imposed",
        "n_orders": len(scored),
        "parser": {
            "rule": "last entity mention after the final </think> delimiter",
            "missing_visible_response": "NA",
        },
        "sources": [
            {"file": path.name, "sha256": sha256(path)} for path in args.raw
        ],
        "per_order": scored,
        "aggregate_by_layer": aggregate,
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {args.out} ({len(scored)} imposed orders)")


if __name__ == "__main__":
    main()
