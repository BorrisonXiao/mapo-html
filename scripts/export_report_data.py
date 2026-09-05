#!/usr/bin/env python3
"""Export the compact evidence snapshot used by the attention-analysis report.

The source experiment artifacts remain in the MAPO working tree.  This script
copies only the curves and summary values needed to regenerate the public page.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


MOSS_NORMALIZED = {
    "MMAU": {
        "mean": [5.257, 1.084, 0.986, 1.329, 0.655, 1.354, 0.285, 3.071],
        "max": [5.095, 1.136, 1.018, 1.337, 0.674, 1.252, 0.275, 2.899],
        "median_mean": 1.20676,
        "median_max": 1.19388,
        "median_prompt_tokens": 185.5,
        "median_audio_tokens": 125.0,
        "median_uniform_first": 0.6703,
        "median_uniform_last": 0.4357,
        "median_expected_last_first": 0.6482,
        "median_observed_first_mean": 0.0023146,
        "median_observed_first_max": 0.0212517,
    },
    "MMAR": {
        "mean": [0.828, 0.389, 2.153, 0.432, 0.713, 1.943, 0.442, 0.528],
        "max": [0.824, 0.485, 2.708, 0.456, 0.689, 2.168, 0.435, 0.600],
        "median_mean": 0.62052,
        "median_max": 0.64479,
        "median_prompt_tokens": 361.0,
        "median_audio_tokens": 275.5,
        "median_uniform_first": 0.79643,
        "median_uniform_last": 0.62806,
        "median_expected_last_first": 0.78934,
        "median_observed_first_mean": 0.0044256,
        "median_observed_first_max": 0.0311584,
    },
    "MMSU": {
        "mean": [0.858, 1.034, 2.590, 3.419, 0.299, 0.878, 0.109, 0.531],
        "max": [0.829, 1.046, 2.448, 3.344, 0.343, 1.041, 0.123, 0.471],
        "median_mean": 0.86772,
        "median_max": 0.93501,
        "median_prompt_tokens": 88.0,
        "median_audio_tokens": 33.5,
        "median_uniform_first": 0.38312,
        "median_uniform_last": 0.19226,
        "median_expected_last_first": 0.50735,
        "median_observed_first_mean": 0.0020416,
        "median_observed_first_max": 0.0164948,
    },
}


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def qwen_snapshot(root: Path) -> dict:
    base = root / "exp/analysis/mmar/Qwen3-Omni-30B-A3B-Thinking/v7-20260415-030139"
    mean_row = read_json(base / "stage4/head_mean/attention_index.json")["rows"][0]
    max_row = read_json(base / "stage4/head_max/attention_index.json")["rows"][0]
    audio_tokens = sum(stop - start for start, stop in mean_row["audio_region_spans"])
    response_start = mean_row["response_start"]
    mean_mass = mean_row["avg_audio_mass"]
    max_mass = max_row["avg_audio_mass"]
    uniform = [audio_tokens / (response_start + index + 1) for index in range(len(mean_mass))]

    def med(values, lo, hi):
        return statistics.median(values[lo:hi])

    mean_lift = [mass / baseline for mass, baseline in zip(mean_mass, uniform)]
    max_lift = [mass / baseline for mass, baseline in zip(max_mass, uniform)]
    windows = {}
    for label, lo, hi in (
        ("description", 80, 125),
        ("transition", 125, 145),
        ("choice_evaluation", 145, 220),
        ("late", 220, 300),
    ):
        windows[label] = {
            "range": [lo, hi],
            "mean_raw_median": med(mean_mass, lo, hi),
            "max_raw_median": med(max_mass, lo, hi),
            "mean_lift_median": med(mean_lift, lo, hi),
            "max_lift_median": med(max_lift, lo, hi),
            "uniform_median": med(uniform, lo, hi),
        }

    return {
        "model": "Qwen3-Omni-30B-A3B-Thinking",
        "benchmark": "MMAR",
        "sample_id": mean_row["id"],
        "question": mean_row["question"],
        "audio_tokens": audio_tokens,
        "response_start": response_start,
        "response_tokens": len(mean_mass),
        "caption_boundary_token": 138,
        "caption_boundary_note": "Manual alignment to the transition phrase ‘Next, evaluating the choices’. A token-level annotation should replace this approximation in production.",
        "mean_mass": mean_mass,
        "max_mass": max_mass,
        "uniform": uniform,
        "first": {
            "uniform": uniform[0],
            "mean_mass": mean_mass[0],
            "max_mass": max_mass[0],
            "mean_lift": mean_lift[0],
            "max_lift": max_lift[0],
        },
        "last_uniform": uniform[-1],
        "windows": windows,
        "source_run": str(base.relative_to(root)),
    }


def moss_snapshot(root: Path) -> list[dict]:
    rows = []
    for benchmark in ("mmau", "mmar", "mmsu"):
        base = root / f"exp/analysis/{benchmark}/MOSS-Audio-4B-Thinking/v1-20260905-entropy-cdf"
        payload = read_json(base / "stage3/modality_retention.json")
        raw_mean = [sample["mean_head_audio_mass"]["late_to_early_ratio"] for sample in payload["samples"]]
        raw_max = [sample["max_head_audio_mass"]["late_to_early_ratio"] for sample in payload["samples"]]
        normalized = MOSS_NORMALIZED[benchmark.upper()]
        raw_decay = sum(x < 0.5 and y < 0.5 for x, y in zip(raw_mean, raw_max))
        normalized_decay = sum(
            x < 0.5 and y < 0.5 for x, y in zip(normalized["mean"], normalized["max"])
        )
        rows.append(
            {
                "benchmark": benchmark.upper(),
                "samples": len(payload["samples"]),
                "raw_mean": statistics.median(raw_mean),
                "raw_max": statistics.median(raw_max),
                "normalized_mean": normalized["median_mean"],
                "normalized_max": normalized["median_max"],
                "raw_decay_cases": raw_decay,
                "normalized_decay_cases": normalized_decay,
                "median_prompt_tokens": normalized["median_prompt_tokens"],
                "median_audio_tokens": normalized["median_audio_tokens"],
                "median_uniform_first": normalized["median_uniform_first"],
                "median_uniform_last": normalized["median_uniform_last"],
                "median_expected_last_first": normalized["median_expected_last_first"],
                "median_observed_first_mean": normalized["median_observed_first_mean"],
                "median_observed_first_max": normalized["median_observed_first_max"],
                "source_run": str(base.relative_to(root)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.mapo_root.resolve()
    payload = {
        "generated_on": "2026-09-05",
        "scope": "Small diagnostic: eight random MOSS examples per benchmark, seed 17; not a benchmark-level estimate.",
        "normalization_note": "MOSS normalized ratios are a diagnostic post-aggregation adjustment. Production analysis should normalize each layer/head before mean or max reduction.",
        "qwen": qwen_snapshot(root),
        "moss": moss_snapshot(root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
