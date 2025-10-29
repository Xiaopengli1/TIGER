"""Utilities for monitoring RQ-VAE code usage statistics."""

from __future__ import annotations

import collections
import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def check_collision(all_indices_str: np.ndarray) -> bool:
    """Return ``True`` if all serialized code strings are unique."""

    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    return tot_item == tot_indice


def get_indices_count(all_indices_str: np.ndarray):
    """Count occurrences for each serialized code string."""

    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str: np.ndarray):
    """Group the sample indices that share the same serialized code."""

    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []
    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups


def compute_layer_metrics(
    indices: np.ndarray,
    num_emb_list: Optional[Sequence[int]] = None,
    layer_names: Optional[Sequence[str]] = None,
    print_table: bool = True,
) -> Dict[str, Any]:
    """Compute collision, utilization, and entropy statistics per quantizer layer."""

    indices = np.asarray(indices)
    if indices.size == 0:
        return {
            "per_layer": [],
            "summary": {
                "N": 0,
                "L": 0,
                "avg_collision_rate": 0.0,
                "avg_collision_item_rate": 0.0,
                "avg_utilization": None,
                "avg_entropy_bits": 0.0,
                "avg_norm_entropy": None,
                "avg_perplexity": 0.0,
            },
        }

    N, L = indices.shape
    if layer_names is None:
        layer_names = [f"layer_{i}" for i in range(L)]

    results: List[Dict[str, Any]] = []
    totals: Dict[str, Any] = {
        "N": int(N),
        "L": int(L),
        "avg_collision_rate": 0.0,
        "avg_collision_item_rate": 0.0,
        "avg_utilization": 0.0 if num_emb_list is not None else None,
        "avg_entropy_bits": 0.0,
        "avg_norm_entropy": 0.0 if num_emb_list is not None else None,
        "avg_perplexity": 0.0,
    }

    for l in range(L):
        x = indices[:, l]
        uniq, counts = np.unique(x, return_counts=True)
        U = int(uniq.size)

        K = int(num_emb_list[l]) if num_emb_list is not None else int(x.max()) + 1

        collision_rate = 1.0 - (U / N)
        collision_item_rate = counts[counts > 1].sum() / N
        utilization = (U / K) if num_emb_list is not None and K > 0 else None

        p = counts / N
        H_bits = float(-(p * np.log2(p)).sum()) if U > 0 else 0.0
        H_norm = (H_bits / math.log2(K)) if num_emb_list is not None and K > 1 else None
        perplexity = float(2.0 ** H_bits)

        results.append(
            {
                "layer": str(layer_names[l]),
                "N": int(N),
                "K": int(K),
                "unique_codes": int(U),
                "collision_rate": float(collision_rate),
                "collision_item_rate": float(collision_item_rate),
                "utilization": float(utilization) if utilization is not None else None,
                "entropy_bits": float(H_bits),
                "norm_entropy": float(H_norm) if H_norm is not None else None,
                "perplexity": float(perplexity),
            }
        )

    totals["avg_collision_rate"] = float(np.mean([r["collision_rate"] for r in results]))
    totals["avg_collision_item_rate"] = float(
        np.mean([r["collision_item_rate"] for r in results])
    )
    totals["avg_entropy_bits"] = float(np.mean([r["entropy_bits"] for r in results]))
    totals["avg_perplexity"] = float(np.mean([r["perplexity"] for r in results]))

    if num_emb_list is not None:
        totals["avg_utilization"] = float(
            np.mean([r["utilization"] for r in results if r["utilization"] is not None])
        )
        norm_list = [r["norm_entropy"] for r in results if r["norm_entropy"] is not None]
        totals["avg_norm_entropy"] = float(np.mean(norm_list)) if norm_list else None

    if print_table:
        head = (
            f"{'layer':<10}{'N':>8}{'K':>8}{'unique':>10}{'coll_rate':>12}"
            f"{'item_coll':>12}{'util':>8}{'H(bits)':>12}{'H_norm':>10}{'pplx':>10}"
        )
        print("\n=== Per-layer metrics ===")
        print(head)
        print("-" * len(head))
        for r in results:
            util = r["utilization"] if r["utilization"] is not None else float("nan")
            norm_entropy = r["norm_entropy"] if r["norm_entropy"] is not None else float("nan")
            print(
                f"{r['layer']:<10}{r['N']:>8}{r['K']:>8}{r['unique_codes']:>10}"
                f"{r['collision_rate']:>12.4f}{r['collision_item_rate']:>12.4f}"
                f"{util:>8.4f}{r['entropy_bits']:>12.4f}{norm_entropy:>10.4f}"
                f"{r['perplexity']:>10.2f}"
            )

        print("\nAverages:")
        print(f"- avg_collision_rate      : {totals['avg_collision_rate']:.4f}")
        print(f"- avg_collision_item_rate : {totals['avg_collision_item_rate']:.4f}")
        if totals["avg_utilization"] is not None:
            print(f"- avg_utilization         : {totals['avg_utilization']:.4f}")
        print(f"- avg_entropy_bits        : {totals['avg_entropy_bits']:.4f}")
        if totals["avg_norm_entropy"] is not None:
            print(f"- avg_norm_entropy        : {totals['avg_norm_entropy']:.4f}")
        print(f"- avg_perplexity          : {totals['avg_perplexity']:.2f}")

    return {"per_layer": results, "summary": totals}


def summarize_layer_max_usage(indices: np.ndarray) -> List[Dict[str, Any]]:
    """Return the most frequent code usage ratio for each layer."""

    indices = np.asarray(indices)
    if indices.size == 0:
        return []

    usage = []
    N = indices.shape[0]
    for l in range(indices.shape[1]):
        _, counts = np.unique(indices[:, l], return_counts=True)
        max_freq = int(counts.max())
        usage.append(
            {
                "layer": l,
                "max_freq": max_freq,
                "max_freq_ratio": float(max_freq / N),
            }
        )

    return usage


__all__ = [
    "check_collision",
    "compute_layer_metrics",
    "get_collision_item",
    "get_indices_count",
    "summarize_layer_max_usage",
]

