"""Context-quality metrics for the ablation benchmark.

These grade the *context*, not the final answer — because if the context is
wrong the model is already doomed.  Metrics are computed from a
:class:`ContextBuildState` plus the task's gold spec.

Retrieval-quality metrics use rank orders:
  * ``retrieved_order`` — post-retrieval order (measures the retriever)
  * ``ranked_order``    — post-rerank order   (measures reranking)
Selection metrics use the final ``selected_items``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List

from .state import ContextBuildState


def recall_at_k(order: List[str], gold: set, k: int) -> float:
    if not gold:
        return 1.0
    return len(set(order[:k]) & gold) / len(gold)


def precision_at_k(order: List[str], gold: set, k: int) -> float:
    topk = order[:k]
    if not topk:
        return 0.0
    return len(set(topk) & gold) / len(topk)


def mrr(order: List[str], gold: set) -> float:
    for i, rid in enumerate(order):
        if rid in gold:
            return 1.0 / (i + 1)
    return 0.0


def _dcg(rels: List[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(rels))


def ndcg_at_k(order: List[str], gold: set, k: int) -> float:
    if not gold:
        return 1.0
    rels = [1.0 if rid in gold else 0.0 for rid in order[:k]]
    ideal = [1.0] * min(len(gold), k) + [0.0] * max(0, k - len(gold))
    idcg = _dcg(ideal)
    return _dcg(rels) / idcg if idcg > 0 else 0.0


def compute_metrics(
    state: ContextBuildState, spec: Dict[str, Any], k: int = 5
) -> Dict[str, float]:
    """Return a dict of context-quality metrics for one built context."""
    gold = set(spec.get("gold_resources", []))
    stale = set(spec.get("stale_resources", []))
    unrelated = set(spec.get("unrelated_resources", []))
    required_rules = set(spec.get("required_rules", []))

    selected = [c.resource_id for c in state.selected_items]
    selected_set = set(selected)
    retrieved = state.retrieved_order
    ranked = state.ranked_order or retrieved
    text = state.formatted_context
    budget = state.budget.total_tokens if state.budget else 0
    selected_rule_ids = {r.id for r in state.selected_rules}

    # citation precision: fraction of selected items cited in the context
    if selected:
        cited = sum(
            1
            for rid in selected
            if f"[{rid}" in text or f'id="{rid}"' in text
        )
        citation_precision = cited / len(selected)
    else:
        citation_precision = 0.0

    stale_leak = (
        len([rid for rid in selected_set & stale]) / len(stale) if stale else 0.0
    )
    # leak only counts if not clearly marked as a conflict/stale in the context
    if stale and "conflict" in text.lower():
        stale_leak = 0.0

    noise = (
        len(selected_set & unrelated) / len(selected) if (unrelated and selected) else 0.0
    )
    diversity = (
        len({c.metadata.get("group") or c.resource_id for c in state.selected_items})
        / len(selected)
        if selected
        else 0.0
    )

    return {
        "retrieval_recall@k": round(recall_at_k(retrieved, gold, k), 4),
        "ranked_ndcg@k": round(ndcg_at_k(ranked, gold, k), 4),
        "ranked_mrr": round(mrr(ranked, gold), 4),
        "final_recall": round(recall_at_k(selected, gold, len(selected) or 1), 4)
        if gold
        else 1.0,
        "final_precision": round(
            len(selected_set & gold) / len(selected), 4
        )
        if selected and gold
        else (1.0 if not gold else 0.0),
        "citation_precision": round(citation_precision, 4),
        "stale_leak_rate": round(stale_leak, 4),
        "noise_rate": round(noise, 4),
        "diversity": round(diversity, 4),
        "required_rule_coverage": round(
            len(required_rules & selected_rule_ids) / len(required_rules), 4
        )
        if required_rules
        else 1.0,
        "budget_ok": 1.0 if (budget == 0 or state.token_estimate <= budget) else 0.0,
        "budget_utilization": round(state.token_estimate / budget, 4) if budget else 0.0,
        "validation_pass": 1.0 if state.validation.get("passed") else 0.0,
        "latency_ms": round(sum(m.duration_ms for m in state.metrics), 3),
        "num_selected": float(len(selected)),
    }


def mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Average a list of metric dicts key-by-key."""
    if not rows:
        return {}
    keys = rows[0].keys()
    return {k: round(sum(r[k] for r in rows) / len(rows), 4) for k in keys}
