"""Non-interactive ablation benchmark for the production pipeline.

Runs every named pipeline in ``registry.PIPELINES`` over every example task and
reports context-quality metrics (recall, NDCG, MRR, citation precision,
stale-leak, budget, ...) so you can see, with and without each operator, how the
context measures.

Usage::

    python -m context_engineering.pipeline.benchmark
    python -m context_engineering.pipeline.benchmark --domain rag_doc_qa --out bench.json
    python -m context_engineering.pipeline.benchmark --pipelines naive_baseline,full_production
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from ..loaders import build_engine, list_domains, load_domain_bundle
from .base import build_pipeline
from .metrics import compute_metrics, mean_metrics
from .registry import PIPELINES, default_registry
from .state import ContextBuildState

# Metrics shown in the summary table (higher is better unless noted).
REPORT_METRICS = [
    "retrieval_recall@k",
    "ranked_ndcg@k",
    "ranked_mrr",
    "final_recall",
    "final_precision",
    "citation_precision",
    "stale_leak_rate",
    "noise_rate",
    "required_rule_coverage",
    "budget_ok",
    "validation_pass",
]


def _default_examples_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


def build_state(engine, task: str, domain: str, user_context: Dict[str, Any] | None = None):
    return ContextBuildState(
        task=task,
        domain=domain,
        user_context=user_context or {},
        resources=engine.resources,
        rules=engine.rules,
        skills=engine.skills,
    )


def run_benchmark(
    examples_dir: str,
    domains: List[str] | None = None,
    pipeline_names: List[str] | None = None,
) -> Dict[str, Any]:
    """Run the ablation and return a structured result dict."""
    domains = domains or list_domains(examples_dir)
    engine = build_engine(examples_dir, domains)
    registry = default_registry()
    pipeline_names = pipeline_names or list(PIPELINES.keys())

    # Collect all (domain, task, gold) tuples.
    tasks: List[Dict[str, Any]] = []
    for domain in domains:
        task_list, gold = load_domain_bundle(examples_dir, domain)
        for t in task_list:
            tasks.append({"domain": domain, "task": t, "gold": gold.get(t["task_id"], {})})

    results: Dict[str, Any] = {"pipelines": {}, "per_task": []}
    per_pipeline_rows: Dict[str, List[Dict[str, float]]] = {n: [] for n in pipeline_names}

    for name in pipeline_names:
        pipeline = build_pipeline(registry, PIPELINES[name])
        for entry in tasks:
            state = build_state(engine, entry["task"]["task"], entry["domain"])
            state = pipeline.run(state)
            m = compute_metrics(state, entry["gold"])
            per_pipeline_rows[name].append(m)
            results["per_task"].append(
                {
                    "pipeline": name,
                    "task_id": entry["task"]["task_id"],
                    "domain": entry["domain"],
                    "selected": [c.resource_id for c in state.selected_items],
                    "metrics": m,
                    "warnings": state.warnings,
                    "errors": state.errors,
                }
            )

    for name in pipeline_names:
        results["pipelines"][name] = {
            "signature": [
                e if isinstance(e, str) else e[0] for e in PIPELINES[name]
            ],
            "mean_metrics": mean_metrics(per_pipeline_rows[name]),
        }
    return results


# --------------------------------------------------------------------------- #
# Reporting                                                                    #
# --------------------------------------------------------------------------- #
_SHORT = {
    "naive_baseline": "naive",
    "full_production": "FULL",
    "full_minus_reranker": "-rerank",
    "full_minus_dense": "-dense",
    "full_minus_conflict": "-conflict",
    "full_minus_permission": "-perm",
    "full_minus_compress_budget": "-compbud",
}


def print_report(results: Dict[str, Any]) -> None:
    names = list(results["pipelines"].keys())

    print("\n" + "=" * 92)
    print("CONTEXT-ENGINEERING ABLATION BENCHMARK")
    print("=" * 92)
    print("(means across all tasks; stale_leak_rate & noise_rate: LOWER is better)")
    print("legend: " + ",  ".join(f"{_SHORT.get(n, n)}={n}" for n in names) + "\n")

    col_w = 11
    labels = [_SHORT.get(n, n) for n in names]
    header = f"{'metric':<24}" + "".join(f"{lab:>{col_w}}" for lab in labels)
    print(header)
    print("-" * len(header))
    for metric in REPORT_METRICS:
        row = f"{metric:<24}"
        for n in names:
            val = results["pipelines"][n]["mean_metrics"].get(metric, 0.0)
            row += f"{val:>{col_w}.3f}"
        print(row)

    # latency separately
    row = f"{'latency_ms (mean)':<24}"
    for n in names:
        val = results["pipelines"][n]["mean_metrics"].get("latency_ms", 0.0)
        row += f"{val:>{col_w}.2f}"
    print(row)

    # Leave-one-out contribution vs full_production.
    if "full_production" in results["pipelines"]:
        full = results["pipelines"]["full_production"]["mean_metrics"]
        print("\n" + "-" * 100)
        print("OPERATOR CONTRIBUTION (full_production minus variant; positive = operator helps)")
        print("-" * 100)
        loo = [n for n in names if n.startswith("full_minus_")]
        summary_metrics = [
            "ranked_ndcg@k", "final_precision", "citation_precision",
            "stale_leak_rate", "budget_ok",
        ]
        head = f"{'removed operator':<28}" + "".join(f"{m[:16]:>18}" for m in summary_metrics)
        print(head)
        print("-" * len(head))
        for n in loo:
            variant = results["pipelines"][n]["mean_metrics"]
            label = n.replace("full_minus_", "-")
            row = f"{label:<28}"
            for m in summary_metrics:
                delta = full.get(m, 0.0) - variant.get(m, 0.0)
                # for "lower is better" metrics, invert sign so positive = helps
                if m in {"stale_leak_rate", "noise_rate"}:
                    delta = -delta
                row += f"{delta:>+18.3f}"
            print(row)

    print("\n" + "=" * 100)
    best = max(
        results["pipelines"].items(),
        key=lambda kv: (
            kv[1]["mean_metrics"].get("final_precision", 0)
            + kv[1]["mean_metrics"].get("ranked_ndcg@k", 0)
            + kv[1]["mean_metrics"].get("citation_precision", 0)
            - kv[1]["mean_metrics"].get("stale_leak_rate", 0)
        ),
    )[0]
    print(f"Best pipeline by (precision + ndcg + citation - stale_leak): {best}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Context-engineering ablation benchmark")
    parser.add_argument("--examples", default=_default_examples_dir())
    parser.add_argument("--domain", default=None, help="restrict to one domain")
    parser.add_argument("--pipelines", default=None, help="comma-separated pipeline names")
    parser.add_argument("--out", default=None, help="write full results JSON here")
    args = parser.parse_args(argv)

    domains = [args.domain] if args.domain else None
    pipeline_names = args.pipelines.split(",") if args.pipelines else None
    results = run_benchmark(args.examples, domains, pipeline_names)
    print_report(results)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nSaved full results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
