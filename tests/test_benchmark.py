"""Integration test: the ablation benchmark measures operator contributions."""
from context_engineering.pipeline.benchmark import run_benchmark


def test_benchmark_shows_operator_value(examples_dir):
    results = run_benchmark(examples_dir)
    full = results["pipelines"]["full_production"]["mean_metrics"]
    naive = results["pipelines"]["naive_baseline"]["mean_metrics"]

    # The full pipeline cites everything; the naive baseline cites nothing.
    assert full["citation_precision"] == 1.0
    assert naive["citation_precision"] == 0.0

    # The conflict detector eliminates stale-version leakage.
    assert full["stale_leak_rate"] == 0.0
    assert naive["stale_leak_rate"] > 0.0

    # The full pipeline passes deterministic context validation; naive does not.
    assert full["validation_pass"] == 1.0
    assert naive["validation_pass"] < 1.0


def test_conflict_detector_is_responsible_for_stale_handling(examples_dir):
    results = run_benchmark(
        examples_dir, pipeline_names=["full_production", "full_minus_conflict"]
    )
    full = results["pipelines"]["full_production"]["mean_metrics"]
    minus = results["pipelines"]["full_minus_conflict"]["mean_metrics"]
    # removing the conflict detector reintroduces stale leakage
    assert minus["stale_leak_rate"] > full["stale_leak_rate"]


def test_reranker_improves_ranking(examples_dir):
    results = run_benchmark(
        examples_dir, pipeline_names=["full_production", "full_minus_reranker"]
    )
    full = results["pipelines"]["full_production"]["mean_metrics"]
    minus = results["pipelines"]["full_minus_reranker"]["mean_metrics"]
    # the cross-encoder reranker improves (or at least does not hurt) NDCG
    assert full["ranked_ndcg@k"] >= minus["ranked_ndcg@k"]
