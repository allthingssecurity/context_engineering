# Architecture

Two layers share the same data models and registries:

1. **MVP engine** (`context_engineering/engine.py`) — ~12 operators, proves the
   `c = F_s(x, K_s, R_s, U, B)` idea end-to-end with human-readable output.
2. **Production pipeline** (`context_engineering/pipeline/`) — a stable,
   pluggable `Operator` interface and the full 19-stage pipeline, with local
   offline defaults and vendor seams.

Everything is registries + registered callables, so the engines contain **no
domain logic** — adding a domain/skill/operator/evaluator never edits them.

---

## Data flow

```
                       x (task)  +  domain hint  +  U (user_context)
                                        │
                    ┌───────────────────▼────────────────────┐
                    │            ContextBuildState            │   ← threaded through every operator
                    │  task, domain, resources, rules, skills │
                    └───────────────────┬────────────────────┘
                                        │
  ROUTE ──▶ CONSTRAIN ──▶ PLAN ──▶ SCOPE ──▶ REWRITE ──▶ RETRIEVE(hybrid)
    │           │           │        │          │             │
 task_router constraint  skill_    scope_    query_     BM25 + hashed-dense
             extractor   planner  resolver  rewriter   fused via RRF
                                        │
  ─▶ PERMISSION-FILTER ─▶ AUTHORITY ─▶ CONFLICT ─▶ RERANK ─▶ DIVERSIFY
        (ACL/tenant)     (freshness)  (recency)  (x-encoder)   (MMR)
                                        │
  ─▶ SELECT(top-k) ─▶ COMPRESS ─▶ BUDGET ─▶ ORDER ─▶ FORMAT ─▶ VALIDATE ─▶ TELEMETRY
                    query-aware   quota   template  XML/cite  secrets/ACL   metrics
                                        │
                                        ▼
                              c  (ContextPackage / formatted context)
                                        │
                    ┌───────────────────┴────────────────────┐
                    │  intrinsic metrics        extrinsic A/B │
                    │  (benchmark.py)           (agent_ab.py) │
                    └─────────────────────────────────────────┘
```

---

## Core modules

| file | responsibility |
| --- | --- |
| `models.py` | dataclasses: `Rule`, `Skill`, `Resource`, `ContextItem`, `ContextPackage`, `EvaluationResult`, `ExperimentResult`, `PipelineState`; `estimate_tokens` |
| `registries.py` | `SkillRegistry`, `RuleRegistry`, `ResourceRegistry` (TF-IDF search), `OperatorRegistry`, `EvaluatorRegistry` |
| `operators.py` | the 12 MVP operators (building blocks of `F_s`) |
| `evaluators.py` | 7 evaluators scoring a `ContextPackage` |
| `engine.py` | `ContextEngine`: selects a skill, runs its pipeline |
| `loaders.py` | load example JSON into a wired engine |
| `run_experiments.py` | build + evaluate every task, print + save |
| `compare_skills.py` | meta experiment: compare skills on one domain |
| `cli.py` | `list-skills`, `list-rules`, `build-context`, `run-experiments`, `compare-skills`, `inspect-trace` |

### Production pipeline (`pipeline/`)

| file | responsibility |
| --- | --- |
| `state.py` | `ContextBuildState`, `Query`, `CandidateResource`, `Budget`, `TraceEvent`, `OperatorMetrics` |
| `base.py` | `Operator` ABC (`name`/`version`/`run`), `OperatorRegistry`, `Pipeline`, `build_pipeline` |
| `operators.py` | 19 production operators (routing → retrieval → filter → rank → assemble → validate → telemetry) |
| `registry.py` | `default_registry()`; named `PIPELINES` (naive, full, leave-one-out ablations) |
| `metrics.py` | context-quality metrics (recall/NDCG/MRR/precision/citation/stale/…) |
| `benchmark.py` | non-interactive ablation study (intrinsic) |
| `agent_ab.py` | downstream A/B via real agents (`codex`/`openai`/`hermes`) — extrinsic |

---

## The operator contract (why it's pluggable)

```python
class Operator(ABC):
    name: str = "operator"
    version: str = "1.0"
    def __init__(self, **config): self.config = config
    @abstractmethod
    def _run(self, state: ContextBuildState) -> ContextBuildState: ...
    def run(self, state):        # timed, fail-safe wrapper
        # records OperatorMetrics(duration, candidates_in/out, error)
        # captures exceptions into state.errors/warnings/trace — never silently fails
        ...
```

Register and reference by name — no engine change:

```python
reg = default_registry()
reg.register("my_reranker", MyReranker)          # subclass of Operator
pipe = build_pipeline(reg, ["task_router", "skill_planner",
                            "hybrid_retriever", "my_reranker",
                            ("evidence_selector", {"top_k": 4}), "xml_formatter"])
```

Pipeline **specs** are ordered lists of `"name"` or `("name", {config})`, so an
ablation is just a spec with one operator removed (`registry.PIPELINES`).

---

## Vendor seams (local default → production swap)

| stage | local default (offline) | swap for |
| --- | --- | --- |
| `task_router` | keyword scoring | LLM JSON router |
| `hybrid_retriever` (dense) | hashed char-n-gram cosine | sentence-transformers / Qdrant / Weaviate |
| `cross_encoder_reranker` | listwise coverage+idf+proximity | Cohere / BGE / Voyage / Jina |
| `query_rewriter` | synonym/symbol map | LLM multi-query / HyDE |
| `query_aware_compressor` | extractive first-N + query overlap | citation-preserving LLM compressor |
| token estimate | `len/4` | the target model's tokenizer |

The `Operator.run` / `ContextBuildState` interfaces do not change across swaps.

---

## Agent backends (extrinsic A/B)

`agent_ab.py` exposes a pluggable `BACKENDS` registry:

| backend | how it runs | notes |
| --- | --- | --- |
| `codex` | `codex exec -s read-only -o <file> -` | ChatGPT-auth, captures final message |
| `openai` | Chat Completions API (`~/.oai_key`) | pick model with `--model` |
| `hermes` | `hermes -z <prompt> --safe-mode` | NousResearch hermes-agent, its configured default model |

Add a backend = one function `(prompt, timeout, model) -> {answer, latency_s, ok}`.
