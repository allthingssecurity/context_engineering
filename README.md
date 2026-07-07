# Context Engineering Framework

A small, runnable, and extensible framework that treats **context construction
as a first-class, testable engineering problem**.

The core idea:

> For any user task `x`, select a suitable **skill** `s`, load its static
> resources and rules `ρ_s`, run its dynamic operators `F_s`, and produce a
> final **context** `c` for an LLM.

Conceptually:

```
c = F_s(x; ρ_s)
```

Implemented in this project as:

```
c = F_s(
    x,     # the current task
    K_s,   # static knowledge/resources (documents, code, logs, paper sections, policies)
    R_s,   # selected rules from a rule registry
    U,     # user / session preferences
    B,     # context/token budget
)
```

where `F_s` is a **pipeline of operators** (search, retrieve, rank, filter,
compress, format) and `c` is a fully-traced **context package**.

Nothing here calls a paid API by default. Retrieval is a local, deterministic
TF‑IDF‑style scorer, so the whole thing runs offline and is trivially testable.

> **Deep dives:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) (system design,
> data flow, operator contract, vendor seams) ·
> [`docs/MATH.md`](docs/MATH.md) (every formula: BM25, RRF, hashed-dense cosine,
> MMR, authority scoring, NDCG/MRR, composite score) ·
> [`docs/EVALUATION.md`](docs/EVALUATION.md) (how to test/validate, the exact
> tasks + gold specs + graders) ·
> **[`docs/FINDINGS.md`](docs/FINDINGS.md)** (the honest results: does it actually
> help? — Codex, OpenAI up to gpt-5.5, hermes-agent, multi-domain) ·
> [`docs/logs/`](docs/logs) (captured run logs).

## Results at a glance

Does context engineering actually help downstream? We tested real agents
(Codex, OpenAI gpt-4.1-nano → **gpt-5.5**, hermes-agent). Multi-domain, repeated,
on gpt-5.5 (4 domains × 5 archetypes × 3 repeats, 120 distractors):

| context | accuracy | real secret leaks | avg tokens | cost |
| --- | --- | --- | --- | --- |
| dump-everything | 0.95 | 0 | 8,644 | $0.251 |
| naive top-k | 0.88 | 3 | 287 | $0.118 |
| **engineered** | **1.00** | **0** | 545 | **$0.084** |

The honest one-paragraph version: a strong model handles *reasoning*-hard tasks
by itself (parity there), but it **cannot** reason around the defects context
engineering removes — **secrets it may disclose, contradictions it can't
resolve, noise that dilutes the answer** — and engineered does the task in **far
fewer, cleaner tokens at lower cost** (clean context ⇒ the model reasons less).
Full story, caveats, and a measurement bug we caught: [`docs/FINDINGS.md`](docs/FINDINGS.md).

---

## What is context engineering?

An LLM's answer is only as good as the context it is given. *Prompt
engineering* tweaks wording; *context engineering* decides **what evidence,
rules, and structure** go into the window in the first place — which documents,
which version, in what order, under which budget, with what citations, and
excluding what noise.

This framework makes that process:

- **Reusable** — a *skill* is a named recipe, not a one-off prompt.
- **Composable** — operators are small functions chained into a pipeline.
- **Governed** — rules constrain retrieval, formatting, and permissions.
- **Measurable** — evaluators score every produced context.
- **Explainable** — every build emits a full trace of *why* each decision was made.

### The abstractions, mapped to code

| Symbol | Meaning | In this repo |
| ------ | ------- | ------------ |
| `x`   | the task | the `task` string passed to `build_context` |
| `s`   | the skill (a context recipe) | `models.Skill`, chosen by `SkillRegistry.select_skill` |
| `ρ_s` | static resource + rule pool for the skill | `ResourceRegistry` (`K_s`) + `RuleRegistry` (`R_s`) |
| `F_s` | the dynamic operator pipeline | `skill.operators`, run by `ContextEngine.build_context` via `OperatorRegistry` |
| `U`   | user/session preferences | `user_prefs` argument (e.g. budget override) |
| `B`   | token budget | `skill.default_budget_tokens` or an override |
| `c`   | the final context package | `models.ContextPackage` |

Three distinctions the code makes deliberately visible:

- **A skill is not the answer.** It is a *reusable context-construction recipe*
  (which operators to run, which rule/resource scopes to draw from, what budget).
- **`ρ_s` is not the final prompt.** It is the static *pool* of resources and
  rules available to the skill; operators select from it.
- **`F_s` is not one function.** It is an ordered pipeline of operators that
  select, transform, rank, filter, compress, and format.

---

## Project structure

```
context_engineering/
    models.py          # dataclasses: Rule, Skill, Resource, ContextItem,
                       #   ContextPackage, EvaluationResult, ExperimentResult, PipelineState
    registries.py      # SkillRegistry, RuleRegistry, ResourceRegistry,
                       #   OperatorRegistry, EvaluatorRegistry (+ TF-IDF search)
    operators.py       # dynamic operators = the building blocks of F_s
    evaluators.py      # scoring functions for a ContextPackage
    engine.py          # ContextEngine: selects a skill, runs its pipeline
    loaders.py         # load example JSON into a wired-up engine
    cli.py             # command-line interface
    run_experiments.py # build + evaluate every example task, save JSON
    compare_skills.py  # meta experiment: compare skills on one domain
    examples/
        rag_doc_qa/         resources|rules|skills|tasks|gold.json
        coding_bugfix/      resources|rules|skills|tasks|gold.json
        paper_explanation/  resources|rules|skills|tasks|gold.json
        compliance_policy/  resources|rules|skills|tasks|gold.json
tests/                 # pytest unit + integration tests
```

---

## Quick start

```bash
# (optional) create a venv, then:
pip install -r requirements.txt      # only pytest; the framework itself has no deps

# run the tests
pytest

# build a context package for a single task
python -m context_engineering.cli build-context \
    --task "What is the refund policy?" --domain rag_doc_qa

# run every example task across all four domains and save results
python -m context_engineering.run_experiments --out results.json

# compare skills within one domain (the meta experiment)
python -m context_engineering.compare_skills --domain rag_doc_qa

# inspect why a task's context was built the way it was
python -m context_engineering.cli inspect-trace \
    --result results.json --task-id rag_task_refund_policy
```

### CLI commands

There are **two ways** to run the commands — no separate install needed for the
first:

**A. Run in place (no install)** — from the repo root, use the module form:

```bash
python -m context_engineering.cli list-skills
python -m context_engineering.cli build-context --task "What is the refund policy?" --domain rag_doc_qa
```

**B. Install the `context-eng` console command** — `pip install -e .` reads
`[project.scripts]` in `pyproject.toml` and puts a `context-eng` executable on
your PATH:

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -e .                                    # creates the `context-eng` command
context-eng list-skills                             # now available directly
```

Either way, the subcommands are identical:

| Command | Purpose |
| ------- | ------- |
| `list-skills` | list registered skills and their operator pipelines |
| `list-rules` | list registered rules with scope/status/priority |
| `build-context --task "..." [--domain D] [--skill S]` | build and print a context package |
| `run-experiments [--out results.json]` | build + evaluate all example tasks |
| `compare-skills --domain D` | meta comparison of skills on one domain |
| `inspect-trace --result results.json --task-id ID` | print the build trace for a task |

So `context-eng list-skills` (installed) ≡ `python -m context_engineering.cli list-skills` (in place).
The production pipeline's benchmark and A/B are separate module entry points:
`python -m context_engineering.pipeline.benchmark` and
`python -m context_engineering.pipeline.agent_ab`.

---

## The four demo domains

| Domain | Example task | What the context build demonstrates |
| ------ | ------------ | ----------------------------------- |
| **rag_doc_qa** | *"What is the refund policy?"* | version filtering (drop stale policy), citations, evidence-only answers |
| **coding_bugfix** | *"Fix refund bug for UPI payment."* | stack-trace-driven file boosting, include failing test, exclude unrelated files, preserve public API |
| **paper_explanation** | *"Explain the equation and method in this paper."* | equation extraction, method + limitations sections, symbol-definition rule |
| **compliance_policy** | *"Can a developer deploy to production without approval?"* | latest approved policy, approval matrix, cited sections, stale version excluded |

Each domain folder ships `resources.json`, `rules.json`, `skills.json`,
`tasks.json`, and `gold.json` (expected resources + required rules per task).

### Example output (`build-context`)

```
Selected skill: rag_doc_qa_v2

Selected rules:
  - rag.answer_only_from_evidence
  - rag.require_citations
  - rag.prefer_latest_version
  ...

Selected resources:
  - refund_policy_2026 (score 0.333) : lexical match score=0.333
  - faq_refunds (score 0.199) : lexical match score=0.199

=== TASK ===
What is the refund policy?

=== RULES ===
- (rag.answer_only_from_evidence) Answer only from cited evidence; ...
- (rag.require_citations) Include citations (source id and section) ...

=== EVIDENCE ===
[refund_policy_2026 §2.1] Refund Policy (v2.0)
Refund Policy. Customers may request a refund within 30 days of purchase. ...

[faq_refunds §1.3] FAQ: Refunds
How long does a refund take under the refund policy? ...

=== EXCLUDED (STALE) ===
- refund_policy_2024: stale: superseded by refund_policy_2026 in group 'refund_policy'

[token estimate: 282 / budget 700]
```

Every build also records a machine-readable **trace** (skill selected, rules
selected, candidate scores, stale filtering, compression, final token
estimate), viewable with `inspect-trace`.

---

## The meta experiment (inner vs outer loop)

`compare_skills.py` runs several skills over the *same* domain tasks:

- `rag_doc_qa_v1` — naive: top‑3 lexical retrieval, no version filter, no citations
- `rag_doc_qa_v2` — curated: retrieve, type‑filter, drop stale, rerank, cite
- `rag_doc_qa_v3` — curated **+** a counter-evidence rule

```
$ python -m context_engineering.compare_skills --domain rag_doc_qa

skill_id                avg_score  budget_pass  gold_recall  citation_pass
--------------------------------------------------------------------------
rag_doc_qa_v2                 1.0          1.0          1.0            1.0
rag_doc_qa_v3                 1.0          1.0          1.0            1.0
rag_doc_qa_v1                 0.8          1.0          1.0            1.0

Best skill (outer loop): rag_doc_qa_v2
```

`v1` scores lower because, lacking a recency filter, it leaks the **stale**
2024 refund policy into the context. This is the point:

- **Inner loop** — for a single task, which context (skill) scored best?
- **Outer loop** — across all tasks, which skill *generalizes* best?

---

## Production pipeline & ablation benchmark

The MVP above uses ~12 operators to prove the architecture. The
`context_engineering/pipeline/` package is the **production-shaped** layer: a
stable, pluggable operator interface and the full stage pipeline

```
task → route → constrain → plan → scope → rewrite → retrieve(hybrid) →
permission-filter → authority/freshness → conflict → rerank → diversify →
select → compress → budget → order → format → validate → telemetry
```

Every stage is an `Operator` with a stable `name`/`version`/`run` contract and a
local, offline, deterministic default — plus a marked seam where a vendor/model
plugs in (Cohere/BGE rerankers, sentence-transformers/Qdrant embeddings, an LLM
router). No paid API is required to run any of it.

- **Hybrid retrieval:** BM25 (sparse) + a local hashed-n-gram dense vector,
  fused with **Reciprocal Rank Fusion**. Set `dense=False` to ablate.
- **Real reranker:** a local listwise cross-encoder *stub* (query coverage +
  idf mass + proximity). Swap `CrossEncoderReranker._run` for a hosted reranker.
- **Deterministic security:** `PermissionFilter` (ACL/tenant/confidentiality)
  and `ContextValidator` (budget, citations, staleness, **secret scan**) — the
  checks you must never delegate to an LLM.

### Add a production operator (fully pluggable)

```python
from context_engineering.pipeline import Operator, default_registry, build_pipeline

class MyReranker(Operator):
    name, version = "my_reranker", "1.0"
    def _run(self, state):
        # reorder state.candidates however you like ...
        state.log(self.name, self.version, "reranked")
        return state

reg = default_registry()
reg.register("my_reranker", MyReranker)          # one line to plug in
pipe = build_pipeline(reg, ["task_router", "skill_planner",
                            "hybrid_retriever", "my_reranker",
                            ("evidence_selector", {"top_k": 4}), "xml_formatter"])
```

The base `Operator.run` wrapper times each stage, records `OperatorMetrics`, and
**never silently fails** — an exception is captured into the state's
`errors`/`warnings`/`trace` and the pipeline continues.

### Measure it — the non-interactive benchmark

`pipeline/benchmark.py` runs several pipeline configurations (a naive baseline,
the full production stack, and leave-one-out ablations) over every example task
and reports **context-quality metrics** — because if the context is wrong the
model is already doomed. It grades the *context*, not the answer.

```bash
python -m context_engineering.pipeline.benchmark                       # all domains
python -m context_engineering.pipeline.benchmark --domain rag_doc_qa
python -m context_engineering.pipeline.benchmark --pipelines naive_baseline,full_production --out bench.json
```

**Downstream evaluation (needs an agent/model key):**

| command | what it does |
| --- | --- |
| `python -m context_engineering.pipeline.agent_ab --backend {codex,openai,hermes}` | A/B one context vs another through a real agent |
| `python -m context_engineering.pipeline.stress_ab --model gpt-4o-mini --suite all` | single-domain stress: dump / naive / engineered, with token+cost+precision metrics |
| `python -m context_engineering.pipeline.multi_domain --model gpt-5.5 --repeats 3` | 4-domain repeated stress (the [FINDINGS](docs/FINDINGS.md) run) |

The `openai`/`hermes` backends read the key from `~/.oai_key` or `$OPENAI_API_KEY`
— nothing is hard-coded. See [`docs/FINDINGS.md`](docs/FINDINGS.md) for results.

Metrics: `retrieval_recall@k`, `ranked_ndcg@k`, `ranked_mrr`, `final_recall`,
`final_precision`, `citation_precision`, `stale_leak_rate` (lower better),
`noise_rate` (lower better), `required_rule_coverage`, `budget_ok`,
`validation_pass`, `latency_ms`.

Example output (means across all example tasks):

```
metric                        naive       FULL    -rerank     -dense  -conflict      -perm   -compbud
-----------------------------------------------------------------------------------------------------
ranked_ndcg@k                 0.928      0.989      0.883      0.953      0.973      0.973      0.989
citation_precision            0.000      1.000      1.000      1.000      1.000      1.000      1.000
stale_leak_rate               0.600      0.000      0.000      0.000      0.400      0.000      0.000
validation_pass               0.000      1.000      1.000      1.000      1.000      0.800      1.000

OPERATOR CONTRIBUTION (full minus variant; positive = operator helps)
-reranker      ranked_ndcg@k +0.106   ...        # reranking is worth +0.106 NDCG
-conflict      stale_leak_rate +0.400 ...        # conflict detector kills stale leakage
-permission    validation_pass +0.200 ...        # removing ACL leaks a confidential secret
```

Read it as: the **naive baseline** cites nothing (`citation_precision 0`),
leaks the stale 2024 policy (`stale_leak_rate 0.6`), and fails validation; the
**conflict detector** is what removes stale leakage; the **permission filter**
is what keeps a confidential secret out of the window (its removal drops
`validation_pass` because the secret scanner fires). This is the inner/outer
loop made quantitative — you can see each operator's marginal contribution.

## Extending the system

The engine contains **no domain logic** — everything is data + registered
callables, so extension never touches `engine.py`.

### Add a new domain

1. Create `context_engineering/examples/<your_domain>/` with the five JSON
   files: `resources.json`, `rules.json`, `skills.json`, `tasks.json`,
   `gold.json`.
2. Give at least one skill `"task_types": ["<your_domain>"]` and
   `"metadata": {"default": true}`.
3. (Optional) add keywords to `operators._DOMAIN_KEYWORDS` so the task
   classifier can route to it without an explicit `--domain` hint.
4. Run `python -m context_engineering.run_experiments` — it auto-discovers any
   domain folder containing a `skills.json`.

### Add a new skill

Add an entry to a domain's `skills.json`. A skill is just:

```json
{
  "id": "my_skill",
  "task_types": ["rag_doc_qa"],
  "rule_scopes": ["rag"],
  "resource_types": ["policy", "document"],
  "operators": ["rule_selector", "lexical_retrieval", "relevance_rank", "context_formatter"],
  "default_budget_tokens": 700,
  "evaluators": ["budget_evaluator", "gold_resource_recall_evaluator"],
  "metadata": {"top_k": 3}
}
```

The `operators` list *is* the pipeline `F_s`. Reorder or swap operators to
change the recipe.

### Add a new operator

Write a function `(state: PipelineState) -> PipelineState` in `operators.py`,
append a `state.log(...)` for traceability, and register it:

```python
def my_operator(state):
    # read state.candidates / state.selected_items, transform them ...
    state.log("my_operator", detail="...")
    return state

# in register_default_operators():
registry.register_operator("my_operator", my_operator)
```

Then reference `"my_operator"` by name in any skill's `operators` list.

### Add a new evaluator

Write a function `(package: ContextPackage, spec: dict) -> EvaluationResult` in
`evaluators.py`, register it in `register_default_evaluators`, and list its
name in a skill's `evaluators`. `spec` is the task's gold entry from
`gold.json`.

---

## Design notes

- **Deterministic by construction.** Retrieval is a pure TF‑IDF‑style scorer;
  token estimation is `len(text)/4`; ranking ties break on id. Tests are stable.
- **No external services.** Zero runtime dependencies; standard library only.
  Swapping in real embeddings, a vector DB, or an LLM is a matter of adding a
  new operator — the interfaces do not change.
- **Traceability first.** The goal is not only to *produce* context but to
  *understand why* it was produced; every operator appends to the trace.

## Running the tests

```bash
pytest                 # 33 deterministic unit + integration tests
pytest tests/test_integration_domains.py -v
```

Integration tests assert the headline behaviors: RAG picks the latest policy
and cites it; the coding task includes the failing test, stack trace, and
relevant source while excluding unrelated files; the paper task includes the
equation, method, and limitations; the compliance task excludes the stale
policy and includes the approval matrix.
