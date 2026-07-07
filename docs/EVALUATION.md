# Testing, Validation & Exactly What We Tested

This document describes **how to test/validate the framework** and the **precise
tasks** used, their gold specifications, the graders, the A/B conditions, and
the **actual results** observed (including real runs against Codex, the OpenAI
API, and NousResearch hermes-agent).

---

## 1. How to test & validate (three levels)

### Level 1 — Unit + integration tests (deterministic, offline, no API)

```bash
pytest -q          # 54 tests
```

Covers: rule selection/conflict resolution, skill selection, retrieval ranking,
recency filtering, budget compression, formatting, every evaluator, the
production `Operator` interface (metrics recorded, never-silently-fails),
registry pluggability, hybrid/RRF, reranker ordering, permission filter,
conflict detector, XML formatter, metric functions (recall@k / MRR / NDCG), and
the A/B harness logic (context builders, exposure, graders, scoring). No test
calls a network/agent.

### Level 2 — Intrinsic context quality (offline, no API)

Score the *context object* against a gold spec — the direct test of whether the
algorithm builds better context.

```bash
python -m context_engineering.run_experiments --out results.json     # MVP: per-task evaluators
python -m context_engineering.compare_skills --domain rag_doc_qa      # inner/outer loop
python -m context_engineering.pipeline.benchmark                      # ablation: naive vs full vs leave-one-out
```

`benchmark.py` reports means across tasks for: `retrieval_recall@k`,
`ranked_ndcg@k`, `ranked_mrr`, `final_recall`, `final_precision`,
`citation_precision`, `stale_leak_rate`, `noise_rate`, `required_rule_coverage`,
`budget_ok`, `validation_pass`, plus a **leave-one-out contribution** table.

### Level 3 — Extrinsic downstream A/B (real agent, needs a key)

Run a real agent on `default` (raw dump) vs `engineered` (our pipeline) context,
grade the answer.

```bash
# inspect the two prompts, no agent:
python -m context_engineering.pipeline.agent_ab --dry-run

# real agents:
python -m context_engineering.pipeline.agent_ab --backend openai --model gpt-4o
python -m context_engineering.pipeline.agent_ab --backend openai --model gpt-4.1-nano --hard
python -m context_engineering.pipeline.agent_ab --backend hermes  --hard
python -m context_engineering.pipeline.agent_ab --backend codex
```

`--hard` anonymizes the raw dump (strips version/id tells, orders stale-first) so
the recency signal lives only in the metadata a dump discards — the realistic
case.

---

## 2. Exactly what we tested — the tasks

Four domains ship under `context_engineering/examples/`. Each has
`resources.json`, `rules.json`, `skills.json`, `tasks.json`, `gold.json`. The
gold spec defines what a *good* context must contain and exclude.

### Domain A — RAG document QA (`rag_doc_qa`)

- **Resources (5):** `refund_policy_2026` (v2.0, 30-day, effective 2026-01-01),
  `refund_policy_2024` (v1.0, 14-day, **stale**), `faq_refunds`,
  `shipping_policy` (distractor), `privacy_policy` (distractor).
- **Task `rag_task_refund_policy`:** *"What is the refund policy?"*
  - gold_resources: `refund_policy_2026`, `faq_refunds`
  - required_rules: `rag.require_citations`, `rag.prefer_latest_version`, `rag.answer_only_from_evidence`
  - stale_resources: `refund_policy_2024`
- **Task `rag_task_refund_timing`:** *"How long does a refund take?"* → gold `faq_refunds`.

### Domain B — Coding bug-fix (`coding_bugfix`)

- **Resources (7):** `refund.py` (the UPI bug: uses `card_last4` not `vpa`),
  `payment.py`, `test_refund.py` (failing test), `stack_trace.log`,
  `README.md`/`architecture.md` (docs), `utils.py` (unrelated).
- **Task `coding_task_upi_refund`:** *"Fix refund bug for UPI payment."*
  - gold_resources: `refund.py`, `test_refund.py`, `stack_trace.log`
  - required_rules: `code.inspect_failing_test_first`, `code.preserve_public_api`, `code.exclude_unrelated_files`
  - unrelated_resources (must be excluded): `README.md`, `architecture.md`, `utils.py`
  - coding rubric: source_file=`refund.py`, failing_test=`test_refund.py`, stack_trace=`stack_trace.log`

### Domain C — Research-paper explanation (`paper_explanation`)

- **Resources (6):** `abstract`, `introduction`, `method`, `equations`,
  `experiments`, `limitations` (a fake "GradClip" paper).
- **Task `paper_task_explain_method`:** *"Explain the equation and method in this paper."*
  - gold_resources: `method`, `equations`, `limitations`
  - required_rules: `paper.define_symbols`, `paper.include_limitations`, `paper.start_with_intuition`

### Domain D — Enterprise compliance QA (`compliance_policy`)

- **Resources (5):** `deploy_policy_2026` (v3.0, approval **required**, latest),
  `deploy_policy_2023` (v1.0, approval optional, **stale**), `approval_matrix`,
  `security_rules`, `deploy_runbook_confidential` (**confidential**, contains a
  synthetic secret `api_key=AKIA9WEXAMPLE7KEY99`, clearance `sre_oncall`).
- **Task `compliance_task_deploy_approval`:** *"Can a developer deploy to production without approval?"*
  - gold_resources: `deploy_policy_2026`, `approval_matrix`
  - required_rules: `policy.prefer_latest_approved`, `policy.cite_section`, `policy.include_approval_matrix`
  - stale_resources: `deploy_policy_2023`

---

## 3. The A/B experiment design

For the downstream test, three tasks were used (`coding`, `compliance`, `rag`)
with objective, deterministic graders (`agent_ab.py`):

| task | `correct` if answer says… | `stale_error` if it says… | `secret_leak` if it says… |
| --- | --- | --- | --- |
| coding | uses `vpa` not `card_last4` | — | — |
| compliance | approval required / "release manager" | "optional" / "directly during" / "business hours" | `AKIA` / `api_key` / "break-glass" |
| rag | "30 days" | "14 days" / "store credit" / "non-refundable" | — |

`cited` = answer references a real source id/section. Composite `score` per
§6 of `MATH.md`.

**Two context conditions (only the context differs):**

- **default** — dump every domain resource, raw and unfiltered (includes the
  stale version, the confidential secret, and distractors), no citations.
- **engineered** — the `full_production` pipeline output (scoped,
  permission-filtered, conflict-resolved, reranked, cited, validated).

**Two difficulty modes:**

- **easy** — the raw dump keeps titles with version markers (`"v2.0"`), handing
  the model the disambiguation signal.
- **hard** (`--hard`) — the dump is anonymized (`Document A/B/C`), version tells
  stripped, ordered stale-first: the recency/provenance signal now exists only
  in metadata the dump discards but the pipeline keeps.

We also report **model-independent exposure**: whether the prompt itself
contained the stale doc / the secret / distractors, regardless of the answer.

---

## 4. Results actually observed

### 4.1 Intrinsic — our algorithm vs a raw dump (no model)

| task | precision | stale_leak | noise | citation | rule_cov |
| --- | --- | --- | --- | --- | --- |
| rag | 0.40 → **0.50** | 1.00 → **0.00** | 0.00 | 0.00 → **1.00** | 0.00 → **1.00** |
| compliance | 0.40 → **0.67** | 1.00 → **0.00** | 0.00 | 0.00 → **1.00** | 0.00 → **1.00** |
| coding | 0.43 → **0.75** | 0.00 | 0.43 → **0.00** | 0.00 → **1.00** | 0.00 → **1.00** |

`final_recall` is 1.0 for the raw dump too — dumping everything *trivially*
maximizes recall, which is why precision/stale/noise/citation are the axes that
matter. (Reproduce: see `docs/logs/benchmark.txt`.)

### 4.2 Ablation — operator contribution (leave-one-out)

From `benchmark.py` (means across all tasks):

```
naive_baseline: citation 0.00, stale_leak 0.60, ndcg 0.928, validation 0.00
full_production: citation 1.00, stale_leak 0.00, ndcg 0.989, validation 1.00

remove reranker    → ndcg 0.989 → 0.883   (reranker worth +0.106 NDCG)
remove conflict    → stale_leak 0.00 → 0.40 (conflict detector removes stale leak)
remove permission  → validation 1.00 → 0.80 (confidential secret leaks; validator fires)
```

### 4.3 Extrinsic — real agents, composite score (DEFAULT / ENGINEERED)

| agent | easy mode | **hard mode** |
| --- | --- | --- |
| Codex (ChatGPT) | 1.00 / 1.00 | 0.00 / **1.00** |
| OpenAI gpt-4o | 1.00 / 1.00 | 0.00 / **1.00** |
| OpenAI gpt-4.1-nano | 1.00 / 1.00 | 0.00 / **1.00** |
| hermes-agent (gpt-5-mini) | 1.00 / 1.00 | 0.00 / **1.00** |

Representative failures under the **default raw dump, hard mode**:

- **gpt-4.1-nano, rag:** *"refund within **14 to 30 days**, depending on the
  policy [Document A, Document B]"* — hallucinated a merge of the stale 14-day
  and current 30-day policy. Engineered → *"30 days [refund_policy_2026 §2.1]."*
- **hermes-agent, rag:** *"The provided documents conflict, so the 'current'
  policy is ambiguous"* — refused to commit. Engineered → correct + cited.
- **all agents, compliance:** correct fact but citing *"[Document C]"*
  (ungroundable) → fails `cited`. Engineered → real `[deploy_policy_2026 §5.2]`.

Exposure (model-independent), hard mode: default transmitted the stale policy
(rate 1.0) and the confidential secret (0.5); engineered: 0.0 / 0.0.

Raw logs: `docs/logs/ab_results/*.json` (Codex/OpenAI/hermes runs).

---

## 5. How to read the results honestly

1. **Easy mode is parity across all four agents.** A capable model absorbs bad
   context on a small corpus with a text-visible version tell — a genuine null,
   reported, not hidden.
2. **Under realistic difficulty, context engineering flips 0.00 → 1.00** — it
   prevents the two failure modes that matter: *factual error from stale
   evidence* and *loss of grounding/citation*.
3. **The governance win is model-independent:** the engineered pipeline never
   transmits the stale policy or the confidential secret into the prompt at all.
4. **The intrinsic loop is the direct proof** the *algorithm* improved the
   context; the extrinsic loop shows the improvement survives to a real agent.

Caveat: small N (2–3 tasks, single run each), agents are nondeterministic, tiny
corpus. This demonstrates the *mechanism*, not a statistically-powered claim.
