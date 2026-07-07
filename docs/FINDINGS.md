# Findings: does context engineering actually help?

This document reports what we measured when we put the framework in front of
**real agents** (Codex, the OpenAI API from gpt-4.1-nano to gpt-5.5, and
NousResearch hermes-agent). It is written to be honest about **where context
engineering helps, where it doesn't, and where our own pipeline has limits.**

TL;DR:

> Context engineering's downstream value is **not** making a smart model smarter
> on reasoning tasks — a strong model handles those itself. It's the defects a
> model **cannot** reason around: **secrets it may disclose, contradictions it
> can't resolve, and noise that dilutes the answer** — plus **doing the task in
> far fewer, cleaner tokens (lower cost)**. Those wins persist up to gpt-5.5.

---

## How we measured (two loops)

1. **Intrinsic** (`pipeline/benchmark.py`, no model) — score the *context object*
   against a gold spec: recall, NDCG, precision, citation, stale-leak. Directly
   tests whether the pipeline builds better context.
2. **Extrinsic** (`pipeline/agent_ab.py`, `stress_ab.py`, `multi_domain.py`) —
   run a real agent on `default` vs `engineered` context and grade the answer.

Only the **context** differs between conditions; the model, task, and
instructions are identical, so any difference is attributable to context
construction.

---

## Experiment 1 — a capable agent over a tiny clean corpus (hermes-agent)

We let hermes-agent do its **own** retrieval (its `read_file`/`search_files`
tools) over a folder of policy docs, vs handing it our pipeline's context.

**Result: parity.** hermes-native picked the latest policy, cited it, and did
not leak — on its own. Over 5 clean files, a capable agent self-serves; our
external pipeline adds little.

> Lesson: context engineering is not "ours beats the agent." Over a small clean
> corpus a good agent needs no help.

---

## Experiment 2 — a model over a noisy corpus (stress_ab.py)

A larger corpus (up to 150 confusable distractors) with varied tasks, feeding
the **same** model **standard top-k retrieval (naive)** vs **our pipeline
(engineered)**.

Headline (gpt-4o-mini and gpt-4o, 150 distractors):

| model | naive | engineered |
| --- | --- | --- |
| gpt-4o-mini | 0.50 correct, 1 leak | 1.00 correct, 0 leaks |
| gpt-4o | 0.50 correct, 1 leak | 0.83 correct, 0 leaks |

**Retrieval recall ≠ good context.** Naive kept the gold doc in context 83% of
the time yet answered wrong half the time — because the context *also* held
noise, a stale contradiction, and a secret. The failures:

- **version conflict** — naive kept both the stale and current value; the model
  couldn't tell which was current. Engineered's `conflict_detector` dropped the
  stale one.
- **dilution** — the gold fact was buried among distractors with confusable
  numbers; engineered's reranker surfaced it.
- **ACL** — naive retrieved a confidential doc; engineered's `permission_filter`
  removed it.

---

## Experiment 3 — gpt-5.5 on harder scenarios (stress_ab.py --suite all)

Adversarial distractors, numeric multi-hop, no-answer traps, exception/negation,
and a temporal-as-of case. Two kinds of "hard" behaved oppositely:

- **Reasoning-hard** (adversarial, numeric, exception, no-answer): gpt-5.5 got
  them right under **both** naive and engineered (0.80/0.80). A frontier model
  reasons through noisy context by itself.
- **Hygiene-hard** (version conflict, ACL): naive **0.50 with a leak**,
  engineered **1.00 with no leak** — even at gpt-5.5. A secret or a contradiction
  *in the prompt* is a property of the context, not the model.

**Honest limitation:** `temporal_as_of` ("as of March 2025, what was the rate
limit?") fails **both** — and engineered makes it *worse*, because its recency
filter drops the historically-correct older version. "Prefer latest" ≠ "prefer
correct-as-of-date." A real fix needs temporal-aware selection.

---

## Experiment 4 — efficiency & quality metrics (the token story)

Adding a third condition, **dump-everything** (no retrieval), and measuring
tokens/cost/precision (gpt-5.5, 150 distractors):

| condition | accuracy | in-tokens | cost | signal density |
| --- | --- | --- | --- | --- |
| dump-everything | 0.82 | 14,519 | $0.219 | 0.00 |
| naive top-k | 0.64 | 369 | $0.031 | 0.12 |
| engineered | 0.82 | 564 | $0.024 | 0.07 |

- **Engineered = dump accuracy at ~1/26th the tokens and ~1/9th the cost.**
- vs naive, engineered spends a few more tokens on scaffolding (rules, citations,
  XML) — which *lowers* raw signal density but *buys* accuracy and safety. Density
  is a lever: strip the scaffolding to raise it, at the cost of grounding.

---

## Experiment 5 — multi-domain, repeated, to be *sure* (multi_domain.py)

Four domains (payments, clinical, legal, security), the same five archetypes in
each, **cross-domain contamination** in the distractors, **3 repeats per cell**,
on **gpt-5.5**, 120 distractors.

| condition | accuracy | real secret leaks | avg in-tokens | total cost |
| --- | --- | --- | --- | --- |
| dump-everything | 0.95 | 0 | 8,644 | $0.251 |
| naive top-k | 0.88 | 3 | 287 | $0.118 |
| **engineered** | **1.00** | **0** | 545 | **$0.084** |

Per-archetype (naive → engineered): needle 1.00→1.00 · version 1.00→1.00 ·
no-answer 1.00→1.00 · **adversarial 0.67→1.00** · **ACL 0.75→1.00**.

Per-domain accuracy reached 0.93–1.00 for engineered in **all four** domains.

Three findings we are now confident in:

1. **Engineered ≥ naive in every domain**, driven by adversarial retrieval and
   governance; parity on tasks a strong model handles alone.
2. **Engineered is the cheapest** — $0.084 vs naive $0.118 — despite bigger
   prompts, because clean context makes the reasoning model **think less**.
   Naive's noisy context nearly doubled gpt-5.5's reasoning (output) tokens.
   Bad context isn't just less accurate, it's more expensive.
3. **Model safety is uneven and unreliable for governance.** gpt-5.5 *refused* to
   output API-key-style secrets (0 leaks in pay/sec/med) but **disclosed the
   confidential legal side-letter 3/3 times** — it didn't recognize a business
   secret as a credential. Engineered leaked 0/12 because the secret never
   entered the context. You cannot rely on model safety; you can rely on
   deterministic filtering.

---

## A measurement bug we caught (why rigor matters)

Our first leak grader matched the **procedure name** "break-glass" (which is in
the question and in *safe* refusals like "the break-glass credential is not
available"). That inflated naive leaks to 10 and dropped ACL accuracy to 0.17.
Re-grading on the **actual secret token** gave the true numbers above (naive 3
real leaks, engineered 0). The grader is now fixed (`docs`-referenced in code)
and covered by `tests/test_stress_harness.py`. Lesson: verify anomalies against
the raw answers before trusting a table.

Another honest caveat: in Experiment 5 the `version_conflict` archetype was
**parity** because the doc titles said "current"/"old", handing naive the
signal. Version resolution is a genuine win only when currency lives in
metadata, not text (as in Experiment 2/3, where naive failed it).

---

## When to reach for context engineering

| situation | does it help? |
| --- | --- |
| Capable agent + small clean corpus | Little — the agent self-serves |
| Any model + noisy / contradictory / sensitive corpus | **Yes** — accuracy + governance |
| Plain LLM call with no retrieval of its own | **Yes** — it *is* the retrieval layer |
| Secrets / ACL / compliance in the corpus | **Yes, essential** — model safety is unreliable |
| Cost / latency sensitive, or large corpus | **Yes** — fewer, cleaner tokens; less reasoning |
| Reasoning-hard tasks a strong model handles | Parity — no answer-quality gain |
| Temporal "as-of" queries | **Not yet** — our recency filter is a known limitation |

Reproduce any of this with the commands in [`EVALUATION.md`](EVALUATION.md); raw
run logs are under [`logs/`](logs).
