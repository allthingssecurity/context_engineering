# The Math of the Context-Engineering Framework

This document states, precisely, every formula the code uses. Nothing here is
hidden behind a hosted API — each function is a small, deterministic operation
you can reproduce by hand.

---

## 1. The core abstraction

For a task `x`, the system produces a context package `c`:

```
c = F_s(x, K_s, R_s, U, B)
```

| symbol | meaning | in code |
| --- | --- | --- |
| `x` | the task string | `state.task` |
| `s` | the selected skill (a recipe) | `SkillRegistry.select_skill` |
| `K_s` | static resources available to `s` | `ResourceRegistry` |
| `R_s` | rules selected for `s` | `RuleRegistry.select_rules` |
| `U` | user/session preferences (tenant, clearances, prefs) | `state.user_context` |
| `B` | token budget | `state.budget` |
| `F_s` | the ordered operator pipeline | `Pipeline.run` |
| `c` | the final context package | `ContextPackage` / formatted context |

`F_s` is a composition of operators `o_i`:

```
c = (o_n ∘ o_{n-1} ∘ … ∘ o_1)(state_0)
```

Each `o_i : State → State` reads and writes a shared `ContextBuildState`.

### Skill selection

Given a domain hint `d`, restrict to skills whose `task_types ∋ d`, prefer one
flagged default, else pick by keyword overlap:

```
s* = argmax_{s}  | tokens(x) ∩ tokens(id_s ∪ desc_s ∪ task_types_s) |
```

### Domain classification (when no hint given)

```
domain(x) = argmax_{d}  | tokens(x) ∩ keywords_d |
```

---

## 2. Tokenization

```
tokenize(t) = regex_findall( [a-z0-9]+ , lowercase(t) )
```

Token estimate (a stable stand-in for a real BPE tokenizer):

```
τ(text) = max(1, round(|text| / 4))        # ~4 chars per token
```

---

## 3. Retrieval scoring

### 3.1 TF-IDF (the simple `ResourceRegistry.search`)

Inverse document frequency over corpus of `N` resources, `df_t` = docs
containing term `t`:

```
idf_t = ln( 1 + N / (1 + df_t) )
```

Score of query `q` against document `d` (with `|d|` = token count):

```
score(q, d) = Σ_{t ∈ q, t ∈ d}  (tf_{t,d} / |d|) · idf_t   +   0.1 · |q ∩ title(d)|
```

### 3.2 BM25 (the production `HybridRetriever`, sparse arm)

With `k1 = 1.5`, `b = 0.75`, average doc length `avgdl`:

```
idf(t)  = ln( 1 + (N − df_t + 0.5) / (df_t + 0.5) )

BM25(q,d) = Σ_{t ∈ q}  idf(t) · [ f(t,d) · (k1 + 1) ]
                                 ─────────────────────────────────────
                                 f(t,d) + k1 · (1 − b + b · |d| / avgdl)
```

### 3.3 Local dense embedding (the dense arm — no API)

A signed, hashed character-n-gram vector of dimension `D = 256`, `n = 3`.
For each n-gram `g` of the text, with a hash `h(g)`:

```
v[ h(g) mod D ]  +=  sign(g),   sign(g) = +1 if bit8(h(g)) else −1
v  ←  v / ‖v‖₂
```

Similarity is cosine:

```
sim(a, b) = Σ_i a_i · b_i          (vectors are unit-normalized)
```

This is a crude but genuine second signal; swapping in a real encoder changes
only `_hash_vector`, not the interface.

### 3.4 Reciprocal Rank Fusion (combining sparse + dense + multi-query)

For a set of rankings `R` (each a list ordered best-first), with `k = 60`:

```
RRF(d) = Σ_{r ∈ R}  1 / ( k + rank_r(d) + 1 )
```

Multi-query expansion feeds several queries (original, keyword, synonym,
symbol); each query × each arm contributes one ranking to `R`.

---

## 4. Post-retrieval scoring

### 4.1 Authority + freshness

For a candidate with metadata, weight `w = 0.15`:

```
a = 1[source_authority = official]
  + 0.5 · 1[approved]
  + max(0, (year(effective_date) − 2020) / 10)

score  ←  score + w · a
```

### 4.2 Cross-encoder reranker (local listwise stub)

Let `q = tokens(x)`, `d = tokens(candidate)`, `M = q ∩ d`:

```
coverage   = |M| / |q|
idf_mass   = Σ_{t ∈ M} idf(t)
title      = |q ∩ tokens(title)|
proximity  = #{ i : (q_i, q_{i+1}) appears adjacently in d }

rerank(x,d) = 2·coverage + idf_mass + 0.3·title + 0.5·proximity
```

Candidates are sorted by `(rerank, authority, id)` descending. Swap this scorer
for Cohere/BGE/Voyage; the `run` interface is unchanged.

### 4.3 MMR diversity (Maximal Marginal Relevance)

Greedily build the ordered set `S` with `λ = 0.7`, `sim` = dense cosine:

```
d* = argmax_{d ∉ S}  [ λ · score(d)  −  (1 − λ) · max_{s ∈ S} sim(d, s) ]
```

### 4.4 Conflict resolution (version/recency)

Group candidates by `metadata.group`. Within a group, keep the winner:

```
winner = argmax_{d ∈ group}  ( effective_date(d), version(d) )     # lexicographic, desc
```

Resolution order for policy status: `latest approved > older approved > draft > archived`.
Losers are marked `stale` and (by default) dropped; a conflict warning is emitted.

---

## 5. Budget & compression

Token estimate per item uses `τ` from §2. The budget allocator drops the
lowest-ranked selected items until the evidence fits:

```
while  Σ_i τ(content_i)  >  B :   drop lowest-ranked item
```

Query-aware extractive compression keeps the first sentence plus any sentence
overlapping the query, up to `max_sentences`:

```
keep(sentences) = { s_0 } ∪ { s : tokens(s) ∩ tokens(x) ≠ ∅ }   truncated to max_sentences
```

Extractive (never abstractive) — safe for policy/legal/code where exact wording
matters.

---

## 6. Evaluation metrics

Let `G` = gold resource set, `order` = ranked ids, `sel` = selected ids, `k`
the cutoff (default 5).

### Retrieval quality

```
recall@k    = |order[:k] ∩ G| / |G|
precision@k = |order[:k] ∩ G| / |order[:k]|
MRR         = 1 / (rank of first gold in order),   0 if none

DCG@k   = Σ_{i=0}^{k-1}  rel_i / log2(i + 2),      rel_i = 1[order_i ∈ G]
IDCG@k  = DCG of the ideal ordering (all gold first)
NDCG@k  = DCG@k / IDCG@k
```

### Selection & safety quality

```
final_recall       = |sel ∩ G| / |G|
final_precision    = |sel ∩ G| / |sel|
citation_precision = |{ i ∈ sel : "[i" appears in context }| / |sel|
stale_leak_rate    = |sel ∩ Stale| / |Stale|         (0 if any stale is marked "conflict")
noise_rate         = |sel ∩ Unrelated| / |sel|
diversity          = |{ group(i) : i ∈ sel }| / |sel|
required_rule_cov  = |Required ∩ selected_rules| / |Required|
budget_ok          = 1[ τ(context) ≤ B ]
```

### Downstream A/B composite score

For an agent's answer graded on booleans `correct`, `cited`, `stale_error`,
`secret_leak`, `distracted`:

```
good = correct ∧ cited
bad  = stale_error ∨ secret_leak ∨ distracted

score = 1.0   if good ∧ ¬bad
        0.5   if good ∧ bad
        0.0   otherwise
```

---

## 7. Two measurement loops

- **Intrinsic** (§6, no model): does `F_s` build *better context*? Measured by
  scoring `c` against the gold spec. Leave-one-out over operators attributes
  each metric gain to a specific operator.
- **Extrinsic** (A/B, §6 composite): does better `c` change the *answer*?
  Measured by running a real agent (Codex / OpenAI / hermes-agent) on
  `default` vs `engineered` context and grading the output.

The intrinsic loop is the direct test of the algorithm; the extrinsic loop
tests whether the improvement survives to a downstream task.
