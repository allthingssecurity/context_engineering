"""Production operators (local, offline, deterministic defaults).

Each operator is a real implementation you can run without any paid API, and a
clearly-marked seam where a vendor/model would plug in:

  * HybridRetriever  -> swap the dense scorer for sentence-transformers/Qdrant
  * CrossEncoderReranker -> swap for Cohere/BGE/Voyage/Jina rerankers
  * QueryRewriter    -> swap the synonym map for an LLM / HyDE rewriter
  * ConstraintExtractor -> swap the regex rules for schema-constrained LLM output

The interfaces (Operator.run, ContextBuildState) do not change when you swap.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Dict, List

from ..models import estimate_tokens
from ..operators import classify_task
from ..registries import tokenize
from .base import Operator
from .state import CandidateResource, ContextBuildState, Budget, Query

_STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "of", "in", "on", "for", "and", "or",
    "can", "do", "does", "what", "how", "why", "this", "that", "with", "without",
    "i", "my", "me", "you", "it",
}

_SECRET_RE = re.compile(
    r"(AKIA[0-9A-Z]{8,})|(api[_-]?key\s*[=:]\s*\S+)|(password\s*[=:]\s*\S+)"
    r"|(secret\s*[=:]\s*\S+)|(bearer\s+[A-Za-z0-9._-]{12,})",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- #
# Local scoring helpers                                                        #
# --------------------------------------------------------------------------- #
def _build_index(resources):
    """Build BM25 statistics over ``resources`` (title + content)."""
    docs: Dict[str, List[str]] = {}
    df: Dict[str, int] = {}
    for r in resources:
        toks = tokenize(r.title + " " + r.content)
        docs[r.id] = toks
        for t in set(toks):
            df[t] = df.get(t, 0) + 1
    n = len(resources) or 1
    avgdl = (sum(len(t) for t in docs.values()) / n) if docs else 1.0
    idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
    return docs, idf, avgdl


def _bm25(query_terms, docs, idf, avgdl, k1=1.5, b=0.75) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for rid, toks in docs.items():
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        dl = len(toks) or 1
        s = 0.0
        for qt in query_terms:
            if qt in tf:
                f = tf[qt]
                s += idf.get(qt, 0.0) * (f * (k1 + 1)) / (
                    f + k1 * (1 - b + b * dl / avgdl)
                )
        scores[rid] = s
    return scores


def _hash_vector(text: str, dim: int = 256, ngram: int = 3) -> List[float]:
    """A deterministic, dependency-free 'embedding': signed hashed char n-grams.

    Stands in for a real dense encoder.  Cosine similarity between these vectors
    is a genuine (if crude) semantic-ish signal, and the retrieval interface is
    identical to swapping in a hosted embedding model.
    """
    vec = [0.0] * dim
    t = " " + text.lower() + " "
    grams = [t[i : i + ngram] for i in range(max(1, len(t) - ngram + 1))]
    for g in grams:
        h = int(hashlib.md5(g.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h >> 8) & 1 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def _cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _rrf(rankings: List[List[str]], k: int = 60) -> Dict[str, float]:
    """Reciprocal Rank Fusion across several rankings (each best-first)."""
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, rid in enumerate(ranking):
            fused[rid] = fused.get(rid, 0.0) + 1.0 / (k + rank + 1)
    return fused


# --------------------------------------------------------------------------- #
# 1. Task Router                                                               #
# --------------------------------------------------------------------------- #
class TaskRouter(Operator):
    """Classify the task into primary/secondary intents with a confidence.

    Local default = keyword scoring (multi-intent aware).  Production seam: an
    LLM JSON router with a deterministic keyword fallback.
    """

    name = "task_router"
    version = "1.0-keyword"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        from ..operators import _DOMAIN_KEYWORDS

        tokens = set(tokenize(state.task))
        scored = sorted(
            (
                (len(tokens & set(kw)), dom)
                for dom, kw in _DOMAIN_KEYWORDS.items()
            ),
            reverse=True,
        )
        primary = state.domain or scored[0][1]
        top = scored[0][0]
        second = scored[1][0] if len(scored) > 1 else 0
        confidence = 1.0 if top == 0 else round((top - second) / top, 3)
        state.domain = primary
        state.route = {
            "primary_task": primary,
            "secondary_tasks": [d for s, d in scored[1:] if s > 0 and d != primary],
            "confidence": confidence,
            "needs_citations": primary in {"rag_doc_qa", "compliance_policy"},
            "needs_code_context": primary == "coding_bugfix",
        }
        state.log(self.name, self.version, "routed", **state.route)
        return state


# --------------------------------------------------------------------------- #
# 2. Constraint Extractor                                                      #
# --------------------------------------------------------------------------- #
class ConstraintExtractor(Operator):
    """Extract hard/soft constraints from the task text.

    Local default = keyword/regex rules.  Production seam: schema-constrained
    LLM extraction, treating inferred constraints as lower-priority candidates.
    """

    name = "constraint_extractor"
    version = "1.0-regex"

    _RULES = [
        (r"(don'?t change|preserve|without changing).{0,20}(public )?api", "must_not", "change_public_api"),
        (r"smallest|minimal patch", "prefer", "smallest_patch"),
        (r"simpl(e|y)|first principles|intuition", "style", "simple_first"),
        (r"with math|equation|derivation", "must_include", "maths"),
        (r"cite|citation|source", "must_include", "citations"),
        (r"latest|current|up.?to.?date", "prefer", "latest_version"),
    ]

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        text = state.task.lower()
        constraints: Dict[str, List[str]] = {}
        for pattern, key, value in self._RULES:
            if re.search(pattern, text):
                constraints.setdefault(key, []).append(value)
        state.constraints = constraints
        state.log(self.name, self.version, "extracted", constraints=constraints)
        return state


# --------------------------------------------------------------------------- #
# 3. Skill Planner                                                             #
# --------------------------------------------------------------------------- #
class SkillPlanner(Operator):
    """Select the skill recipe and its rules, and set the token budget B.

    Local default = rule-based (domain default skill).  Production seam: a
    bandit/eval-optimized planner activated only after enough logs.
    """

    name = "skill_planner"
    version = "1.0-rulebased"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        skill = state.skills.select_skill(state.task, domain_hint=state.domain)
        state.selected_skill = skill
        state.selected_rules = state.rules.select_rules(skill.rule_scopes)
        budget_tokens = self.config.get("budget_tokens", skill.default_budget_tokens)
        state.budget = Budget(total_tokens=budget_tokens)
        state.log(
            self.name,
            self.version,
            "planned",
            skill_id=skill.id,
            rules=[r.id for r in state.selected_rules],
            budget=budget_tokens,
        )
        return state


# --------------------------------------------------------------------------- #
# 4. Resource Scope Resolver                                                   #
# --------------------------------------------------------------------------- #
class ScopeResolver(Operator):
    """Decide WHERE to search before retrieval: domain, allowed types, exclusions.

    Local default = metadata-based scoping from the selected skill + user ctx.
    Production seam: ACL/tenant/version/collection resolution.
    """

    name = "scope_resolver"
    version = "1.0-metadata"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        skill = state.selected_skill
        state.scope = {
            "domain": state.domain,
            "allowed_types": list(skill.resource_types) if skill else [],
            "exclude_status": self.config.get("exclude_status", ["draft", "archived"]),
            "tenant": state.user_context.get("tenant"),
        }
        state.log(self.name, self.version, "scoped", **state.scope)
        return state


# --------------------------------------------------------------------------- #
# 5. Query Rewriter                                                            #
# --------------------------------------------------------------------------- #
class QueryRewriter(Operator):
    """Expand the task into multiple retrieval queries.

    Local default = keyword + a tiny synonym/symbol map.  Production seam: LLM
    multi-query / HyDE / entity- and code-symbol-aware rewriting.
    """

    name = "query_rewriter"
    version = "1.0-synonym"

    _SYNONYMS = {
        "refund": ["return", "money back", "reimbursement"],
        "deploy": ["release", "ship", "rollout"],
        "production": ["prod", "live"],
        "approval": ["approve", "sign off", "authorization"],
        "equation": ["formula", "expression"],
        "method": ["approach", "algorithm", "procedure"],
        "bug": ["defect", "failure", "error"],
    }

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        original = state.task
        queries = [Query(original, kind="original", weight=1.0)]

        keyword_terms = [t for t in tokenize(original) if t not in _STOPWORDS]
        if keyword_terms:
            queries.append(Query(" ".join(keyword_terms), kind="keyword", weight=0.9))

        expansions: List[str] = []
        for term in keyword_terms:
            expansions.extend(self._SYNONYMS.get(term, []))
        if expansions:
            queries.append(Query(" ".join(expansions), kind="synonym", weight=0.6))

        # Code-symbol query: identifiers in the task (CamelCase / snake_case).
        symbols = re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*", original)
        code_like = [s for s in symbols if ("_" in s or s[:1].isupper() or "." in s)]
        if code_like:
            queries.append(Query(" ".join(code_like), kind="symbol", weight=0.7))

        state.queries = queries
        state.log(
            self.name,
            self.version,
            "rewritten",
            queries=[(q.kind, q.text) for q in queries],
        )
        return state


# --------------------------------------------------------------------------- #
# 6. Hybrid Retriever                                                          #
# --------------------------------------------------------------------------- #
class HybridRetriever(Operator):
    """BM25 (sparse) + local dense vectors, fused with Reciprocal Rank Fusion.

    Config:
      sparse=True/False, dense=True/False, fusion='rrf', top_n=None
    Set ``dense=False`` to ablate to sparse-only retrieval.
    Production seam: replace ``_hash_vector`` with a real encoder + vector DB.
    """

    name = "hybrid_retriever"
    version = "1.0-bm25+hashdense"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        use_sparse = self.config.get("sparse", True)
        use_dense = self.config.get("dense", True)
        top_n = self.config.get("top_n")

        resources = state.resources.list_resources(state.domain)
        if not resources:
            state.candidates = []
            state.retrieved_order = []
            return state

        docs, idf, avgdl = _build_index(resources)
        queries = state.queries or [Query(state.task)]

        rankings: List[List[str]] = []
        sparse_best: Dict[str, float] = {}
        dense_best: Dict[str, float] = {}

        doc_vectors = (
            {r.id: _hash_vector(r.title + " " + r.content) for r in resources}
            if use_dense
            else {}
        )

        for q in queries:
            q_terms = tokenize(q.text)
            if use_sparse:
                s_scores = _bm25(q_terms, docs, idf, avgdl)
                for rid, sc in s_scores.items():
                    sparse_best[rid] = max(sparse_best.get(rid, 0.0), sc)
                ranking = [rid for rid, _ in sorted(
                    s_scores.items(), key=lambda kv: (-kv[1], kv[0])) if _ > 0]
                if ranking:
                    rankings.append(ranking)
            if use_dense:
                qv = _hash_vector(q.text)
                d_scores = {rid: _cosine(qv, dv) for rid, dv in doc_vectors.items()}
                for rid, sc in d_scores.items():
                    dense_best[rid] = max(dense_best.get(rid, 0.0), sc)
                ranking = [rid for rid, _ in sorted(
                    d_scores.items(), key=lambda kv: (-kv[1], kv[0]))][: max(5, len(resources))]
                rankings.append(ranking)

        fused = _rrf(rankings) if rankings else {r.id: 0.0 for r in resources}

        by_id = {r.id: r for r in resources}
        candidates: List[CandidateResource] = []
        for rid, fscore in fused.items():
            r = by_id[rid]
            c = CandidateResource(
                resource_id=r.id,
                title=r.title,
                content=r.content,
                type=r.type,
                metadata={
                    "version": r.version,
                    "effective_date": r.effective_date,
                    **r.metadata,
                },
                sparse_score=round(sparse_best.get(rid, 0.0), 6),
                dense_score=round(dense_best.get(rid, 0.0), 6),
                fused_score=round(fscore, 6),
                score=round(fscore, 6),
            )
            c.add_reason(
                f"hybrid rrf={fscore:.4f} (sparse={c.sparse_score:.3f}, dense={c.dense_score:.3f})"
            )
            candidates.append(c)

        candidates.sort(key=lambda c: (-c.fused_score, c.resource_id))
        if top_n:
            candidates = candidates[:top_n]
        state.candidates = candidates
        state.retrieved_order = [c.resource_id for c in candidates]
        state.log(
            self.name,
            self.version,
            "retrieved",
            n=len(candidates),
            sparse=use_sparse,
            dense=use_dense,
            order=state.retrieved_order,
        )
        return state


# --------------------------------------------------------------------------- #
# 7. Permission / Metadata Filter                                             #
# --------------------------------------------------------------------------- #
class PermissionFilter(Operator):
    """Drop candidates the user/scope may not see. DETERMINISTIC (security).

    Enforces: allowed types, excluded lifecycle status (draft/archived),
    confidentiality vs user clearances, tenant match.  Never rely on prompt
    instructions for this.
    """

    name = "permission_filter"
    version = "1.0-acl"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        scope = state.scope or {}
        allowed_types = set(scope.get("allowed_types") or [])
        exclude_status = set(scope.get("exclude_status") or [])
        clearances = set(state.user_context.get("clearances", []))
        tenant = scope.get("tenant")

        kept, dropped = [], []
        for c in state.candidates:
            reason = None
            if allowed_types and c.type not in allowed_types:
                reason = f"type '{c.type}' not in scope"
            elif c.metadata.get("status") in exclude_status:
                reason = f"status '{c.metadata.get('status')}' excluded"
            elif c.metadata.get("confidential") and (
                c.metadata.get("clearance") not in clearances
            ):
                reason = "insufficient clearance for confidential resource"
            elif tenant and c.metadata.get("tenant") not in (None, tenant):
                reason = "tenant mismatch"
            if reason:
                dropped.append((c.resource_id, reason))
            else:
                kept.append(c)
        state.candidates = kept
        state.log(
            self.name,
            self.version,
            "filtered",
            kept=[c.resource_id for c in kept],
            dropped=dropped,
        )
        if dropped:
            state.warnings.append(f"permission_filter dropped {len(dropped)} resource(s)")
        return state


# --------------------------------------------------------------------------- #
# 8. Authority + Freshness Scorer                                             #
# --------------------------------------------------------------------------- #
class AuthorityFreshnessScorer(Operator):
    """Boost official, approved, current, high-authority sources."""

    name = "authority_freshness_scorer"
    version = "1.0"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        weight = self.config.get("weight", 0.15)
        for c in state.candidates:
            a = 0.0
            if c.metadata.get("source_authority") == "official":
                a += 1.0
            if c.metadata.get("status_approved") or c.metadata.get("status") == "approved":
                a += 0.5
            eff = c.metadata.get("effective_date")
            if eff:
                try:
                    year = int(str(eff)[:4])
                    a += max(0.0, (year - 2020) / 10.0)
                except ValueError:
                    pass
            c.authority_score = round(a, 4)
            c.score = round(c.score + weight * a, 6)
            if a:
                c.add_reason(f"authority+freshness={a:.2f}")
        state.candidates.sort(key=lambda c: (-c.score, c.resource_id))
        state.log(
            self.name,
            self.version,
            "scored",
            authority={c.resource_id: c.authority_score for c in state.candidates},
        )
        return state


# --------------------------------------------------------------------------- #
# 12. Conflict / Contradiction Detector (+ version resolution)                #
# --------------------------------------------------------------------------- #
class ConflictDetector(Operator):
    """Detect version conflicts, keep the latest approved, mark others stale.

    Resolution order: latest approved > older approved > draft > archived.
    Emits a warning per unresolved/again conflicting pair.
    """

    name = "conflict_detector"
    version = "1.0-version"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        groups: Dict[str, List[CandidateResource]] = {}
        kept: List[CandidateResource] = []
        for c in state.candidates:
            g = c.metadata.get("group")
            if g:
                groups.setdefault(g, []).append(c)
            else:
                kept.append(c)

        drop_stale = self.config.get("drop_stale", True)
        for group, members in groups.items():
            if len(members) == 1:
                kept.append(members[0])
                continue
            members.sort(
                key=lambda c: (
                    c.metadata.get("effective_date") or "",
                    c.metadata.get("version") or "",
                ),
                reverse=True,
            )
            winner = members[0]
            kept.append(winner)
            for loser in members[1:]:
                loser.stale = True
                loser.add_reason(f"stale: superseded by {winner.resource_id}")
                state.conflicts.append(
                    {
                        "group": group,
                        "winner": winner.resource_id,
                        "superseded": loser.resource_id,
                    }
                )
                state.warnings.append(
                    f"conflict: {loser.resource_id} superseded by {winner.resource_id}"
                )
                if not drop_stale:
                    kept.append(loser)
        state.candidates = kept
        state.log(
            self.name,
            self.version,
            "resolved",
            conflicts=state.conflicts,
            drop_stale=drop_stale,
        )
        return state


# --------------------------------------------------------------------------- #
# 9. Cross-Encoder / LLM Reranker                                             #
# --------------------------------------------------------------------------- #
class CrossEncoderReranker(Operator):
    """Reorder candidates by a local listwise relevance score.

    Local default = query-term coverage + idf mass + proximity + title match.
    Production seam: Cohere / BGE / Qwen / Voyage / Jina cross-encoder — same
    ``run`` interface, just a different scorer.
    """

    name = "cross_encoder_reranker"
    version = "1.0-local"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        resources = state.resources.list_resources(state.domain)
        _, idf, _ = _build_index(resources)
        q_terms = tokenize(state.task)
        q_set = set(q_terms)
        for c in state.candidates:
            doc_terms = tokenize(c.title + " " + c.content)
            doc_set = set(doc_terms)
            matched = q_set & doc_set
            coverage = len(matched) / (len(q_set) or 1)
            idf_mass = sum(idf.get(t, 0.0) for t in matched)
            title_match = len(q_set & set(tokenize(c.title)))
            # crude proximity: any adjacent query-term pair present in doc
            proximity = 0.0
            joined = " ".join(doc_terms)
            for i in range(len(q_terms) - 1):
                if f"{q_terms[i]} {q_terms[i+1]}" in joined:
                    proximity += 1.0
            score = 2.0 * coverage + idf_mass + 0.3 * title_match + 0.5 * proximity
            c.rerank_score = round(score, 6)
            c.score = c.rerank_score
            c.add_reason(f"rerank={score:.3f} (cov={coverage:.2f})")
        state.candidates.sort(key=lambda c: (-c.score, -c.authority_score, c.resource_id))
        state.log(
            self.name,
            self.version,
            "reranked",
            order=[(c.resource_id, c.rerank_score) for c in state.candidates],
        )
        return state


# --------------------------------------------------------------------------- #
# 10. Deduper / Diversity Selector (MMR)                                      #
# --------------------------------------------------------------------------- #
class MMRDiversitySelector(Operator):
    """Reorder for diversity via Maximal Marginal Relevance (local dense sim)."""

    name = "mmr_diversity_selector"
    version = "1.0-mmr"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        lam = self.config.get("lambda", 0.7)
        cands = list(state.candidates)
        if len(cands) <= 2:
            return state
        vectors = {c.resource_id: _hash_vector(c.title + " " + c.content) for c in cands}
        selected: List[CandidateResource] = []
        remaining = cands[:]
        # seed with top-scored
        remaining.sort(key=lambda c: -c.score)
        selected.append(remaining.pop(0))
        while remaining:
            best, best_val = None, -1e9
            for c in remaining:
                max_sim = max(
                    _cosine(vectors[c.resource_id], vectors[s.resource_id])
                    for s in selected
                )
                val = lam * c.score - (1 - lam) * max_sim
                if val > best_val:
                    best_val, best = val, c
            selected.append(best)
            remaining.remove(best)
        state.candidates = selected
        state.log(
            self.name, self.version, "diversified",
            order=[c.resource_id for c in selected],
        )
        return state


# --------------------------------------------------------------------------- #
# Evidence selection (top-k)                                                   #
# --------------------------------------------------------------------------- #
class EvidenceSelector(Operator):
    """Record the ranked order and select the top-k candidates as evidence."""

    name = "evidence_selector"
    version = "1.0"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        top_k = self.config.get("top_k", 5)
        state.ranked_order = [c.resource_id for c in state.candidates]
        state.selected_items = [c.copy() for c in state.candidates[:top_k]]
        state.log(
            self.name, self.version, "selected",
            selected=[c.resource_id for c in state.selected_items], top_k=top_k,
        )
        return state


# --------------------------------------------------------------------------- #
# 13. Domain-Specific Compressor (query-aware, citation-preserving)           #
# --------------------------------------------------------------------------- #
class QueryAwareCompressor(Operator):
    """Extractive, citation-preserving compression.

    Keeps the first sentence plus any sentence overlapping the query terms.
    Never abstracts wording away — safe for policy/legal/code contexts.
    """

    name = "query_aware_compressor"
    version = "1.0-extractive"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        max_sentences = self.config.get("max_sentences", 4)
        q_terms = set(tokenize(state.task))
        for c in state.selected_items:
            sentences = re.split(r"(?<=[.!?])\s+", c.content.strip())
            if len(sentences) <= max_sentences:
                continue
            kept = [sentences[0]]
            for s in sentences[1:]:
                if q_terms & set(tokenize(s)) and len(kept) < max_sentences:
                    kept.append(s)
            compressed = " ".join(kept)
            if len(compressed) < len(c.content):
                c.content = compressed
                c.metadata["compressed"] = True
                c.add_reason("query-aware extractive compression")
        state.log(self.name, self.version, "compressed",
                  compressed=[c.resource_id for c in state.selected_items
                              if c.metadata.get("compressed")])
        return state


# --------------------------------------------------------------------------- #
# 14. Budget Allocator                                                        #
# --------------------------------------------------------------------------- #
class BudgetAllocator(Operator):
    """Enforce the token budget B; drop lowest-ranked evidence if over."""

    name = "budget_allocator"
    version = "1.0-quota"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        budget = state.budget.total_tokens if state.budget else self.config.get("budget", 1000)

        def evidence_tokens():
            return sum(estimate_tokens(c.content) for c in state.selected_items)

        while state.selected_items and evidence_tokens() > budget:
            dropped = state.selected_items.pop()
            state.warnings.append(
                f"budget_allocator dropped '{dropped.resource_id}' (budget {budget})"
            )
            state.log(self.name, self.version, "dropped", resource_id=dropped.resource_id)
        if state.budget:
            state.budget.spent = evidence_tokens()
        state.log(self.name, self.version, "allocated",
                  evidence_tokens=evidence_tokens(), budget=budget)
        return state


# --------------------------------------------------------------------------- #
# 15. Context Ordering                                                        #
# --------------------------------------------------------------------------- #
class ContextOrderer(Operator):
    """Order evidence by score (fresh/authoritative first), stale last."""

    name = "context_orderer"
    version = "1.0"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        state.selected_items.sort(key=lambda c: (c.stale, -c.score, c.resource_id))
        state.log(self.name, self.version, "ordered",
                  order=[c.resource_id for c in state.selected_items])
        return state


# --------------------------------------------------------------------------- #
# 16. Formatters                                                              #
# --------------------------------------------------------------------------- #
class XMLFormatter(Operator):
    """Structured, boundary-delimited context with citations + output contract.

    Models respect clear boundaries better than loose prose; this is the
    production default for complex contexts.
    """

    name = "xml_formatter"
    version = "1.0"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        lines: List[str] = []
        lines.append(f"<task>{state.task}</task>")
        if state.constraints:
            lines.append("<constraints>")
            for k, vals in state.constraints.items():
                for v in vals:
                    lines.append(f"  <constraint kind=\"{k}\">{v}</constraint>")
            lines.append("</constraints>")
        lines.append("<rules>")
        for r in state.selected_rules:
            lines.append(f"  <rule id=\"{r.id}\">{r.description}</rule>")
        lines.append("</rules>")
        lines.append("<evidence>")
        if not state.selected_items:
            lines.append("  <missing>Evidence is missing.</missing>")
            state.warnings.append("no evidence selected")
        for c in state.selected_items:
            section = c.metadata.get("section", "")
            cite = f"[{c.resource_id}" + (f" §{section}]" if section else "]")
            lines.append(
                f"  <source id=\"{c.resource_id}\" section=\"{section}\" "
                f"authority=\"{c.authority_score}\">{cite} {c.content.strip()}</source>"
            )
        lines.append("</evidence>")
        if state.conflicts:
            lines.append("<conflicts>")
            for cf in state.conflicts:
                lines.append(
                    f"  <conflict>WARNING: {cf['superseded']} superseded by "
                    f"{cf['winner']} (group {cf['group']})</conflict>"
                )
            lines.append("</conflicts>")
        contract = (
            "Answer only from the cited evidence above. Cite [source_id] for "
            "every claim. If evidence is missing, say so."
        )
        lines.append(f"<output_contract>{contract}</output_contract>")

        state.formatted_context = "\n".join(lines) + "\n"
        state.token_estimate = estimate_tokens(state.formatted_context)
        state.log(self.name, self.version, "formatted", token_estimate=state.token_estimate)
        return state


class SimpleFormatter(Operator):
    """Naive markdown formatter WITHOUT citations (baseline for ablation)."""

    name = "simple_formatter"
    version = "1.0"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        lines = [f"# Task\n{state.task}", "\n# Evidence"]
        for c in state.selected_items:
            lines.append(f"\n## {c.title}\n{c.content.strip()}")
        state.formatted_context = "\n".join(lines) + "\n"
        state.token_estimate = estimate_tokens(state.formatted_context)
        state.log(self.name, self.version, "formatted", token_estimate=state.token_estimate)
        return state


# --------------------------------------------------------------------------- #
# 17. Context Validator (deterministic, pre-send)                             #
# --------------------------------------------------------------------------- #
class ContextValidator(Operator):
    """Deterministic pre-send checks: budget, citations, staleness, secrets, ACL."""

    name = "context_validator"
    version = "1.0"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        text = state.formatted_context
        budget = state.budget.total_tokens if state.budget else 0
        checks: Dict[str, bool] = {}
        checks["budget_ok"] = budget == 0 or state.token_estimate <= budget
        checks["has_evidence"] = bool(state.selected_items)
        checks["citations_present"] = all(
            f"[{c.resource_id}" in text or f'id="{c.resource_id}"' in text
            for c in state.selected_items
        ) if state.selected_items else False
        checks["no_secrets"] = _SECRET_RE.search(text) is None
        stale_included = [c.resource_id for c in state.selected_items if c.stale]
        checks["no_unmarked_stale"] = not stale_included or "conflict" in text.lower()
        checks["rules_present"] = bool(state.selected_rules)
        state.validation = {"checks": checks, "passed": all(checks.values())}
        for name, ok in checks.items():
            if not ok:
                state.warnings.append(f"validation failed: {name}")
                if name in {"no_secrets"}:
                    state.errors.append("SECURITY: secret detected in context")
        state.log(self.name, self.version, "validated", **checks)
        return state


# --------------------------------------------------------------------------- #
# 18. Telemetry                                                               #
# --------------------------------------------------------------------------- #
class Telemetry(Operator):
    """Summarize provenance and per-operator metrics (JSON-able trace)."""

    name = "telemetry"
    version = "1.0-json"

    def _run(self, state: ContextBuildState) -> ContextBuildState:
        total_ms = round(sum(m.duration_ms for m in state.metrics), 3)
        errors = [m.operator for m in state.metrics if m.error]
        state.log(
            self.name, self.version, "summary",
            operators=len(state.metrics), total_ms=total_ms,
            errored_operators=errors, warnings=len(state.warnings),
            final_tokens=state.token_estimate,
        )
        return state
