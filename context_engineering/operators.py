"""Dynamic operators — the building blocks of a skill's pipeline ``F_s``.

Every operator has the signature ``(state: PipelineState) -> PipelineState``.
Operators select, transform, rank, filter, compress and format resources and
rules.  They append trace events so the *why* of every decision is recoverable.

The public :func:`register_default_operators` wires all of them into an
:class:`OperatorRegistry`.
"""
from __future__ import annotations

import re
from typing import Dict, List

from .models import ContextItem, PipelineState, estimate_tokens
from .registries import OperatorRegistry, tokenize

# --------------------------------------------------------------------------- #
# Domain classification (used when no domain hint is supplied)                #
# --------------------------------------------------------------------------- #
_DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "coding_bugfix": ["fix", "bug", "error", "stack", "trace", "test", "crash"],
    "paper_explanation": ["equation", "method", "paper", "explain", "symbol",
                          "derivation", "theorem"],
    "compliance_policy": ["deploy", "production", "approval", "approve",
                          "compliance", "audit", "permission"],
    "rag_doc_qa": ["refund", "faq", "document", "policy", "shipping",
                  "return", "warranty"],
}


def classify_task(task: str) -> str:
    """Deterministically map a task string to a domain via keyword scoring."""
    tokens = set(tokenize(task))
    best_domain = "rag_doc_qa"
    best_score = -1
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        score = len(tokens & set(keywords))
        if score > best_score:
            best_score = score
            best_domain = domain
    return best_domain


def task_classifier_operator(state: PipelineState) -> PipelineState:
    """Classify the task into a domain (if not already set)."""
    if not state.domain:
        state.domain = classify_task(state.task)
    state.log("task_classified", domain=state.domain)
    return state


# --------------------------------------------------------------------------- #
# Rule selection                                                              #
# --------------------------------------------------------------------------- #
def rule_selector_operator(state: PipelineState) -> PipelineState:
    """Select active rules for the skill's rule scopes (part of ``R_s``)."""
    rules = state.rule_registry.select_rules(state.skill.rule_scopes)
    state.selected_rules = rules
    state.log(
        "rules_selected",
        rule_ids=[r.id for r in rules],
        scopes=state.skill.rule_scopes,
    )
    return state


# --------------------------------------------------------------------------- #
# Retrieval                                                                   #
# --------------------------------------------------------------------------- #
def lexical_retrieval_operator(state: PipelineState) -> PipelineState:
    """Retrieve candidate resources via local TF-IDF-style lexical search."""
    results = state.resource_registry.search(state.task, domain=state.domain)
    candidates: List[ContextItem] = []
    for res, score in results:
        candidates.append(
            ContextItem(
                resource_id=res.id,
                title=res.title,
                content=res.content,
                reason_selected=f"lexical match score={score:.3f}",
                score=score,
                metadata={
                    "type": res.type,
                    "version": res.version,
                    "effective_date": res.effective_date,
                    "group": res.metadata.get("group"),
                    "section": res.metadata.get("section"),
                    **res.metadata,
                },
            )
        )
    state.candidates = candidates
    state.log(
        "retrieved_candidates",
        count=len(candidates),
        scores={c.resource_id: c.score for c in candidates},
    )
    return state


# --------------------------------------------------------------------------- #
# Filters                                                                      #
# --------------------------------------------------------------------------- #
def type_filter_operator(state: PipelineState) -> PipelineState:
    """Keep only resources whose type is required by the skill."""
    wanted = set(state.skill.resource_types)
    if not wanted:
        return state
    kept, dropped = [], []
    for c in state.candidates:
        if c.metadata.get("type") in wanted:
            kept.append(c)
        else:
            dropped.append(c.resource_id)
    state.candidates = kept
    state.log("type_filter", kept=[c.resource_id for c in kept], dropped=dropped)
    return state


def recency_filter_operator(state: PipelineState) -> PipelineState:
    """Within a version group, keep the latest; mark older ones stale.

    Resources are grouped by ``metadata['group']``.  When a group has more than
    one candidate, the one with the newest ``effective_date`` (falling back to
    ``version``) wins; the rest are moved to ``state.stale_items`` and dropped
    from candidates.  A warning is recorded per stale item.
    """
    groups: Dict[str, List[ContextItem]] = {}
    ungrouped: List[ContextItem] = []
    for c in state.candidates:
        group = c.metadata.get("group")
        if group:
            groups.setdefault(group, []).append(c)
        else:
            ungrouped.append(c)

    kept: List[ContextItem] = list(ungrouped)
    for group, members in groups.items():
        if len(members) == 1:
            kept.append(members[0])
            continue

        def sort_key(item: ContextItem):
            return (
                item.metadata.get("effective_date") or "",
                item.metadata.get("version") or "",
            )

        members_sorted = sorted(members, key=sort_key, reverse=True)
        winner = members_sorted[0]
        kept.append(winner)
        for loser in members_sorted[1:]:
            loser.metadata["stale"] = True
            loser.reason_selected = (
                f"stale: superseded by {winner.resource_id} in group '{group}'"
            )
            state.stale_items.append(loser)
            state.warnings.append(
                f"stale resource excluded: {loser.resource_id} "
                f"(superseded by {winner.resource_id})"
            )
            state.log(
                "stale_filtered",
                resource_id=loser.resource_id,
                superseded_by=winner.resource_id,
                group=group,
            )
    state.candidates = kept
    return state


def dedupe_operator(state: PipelineState) -> PipelineState:
    """Remove exact/near-duplicate candidates by content fingerprint."""
    seen: Dict[str, str] = {}
    kept: List[ContextItem] = []
    removed: List[str] = []
    for c in state.candidates:
        fingerprint = " ".join(tokenize(c.content))[:200]
        if fingerprint in seen:
            removed.append(c.resource_id)
            continue
        seen[fingerprint] = c.resource_id
        kept.append(c)
    state.candidates = kept
    if removed:
        state.log("deduped", removed=removed)
    return state


# --------------------------------------------------------------------------- #
# Ranking                                                                      #
# --------------------------------------------------------------------------- #
def relevance_rank_operator(state: PipelineState) -> PipelineState:
    """Sort candidates by score and truncate to the skill's ``top_k``.

    The surviving candidates become ``state.selected_items``.
    """
    ranked = sorted(
        state.candidates, key=lambda c: (-c.score, c.resource_id)
    )
    top_k = state.skill.metadata.get("top_k")
    if top_k is not None:
        ranked = ranked[:top_k]
    state.selected_items = ranked
    state.log(
        "ranked",
        order=[(c.resource_id, round(c.score, 4)) for c in ranked],
        top_k=top_k,
    )
    return state


# --------------------------------------------------------------------------- #
# Compression / budget                                                        #
# --------------------------------------------------------------------------- #
def _first_n_sentences(text: str, n: int) -> str:
    """Return the first ``n`` sentences of ``text`` (deterministic split)."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sentences[:n]).strip()


def budget_compressor_operator(state: PipelineState) -> PipelineState:
    """Ensure the selected evidence fits the token budget ``B``.

    Strategy (simple and deterministic):
      1. Estimate tokens of the concatenated selected items.
      2. While over budget, compress the *lowest-ranked* item to its first
         sentences; if it cannot shrink further, drop it entirely.
    """
    budget = state.budget

    def total_tokens() -> int:
        return sum(estimate_tokens(c.content) for c in state.selected_items)

    n_sentences = 3
    while state.selected_items and total_tokens() > budget:
        # Operate on the lowest-ranked item (end of list).
        item = state.selected_items[-1]
        compressed = _first_n_sentences(item.content, n_sentences)
        if compressed and estimate_tokens(compressed) < estimate_tokens(
            item.content
        ):
            item.content = compressed
            item.metadata["compressed"] = True
            state.log(
                "compressed",
                resource_id=item.resource_id,
                sentences=n_sentences,
                new_tokens=estimate_tokens(item.content),
            )
            if n_sentences > 1:
                n_sentences -= 1
        else:
            dropped = state.selected_items.pop()
            state.warnings.append(
                f"dropped '{dropped.resource_id}' to satisfy budget "
                f"({budget} tokens)"
            )
            state.log("dropped_for_budget", resource_id=dropped.resource_id)
            n_sentences = 3
    return state


# --------------------------------------------------------------------------- #
# Domain-specific operators                                                    #
# --------------------------------------------------------------------------- #
_STACK_FILE_RE = re.compile(r'([\w./-]+\.py)"?,?\s*line\s*(\d+)', re.IGNORECASE)
_STACK_FILE_SIMPLE_RE = re.compile(r'([\w./-]+\.py)')


def coding_stacktrace_operator(state: PipelineState) -> PipelineState:
    """Extract filenames from log/stack-trace candidates and boost matches.

    Any candidate whose resource_id or title matches a file mentioned in a
    ``log`` resource has its score boosted so the failing code surfaces.
    """
    referenced: List[str] = []
    for c in state.candidates:
        if c.metadata.get("type") == "log":
            for match in _STACK_FILE_RE.finditer(c.content):
                referenced.append(match.group(1))
            for match in _STACK_FILE_SIMPLE_RE.finditer(c.content):
                referenced.append(match.group(1))
    referenced_bases = {r.split("/")[-1] for r in referenced}

    if referenced_bases:
        for c in state.candidates:
            base = c.resource_id.split("/")[-1]
            if base in referenced_bases or any(
                base in ref for ref in referenced_bases
            ):
                c.score += 1.0
                c.reason_selected += " | boosted: named in stack trace"
        state.log(
            "stacktrace_boost",
            referenced_files=sorted(referenced_bases),
        )
    return state


_EQUATION_RE = re.compile(r"[=≈≤≥∑∏∫√±·×]|\\[a-zA-Z]+|\b[a-zA-Z]\s*=")


def paper_equation_extractor_operator(state: PipelineState) -> PipelineState:
    """Boost paper candidates that contain equations/symbols.

    Also annotates each candidate with the equation lines it contains so the
    formatter and evaluators can rely on them.
    """
    for c in state.candidates:
        eq_lines = [
            line.strip()
            for line in c.content.splitlines()
            if _EQUATION_RE.search(line)
        ]
        if eq_lines:
            c.metadata["equations"] = eq_lines
            c.score += 0.5
            c.reason_selected += " | contains equation(s)"
    state.log(
        "equation_extract",
        with_equations=[
            c.resource_id for c in state.candidates if c.metadata.get("equations")
        ],
    )
    return state


# --------------------------------------------------------------------------- #
# Formatting                                                                   #
# --------------------------------------------------------------------------- #
def citation_formatter_operator(state: PipelineState) -> PipelineState:
    """Attach a citation label ``[<id> §<section>]`` to each selected item."""
    for c in state.selected_items:
        section = c.metadata.get("section")
        label = f"[{c.resource_id}"
        if section:
            label += f" §{section}"
        label += "]"
        c.metadata["citation"] = label
    state.log(
        "citations_added",
        citations=[c.metadata.get("citation") for c in state.selected_items],
    )
    return state


def context_formatter_operator(state: PipelineState) -> PipelineState:
    """Render task, rules, evidence and a trace summary into final context.

    The output is human-readable and is what would be sent to the LLM as ``c``.
    """
    lines: List[str] = []
    lines.append("=== TASK ===")
    lines.append(state.task)
    lines.append("")

    lines.append("=== RULES ===")
    if state.selected_rules:
        for r in state.selected_rules:
            lines.append(f"- ({r.id}) {r.description}")
    else:
        lines.append("- (none selected)")
    lines.append("")

    lines.append("=== EVIDENCE ===")
    if state.selected_items:
        for c in state.selected_items:
            citation = c.metadata.get("citation")
            header = citation if citation else f"[{c.resource_id}]"
            title = f" {c.title}" if c.title else ""
            lines.append(f"{header}{title}")
            lines.append(c.content.strip())
            lines.append("")
    else:
        lines.append("No evidence selected. Evidence is missing.")
        lines.append("")
        state.warnings.append("no evidence selected for task")

    if state.stale_items:
        lines.append("=== EXCLUDED (STALE) ===")
        for c in state.stale_items:
            lines.append(f"- {c.resource_id}: {c.reason_selected}")
        lines.append("")

    formatted = "\n".join(lines).rstrip() + "\n"
    state.formatted_context = formatted
    state.token_estimate = estimate_tokens(formatted)
    state.log("formatted", token_estimate=state.token_estimate)
    return state


# --------------------------------------------------------------------------- #
# Registration                                                                 #
# --------------------------------------------------------------------------- #
def register_default_operators(registry: OperatorRegistry) -> OperatorRegistry:
    """Register every built-in operator into ``registry`` and return it."""
    registry.register_operator("task_classifier", task_classifier_operator)
    registry.register_operator("rule_selector", rule_selector_operator)
    registry.register_operator("lexical_retrieval", lexical_retrieval_operator)
    registry.register_operator("type_filter", type_filter_operator)
    registry.register_operator("recency_filter", recency_filter_operator)
    registry.register_operator("dedupe", dedupe_operator)
    registry.register_operator("relevance_rank", relevance_rank_operator)
    registry.register_operator("budget_compressor", budget_compressor_operator)
    registry.register_operator("coding_stacktrace", coding_stacktrace_operator)
    registry.register_operator(
        "paper_equation_extractor", paper_equation_extractor_operator
    )
    registry.register_operator("citation_formatter", citation_formatter_operator)
    registry.register_operator("context_formatter", context_formatter_operator)
    return registry
