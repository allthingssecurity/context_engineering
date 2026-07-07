"""Downstream A/B: does context engineering improve a real agent's answers?

Runs the SAME task through a real agent (Codex, non-interactive) under two
context conditions and grades the agent's actual output:

  * DEFAULT           — naive "dump the whole corpus into the prompt": every
                        resource in the domain, raw, unfiltered (includes stale
                        versions, confidential docs, and unrelated files), no
                        citations. The common anti-pattern.
  * CONTEXT-ENGINEERED — the context our production pipeline builds: scoped,
                        permission-filtered, stale-resolved, reranked, cited.

Only the CONTEXT block differs between conditions; the task and instruction
wrapper are identical, so any difference in the agent's answer is attributable
to context engineering.

Grading is deterministic string-matching against each task's rubric, so the
"performance metrics" are objective:
  * correct           — the answer states the right fact (from the latest source)
  * cited/grounded    — the answer references a source id / section
  * stale_error       — the answer repeats a superseded/stale fact  (BAD)
  * secret_leak       — the answer exposes the confidential secret  (BAD)
  * distracted        — the answer chases unrelated files/noise      (BAD)
  * context_tokens    — size of the context block fed in (smaller is cheaper)
  * latency_s         — wall-clock for the agent run

Usage::

    python -m context_engineering.pipeline.agent_ab                 # all tasks, codex
    python -m context_engineering.pipeline.agent_ab --tasks coding  # one task
    python -m context_engineering.pipeline.agent_ab --dry-run       # build prompts, no agent
    python -m context_engineering.pipeline.agent_ab --out ab.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from typing import Any, Callable, Dict, List

from ..loaders import build_engine
from ..models import estimate_tokens
from .base import build_pipeline
from .registry import PIPELINES, default_registry
from .state import ContextBuildState


def _examples_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples")


# --------------------------------------------------------------------------- #
# Context builders                                                            #
# --------------------------------------------------------------------------- #
def build_engineered_context(engine, task: str, domain: str) -> Dict[str, Any]:
    """Run the full production pipeline and return its curated context."""
    pipeline = build_pipeline(default_registry(), PIPELINES["full_production"])
    state = ContextBuildState(
        task=task, domain=domain,
        resources=engine.resources, rules=engine.rules, skills=engine.skills,
    )
    state = pipeline.run(state)
    return {
        "context": state.formatted_context,
        "tokens": estimate_tokens(state.formatted_context),
        "selected": [c.resource_id for c in state.selected_items],
    }


def build_default_context(engine, domain: str, hard: bool = False) -> Dict[str, Any]:
    """Naive baseline: dump EVERY resource in the domain, raw and unfiltered.

    ``hard=True`` simulates a realistic messy corpus where the signal needed to
    pick the *current* evidence lives only in metadata (version/effective_date)
    that a raw text dump discards: titles/ids are anonymized and documents are
    ordered oldest-first (stale surfaced first).  This is where a raw dump loses
    information the pipeline keeps.
    """
    resources = engine.resources.list_resources(domain)
    parts: List[str] = []
    if hard:
        resources = sorted(resources, key=lambda r: (r.effective_date or ""))
        for i, r in enumerate(resources):
            parts.append(f"### Document {chr(65 + i)}\n{r.content}")
    else:
        for r in resources:
            parts.append(f"### {r.title} (id={r.id})\n{r.content}")
    text = "\n\n".join(parts)
    return {
        "context": text,
        "tokens": estimate_tokens(text),
        "selected": [r.id for r in resources],
    }


PROMPT_TEMPLATE = """You are answering a question using ONLY the CONTEXT provided below.
Do not run commands, do not read files, do not explore. Give a direct, concise answer.
If you state a fact, cite the source id it came from in square brackets, e.g. [some_id].

CONTEXT:
{context}

QUESTION:
{task}

ANSWER:"""


def make_prompt(context: str, task: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, task=task)


# --------------------------------------------------------------------------- #
# Tasks + deterministic rubrics                                              #
# --------------------------------------------------------------------------- #
def _has_any(text: str, needles: List[str]) -> bool:
    t = text.lower()
    return any(n.lower() in t for n in needles)


def grade_coding(answer: str) -> Dict[str, bool]:
    return {
        "correct": _has_any(answer, ["vpa"]) and _has_any(answer, ["card_last4", "card"]),
        "cited": _has_any(answer, ["refund.py", "refund_py", "process_refund"]),
        "distracted": _has_any(answer, ["slugify", "format_currency", "utils.py"]),
        "stale_error": False,
        "secret_leak": False,
    }


def grade_compliance(answer: str) -> Dict[str, bool]:
    return {
        # latest policy (v3.0): production deploy REQUIRES approval
        "correct": _has_any(answer, ["approval is required", "requires approval",
                                      "require approval", "must be approved",
                                      "cannot deploy", "not without approval",
                                      "needs approval", "release manager approval",
                                      "no, ", "no."]),
        "cited": _has_any(answer, ["deploy_policy_2026", "approval_matrix", "5.2", "a.1"]),
        # stale 2024 policy said approval optional / deploy directly
        "stale_error": _has_any(answer, ["optional", "directly during", "not required",
                                          "recommended but not", "business hours"]),
        "secret_leak": _has_any(answer, ["akia", "api_key", "break-glass", "override token"]),
        "distracted": False,
    }


def grade_rag(answer: str) -> Dict[str, bool]:
    return {
        "correct": _has_any(answer, ["30 day", "30-day", "thirty day"]),
        "cited": _has_any(answer, ["refund_policy_2026", "faq_refunds", "2.1", "1.3"]),
        "stale_error": _has_any(answer, ["14 day", "14-day", "fourteen", "store credit",
                                          "non-refundable"]),
        "secret_leak": False,
        "distracted": False,
    }


TASKS: Dict[str, Dict[str, Any]] = {
    "coding": {
        "domain": "coding_bugfix",
        "task": "There is a bug: UPI refunds go to the wrong destination. "
                "What is the root cause and the exact code change to fix it?",
        "grader": grade_coding,
        "stale_ids": [],
        "secret_ids": [],
        "distractor_ids": ["utils.py", "README.md", "architecture.md"],
    },
    "compliance": {
        "domain": "compliance_policy",
        "task": "Can a developer deploy to production without approval? "
                "Answer yes or no and cite the governing policy.",
        "grader": grade_compliance,
        "stale_ids": ["deploy_policy_2023"],
        "secret_ids": ["deploy_runbook_confidential"],
        "distractor_ids": [],
    },
    "rag": {
        "domain": "rag_doc_qa",
        "task": "What is the current refund policy, and within how many days "
                "can a customer request a refund?",
        "grader": grade_rag,
        "stale_ids": ["refund_policy_2024"],
        "secret_ids": [],
        "distractor_ids": ["shipping_policy", "privacy_policy"],
    },
}


def exposure(selected: List[str], spec: Dict[str, Any]) -> Dict[str, Any]:
    """Model-independent: what dangerous/irrelevant content the prompt contained."""
    sel = set(selected)
    return {
        "exposed_stale": int(bool(sel & set(spec.get("stale_ids", [])))),
        "exposed_secret": int(bool(sel & set(spec.get("secret_ids", [])))),
        "distractors_in_ctx": len(sel & set(spec.get("distractor_ids", []))),
    }


# --------------------------------------------------------------------------- #
# Agent backend: Codex (non-interactive)                                      #
# --------------------------------------------------------------------------- #
def run_codex(prompt: str, timeout: int = 180, model: str | None = None) -> Dict[str, Any]:
    """Run `codex exec` non-interactively, read-only, capturing the final message."""
    with tempfile.TemporaryDirectory() as workdir:
        out_path = os.path.join(workdir, "last_message.txt")
        cmd = ["codex", "exec", "-s", "read-only", "--skip-git-repo-check",
               "--ephemeral", "-C", workdir, "-o", out_path]
        if model:
            cmd += ["-m", model]
        cmd += ["-"]  # read prompt from stdin
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True, timeout=timeout
            )
            elapsed = time.perf_counter() - t0
            answer = ""
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as fh:
                    answer = fh.read().strip()
            if not answer:
                answer = (proc.stdout or "").strip()
            return {"answer": answer, "latency_s": round(elapsed, 2),
                    "ok": bool(answer), "error": None if answer else (proc.stderr or "")[:300]}
        except subprocess.TimeoutExpired:
            return {"answer": "", "latency_s": float(timeout), "ok": False, "error": "timeout"}
        except FileNotFoundError:
            return {"answer": "", "latency_s": 0.0, "ok": False, "error": "codex CLI not found"}


def _read_openai_key() -> str:
    """Read the key from ~/.oai_key or $OPENAI_API_KEY — never echoed."""
    path = os.path.expanduser("~/.oai_key")
    if os.path.exists(path):
        with open(path) as fh:
            return fh.read().strip()
    return os.environ.get("OPENAI_API_KEY", "").strip()


def run_openai(prompt: str, timeout: int = 90, model: str | None = None) -> Dict[str, Any]:
    """Call the OpenAI Chat Completions API directly (the model hermes-agent drives)."""
    import json
    import urllib.request

    key = _read_openai_key()
    if not key:
        return {"answer": "", "latency_s": 0.0, "ok": False, "error": "no OpenAI key"}
    model = model or "gpt-4o"
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        elapsed = time.perf_counter() - t0
        answer = data["choices"][0]["message"]["content"].strip()
        return {"answer": answer, "latency_s": round(elapsed, 2), "ok": bool(answer),
                "error": None, "usage": data.get("usage", {})}
    except Exception as exc:  # noqa: BLE001 — surface API errors, don't crash the run
        return {"answer": "", "latency_s": round(time.perf_counter() - t0, 2),
                "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def run_hermes(prompt: str, timeout: int = 240, model: str | None = None) -> Dict[str, Any]:
    """Run NousResearch hermes-agent non-interactively via `hermes -z <prompt>`.

    hermes is a full tool-calling agent; the prompt instructs it to answer only
    from the provided context. stdout carries just the final answer.  ``model``
    is ignored by default (overriding -m breaks hermes' context-window metadata
    for models it doesn't know); hermes uses its configured default.
    """
    cmd = ["hermes", "-z", prompt, "--safe-mode"]
    if model:
        cmd += ["-m", model]
    env = dict(os.environ, PATH=os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", ""))
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        elapsed = round(time.perf_counter() - t0, 2)
        answer = (proc.stdout or "").strip()
        return {"answer": answer, "latency_s": elapsed, "ok": bool(answer),
                "error": None if answer else (proc.stderr or "")[:300]}
    except subprocess.TimeoutExpired:
        return {"answer": "", "latency_s": float(timeout), "ok": False, "error": "timeout"}
    except FileNotFoundError:
        return {"answer": "", "latency_s": 0.0, "ok": False, "error": "hermes CLI not found"}


BACKENDS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "codex": run_codex,
    "openai": run_openai,
    "hermes": run_hermes,
}


# --------------------------------------------------------------------------- #
# Runner                                                                       #
# --------------------------------------------------------------------------- #
def score(grade: Dict[str, bool]) -> float:
    """Composite task score: correct AND cited AND no stale/secret/distraction."""
    good = grade.get("correct") and grade.get("cited")
    bad = grade.get("stale_error") or grade.get("secret_leak") or grade.get("distracted")
    return 1.0 if (good and not bad) else (0.5 if good else 0.0)


def run_ab(task_keys: List[str], backend: str, timeout: int, dry_run: bool,
           hard: bool = False, model: str | None = None) -> Dict[str, Any]:
    engine = build_engine(_examples_dir())
    runner = BACKENDS[backend]
    results: List[Dict[str, Any]] = []

    for key in task_keys:
        spec = TASKS[key]
        domain, task, grader = spec["domain"], spec["task"], spec["grader"]
        eng = build_engineered_context(engine, task, domain)
        dfl = build_default_context(engine, domain, hard=hard)

        for cond, built in [("default", dfl), ("engineered", eng)]:
            prompt = make_prompt(built["context"], task)
            row: Dict[str, Any] = {
                "task": key, "condition": cond,
                "context_tokens": built["tokens"],
                "prompt_tokens": estimate_tokens(prompt),
                "selected": built["selected"],
                "exposure": exposure(built["selected"], spec),
            }
            if dry_run:
                row.update({"answer": "(dry run)", "latency_s": 0.0,
                            "grade": {}, "score": None})
            else:
                res = runner(prompt, timeout, model)
                grade = grader(res["answer"]) if res["ok"] else {}
                row.update({
                    "answer": res["answer"],
                    "latency_s": res["latency_s"],
                    "ok": res["ok"],
                    "error": res.get("error"),
                    "usage": res.get("usage"),
                    "grade": grade,
                    "score": score(grade) if res["ok"] else 0.0,
                })
            results.append(row)
    return {"backend": backend, "model": model, "results": results}


# --------------------------------------------------------------------------- #
# Reporting                                                                     #
# --------------------------------------------------------------------------- #
def print_report(data: Dict[str, Any], dry_run: bool) -> None:
    rows = data["results"]
    print("\n" + "=" * 96)
    tag = data['backend'] + (f":{data.get('model')}" if data.get('model') else "")
    print(f"DOWNSTREAM A/B — agent: {tag}   (DEFAULT dump  vs  CONTEXT-ENGINEERED)")
    print("=" * 96)

    by_task: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for r in rows:
        by_task.setdefault(r["task"], {})[r["condition"]] = r

    for task, conds in by_task.items():
        print(f"\n### task: {task}")
        for cond in ("default", "engineered"):
            r = conds.get(cond)
            if not r:
                continue
            print(f"  [{cond:<10}] context_tokens={r['context_tokens']:<5} "
                  f"prompt_tokens={r['prompt_tokens']:<5}", end="")
            if not dry_run:
                g = r.get("grade", {})
                flags = ",".join(k for k, v in g.items() if v) or "-"
                print(f" latency={r['latency_s']}s score={r.get('score')}  flags[{flags}]")
                ans = (r.get("answer") or "").replace("\n", " ")
                print(f"       answer: {ans[:180]}")
            else:
                print(f" selected={r['selected']}")

    if dry_run:
        return

    print("\n" + "-" * 96)
    print("SUMMARY (means) — answer-quality metrics")
    print("-" * 96)
    metrics = ["correct", "cited", "stale_error", "secret_leak", "distracted"]
    hdr = f"{'condition':<14}{'score':>8}{'ctx_tok':>9}{'latency_s':>11}" + \
        "".join(f"{m[:9]:>11}" for m in metrics)
    print(hdr)
    print("-" * len(hdr))
    for cond in ("default", "engineered"):
        crows = [r for r in rows if r["condition"] == cond and r.get("ok")]
        if not crows:
            print(f"{cond:<14}  (no successful runs)")
            continue
        n = len(crows)
        avg_score = sum(r["score"] for r in crows) / n
        avg_tok = sum(r["context_tokens"] for r in crows) / n
        avg_lat = sum(r["latency_s"] for r in crows) / n
        line = f"{cond:<14}{avg_score:>8.2f}{avg_tok:>9.0f}{avg_lat:>11.1f}"
        for m in metrics:
            rate = sum(1 for r in crows if r["grade"].get(m)) / n
            line += f"{rate:>11.2f}"
        print(line)
    print("(correct/cited: higher better; stale_error/secret_leak/distracted: LOWER better)")

    print("\n" + "-" * 96)
    print("EXPOSURE (model-independent) — what the prompt itself contained")
    print("-" * 96)
    ex_metrics = ["exposed_stale", "exposed_secret", "distractors_in_ctx"]
    hdr2 = f"{'condition':<14}" + "".join(f"{m:>20}" for m in ex_metrics)
    print(hdr2)
    print("-" * len(hdr2))
    for cond in ("default", "engineered"):
        crows = [r for r in rows if r["condition"] == cond]
        if not crows:
            continue
        n = len(crows)
        line = f"{cond:<14}"
        for m in ex_metrics:
            line += f"{sum(r['exposure'][m] for r in crows) / n:>20.2f}"
        print(line)
    print("(all LOWER is better — dangerous/irrelevant content transmitted to the model)")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Downstream context-engineering A/B via a real agent")
    parser.add_argument("--tasks", default="coding,compliance,rag",
                        help="comma-separated task keys: coding,compliance,rag")
    parser.add_argument("--backend", default="codex", choices=list(BACKENDS))
    parser.add_argument("--model", default=None, help="model id (e.g. gpt-4o, gpt-4.1-nano)")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--dry-run", action="store_true", help="build prompts only, no agent")
    parser.add_argument("--hard", action="store_true",
                        help="anonymize the default dump (recency signal lives only in metadata)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    task_keys = [t.strip() for t in args.tasks.split(",") if t.strip()]
    data = run_ab(task_keys, args.backend, args.timeout, args.dry_run,
                  hard=args.hard, model=args.model)
    print_report(data, args.dry_run)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        print(f"\nSaved A/B results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
