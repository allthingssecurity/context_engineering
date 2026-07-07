# Downstream A/B raw logs

These JSON files are the actual outputs of `agent_ab.py` runs against real
agents on the maintainer's key. Only the context differs between conditions
(`default` raw dump vs `engineered` pipeline output). No secret is stored in
these files (the synthetic `AKIA…EXAMPLE` never appears in any answer).

| file | agent / model | mode | headline |
| --- | --- | --- | --- |
| `hermes_hard.json` | hermes-agent (gpt-5-mini) | hard | default 0.00 → engineered 1.00 |
| `hermes_easy.json` | hermes-agent (gpt-5-mini) | easy | 1.00 / 1.00 (parity) |
| `nano_hard.json` | openai gpt-4.1-nano | hard | default 0.00 (hallucinated "14–30 days") → engineered 1.00 |
| `nano_easy.json` | openai gpt-4.1-nano | easy | 1.00 / 1.00 (parity) |
| `gpt4o_hard.json` | openai gpt-4o | hard | default 0.00 → engineered 1.00 |
| `ab_hard.json` | codex (ChatGPT) | hard | default 0.00 → engineered 1.00 |

Each record has: `condition`, `context_tokens`, `prompt_tokens`, `selected`
(resource ids in context), `exposure` (stale/secret/distractors transmitted),
`answer` (the agent's actual text), `grade`, and `score`.

Reproduce (needs your own key in `~/.oai_key` or `$OPENAI_API_KEY`):

```bash
python -m context_engineering.pipeline.agent_ab --backend openai --model gpt-4.1-nano --hard
python -m context_engineering.pipeline.agent_ab --backend hermes --hard
```

See `docs/EVALUATION.md` for the full task definitions and grading rubric.
