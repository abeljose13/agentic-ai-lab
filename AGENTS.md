# AGENTS.md

Personal learning lab for an AI agent developer training program. uv-managed, Python 3.12 only (>=3.12,<3.13), not an installable package (`package = false`).

## Commands

```bash
uv sync                                        # install deps from uv.lock
uv run python tests/test_checkpoint_3.py      # run a "test" script
uv run python tests/test_llm_client.py        # basic client smoke script
```

- There is no pytest, lint, format, or typecheck setup. Files under `tests/` are manual scripts, not pytest tests — run them directly with `uv run python`.
- Run everything from the repo root; imports use the `src.*` prefix.

## Environment (required)

- Scripts make **live LLM API calls** (real requests, real cost) unless provider is `ollama`. Never invoke them as a routine verification step.
- A local `.env` (gitignored) supplies keys; template in `.env.example`.
- `LLM_PROVIDER` selects the backend. Valid values: `openai | anthropic | google | ollama | groq`. Note: `.env.example` says `gemini`, which is **not** a valid value — use `google` (code default is `google`).
- Google key is read from `GEMINI_API_KEY` or `GOOGLE_API_KEY`. Other providers use `<NAME>_API_KEY`. Model overrides: `OPENAI_MODEL`, `ANTHROPIC_MODEL`, `GEMINI_MODEL`, `GROQ_MODEL`, `OLLAMA_MODEL`.
- `ollama` targets a LAN server (`MAC_SERVER_IP`/`OLLAMA_PORT`, default 192.168.1.14:11434) — unreachable unless that host is up.

## Architecture

- `src/llm_client/config.py` — `load_env()` (idempotent `.env` loading) and `get_provider_name()` (reads + validates `LLM_PROVIDER`, default `google`); single source for provider config.
- `src/llm_client/llm_client.py` — `LLMClient`: multi-provider calls with exponential-backoff retries (3 attempts) on rate-limit/network errors; non-transient errors re-raise immediately.
- `src/llm_client/providers.py` — `get_provider()` factory returning `LLMProvider` subclasses per `LLM_PROVIDER`.
- `src/llm_client/llm_provider.py` — `LLMProvider` ABC + `generate_structured()`: JSON-schema prompting with Pydantic validation and re-prompt retries.
- `src/llm_client/service.py` — domain example (`classify()` intent classifier).
- `src/api/`, `src/tools/`, `src/workflows/` — empty placeholders for future course checkpoints.
- Env is loaded automatically: both `LLMClient.__init__` and `get_provider()` go through `config.py` — new entrypoints do not need to call `load_dotenv()` manually.

## Conventions

- Docstrings, comments, and prompts are in Spanish; keep that style.
- Anthropic path clamps temperature to [0.0, 1.0]; general validation allows 0.0–2.0 — don't "fix" the clamp.
