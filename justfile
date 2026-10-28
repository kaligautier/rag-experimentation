#!/usr/bin/env -S just --justfile
# ADK Agent Template

# Load environment variables from .env file
set dotenv-load

promptfoo-version := "0.125.1"

# Internal helper to check UV is available
_check-uv:
    #!/usr/bin/env bash
    if ! command -v uv &> /dev/null; then
        echo "ERROR: UV is required but not installed."
        echo "Please install UV from: https://docs.astral.sh/uv/"
        exit 1
    fi

# Install dependencies using UV
[group('setup')]
install: _check-uv
    echo "Installing dependencies with UV..."
    cd {{ source_directory() }} && uv sync

# Apply Liquibase migrations to the docker-compose database
[group('setup')]
migrate:
    cd {{ source_directory() }} && docker compose run --rm liquibase

# Launch agent as API server
[group('run')]
api port="8000": _check-uv
    cd {{ source_directory() }}/src && uv run uvicorn app.main:app --reload --host 127.0.0.1 --port {{ port }}

# Format code with ruff
[group('quality')]
format: _check-uv
    cd {{ source_directory() }} && uv run ruff format src/ eval/

[group('quality')]
sort-imports: _check-uv
    cd {{ source_directory() }} && uv run ruff check --select I --fix src/ eval/

# Lint code with ruff
[group('quality')]
lint: _check-uv
    cd {{ source_directory() }} && uv run ruff check src/ eval/

# Run full test suite with coverage
[group('quality')]
test *args: _check-uv
    cd {{ source_directory() }}/src && uv run pytest --cov=app --cov-report=term-missing {{ args }}

# Integration tests use isolated schemas in the local pgvector database.
[group('quality')]
test-integration *args: _check-uv
    cd {{ source_directory() }}/src && RAG_TEST_DB_URL="${RAG_TEST_DB_URL:-${RAG_DB_URL}}" uv run pytest test/integration -q {{ args }}

# Run all quality checks (format + lint + test) - required before commit
[group('quality')]
pre-commit: format lint test

# Build Docker image
[group('docker')]
build:
    cd {{ source_directory() }} && docker build --no-cache -t adk-agent-template:latest .

# Run Docker container with environment variables from .env
[group('docker')]
run:
    cd {{ source_directory() }} && docker run -p 8000:8000 \
        -e GOOGLE_GENAI_USE_VERTEXAI="${GOOGLE_GENAI_USE_VERTEXAI}" \
        -e GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT}" \
        -e GOOGLE_CLOUD_LOCATION="${GOOGLE_CLOUD_LOCATION}" \
        adk-agent-template:latest local

# Local Promptfoo scaffold; no API server is needed for the Python provider.
[group('eval')]
eval-check:
    just _promptfoo validate config -c eval/rag/promptfoo/config.cjs

[group('eval')]
eval-smoke:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{ source_directory() }}
    status=0
    just _promptfoo eval -c eval/rag/promptfoo/tests/config.cjs --no-cache -o .promptfoo/smoke.json || status=$?
    # Promptfoo exits 100 for the intentionally failing fixture cases.
    if [[ "$status" != 100 ]]; then
      echo "Expected an evaluation failure exit (100), got $status" >&2
      exit 1
    fi
    node eval/rag/promptfoo/tests/check-results.cjs .promptfoo/smoke.json

[group('eval')]
eval-rag:
    #!/usr/bin/env bash
    set -euo pipefail
    : "${GOOGLE_CLOUD_PROJECT:?Set GOOGLE_CLOUD_PROJECT in .env}"
    : "${GOOGLE_CLOUD_LOCATION:?Set GOOGLE_CLOUD_LOCATION in .env}"
    : "${RAG_EVAL_JUDGE_MODEL:?Set a Vertex judge model in .env}"
    if [[ "$RAG_EVAL_JUDGE_MODEL" == "${MODEL:-gemini-2.5-flash}" ]]; then
      echo "Choose a judge model different from the agent model" >&2
      exit 2
    fi
    just _promptfoo eval -c eval/rag/promptfoo/config.cjs --no-cache -o .promptfoo/rag.json

[group('eval')]
eval-view:
    just _promptfoo view --host 127.0.0.1 --no

_promptfoo +args: _check-uv
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{ source_directory() }}
    mkdir -p .promptfoo
    export PROMPTFOO_CONFIG_DIR="{{ source_directory() }}/.promptfoo"
    export PROMPTFOO_DISABLE_TELEMETRY=1 PROMPTFOO_DISABLE_UPDATE=1
    export PROMPTFOO_PYTHON="{{ source_directory() }}/.venv/bin/python"
    export PYTHONPATH="{{ source_directory() }}/src:{{ source_directory() }}"
    export LOG_LEVEL=error
    uv run --frozen npx --yes promptfoo@{{promptfoo-version}} {{args}}
