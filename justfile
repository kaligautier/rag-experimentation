#!/usr/bin/env -S just --justfile
# ADK Agent Template

# Load environment variables from .env file
set dotenv-load

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

# Direct ADK evaluation; each YAML contains one native RAG rubric.
[group('eval')]
eval:
    #!/usr/bin/env bash
    set -uo pipefail
    status=0
    for metric in correctness relevance groundedness retrieval_relevance; do
        just _eval-suite "$metric" || status=$?
    done
    exit "$status"

[group('eval')]
eval-rag: eval

[group('eval')]
eval-correctness: (_eval-suite "correctness")

[group('eval')]
eval-relevance: (_eval-suite "relevance")

[group('eval')]
eval-groundedness: (_eval-suite "groundedness")

[group('eval')]
eval-retrieval-relevance: (_eval-suite "retrieval_relevance")

[group('eval')]
eval-check:
    #!/usr/bin/env bash
    set -euo pipefail
    for metric in correctness relevance groundedness retrieval_relevance; do
        just _promptfoo validate config -c "eval/online/promptfooconfig.$metric.yaml"
    done

[group('eval')]
eval-view ui_port="15500":
    just _promptfoo view --no --host 127.0.0.1 --port {{ ui_port }}

_promptfoo +args: _check-uv
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{ source_directory() }}
    mkdir -p eval/.promptfoo/results
    export PROMPTFOO_CONFIG_DIR="$PWD/eval/.promptfoo"
    export PROMPTFOO_DISABLE_TELEMETRY=1 PROMPTFOO_DISABLE_UPDATE=1
    export PROMPTFOO_ENABLE_OTEL=false
    export OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT="${OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT:-SPAN_ONLY}"
    export ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS="${ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS:-true}"
    export PROMPTFOO_PYTHON="$PWD/.venv/bin/python"
    export PYTHONPATH="$PWD/src:$PWD"
    export VERTEX_PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-}"
    export VERTEX_REGION="${GOOGLE_CLOUD_LOCATION:-global}"
    export PROMPTFOO_GRADER_MODEL="${PROMPTFOO_GRADER_MODEL:-gemini-2.5-pro}"
    export PROMPTFOO_EMBEDDING_MODEL="${PROMPTFOO_EMBEDDING_MODEL:-gemini-embedding-001}"
    export PROMPTFOO_EMBEDDING_REGION="${PROMPTFOO_EMBEDDING_REGION:-europe-west1}"
    export LOG_LEVEL=info
    uv run --project "$PWD" --frozen npx --yes promptfoo@0.125.1 {{ args }}

_eval-suite metric:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{ source_directory() }}
    case "{{ metric }}" in correctness|relevance|groundedness|retrieval_relevance) ;; *) exit 2 ;; esac
    # Delete only this campaign's generated report, so stale results cannot mask failure.
    rm -f "eval/.promptfoo/results/{{ metric }}.json"
    just _promptfoo eval -c eval/online/promptfooconfig.{{ metric }}.yaml --no-cache --max-concurrency 1 -o eval/.promptfoo/results/{{ metric }}.json
