#!/bin/bash

# Production server startup script for ADK agent
# Supports both production (Gunicorn) and development (Uvicorn) modes

set -e  # Exit on error

# Configuration
APP_MODULE="app.main:app"
HOST="0.0.0.0"
PORT="8000"
LOG_CONFIG="logging.conf"

# Change to src directory (where app module is located)
cd /app/src || cd src

# Determine mode: production or development
if [ "$1" = "local" ]; then
    echo "Starting in DEVELOPMENT mode..."
    echo "Using Uvicorn with auto-reload"

    exec uvicorn ${APP_MODULE} \
        --host ${HOST} \
        --port ${PORT} \
        --log-config ${LOG_CONFIG} \
        --reload
else
    echo "Starting in PRODUCTION mode..."
    echo "Using Gunicorn with Uvicorn workers"

    exec gunicorn ${APP_MODULE} \
        --bind ${HOST}:${PORT} \
        --worker-class uvicorn.workers.UvicornWorker \
        --workers 1 \
        --timeout 600 \
        --access-logfile - \
        --log-config ${LOG_CONFIG}
fi
