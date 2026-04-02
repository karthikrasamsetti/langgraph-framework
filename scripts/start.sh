#!/bin/bash
set -e

echo "Starting LangGraph Agent Framework"
echo "Provider: ${LLM_PROVIDER:-not set}"
echo "RAG: ${RAG_ENABLED:-false}"
echo "LangSmith: ${LANGSMITH_TRACING:-false}"

# Create required directories if they don't exist
mkdir -p /app/vector_db /app/my_docs /app/logs

# Start the server
exec uvicorn api.server:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level warning