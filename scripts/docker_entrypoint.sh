#!/bin/sh
# Render and other PaaS hosts inject PORT; default keeps local Docker behavior.
set -e

PORT="${PORT:-8000}"

case "${SERVICE_MODE:-api}" in
  streamlit)
    exec streamlit run app/app.py \
      --server.address=0.0.0.0 \
      --server.port="${PORT}" \
      --server.headless=true \
      --browser.gatherUsageStats=false
    ;;
  api|*)
    exec uvicorn api.main:app --host 0.0.0.0 --port "${PORT}"
    ;;
esac
