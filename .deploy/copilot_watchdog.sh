#!/bin/bash
# Watchdog: keeps the Agentic Support Copilot API + localhost.run tunnel alive.
# Runs every 10 minutes via cron (copilot-tunnel-watchdog). Idempotent.
# Adapted from ~/workspace/ollama-rag-api/.deploy/tunnel_watchdog.sh.
set -u
REPO="$HOME/workspace/agentic-support-copilot"
DEPLOY_RAG="$HOME/workspace/ollama-rag-api/.deploy"  # lr_key + pc.py live here
URL_FILE="$HOME/workspace/your_files/agentic-support-copilot-url.txt"
LOG="/tmp/copilot_watchdog.log"

if ! pgrep -f "[u]vicorn app.main:app --host 0.0.0.0 --port 8001" >/dev/null 2>&1; then
  # double-fork detach: subshell exits immediately, init adopts the grandchild
  # (setsid alone left the daemon as our child -> do_wait hang; use `;` not `&&`).
  ( cd "$REPO/backend"; setsid nohup env API_KEY=demo-key LLM_PROVIDER=fake DEMO_MODE=true LOG_LEVEL=INFO ../.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001 > /tmp/copilot_api.log 2>&1 < /dev/null & )
  sleep 8
fi

# NOTE: the pattern must match ONLY this tunnel (port 8001), not the rag-api one (8000).
if ! pgrep -f "[l]ocalhost:8001" >/dev/null 2>&1; then
  rm -f /tmp/copilot_localrun.log
  setsid nohup ssh -i "$DEPLOY_RAG/lr_key" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
    -o ProxyCommand="python3 $DEPLOY_RAG/pc.py %h %p" \
    -R 80:localhost:8001 nokey@localhost.run > /tmp/copilot_localrun.log 2>&1 < /dev/null &
  sleep 25
fi

# Use the LAST URL in the log — after a reconnect it can hold a stale banner URL too.
URL=$(grep -aoE "https?://[a-zA-Z0-9.-]+\.lhr\.life" /tmp/copilot_localrun.log 2>/dev/null | tail -1)
if [ -n "$URL" ]; then
  echo "$URL" > "$URL_FILE"
  echo "$(date -u +%FT%TZ) ok $URL" >> "$LOG"
else
  echo "$(date -u +%FT%TZ) FAIL no URL in tunnel log" >> "$LOG"
fi
