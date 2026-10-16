#!/usr/bin/env bash
# Always-on НИР7 data collection (Linux/macOS). Restarts the service if Python exits.
# Prefer the systemd unit from docs/данные.md on a server; this loop is the fallback.
cd "$(dirname "$0")" || exit 1
PYTHON="${PYTHON:-python3}"
mkdir -p data/raw
while true; do
  "$PYTHON" -u -m collector.service >> data/raw/service.log 2>&1
  echo "$(date -Is) service exited, restarting in 30 s" >> data/raw/service.log
  sleep 30
done
