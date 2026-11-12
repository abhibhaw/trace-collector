#!/usr/bin/env bash
# Copy repo code (harness, agents, scripts) to /work/repo on the shared volume. Excludes venvs and results.
set -euo pipefail
cd "$(dirname "$0")/.."
K="kubectl --context gepton-base -n experiments"
POD=$($K get pod -l app=toolbox -o jsonpath='{.items[0].metadata.name}')
COPYFILE_DISABLE=1 tar --no-xattrs -czf - --exclude='.venv' --exclude='__pycache__' --exclude='results' --exclude='*.pyc' harness agents scripts CONTRACT.md \
  | $K exec -i "$POD" -- bash -c 'rm -rf /work/repo.new && mkdir -p /work/repo.new && tar xzf - -C /work/repo.new && rm -rf /work/repo && mv /work/repo.new /work/repo && echo synced: $(find /work/repo/agents -name agent.py | wc -l) agents'
