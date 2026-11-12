#!/usr/bin/env bash
# Build one venv per requirements file on the shared volume: /work/venvs/<name>.
# agents/<family>/requirements.txt          -> venv <family>
# agents/<family>/requirements-<x>.txt      -> venv <family>-<x>
# Usage: scripts/build-venvs.sh [family ...]   (default: all)
set -euo pipefail
K="kubectl --context gepton-base -n experiments"
POD=$($K get pod -l app=toolbox -o jsonpath='{.items[0].metadata.name}')
$K exec -i "$POD" -- bash -s -- "$@" <<'INNER'
set -euo pipefail
cd /work/repo
export UV_HTTP_TIMEOUT=300 UV_CONCURRENT_DOWNLOADS=4 UV_HTTP_RETRIES=6
fams=("$@"); [ ${#fams[@]} -eq 0 ] && fams=($(ls agents))
for fam in "${fams[@]}"; do
  for req in agents/$fam/requirements*.txt; do
    [ -f "$req" ] || continue
    base=$(basename "$req" .txt); suffix=${base#requirements}; name="$fam${suffix}"
    echo "=== $name from $req"
    rm -rf /work/venvs/$name
    uv venv -q -p 3.12 /work/venvs/$name
    ok=0; for attempt in 1 2 3; do uv pip install -q -p /work/venvs/$name -r "$req" /work/repo/harness && { ok=1; break; }; echo "retry $attempt for $name"; sleep 10; done
    if [ $ok = 1 ]; then
      /work/venvs/$name/bin/python -c "import peira_harness, langfuse; print('ok $name')"
    else
      echo "FAILED $name"
    fi
  done
done
INNER
