#!/usr/bin/env bash
# Idempotent install of the tools service into the experiments namespace (kube context gepton-base).
# Usage: infra/tools/install.sh            # apply everything and wait for rollout
set -euo pipefail
cd "$(dirname "$0")"
K="kubectl --context gepton-base -n experiments"

# 1. valkey password: create once, never print it.
if ! $K get secret tools-valkey >/dev/null 2>&1; then
  $K create secret generic tools-valkey --from-literal=password="$(openssl rand -hex 24)" >/dev/null
  echo "created secret tools-valkey"
fi

# 2. App code ConfigMap (all files under app/, flat keys).
$K create configmap tools-app --from-file=app/ --dry-run=client -o yaml | $K apply -f - >/dev/null
SUM=$(find app -maxdepth 1 -type f | sort | xargs cat | shasum -a 256 | cut -c1-16)
echo "app checksum $SUM"

# 3. valkey, then the tools deployment (checksum annotation rolls pods when code changes).
$K apply -f k8s/valkey.yaml
$K rollout status deploy/tools-valkey --timeout=600s
sed "s/__APP_CHECKSUM__/$SUM/" k8s/tools.yaml | $K apply -f -
$K rollout status deploy/tools --timeout=900s
$K get pods -l part-of=peira-tools -o wide
