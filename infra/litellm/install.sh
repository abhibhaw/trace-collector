#!/usr/bin/env bash
# Idempotent install of the LiteLLM proxy (+ its postgres) in namespace experiments.
# Never prints secret values. Safe to re-run.
set -euo pipefail

CTX=gepton-base
NS=experiments
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K=(kubectl --context "$CTX" -n "$NS")
PF_PORT="${PF_PORT:-14000}"
KEY_WAIT_SECONDS="${KEY_WAIT_SECONDS:-1800}"

log() { printf '[litellm-install] %s\n' "$*" >&2; }
have_secret() { "${K[@]}" get secret "$1" -o name >/dev/null 2>&1; }

# 1. Internal secrets. Created once; never rotated (postgres password is bound to the PVC at initdb).
if ! have_secret litellm-secrets; then
  log "creating secret litellm-secrets"
  "${K[@]}" create secret generic litellm-secrets \
    --from-literal=LITELLM_MASTER_KEY="sk-$(openssl rand -hex 32)" \
    --from-literal=POSTGRES_PASSWORD="$(openssl rand -hex 24)" >/dev/null
else
  log "secret litellm-secrets exists"
fi

# 2. Postgres.
"${K[@]}" apply -f "$DIR/postgres.yaml"
"${K[@]}" rollout status statefulset/litellm-postgres --timeout=600s

# 3. Config.
"${K[@]}" create configmap litellm-config --from-file=config.yaml="$DIR/config.yaml" \
  --dry-run=client -o yaml | "${K[@]}" apply -f -
CONFIG_HASH="$(shasum -a 256 "$DIR/config.yaml" | cut -c1-16)"

# 4. Proxy. On first install start with 1 replica so only one pod runs DB migrations.
FIRST=0
"${K[@]}" get deploy litellm >/dev/null 2>&1 || FIRST=1
if [ "$FIRST" = 1 ]; then
  sed 's/^  replicas: 2$/  replicas: 1/' "$DIR/deploy.yaml" | "${K[@]}" apply -f -
else
  "${K[@]}" apply -f "$DIR/deploy.yaml"
fi
KEYS_PRESENT=no
have_secret llm-keys && KEYS_PRESENT=yes
# Pod-template annotations: a config change or the arrival of llm-keys restarts the pods.
"${K[@]}" patch deploy litellm --type merge -p \
  "{\"spec\":{\"template\":{\"metadata\":{\"annotations\":{\"peira.tech/config-hash\":\"$CONFIG_HASH\",\"peira.tech/llm-keys-present\":\"$KEYS_PRESENT\"}}}}}" >/dev/null
"${K[@]}" rollout status deploy/litellm --timeout=900s
if [ "$FIRST" = 1 ]; then
  "${K[@]}" scale deploy/litellm --replicas=2
  "${K[@]}" rollout status deploy/litellm --timeout=900s
fi

# 5. Port-forward for admin calls.
"${K[@]}" port-forward svc/litellm "$PF_PORT:4000" >/dev/null 2>&1 &
PF_PID=$!
trap 'kill $PF_PID 2>/dev/null || true' EXIT
for _ in $(seq 1 30); do
  curl -fsS "http://localhost:$PF_PORT/health/liveliness" >/dev/null 2>&1 && break
  sleep 1
done

auth_header() {
  "${K[@]}" get secret litellm-secrets -o jsonpath='{.data.LITELLM_MASTER_KEY}' | base64 -d | sed 's/^/Authorization: Bearer /'
}

# 6. Client virtual key: created once. Each new key would be another $300 budget.
if ! have_secret litellm-client; then
  log "creating virtual key (max_budget 300) into secret litellm-client"
  auth_header | curl -fsS -X POST "http://localhost:$PF_PORT/key/generate" \
      -H @- -H 'Content-Type: application/json' \
      -d '{"key_alias":"peira-agents","max_budget":300,"models":["claude-haiku-5-5","claude-sonnet-5-5"],"metadata":{"owner":"infra-litellm"}}' \
    | jq -je '.key' \
    | "${K[@]}" create secret generic litellm-client --from-file=LLM_PROXY_KEY=/dev/stdin >/dev/null
  log "secret litellm-client created"
else
  log "secret litellm-client exists"
fi

# 7. Upstream key. If missing, wait for the user to create it, then restart pods to pick it up.
if [ "$KEYS_PRESENT" = no ]; then
  log "secret llm-keys missing; polling every 60s for up to $((KEY_WAIT_SECONDS / 60)) min"
  waited=0
  until have_secret llm-keys; do
    if [ "$waited" -ge "$KEY_WAIT_SECONDS" ]; then
      log "llm-keys still missing. Create it, then re-run install.sh."
      exit 2
    fi
    sleep 60
    waited=$((waited + 60))
  done
  log "llm-keys found; restarting proxy"
  "${K[@]}" patch deploy litellm --type merge -p \
    '{"spec":{"template":{"metadata":{"annotations":{"peira.tech/llm-keys-present":"yes"}}}}}' >/dev/null
  "${K[@]}" rollout status deploy/litellm --timeout=900s
fi

log "done. Service: http://litellm.$NS.svc.cluster.local:4000"
