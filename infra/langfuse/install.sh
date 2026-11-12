#!/usr/bin/env bash
# Install or upgrade self-hosted Langfuse in namespace `experiments` on context `gepton-base`.
# Re-runnable: secrets are created only when missing; ClickHouse and the Helm release are applied idempotently.
# No secret value is printed or written to disk.
set -euo pipefail

CTX=gepton-base
NS=experiments
RELEASE=langfuse
CHART="${LANGFUSE_CHART:-langfuse/langfuse}"   # set LANGFUSE_CHART to a local chart dir to install offline
CHART_VERSION=2.2.4
BASE_URL="http://langfuse-web.${NS}.svc.cluster.local:3000"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

k() { kubectl --context "$CTX" -n "$NS" "$@"; }

secret_exists() { k get secret "$1" >/dev/null 2>&1; }

hex() { openssl rand -hex "$1"; }

ensure_secrets() {
  # Langfuse app credentials + headless-init values.
  if ! secret_exists langfuse-app-secrets; then
    k create secret generic langfuse-app-secrets \
      --from-literal=salt="$(hex 32)" \
      --from-literal=encryption-key="$(hex 32)" \
      --from-literal=nextauth-secret="$(hex 32)" \
      --from-literal=init-user-password="$(hex 16)" \
      --from-literal=init-public-key="pk-lf-$(hex 16)" \
      --from-literal=init-secret-key="sk-lf-$(hex 24)"
  fi

  # Postgres: superuser + application user (keys match the groundhog2k postgres chart).
  if ! secret_exists langfuse-pg-auth; then
    k create secret generic langfuse-pg-auth \
      --from-literal=POSTGRES_USER=postgres \
      --from-literal=POSTGRES_PASSWORD="$(hex 24)" \
      --from-literal=POSTGRES_DB=langfuse \
      --from-literal=USERDB_USER=langfuse \
      --from-literal=USERDB_PASSWORD="$(hex 24)"
  fi

  # Valkey: one key per ACL user.
  if ! secret_exists langfuse-valkey-auth; then
    k create secret generic langfuse-valkey-auth --from-literal=default="$(hex 24)"
  fi

  # ClickHouse `default` user.
  if ! secret_exists langfuse-clickhouse-auth; then
    k create secret generic langfuse-clickhouse-auth --from-literal=password="$(hex 24)"
  fi

  # SeaweedFS S3: access key, secret key, and the IAM config SeaweedFS reads (same secret key).
  if ! secret_exists langfuse-s3-creds; then
    (
      sk="$(hex 24)"
      k create secret generic langfuse-s3-creds \
        --from-literal=accessKey=langfuse \
        --from-literal=secretKey="$sk" \
        --from-literal=seaweedfs_s3_config="$(jq -cn --arg sk "$sk" \
          '{identities:[{name:"langfuse",credentials:[{accessKey:"langfuse",secretKey:$sk}],actions:["Admin","Read","Write","List","Tagging"]}]}')"
    )
  fi
}

ensure_client_secret() {
  # Copy the init project keys into the secret agent pods load with envFrom. Values never leave the pipe.
  if ! secret_exists langfuse-client; then
    k get secret langfuse-app-secrets -o json \
      | jq --arg ns "$NS" --arg url "$BASE_URL" '{
          apiVersion: "v1", kind: "Secret", type: "Opaque",
          metadata: {name: "langfuse-client", namespace: $ns, labels: {"app.kubernetes.io/part-of": "langfuse"}},
          data: {
            LANGFUSE_PUBLIC_KEY: .data["init-public-key"],
            LANGFUSE_SECRET_KEY: .data["init-secret-key"],
            LANGFUSE_BASE_URL: ($url | @base64)
          }}' \
      | k create -f -
  fi
}

ensure_bucket() {
  # The chart's bucket-creation hook Job has no nodeSelector/tolerations, so hooks are disabled
  # and the bucket is created here, inside the SeaweedFS pod.
  k rollout status deploy/${RELEASE}-s3-all-in-one --timeout=10m
  k exec deploy/${RELEASE}-s3-all-in-one -- sh -ec '
    if echo "s3.bucket.list" | weed shell 2>/dev/null | awk "{print \$1}" | grep -Fxq langfuse; then
      echo "bucket langfuse exists"
    else
      echo "s3.bucket.create --name langfuse" | weed shell
    fi'
}

main() {
  kubectl config get-contexts "$CTX" >/dev/null
  ensure_secrets

  k apply -f "$DIR/clickhouse.yaml"
  k rollout status statefulset/clickhouse --timeout=15m

  local version_args=()
  [[ -d "$CHART" ]] || version_args=(--version "$CHART_VERSION")
  helm --kube-context "$CTX" upgrade --install "$RELEASE" "$CHART" ${version_args[@]+"${version_args[@]}"} \
    -n "$NS" -f "$DIR/values.yaml" --no-hooks --wait --timeout 20m

  ensure_bucket
  ensure_client_secret

  k get pods -l 'app.kubernetes.io/instance=langfuse' -o wide
  k get pods -l 'app.kubernetes.io/name=clickhouse' -o wide
  echo "Langfuse web: $BASE_URL"
}

main "$@"
