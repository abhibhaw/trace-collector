# Self-hosted Langfuse

Langfuse v4 (chart `langfuse/langfuse` 2.2.4, app 4.77.0) runs in namespace `experiments` on context `gepton-base`.
In-cluster URL: `http://langfuse-web.experiments.svc.cluster.local:3000`.
Organization `peira`, project `peira-traces`, and its API keys exist from the first boot (headless init).
Sign-up and telemetry are disabled.

## Files

| File | Content |
|---|---|
| `values.yaml` | Helm values. No secret values. |
| `clickhouse.yaml` | Single-node ClickHouse StatefulSet, Service, and config. |
| `install.sh` | Re-runnable install. Creates secrets only if they are missing. |

## Install or upgrade

```bash
infra/langfuse/install.sh
```

The script does these steps:

1. It creates the secrets with `openssl rand` if they do not exist.
2. It applies `clickhouse.yaml` and waits for ClickHouse.
3. It runs `helm upgrade --install --no-hooks`.
4. It creates the S3 bucket `langfuse` inside the SeaweedFS pod.
5. It creates `langfuse-client` from the init project keys.

The chart's bucket hook Job has no nodeSelector or tolerations.
Thus the install uses `--no-hooks`, and the script creates the bucket with `kubectl exec`.

## Open the UI

```bash
kubectl --context gepton-base -n experiments port-forward svc/langfuse-web 3000:3000
```

Open http://localhost:3000.
Log in as `admin@peira.local`.
To read the admin password, run this command:

```bash
kubectl --context gepton-base -n experiments get secret langfuse-app-secrets \
  -o jsonpath='{.data.init-user-password}' | base64 -d; echo
```

## Use the API locally

Keep the port-forward open.
Then load the client keys into your shell:

```bash
export LANGFUSE_PUBLIC_KEY="$(kubectl --context gepton-base -n experiments get secret langfuse-client -o jsonpath='{.data.LANGFUSE_PUBLIC_KEY}' | base64 -d)"
export LANGFUSE_SECRET_KEY="$(kubectl --context gepton-base -n experiments get secret langfuse-client -o jsonpath='{.data.LANGFUSE_SECRET_KEY}' | base64 -d)"
export LANGFUSE_BASE_URL=http://localhost:3000
```

Health check: `curl -s http://localhost:3000/api/public/health`.

## Read API in Langfuse v4

This deployment runs in Langfuse v4 `events_only` mode.
Ingestion goes through OpenTelemetry (`POST /api/public/otel/v1/traces`); the v3+ Python SDK uses it automatically.
The legacy trace endpoints `GET /api/public/traces` and `GET /api/public/traces/{id}` return 404 with an "events_only" message.
To read a trace back, use `GET /api/public/v2/observations?traceId=<id>&fields=core,basic,usage`.
The `tags` query parameter on that endpoint has no effect; filter by `traceId`, or use the UI.
Tags and session IDs are stored on every observation (ClickHouse table `events_core`).
The UI shows traces, tags, and sessions as usual.

## Secrets

All values come from `openssl rand`.
No value is in Git or in a file.

| Secret | Keys | Used by |
|---|---|---|
| `langfuse-app-secrets` | `salt`, `encryption-key`, `nextauth-secret`, `init-user-password`, `init-public-key`, `init-secret-key` | web, worker |
| `langfuse-pg-auth` | `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `USERDB_USER`, `USERDB_PASSWORD` | postgres, web, worker |
| `langfuse-valkey-auth` | `default` | valkey, web, worker |
| `langfuse-clickhouse-auth` | `password` | clickhouse, web, worker |
| `langfuse-s3-creds` | `accessKey`, `secretKey`, `seaweedfs_s3_config` | seaweedfs, web, worker |
| `langfuse-client` | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` | agent pods (`envFrom`) |

The chart also makes an empty `langfuse-app` secret.
Do not delete `langfuse-app-secrets`.
If `salt` or `encryption-key` is lost, the stored API keys and encrypted data become unusable.

## Component sizing

All pods use the `experiments` Karpenter pool (arm64) and have the contract nodeSelector and toleration.
All pods have `karpenter.sh/do-not-disrupt: "true"`, so consolidation does not restart them during ingestion.

| Component | Kind | CPU request / limit | Memory request / limit | Storage (`managed-csi`) |
|---|---|---|---|---|
| langfuse-web | Deployment, 1 replica | 500m / 1 | 1.5Gi / 4Gi | - |
| langfuse-worker | Deployment, 1 replica | 500m / 1 | 1Gi / 3Gi | - |
| clickhouse | StatefulSet, 1 replica | 1 / 2 | 3Gi / 6Gi | 50Gi |
| langfuse-postgresql | StatefulSet, 1 replica | 250m / 1 | 512Mi / 1Gi | 10Gi |
| langfuse-redis (valkey) | Deployment, 1 replica | 100m / 500m | 256Mi / 512Mi | 5Gi |
| langfuse-s3-all-in-one (seaweedfs) | Deployment, 1 replica | 250m / 1 | 512Mi / 1Gi | 30Gi |
| **Total** | | **2.6 CPU** | **~6.8Gi** | 95Gi |

The target load is about 10,000 traces with 5-60 observations each over a few hours.
That is fewer than 100 events per second, which one web and one worker handle.
If the worker queue grows, raise `langfuse.worker.replicas`.
The web limit is 4Gi because startup peaked above 2Gi and was OOMKilled once; steady state is about 0.7Gi.

ClickHouse runs without the operator and without Keeper (`CLICKHOUSE_CLUSTER_ENABLED=false`).
Its system log tables keep 3 days of data, and the noisy ones are off (see `clickhouse.yaml`).

## Uninstall

```bash
helm --kube-context gepton-base -n experiments uninstall langfuse
kubectl --context gepton-base -n experiments delete -f infra/langfuse/clickhouse.yaml
```

PVCs and secrets stay.
Delete them by hand only if you want to lose all trace data.
