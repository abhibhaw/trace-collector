# LiteLLM proxy

Service: `http://litellm.experiments.svc.cluster.local:4000`.
Owner: infra-litellm.

## What it does

- Holds the Anthropic key (secret `llm-keys`, key `ANTHROPIC_API_KEY`, created by the user).
- Serves `claude-haiku-5-5` and `claude-sonnet-5-5` in OpenAI format (`/v1/chat/completions`) and Anthropic format (`/v1/messages`).
- Enforces a $300 total spend cap with two guards:
  - The client virtual key (secret `litellm-client`) has `max_budget: 300`.
  - The proxy-wide `max_budget: 300` is stored in Postgres (`litellm_proxy_budget` row).
    LiteLLM requires `budget_duration` with it, so it is `365d` (no reset during the project).
- Strips `temperature`, `top_p` and `top_k` on both routes (`additional_drop_params`).
- Sends no callbacks anywhere (no Langfuse, no OTEL). Telemetry is off (`LITELLM_TELEMETRY_DISABLED=true`).

## Files

| File | Purpose |
|---|---|
| `config.yaml` | Proxy config, source of ConfigMap `litellm-config`. No secrets. |
| `deploy.yaml` | Deployment (2 replicas) and Service `litellm`. |
| `postgres.yaml` | StatefulSet and Service `litellm-postgres` (postgres:16, managed-csi 5Gi). |
| `install.sh` | Idempotent installer. |

## Install

```bash
infra/litellm/install.sh
```

The script does these steps:

1. Creates secret `litellm-secrets` (`LITELLM_MASTER_KEY`, `POSTGRES_PASSWORD`) once, from `openssl rand`.
   It never rotates them, because the Postgres password is bound to the PVC at initdb.
2. Applies Postgres, the ConfigMap and the Deployment.
   A pod-template annotation holds the config hash, so a config change rolls the pods.
3. Creates the virtual key once and stores it in secret `litellm-client` (`LLM_PROXY_KEY`).
   The key goes from curl through jq into kubectl and is never printed.
   A re-run does not create a second key, because each key is another $300 budget.
4. If `llm-keys` is missing, it polls every 60s for up to 30 min, then restarts the pods.

## Image

`ghcr.io/berriai/litellm:v1.109.1`, pinned by index digest `sha256:11175975...`.
The index has `linux/arm64` and `linux/amd64`, checked with the GHCR registry API.

## Limits and retries

- rpm/tpm are per replica, because there is no Redis.
  Per replica: haiku 200 rpm / 1M tpm, sonnet 50 rpm / 400K tpm.
  Total with 2 replicas: haiku 400 rpm, sonnet 100 rpm.
- `num_retries: 3`.
  Retries on 429 (`RateLimitError`) and 500/529 (`InternalServerError`).
  The minimum backoff is 2s, and LiteLLM honors `retry-after`.
- `disable_cooldowns: true`.
  Each model has one deployment, so a cooldown would fail every request for that model.
- `request_timeout: 600`.

## Pricing (model_info)

| Model | Input / 1M | Output / 1M | Cache read / 1M |
|---|---|---|---|
| claude-haiku-5-5 | 0.10 | 0.50 | LiteLLM map (0.01) |
| claude-sonnet-5-5 | 2.00 | 10.00 | 0.20 (overrides map value 0.10) |

## Admin access

```bash
kubectl --context gepton-base -n experiments port-forward svc/litellm 4000:4000 &
kubectl --context gepton-base -n experiments get secret litellm-secrets -o jsonpath='{.data.LITELLM_MASTER_KEY}' \
  | base64 -d | sed 's/^/Authorization: Bearer /' \
  | curl -s -H @- 'http://localhost:4000/key/info?key=...' | jq .
```

Spend for the client key: `GET /key/info` with the client key itself as bearer.

## Smoke test

```bash
source scripts/dev-env.sh
uv run --with anthropic --with openai python scripts/litellm-smoke.py
```

## Smoke results

Pending.
Secret `llm-keys` did not exist within the 30 min wait window, so no upstream call has run yet.
After the user creates it, run `infra/litellm/install.sh` (it restarts the pods) and then the smoke test.

## Thinking and multi-turn tool use

Risk: frameworks that use the OpenAI route send back `content` + `tool_calls` only, without thinking blocks.
`scripts/litellm-smoke.py` tests this (`tools3`) and the full-replay variant (`tools3_full`).
If `tools3` fails on the OpenAI route, the planned proxy fix is per deployment in `config.yaml`:

- `claude-sonnet-5-5`: `thinking: {type: between_tools}` (valid at effort high or below).
- `claude-haiku-5-5`: `thinking: {type: disabled}`.

Do not set `litellm_settings.modify_params: true`.
With it, LiteLLM drops the deployment `thinking` param when the history has no thinking blocks, which would undo the fix.
