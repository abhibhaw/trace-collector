# shellcheck shell=bash
# Source this file:  source scripts/dev-env.sh
#
# - Starts background port-forwards (if not already running) for:
#     litellm 4000, langfuse-web 3000, tools 8080
#   Each forward runs in a restart loop, so it survives pod restarts.
#   Safe when many shells source this at once: a per-port lock (atomic mkdir) plus a
#   listening-port check. Existing forwards are never killed.
# - Exports LLM_PROXY_URL, LLM_PROXY_KEY, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
#   LANGFUSE_BASE_URL, TOOLS_URL. Values are never printed.
# - Missing services or secrets are tolerated (a short note goes to stderr).
#
# To stop all forwards: pkill -f 'peira-pf-loop'

_peira_ctx=gepton-base
_peira_ns=experiments
_peira_lockroot="${TMPDIR:-/tmp}/peira-pf"
mkdir -p "$_peira_lockroot" 2>/dev/null

_peira_listening() { nc -z 127.0.0.1 "$1" >/dev/null 2>&1; }

# _peira_forward <service> <local_port> <remote_port>
_peira_forward() {
  local svc=$1 lport=$2 rport=$3 lock="$_peira_lockroot/$2.lock" pid
  _peira_listening "$lport" && return 0
  if ! mkdir "$lock" 2>/dev/null; then
    pid=$(cat "$lock/pid" 2>/dev/null)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      return 0 # another shell owns a live loop for this port
    fi
    # Stale lock: the owning loop is gone. Re-take it atomically.
    rm -rf "$lock" 2>/dev/null
    mkdir "$lock" 2>/dev/null || return 0
  fi
  if ! kubectl --context "$_peira_ctx" -n "$_peira_ns" get svc "$svc" -o name >/dev/null 2>&1; then
    rmdir "$lock" 2>/dev/null
    echo "dev-env: service $svc not found; skipping port $lport" >&2
    return 0
  fi
  nohup bash -c "exec -a peira-pf-loop-$lport bash -c 'while true; do kubectl --context $_peira_ctx -n $_peira_ns port-forward svc/$svc $lport:$rport >/dev/null 2>&1; sleep 2; done'" \
    >/dev/null 2>&1 &
  echo $! >"$lock/pid"
  disown 2>/dev/null
  return 0
}

# _peira_secret <secret> <key>  -> value on stdout (captured, never echoed)
_peira_secret() {
  kubectl --context "$_peira_ctx" -n "$_peira_ns" get secret "$1" -o "jsonpath={.data.$2}" 2>/dev/null | base64 -d 2>/dev/null
}

_peira_forward litellm 4000 4000
_peira_forward langfuse-web 3000 3000
_peira_forward tools 8080 8080

export LLM_PROXY_URL=http://localhost:4000
export LANGFUSE_BASE_URL=http://localhost:3000
export TOOLS_URL=http://localhost:8080

_peira_v=$(_peira_secret litellm-client LLM_PROXY_KEY)
if [ -n "$_peira_v" ]; then export LLM_PROXY_KEY="$_peira_v"; else echo "dev-env: secret litellm-client missing" >&2; fi
_peira_v=$(_peira_secret langfuse-client LANGFUSE_PUBLIC_KEY)
if [ -n "$_peira_v" ]; then export LANGFUSE_PUBLIC_KEY="$_peira_v"; else echo "dev-env: secret langfuse-client missing" >&2; fi
_peira_v=$(_peira_secret langfuse-client LANGFUSE_SECRET_KEY)
[ -n "$_peira_v" ] && export LANGFUSE_SECRET_KEY="$_peira_v"
unset _peira_v

# Wait up to ~10s for the proxy forward so callers can use it immediately.
for _peira_i in 1 2 3 4 5 6 7 8 9 10; do
  _peira_listening 4000 && break
  sleep 1
done
unset _peira_i
_peira_listening 4000 || echo "dev-env: localhost:4000 not listening yet" >&2
true
