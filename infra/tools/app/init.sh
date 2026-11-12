#!/bin/sh
# Init container: build the venv and download the SQLite sample databases into shared emptyDirs.
set -eu
# UV_NATIVE_TLS: uv's bundled rustls stack timed out against PyPI from this cluster; native TLS works.
export UV_CACHE_DIR=/work/uv-cache UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never UV_NATIVE_TLS=1 UV_HTTP_TIMEOUT=120 UV_CONCURRENT_DOWNLOADS=8
if [ ! -x /work/venv/bin/uvicorn ]; then
  n=0
  until { uv venv --clear --python /usr/local/bin/python3.12 /work/venv && uv pip install --python /work/venv/bin/python --only-binary :all: -r /app/requirements.txt; }; do
    n=$((n + 1))
    if [ "$n" -ge 3 ]; then
      echo "uv failed 3 times; falling back to pip"
      /usr/local/bin/python3.12 -m venv --clear /work/venv
      /work/venv/bin/pip install --no-cache-dir --only-binary :all: --retries 10 --timeout 60 -r /app/requirements.txt
      break
    fi
    sleep $((n * 5))
  done
fi
rm -rf /work/uv-cache
fetch() { # url dest
  [ -s "$2" ] && return 0
  n=0
  until /work/venv/bin/python -I - "$1" "$2" <<'PY'
import sys, urllib.request
url, dest = sys.argv[1], sys.argv[2]
req = urllib.request.Request(url, headers={"User-Agent": "peira-trace-collector/0.1 (research experiment; https://peira.tech)"})
data = urllib.request.urlopen(req, timeout=60).read()
assert data[:16] == b"SQLite format 3\x00", "not a SQLite file"
open(dest + ".tmp", "wb").write(data)
import os; os.replace(dest + ".tmp", dest)
print("downloaded", dest, len(data))
PY
  do n=$((n + 1)); [ "$n" -ge 6 ] && exit 1; sleep $((n * 5)); done
}
fetch https://github.com/lerocha/chinook-database/releases/download/v1.4.5/Chinook_Sqlite.sqlite /data/chinook.sqlite
fetch https://github.com/jpwhite3/northwind-SQLite3/raw/main/dist/northwind.db /data/northwind.sqlite
chmod 0444 /data/*.sqlite
echo "init done"
