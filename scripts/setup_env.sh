#!/usr/bin/env bash
# Sudo-free Python environment bootstrap.
#
# This box has no `pip` and no `ensurepip`, and `sudo` needs a password, so
# apt-get is not an option. get-pip.py is the supported way in.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.venv"
PY="$VENV/bin/python"

log() { printf '\033[1;34m[setup_env]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[setup_env] %s\033[0m\n' "$*" >&2; exit 1; }

command -v python3 >/dev/null || die "python3 not found"

if [[ ! -x "$PY" ]]; then
  log "creating venv at $VENV"
  # --without-pip: this interpreter has no ensurepip module.
  python3 -m venv --without-pip "$VENV" 2>/dev/null \
    || die "failed to create venv"
fi

if ! "$PY" -m pip --version >/dev/null 2>&1; then
  log "bootstrapping pip via get-pip.py (no sudo required)"
  tmp="$(mktemp -d)"
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$tmp/get-pip.py"
  "$PY" "$tmp/get-pip.py" --quiet
  rm -rf "$tmp"
fi

log "pip: $("$PY" -m pip --version)"

# CPU-only torch: this host has no NVIDIA driver, so the default (CUDA)
# wheels are several GB of code that can never execute.
TORCH_INDEX="https://download.pytorch.org/whl/cpu"

log "installing core dependencies"
"$PY" -m pip install --quiet --upgrade pip wheel

log "installing CPU-only torch (no GPU on this host)"
if ! "$PY" -m pip install --quiet --index-url "$TORCH_INDEX" torch; then
  log "cpu-index torch install failed; falling back to default index"
  "$PY" -m pip install --quiet torch
fi

log "installing -r requirements.txt"
"$PY" -m pip install --quiet -r "$ROOT/requirements.txt"

log "installing -r requirements-embeddings.txt"
"$PY" -m pip install --quiet -r "$ROOT/requirements-embeddings.txt" \
  || log "WARNING: embedding deps failed to install; Phase 6 will report FAILED"

log "verifying"
"$PY" - <<'PYEOF'
import importlib
mods = [
    ("pdfplumber", "text/table extraction"),
    ("trimitdown_pdf", "is_real_table gate"),
    ("fitz", "page rendering"),
    ("PIL", "image IO"),
    ("yaml", "config"),
    ("qdrant_client", "vector db"),
    ("rank_bm25", "keyword retrieval"),
]
bad = []
for m, why in mods:
    try:
        importlib.import_module(m)
        print(f"  ok    {m:16s} {why}")
    except Exception as exc:
        bad.append((m, exc))
        print(f"  FAIL  {m:16s} {why}: {exc}")
try:
    import sentence_transformers
    print(f"  ok    {'sentence_transf':16s} embeddings")
except Exception as exc:
    print(f"  FAIL  {'sentence_transf':16s} embeddings: {exc}")
    bad.append(("sentence_transformers", exc))
if bad:
    print(f"\n{len(bad)} import(s) failed")
PYEOF

log "done. activate with: source $VENV/bin/activate"