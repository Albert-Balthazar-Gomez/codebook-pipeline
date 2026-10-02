#!/usr/bin/env bash
# Builds the external figure-extractor dependency (Phase 4).
#
# figure-extractor is a Python wrapper around PDFFigures 2.0. PDFFigures 2.0
# publishes NO release artifacts, so its jar must be compiled from source with
# sbt. This script does that without sudo, because this host's sudo requires a
# password and non-interactive apt is therefore unavailable.
#
#   1. locate a Java 11 JDK  (PDFFigures 2.0 is built against Java 11)
#   2. install sbt 1.x locally (sbt 2.x drops `bintrayPackage`, which
#      pdffigures2's build.sbt still uses, so the major version is pinned)
#   3. clone figure-extractor
#   4. clone + `sbt assembly` pdffigures2 -> pdffigures2/pdffigures2.jar
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# sbt's launcher word-splits SBT_OPTS, so any path containing whitespace passed
# through it is torn into separate arguments. A repo checked out under, say,
# "Vector db" therefore cannot host the sbt/JDK/clones in-tree. Detect that and
# relocate the heavy build tooling to a space-free cache directory instead.
# Override with CODEBOOK_TOOLS_DIR to force a location.
if [[ -n "${CODEBOOK_TOOLS_DIR:-}" ]]; then
  TOOLS="$CODEBOOK_TOOLS_DIR"
elif [[ "$ROOT" =~ [[:space:]] ]]; then
  TOOLS="${XDG_CACHE_HOME:-$HOME/.cache}/codebook-pipeline/tools"
  echo "[setup_deps] repo path contains whitespace; relocating build tools to $TOOLS" >&2
else
  TOOLS="$ROOT/tools"
fi

SBT_VERSION="${SBT_VERSION:-1.13.0}"
SBT_DIR="$TOOLS/sbt"
FE_DIR="$TOOLS/figure-extractor"
LOCAL_JDK="$TOOLS/jdk11"

log()  { printf '\033[1;34m[setup_deps]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup_deps] %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m[setup_deps] %s\033[0m\n' "$*" >&2; exit 1; }

mkdir -p "$TOOLS"

# ---------------------------------------------------------------- Java 11 ---
# sbt must COMPILE Scala, so a JRE is not enough -- `javac` has to exist. This
# host ships JRE-only packages for 11/17/21, so a full JDK usually has to be
# fetched. That is done into tools/ rather than via apt, because sudo here needs
# a password and nothing can prompt for one.
JAVA_MAJOR="${JAVA_MAJOR:-11}"

# A JDK (has javac) vs a JRE (does not).
is_jdk() { [[ -n "$1" && -x "$1/bin/javac" && -x "$1/bin/java" ]]; }

jdk_major() {
  local v
  v="$("$1/bin/java" -version 2>&1 | head -1)" || return 1
  # `openjdk version "11.0.32" 2026-07-21` or `openjdk version "1.8.0_..."`
  v="${v#*version \"}"; v="${v%%[!0-9.]*}"
  v="${v%%.*}"
  [[ "$v" == "1" ]] && { v="$(echo "$1" | grep -oE '(8|9|1[0-9]|2[0-9])' | tail -1)"; }
  echo "$v"
}

# Order of preference: explicit JAVA_HOME, a previously downloaded tools/jdk11,
# any distro JDK of the right major version, then any distro JDK at all.
find_java_home() {
  local c
  is_jdk "${JAVA_HOME:-}" && { echo "$JAVA_HOME"; return 0; }
  is_jdk "$LOCAL_JDK" && { echo "$LOCAL_JDK"; return 0; }
  for c in /usr/lib/jvm/*openjdk-* /usr/lib/jvm/*temurin* /usr/lib/jvm/*zulu* \
           /usr/lib/jvm/*corretto* /Library/Java/JavaVirtualMachines/*/Contents/Home; do
    is_jdk "$c" || continue
    [[ "$(jdk_major "$c")" == "$JAVA_MAJOR" ]] && { echo "$c"; return 0; }
  done
  for c in /usr/lib/jvm/*openjdk-* /usr/lib/jvm/*temurin* /usr/lib/jvm/*zulu* \
           /usr/lib/jvm/*corretto* /Library/Java/JavaVirtualMachines/*/Contents/Home; do
    is_jdk "$c" && { echo "$c"; return 0; }
  done
  return 1
}

if ! JH="$(find_java_home)"; then
  log "no full JDK found (a JRE cannot compile Scala) -- fetching Temurin $JAVA_MAJOR into tools/jdk11 (~190MB, no sudo)"
  tmp="$(mktemp -d)"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    JDK_URL="https://api.adoptium.net/v3/binary/latest/${JAVA_MAJOR}/ga/mac/x64/jdk/hotspot/normal/eclipse"
  else
    JDK_URL="https://api.adoptium.net/v3/binary/latest/${JAVA_MAJOR}/ga/linux/x64/jdk/hotspot/normal/eclipse"
  fi
  curl -fsSL "$JDK_URL" -o "$tmp/jdk.tar.gz"
  mkdir -p "$LOCAL_JDK"
  tar xzf "$tmp/jdk.tar.gz" -C "$LOCAL_JDK" --strip-components=1
  rm -rf "$tmp"
  is_jdk "$LOCAL_JDK" || die "downloaded JDK is missing javac at $LOCAL_JDK/bin/javac"
  JH="$LOCAL_JDK"
fi
export JAVA_HOME="$JH"
export PATH="$JH/bin:$PATH"

JVER="$("$JH/bin/java" -version 2>&1 | head -1)"
log "using JAVA_HOME=$JH"
log "  $JVER"
[[ -x "$JH/bin/javac" ]] && log "  javac: $("$JH/bin/javac" -version 2>&1)" \
  || die "no javac in $JH -- sbt cannot build without a JDK"

# ------------------------------------------------------------------ OCR ---
# tesseract needs root, and this host's sudo requires a password that nothing
# here can supply. So: detect it, and if missing print the exact command rather
# than failing the build or silently leaving Phase 8 without OCR.
if command -v tesseract >/dev/null 2>&1; then
  log "tesseract: $(tesseract --version 2>&1 | head -1)"
else
  warn "tesseract is NOT installed. Phase 8 (OCR for scanned pages) will report"
  warn "DEGRADED and name this binary until it is available."
  warn ""
  warn "  Run this yourself in another terminal (it needs your sudo password):"
  warn ""
  warn "      sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-eng"
  warn ""
  warn "  poppler-utils (pdftoppm/pdfimages) is already present: $(command -v pdftoppm >/dev/null && echo yes || echo 'NO - sudo apt install -y poppler-utils')"
  warn ""
  warn "  Then re-run: bash scripts/setup_deps.sh"
fi

# The Python OCR binding is pip-installable without root.
if ! "$ROOT/.venv/bin/python" -c "import pytesseract" 2>/dev/null; then
  log "installing pytesseract (Python binding; needs the binary above)"
  "$ROOT/.venv/bin/python" -m pip install --quiet pytesseract pdf2image || \
    warn "pytesseract install failed; Phase 8 will report it"
fi

# -------------------------------------------------------------------- sbt ---
if [[ ! -x "$SBT_DIR/bin/sbt" ]]; then
  log "downloading sbt $SBT_VERSION (no sudo: unpacking into tools/)"
  tmp="$(mktemp -d)"
  curl -fsSL "https://github.com/sbt/sbt/releases/download/v$SBT_VERSION/sbt-$SBT_VERSION.tgz" \
    -o "$tmp/sbt.tgz"
  # The tarball's top-level dir is `sbt/`, so unpack straight into $SBT_DIR.
  mkdir -p "$SBT_DIR"
  tar xzf "$tmp/sbt.tgz" -C "$SBT_DIR" --strip-components=1
  rm -rf "$tmp"
fi
log "sbt ready: $("$SBT_DIR/bin/sbt" --script-version 2>/dev/null || echo "$SBT_DIR/bin/sbt")"

# -------------------------------------------------------- figure-extractor ---
if [[ ! -d "$FE_DIR/.git" ]]; then
  log "cloning Huang-lab/figure-extractor"
  git clone --depth 1 https://github.com/Huang-lab/figure-extractor "$FE_DIR"
else
  log "figure-extractor already cloned"
fi

# -------------------------------------------------------------- pdffigures2 ---
# figure-extractor expects the jar at <repo>/pdffigures2/pdffigures2.jar, which
# matches assemblyOutputPath in allenai/pdffigures2's build.sbt.
P2_DIR="$FE_DIR/pdffigures2"
JAR="$P2_DIR/pdffigures2.jar"

if [[ -f "$JAR" ]]; then
  log "pdffigures2.jar already built ($JAR)"
else
  if [[ ! -d "$P2_DIR/.git" ]]; then
    log "cloning allenai/pdffigures2"
    git clone --depth 1 https://github.com/allenai/pdffigures2.git "$P2_DIR"
  fi

  log "building pdffigures2 with sbt assembly (~5-15 min, first run downloads Scala deps)"
  # sbt needs a writable home for its cache/ivy dirs.
  export SBT_OPTS="${SBT_OPTS:-} -Dsbt.ivy.home=$P2_DIR/.ivy -Dsbt.boot.directory=$P2_DIR/.sbt-boot"
  # Force the JDK we found above; sbt otherwise picks the default `java` (21).
  export JAVA_HOME="$JH"
  export PATH="$JH/bin:$PATH"

  ( cd "$P2_DIR" && "$SBT_DIR/bin/sbt" -batch assembly )

  # build.sbt writes to the repo root; tolerate older layouts anyway.
  if [[ ! -f "$JAR" ]]; then
    found="$(find "$P2_DIR/target" -name 'pdffigures2-assembly-*.jar' 2>/dev/null | head -1 || true)"
    [[ -n "$found" ]] && { log "jar found at $found, moving to $JAR"; mv "$found" "$JAR"; }
  fi
  [[ -f "$JAR" ]] || die "assembly finished but no jar at $JAR"
fi

log "jar: $JAR ($(du -h "$JAR" | cut -f1))"

# figure-extractor's own runtime deps (Flask stack, only needed for the API
# server -- local mode needs none of it, but installing keeps `python
# setup_local.py` working).
if ! "$ROOT/.venv/bin/python" -c "import requests" 2>/dev/null; then
  log "installing figure-extractor python requirements"
  "$ROOT/.venv/bin/python" -m pip install --quiet -r "$FE_DIR/requirements.txt" || \
    warn "figure-extractor requirements failed (local mode does not need them)"
fi

# Smoke test: the jar must actually start.
log "smoke-testing pdffigures2"
if "$JH/bin/java" -Xmx2g \
      -Dsun.java2d.cmm=sun.java2d.cmm.kcms.KcmsServiceProvider \
      -jar "$JAR" 2>&1 | head -5; then
  log "pdffigures2 invoked successfully"
fi

log "Phase 4 dependency ready."
log "  JAVA_HOME=$JH"
log "  figure-extractor=$FE_DIR"
log "  pdffigures2.jar=$JAR"

# The pipeline discovers these paths from a locator file rather than from the
# environment, so a later `python -m src.pipeline.run` needs no exports at all.
# The Python side can override any of this via config `paths.tools`.
LOCATOR="$ROOT/.tools_locator.json"
cat > "$LOCATOR" <<JSON
{
  "tools_dir": "$TOOLS",
  "figure_extractor_dir": "$FE_DIR",
  "pdffigures2_jar": "$JAR",
  "java_home": "$JH",
  "sbt_dir": "$SBT_DIR"
}
JSON
log "wrote $LOCATOR"
log "(add 'export JAVA_HOME=$JH' to your shell profile for manual figure-extractor runs)"