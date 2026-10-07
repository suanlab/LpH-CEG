#!/usr/bin/env bash
# Regenerate MANIFEST.sha256 over the canonical SOURCE surface.
#
# The manifest lets an artifact evaluator verify source provenance with
#   sha256sum -c MANIFEST.sha256
# without a git checkout. It covers hand-written source only -- Python
# modules, scripts, tests, the paper source, and environment manifests --
# and deliberately excludes generated artifacts (figures, result CSVs,
# build outputs, caches) and the external symlinked datasets, which are
# not part of the tracked source and would drift on every run.
#
# Usage:  bash scripts/gen_manifest.sh   (run from anywhere; cds to repo root)

set -euo pipefail
cd "$(dirname "$0")/.."

out=MANIFEST.sha256

# Deterministic, sorted list of canonical source files.
mapfile -t files < <(
  {
    find src scripts tests -type f \( -name '*.py' -o -name '*.sh' \) \
      ! -path '*__pycache__*'
    printf '%s\n' \
      docs/paper/main.tex docs/paper/references.bib docs/paper/extended_version.tex \
      pyproject.toml requirements.txt requirements.lock Dockerfile \
      configs/experiments.yaml
  } | sort -u
)

# Comment-free so `sha256sum -c MANIFEST.sha256` verifies with zero
# warnings (provenance notes live in this generator, not the manifest).
: > "$out"
for f in "${files[@]}"; do
  [ -f "$f" ] && sha256sum "$f" >> "$out"
done

n=$(grep -cE '^[0-9a-f]{64}  ' "$out")
echo "Wrote $out ($n files)."
