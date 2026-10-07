#!/usr/bin/env bash
# Build a clean, publishable snapshot of the artifact (for the ICDE
# supplemental-material URL) from the current commit.
#
# The snapshot is the committed tree minus internal working notes (plans,
# audits, strategy memos, the Korean draft), with fresh git history so the
# development history and its author metadata are not published. It refuses
# to finish if absolute machine paths or e-mail addresses remain.
#
# Usage:  bash scripts/export_artifact.sh [OUT_DIR]   (default ../LpH-CEG-artifact)
# Then review OUT_DIR, commit with the identity you want, and push.

set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-../LpH-CEG-artifact}"

if [ -e "$out" ]; then
    echo "REFUSING: $out exists; remove it or pass another directory." >&2
    exit 1
fi
mkdir -p "$out"
git archive HEAD | tar -x -C "$out"

internal=(
    AUDIT.md CHECK.md CLAUDE.md ENHANCEMENT_PLAN.md ICDE_PLAN.md PLAN.md
    REVIEW.md TOP_PLAN.md docs/proof_review_request.md
    docs/BREAKTHROUGH_STRATEGY_2026Q3.md docs/PORTFOLIO_STRATEGY_2026Q2.md
    docs/archive
    docs/paper/korean.tex docs/paper/korean.pdf docs/paper/korean.xdv
)
for f in "${internal[@]}"; do rm -rf "${out:?}/$f"; done

# The README's pointer to the internal notes no longer applies.
python3 - "$out/README.md" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
s = s.replace(
    "Internal planning documents (`*_PLAN.md`, `AUDIT.md`, `CHECK.md`,\n"
    "`REVIEW.md`) are working notes, not documentation.\n\n", "")
open(p, "w").write(s)
PY

# Leak checks: absolute machine paths and e-mail addresses.
leaks=$(grep -rIlE "/projects/|/home/[a-z]|/Users/[A-Za-z]" "$out" \
        --exclude=export_artifact.sh || true)
mails=$(grep -rIhoE "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[a-z]{2,}" "$out" \
        --exclude="*.bib" --exclude=export_artifact.sh | grep -v -e "noreply@anthropic.com" -e "suanlee@semyung.ac.kr" | sort -u || true)
if [ -n "$leaks$mails" ]; then
    echo "REFUSING: possible leaks in the snapshot:" >&2
    [ -n "$leaks" ] && echo "$leaks" >&2
    [ -n "$mails" ] && echo "$mails" >&2
    exit 1
fi

git -C "$out" init -q -b main
git -C "$out" add -A
echo "Snapshot ready in $out ($(git -C "$out" ls-files | wc -l) files, staged, not committed)."
echo "Next: cd $out && git commit -m 'LpH-CEG artifact (ICDE 2027)' && gh repo create ..."
