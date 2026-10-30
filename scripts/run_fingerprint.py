"""Run provenance for resumable experiments.

A resumed run appends to rows written earlier, so it must run the same
computation: the same estimator code, the same registered queries and data
and the same parameters.  ``guard`` stores a fingerprint of those inputs next
to the output (``<output>.fingerprint.json``) on a fresh run and refuses a
``--resume`` whose fingerprint differs, naming the inputs that changed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(
    script: str | Path, inputs: list[Path], params: dict[str, object]
) -> dict[str, str]:
    """Hashes of the script, every ``src/`` module, the inputs and the params."""
    fp: dict[str, str] = {}
    script = Path(script).resolve()
    fp[f"script:{script.relative_to(ROOT)}"] = _sha(script)
    src = hashlib.sha256()
    for p in sorted((ROOT / "src").rglob("*.py")):
        src.update(str(p.relative_to(ROOT)).encode())
        src.update(p.read_bytes())
    fp["src"] = src.hexdigest()
    for p in inputs:
        p = Path(p).resolve()
        label = p.relative_to(ROOT) if p.is_relative_to(ROOT) else p
        fp[f"input:{label}"] = _sha(p) if p.exists() else "missing"
    fp["params"] = hashlib.sha256(
        json.dumps(params, sort_keys=True, default=repr).encode()
    ).hexdigest()
    return fp


def module_params(namespace: dict[str, object]) -> dict[str, object]:
    """A script's upper-case constants (seeds, caps, p-sets, workers ...)."""
    return {
        k: v
        for k, v in namespace.items()
        if k.isupper() and isinstance(v, (int, float, str, tuple, list, frozenset))
    }


def guard(
    output: Path,
    script: str | Path,
    inputs: list[Path],
    params: dict[str, object],
    *,
    resume: bool,
) -> None:
    """Record the run's fingerprint, or refuse a resume across a change.

    A resume against an output without a stored fingerprint (written before
    this guard existed) is refused too: its provenance cannot be checked.
    """
    side = output.with_name(output.name + ".fingerprint.json")
    now = fingerprint(script, inputs, params)
    if resume and output.exists():
        if not side.exists():
            raise SystemExit(
                f"REFUSING TO RESUME: {output.name} has no fingerprint; rerun "
                "without --resume"
            )
        old = json.loads(side.read_text())
        changed = sorted(k for k in old.keys() | now.keys() if old.get(k) != now.get(k))
        if changed:
            raise SystemExit(
                f"REFUSING TO RESUME {output.name}: changed since the stored "
                f"rows were written: {', '.join(changed)}"
            )
        return
    side.write_text(json.dumps(now, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(fingerprint(sys.argv[1], [], {}), indent=2))
