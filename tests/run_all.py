"""Zero-dependency test runner: `python tests/run_all.py`.

Runs every `test_*` function in tests/test_*.py, prints a pass line per test.
Exists so the suite works on a bare interpreter before the cloud image is built;
`pytest tests -q` produces the same result when pytest is available.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).parent


def load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    files = sorted(p for p in HERE.glob("test_*.py"))
    total = failed = 0
    t0 = time.time()
    for f in files:
        mod = load(f)
        tests = [(k, v) for k, v in vars(mod).items() if k.startswith("test_") and callable(v)]
        print(f"\n{f.name}  ({len(tests)} tests)")
        for name, fn in tests:
            total += 1
            try:
                fn()
                print(f"  ok    {name}")
            except Exception:
                failed += 1
                print(f"  FAIL  {name}")
                traceback.print_exc()
    print(f"\n{total - failed}/{total} passed in {time.time() - t0:.2f}s")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
