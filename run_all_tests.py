"""
run_all_tests.py — Run the full V4 test suite.
Usage: python run_all_tests.py
"""

import os
import sys
import time
import importlib
import traceback

TESTS = [
    "test_config",
    "test_entropy",
    "test_bfs",
    "test_seed_consistency",
    "test_control",
    "test_nonregression_chao_u",
]


def main():
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    passed = 0
    failed = []
    t0 = time.time()

    for name in TESTS:
        mod = importlib.import_module(f"tests.{name}")
        for attr in dir(mod):
            if attr.startswith("test_"):
                fn = getattr(mod, attr)
                try:
                    fn()
                    passed += 1
                    print(f"  PASS  {name}.{attr}")
                except Exception as e:
                    failed.append(f"{name}.{attr}")
                    print(f"  FAIL  {name}.{attr}: {e}")
                    traceback.print_exc()

    print(f"\n{passed} passed, {len(failed)} failed "
          f"in {time.time() - t0:.2f}s")
    if failed:
        print("FAILED:", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
