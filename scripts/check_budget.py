"""Check the total floppy-package size against the 1MB budget.

The runnable package is: host/floppy (stripped binary, vocab embedded) +
model/model.bin. Exit code 0 if under budget, 1 otherwise.
"""

import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(__file__), "..")
BUDGET = 1_000_000


def main() -> int:
    binary = os.path.join(ROOT, "host", "floppy")
    weights = os.path.join(ROOT, "model", "model.bin")

    if not os.path.exists(binary) or not os.path.exists(weights):
        print("missing artifacts: build host/floppy and export model/model.bin first")
        return 1

    # strip a copy to measure the distributable size
    stripped = binary + ".stripped"
    subprocess.run(["cp", binary, stripped], check=True)
    subprocess.run(["strip", stripped], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    sizes = {
        "host/floppy (stripped, vocab embedded)": os.path.getsize(stripped),
        "model/model.bin (int8 weights + scales)": os.path.getsize(weights),
    }
    os.remove(stripped)

    total = sum(sizes.values())
    print()
    for name, sz in sizes.items():
        print(f"  {name:46s} {sz:>10,} bytes")
    print(f"  {'TOTAL':46s} {total:>10,} bytes")
    print(f"  {'Budget':46s} {BUDGET:>10,} bytes")
    if total < BUDGET:
        print(f"\n  FITS ON FLOPPY (1MB): {100 * total / BUDGET:.1f}% of budget, "
              f"{BUDGET - total:,} bytes spare")
        return 0
    print("\n  OVER BUDGET by", f"{total - BUDGET:,} bytes")
    return 1


if __name__ == "__main__":
    sys.exit(main())
