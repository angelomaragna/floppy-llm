import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts import check_budget  # noqa: E402


def test_package_fits_1mb():
    rc = check_budget.main()
    assert rc == 0, "floppy package exceeds the 1,000,000-byte budget"
