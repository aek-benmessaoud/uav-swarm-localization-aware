"""utils/paths.py — Canonical raw-CSV path builder (single source of truth).

The method name in the filename is sanitized the SAME way everywhere so the
runner, stats, and p-value scripts always agree on the file name.
"""

import os
import re


def sanitize_method_name(method):
    return re.sub(r"[^\w\-]", "_", method)


def raw_csv_path(out_dir, info_model, method, alpha=None, variant=None):
    safe = sanitize_method_name(method)
    if alpha is not None:
        safe = f"{safe}@{alpha}"
    elif variant is not None:
        safe = f"{safe}@{variant}"
    return os.path.join(out_dir, f"raw_{info_model}__{safe}.csv")
