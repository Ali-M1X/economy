"""Render the dashboard headlessly against a snapshot directory and fail on any exception.

    python -m dashboard.smoke output        # after a data-availability run
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main(root: str) -> int:
    from streamlit.testing.v1 import AppTest

    os.environ["MACRO_PULSE_SNAPSHOT_DIR"] = str(Path(root).resolve())
    os.environ["MACRO_PULSE_NO_LIVE"] = "1"
    at = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=600).run()
    for e in at.exception:
        print(f"::error::dashboard exception: {e.value}\n{''.join(e.stack_trace)}")
    empty = sum(1 for m in at.metric if m.value == "—")
    print(f"dashboard render: {len(at.tabs)} tabs, {len(at.metric)} metrics ({empty} without data), "
          f"{len(at.dataframe)} tables, {len(at.warning)} warnings, {len(at.exception)} exceptions")
    for w in at.warning:
        print(f"  warning: {w.value}")
    return 1 if at.exception else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "output"))
