"""Send the experimental advisory report (reports.advisor) to Telegram.

    python -m jobs.advisor --root output [--dry-run]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from reports import advisor, facts


def main(argv: list[str] | None = None) -> int:
    from notify import telegram

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    f = facts.load(Path(args.root))
    if not f.signals:
        print("::error::no signals.json in the snapshot; nothing to analyse")
        return 1
    for i, msg in enumerate(advisor.build(f, pd.Timestamp.now(tz="UTC"))):
        r = telegram.send(msg, kind=f"advisor-{i + 1}", dry_run=args.dry_run)
        print(f"advisor message {i + 1}: {'sent' if r.sent else 'dry run → ' + r.where}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
