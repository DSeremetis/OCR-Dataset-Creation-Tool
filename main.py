#!/usr/bin/env python3
"""Start the standalone OCR annotation UI.

Run ``python3 main.py`` from this folder. It opens this repository as the
annotation project, using ``source/images`` and ``source/transcripts``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from Line_Breaking_Tool.line_breaking_tool import main as start_annotation_ui


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the OCR dataset annotation UI.")
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Project folder to open (default: this repository).",
    )
    args = parser.parse_args()
    sys.argv = ["line_breaking_tool.py", "--project", str(args.project)]
    return start_annotation_ui()


if __name__ == "__main__":
    raise SystemExit(main())
