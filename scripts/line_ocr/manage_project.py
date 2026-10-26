#!/usr/bin/env python3
"""Create a standalone annotation project or import one page/transcript pair."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.line_ocr.project_core import ensure_project, import_page


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Create the project directory layout.")
    create.add_argument("project", type=Path)
    add = commands.add_parser("add-page", help="Copy one page image and its transcript into a project.")
    add.add_argument("project", type=Path)
    add.add_argument("image", type=Path)
    add.add_argument("transcript", type=Path)
    add.add_argument("--page-id")
    args = parser.parse_args()
    if args.command == "create":
        ensure_project(args.project)
        print(f"Created annotation project: {args.project.resolve()}")
    else:
        print(import_page(args.project, args.image, args.transcript, args.page_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
