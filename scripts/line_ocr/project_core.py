"""Small, model-independent project storage helpers for the annotation UI."""

from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def layout(root: Path) -> dict[str, Path]:
    """Return the directories used by the standalone annotation core."""
    root = root.resolve()
    return {
        "root": root,
        "images": root / "source" / "images",
        "transcripts": root / "source" / "transcripts",
        "manifest": root / "source" / "manifest.csv",
        "annotations": root / "annotations",
        "pages": root / "annotations" / "pages",
        "line_pairs": root / "annotations" / "line_pairs",
        "proposals": root / "annotations" / "proposals",
        "drafts": root / "annotations" / "drafts",
    }


def ensure_project(root: Path) -> dict[str, Path]:
    paths = layout(root)
    for key in ("images", "transcripts", "pages", "line_pairs", "proposals", "drafts"):
        paths[key].mkdir(parents=True, exist_ok=True)
    if not paths["manifest"].exists():
        with paths["manifest"].open("w", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=["page_id", "image", "transcript"]).writeheader()
    return paths


def clean_page_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not value:
        raise ValueError("The page filename does not contain a usable page ID.")
    return value


def read_manifest(root: Path) -> list[tuple[str, Path, Path]]:
    paths = ensure_project(root)
    rows: list[tuple[str, Path, Path]] = []
    with paths["manifest"].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            page_id = str(row.get("page_id", "")).strip()
            image = paths["root"] / str(row.get("image", ""))
            transcript = paths["root"] / str(row.get("transcript", ""))
            if page_id and image.is_file() and transcript.is_file():
                rows.append((page_id, image, transcript))
    if rows:
        return rows

    # A freshly arranged project can be used immediately: scans and matching
    # transcripts placed directly in source/ are paired only when their stems
    # are identical.  Once a page is imported through the UI/CLI, manifest
    # rows become the explicit source of truth.
    transcripts = {path.stem: path for path in paths["transcripts"].glob("*.txt")}
    return [
        (image.stem, image, transcripts[image.stem])
        for image in sorted(paths["images"].iterdir())
        if image.is_file() and image.suffix.lower() in IMAGE_SUFFIXES and image.stem in transcripts
    ]


def import_page(root: Path, image: Path, transcript: Path, page_id: str | None = None) -> str:
    """Copy one source pair into a project and append a stable manifest row."""
    paths = ensure_project(root)
    if image.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError("Page scans must be PNG, JPEG, TIFF, or TIF files.")
    if transcript.suffix.lower() != ".txt":
        raise ValueError("Page transcripts must be UTF-8 .txt files.")
    try:
        transcript.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Page transcripts must be UTF-8 text files.") from error
    page_id = clean_page_id(page_id or image.stem)
    existing = {item[0] for item in read_manifest(root)}
    if page_id in existing:
        raise ValueError(f"A page named '{page_id}' already exists in this project.")
    image_destination = paths["images"] / f"{page_id}{image.suffix.lower()}"
    transcript_destination = paths["transcripts"] / f"{page_id}.txt"
    shutil.copy2(image, image_destination)
    shutil.copy2(transcript, transcript_destination)
    with paths["manifest"].open("a", encoding="utf-8", newline="") as handle:
        csv.DictWriter(handle, fieldnames=["page_id", "image", "transcript"]).writerow({
            "page_id": page_id,
            "image": image_destination.relative_to(paths["root"]).as_posix(),
            "transcript": transcript_destination.relative_to(paths["root"]).as_posix(),
        })
    return page_id
