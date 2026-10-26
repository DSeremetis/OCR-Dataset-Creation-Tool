#!/usr/bin/env python3
"""Convert Kraken native segmentation JSON into review-tool proposal JSON.

The converter preserves Kraken's reported reading order. It assigns transcript
lines automatically only when the detected line count exactly matches the
non-empty source-line count; all other pages deliberately remain unpaired for
human review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def bounds(points) -> tuple[float, float, float, float]:
    """Return x, y, width, height from a Kraken polygon or rectangle."""
    if len(points) == 4 and all(isinstance(value, (int, float)) for value in points):
        left, top, right, bottom = map(float, points)
        return left, top, right - left, bottom - top
    coordinates = [(float(point[0]), float(point[1])) for point in points]
    left = min(point[0] for point in coordinates)
    right = max(point[0] for point in coordinates)
    top = min(point[1] for point in coordinates)
    bottom = max(point[1] for point in coordinates)
    return left, top, right - left, bottom - top


def regions_from_kraken(data: dict) -> list[dict]:
    """Handle baseline and legacy-box JSON emitted by supported Kraken releases."""
    source_lines = data.get("lines", [])
    regions = []
    if source_lines:
        for line in source_lines:
            if not isinstance(line, dict):
                raise ValueError("Unsupported Kraken line record; expected an object with boundary or bbox.")
            geometry = line.get("boundary") or line.get("polygon") or line.get("bbox") or line.get("box")
            if not geometry:
                raise ValueError("A Kraken line has no boundary/bounding-box geometry.")
            x, y, width, height = bounds(geometry)
            regions.append({"x": x, "y": y, "width": width, "height": height, "text_index": None})
        return regions
    # Kraken's legacy segmenter emits reading-order rectangles in ``boxes``.
    if data.get("boxes"):
        for box in data["boxes"]:
            x, y, width, height = bounds(box)
            regions.append({"x": x, "y": y, "width": width, "height": height, "text_index": None})
        return regions
    raise ValueError("No recognized Kraken line geometry found (expected lines or boxes).")


def transcript_lines(label_path: Path) -> list[str]:
    return [line.strip() for line in label_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segments-dir", type=Path, default=root / "outputs/kraken_native_segments")
    parser.add_argument("--labels", type=Path, default=root / "dataset/labels/clean_from_doc")
    parser.add_argument("--output-dir", type=Path, default=root / "outputs/line_annotations/proposals")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    converted = unpaired = skipped = 0
    for segment_path in sorted(args.segments_dir.glob("*.json")):
        destination = args.output_dir / segment_path.name
        if destination.exists() and not args.force:
            skipped += 1
            continue
        data = json.loads(segment_path.read_text(encoding="utf-8"))
        regions = regions_from_kraken(data)
        label_path = args.labels / f"{segment_path.stem}_ground_truth.txt"
        lines = transcript_lines(label_path) if label_path.exists() else []
        paired = len(lines) == len(regions) and bool(lines)
        if paired:
            for index, region in enumerate(regions):
                region["text_index"] = index
        else:
            unpaired += 1
        payload = {
            "version": 1,
            "source": "kraken-native",
            "source_file": str(segment_path),
            "transcript_lines": lines,
            "regions": regions,
            "automatically_paired": paired,
        }
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        converted += 1
    print({"converted": converted, "unpaired_for_review": unpaired, "skipped": skipped, "output_dir": str(args.output_dir)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
