# OCR Dataset Annotation — Core

This is a standalone, model-independent UI for turning page scans and known
page transcripts into reviewed OCR line-image/text pairs. Its deliberately
small scope is:

```text
page image + page transcript
  → optional Kraken line proposals
  → manual rotation, box, and transcript-line correction
  → reviewed line-image/text pairs
```

Training, active learning, corpus rebuilding, and OCR-model-assisted text
splitting are not part of this core workflow.

## Install on Arch Linux

```bash
sudo pacman -S python-pyqt5
```

## Run

Place scans in `source/images/` and their matching UTF-8 transcripts in
`source/transcripts/`. Matching names are paired automatically: for example,
`page_001.png` pairs with `page_001.txt`. Then run:

```bash
python3 main.py
```

The default project is this repository. To open a project elsewhere, pass its
location:

```bash
python3 main.py --project /path/to/my_ocr_project
```

Use **Import page + transcript** in the UI to add each pair. The image and
UTF-8 `.txt` transcript are copied into the project and recorded in its
manifest. You can also import a pair without opening the UI:

```text
python scripts/line_ocr/manage_project.py add-page /path/to/my_ocr_project scan.jpg page.txt
```

The core creates and uses this layout:

```bash
my_ocr_project/
  source/images/
  source/transcripts/
  source/manifest.csv
  annotations/pages/
  annotations/line_pairs/
  annotations/proposals/
  annotations/drafts/
```

## Annotating a page

1. Import a page scan and its transcript, then select it in the review queue.
2. Review the transcript rows. They initially come from the existing newline characters in the `.txt` file; those newlines may not match the handwritten line breaks.
3. Double-click a row to edit its text.
4. Use **Insert text line below** and **Delete selected text line** to split or merge the transcript into the correct physical lines. Existing text can be cut and pasted between rows.
5. Click **Re-segment current orientation with Kraken** when Kraken is installed and you want a proposal. **Auto-detect line regions** is a simple local fallback. Neither is ground truth.
6. Review and correct the proposed regions. Select a transcript row and use **Draw line region**, or select an existing region and use **Assign text to selected region**.
7. Drag inside a rectangle to move it.
8. Drag a corner handle to resize its start/end and top/bottom.
9. Use arrow keys to move the selected region; hold Shift for 5-pixel steps.
10. If the scan is slightly tilted, enter a small value in **Deskew** (start
    with 0.25° to 1°) and click **Apply deskew**. The page and all current
    boxes rotate together. Fine deskew should be done before final box edits;
    save the resulting angle with the page annotation.
11. Press **Export reviewed page** only after every text row has exactly one assigned box. This is the gate that produces reviewed pairs.

Use **Previous** and **Next**, or Page Up/Page Down, to move between pages.

## Saved files

For each reviewed page, the tool writes:

```text
annotations/pages/<page-id>.json
annotations/pages/<page-id>_preview.png
annotations/line_pairs/<page-id>/manifest.json
annotations/line_pairs/<page-id>/<page-id>_line_001.png
annotations/line_pairs/<page-id>/<page-id>_line_001.txt
```

The JSON stores image dimensions, corrected transcript lines, each rectangle's
pixel coordinates, its transcript line assignment, and the text itself. The preview PNG is a visual
record of the committed line regions. This format is intentionally independent
of Tesseract so it can later be used to generate Tesseract `.lstmf` data or a
Keras training dataset.

`Ctrl+S` saves work in progress. The export button requires complete,
one-to-one assignments before it reports the line pairs as reviewed.

## Current scope

Kraken proposals should be preferred for automatic detection. The built-in
horizontal ink projection remains available as a fallback and can be confused
by page artifacts, margin notes, or touching lines. Neither proposal source is
ground truth until reviewed and saved.
