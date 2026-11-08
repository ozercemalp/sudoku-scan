# COMP 482 Project 1 — SudokuScan development set

20 images of printed sudoku puzzles with ground-truth corners and grids, for developing and
testing your pipeline. They are **not** part of the grade.

These images are **synthetic photographs**: each puzzle was rendered on a page (newspaper,
puzzle book or printout) and placed in front of a simulated phone camera with real perspective
projection, then given a desk background, clutter, lighting, shadows, lens blur, sensor noise and
JPEG compression. This makes the ground truth exact. The hidden test set and your classmates'
personal datasets are **real** phone photos, so do not tune your system to the look of these
images; use your own photographs as well.

## Contents

```
images/        dev_01.jpg ... dev_20.jpg       (1600×1200 up to 3200×2400, landscape and portrait)
annotations/   dev_01.json ... dev_20.json     (format of Appendix A in the assignment)
index.csv      one row per image: size, conditions, camera angles, layout, number of givens, solution
preview_ground_truth.jpg   all 20 images with the ground-truth corners drawn
```

## Conventions

- **corners**: `[[x, y], ...]` in the order **TL, TR, BR, BL of the puzzle**, in original-image
  pixels with sub-pixel precision. A corner is the **outer corner of the thick outer border**
  of the grid. Coordinates follow the OpenCV convention: the centre of the top-left pixel is (0, 0).
  So mapping the corners to `(-0.5, -0.5)`, `(449.5, -0.5)`, `(449.5, 449.5)`, `(-0.5, 449.5)`
  makes each cell of a 450×450 rectification exactly 50×50 px.
- **grid**: 9 rows of 9 integers, top row first, 0 for an empty cell.
- **conditions**: the C1–C5 labels from Section 4.2 of the assignment.
- **notes** (optional): dev_19 and dev_20 have a **gentle page curl**. No 3×3 transformation can
  remove it exactly, which makes them good failure cases for your report.

## index.csv columns

| Column | Meaning |
|---|---|
| `tilt_deg`, `pan_deg`, `roll_deg` | True camera rotation of the page: tilt about the horizontal axis, pan about the vertical axis, roll in the image plane |
| `side_ratio` | Largest ratio between opposite sides of the ground-truth quadrilateral. It is one possible obliqueness measure; you still have to define and justify your own for Stage 4 |
| `layout` | `newspaper`, `book` or `printout` |
| `givens` | Number of printed digits |
| `solution` | The unique solution as 81 digits, row by row (useful for bonus B1/B2) |

## Coverage

| Condition | Images |
|---|---|
| C1 near-frontal | 01, 02, 03, 04, 19 |
| C2 oblique (about 34–55°) | 05, 06, 07, 08, 09, 10, 11, 18, 20 |
| C3 in-plane rotation (18–30°) | 11, 12, 13, 14 |
| C4 uneven or low light | 04, 09, 14, 15, 16, 20 |
| C5 clutter (crosswords, tables, notes box, objects) | 03, 10, 13, 16, 17, 18 |

Every puzzle is a valid sudoku with a unique solution. Twenty different digit fonts are used,
and grid styles vary (plain, shaded boxes, blue ink, thin outer border).
