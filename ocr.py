"""Crop cells, identify empty cells, and recognise printed digits with Tesseract."""

import os
from time import perf_counter

import cv2
import numpy as np
import pytesseract


def extract_cells(image):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    height, width = gray.shape
    cells = []
    for row in range(9):
        for column in range(9):
            # Divide into nine equal rows and columns, rounding the shared edges.
            x0, x1 = round(column * width / 9), round((column + 1) * width / 9)
            y0, y1 = round(row * height / 9), round((row + 1) * height / 9)
            # Remove 10% from each edge to keep grid lines out of the digit crop.
            margin_x = round((x1 - x0) * 0.10)
            margin_y = round((y1 - y0) * 0.10)
            cell_img = gray[y0 + margin_y:y1 - margin_y, x0 + margin_x:x1 - margin_x]
            cells.append(cell_img.copy())
    return cells


def prepare_cell(cell_img):
    height, width = cell_img.shape
    threshold, _ = cv2.threshold(cell_img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # Empty JPEG cells still have small intensity variations. Require ink to
    # contrast with the local background before accepting a component.
    # Use the stricter of Otsu, background minus 35, and 200 as the ink cutoff.
    cutoff = max(0, min(threshold, float(np.percentile(cell_img, 90)) - 35, 200))
    _, ink = cv2.threshold(cell_img, cutoff, 255, cv2.THRESH_BINARY_INV)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
    best_component = None
    best_area = 0
    # Label 0 is background; inspect only the connected ink regions.
    for component in range(1, count):
        x, y, w, h, area = stats[component]
        # Regions touching the crop edge are likely leftover grid lines.
        if x == 0 or y == 0 or x + w == width or y + h == height:
            continue
        # Reject tiny specks: require 1.2% ink area (at least 8 pixels),
        # 22% of the crop height, and a width of at least 2 pixels.
        if area < max(8, cell_img.size * 0.012) or h < height * 0.22 or w < 2:
            continue
        # Keep the largest plausible digit component.
        if area > best_area:
            best_component, best_area = component, area
    tile = np.full((96, 96), 255, np.uint8)
    if best_component is None:
        return True, tile
    x, y, w, h, _ = stats[best_component]
    component_mask = labels[y:y + h, x:x + w] == best_component
    glyph = np.full((h, w), 255, np.uint8)
    glyph[component_mask] = 0
    # Fit the digit within 64x56 pixels without changing its aspect ratio.
    scale = min(64 / h, 56 / w)
    glyph = cv2.resize(glyph, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_CUBIC)
    glyph_height, glyph_width = glyph.shape
    # Centre the digit on a white 96x96 tile so OCR has clear padding.
    top = (96 - glyph_height) // 2
    left = (96 - glyph_width) // 2
    tile[top:top + glyph_height, left:left + glyph_width] = glyph
    return False, tile


def recognise_cells(cells, prepare=True, deadline=None):
    if len(cells) != 81:
        raise ValueError('expected 81 cell images')
    # Limit OCR threads unless the caller already chose a setting.
    os.environ.setdefault('OMP_THREAD_LIMIT', '1')
    if deadline is None:
        # Share a five-second budget between the atlas and any digit retries.
        deadline = perf_counter() + 5
    grid = [[0] * 9 for _ in range(9)]
    empty_mask = [[True] * 9 for _ in range(9)]
    tiles = []
    filled_indices = []
    for index, cell_img in enumerate(cells):
        # Decide empty versus filled before calling OCR, even in comparison mode.
        is_empty, tile = prepare_cell(cell_img)
        empty_mask[index // 9][index % 9] = is_empty
        if not prepare:
            # Keep the raw crop's content for comparison, with a white border.
            tile = cv2.resize(cell_img, (72, 72))
            tile = cv2.copyMakeBorder(tile, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        tiles.append(tile)
        if not is_empty:
            filled_indices.append(index)
    if not filled_indices:
        return {'grid': grid, 'empty_mask': empty_mask, 'unread_cells': []}

    # OCR only filled cells. A spaced atlas avoids starting a process per digit.
    # A 128-pixel slot holds a 96-pixel tile with 16 pixels of space on each side.
    stride = 128
    # Round up to enough rows of nine slots for all filled cells.
    rows = (len(filled_indices) + 8) // 9
    atlas = np.full((rows * stride, 9 * stride), 255, np.uint8)
    for slot, index in enumerate(filled_indices):
        x = (slot % 9) * stride + 16
        y = (slot // 9) * stride + 16
        atlas[y:y + 96, x:x + 96] = tiles[index]
    remaining = deadline - perf_counter()
    if remaining < 0.2:
        raise TimeoutError('OCR time budget exhausted')
    # PSM 6 reads the atlas as a text block; only Sudoku digits 1-9 are allowed.
    config = '--psm 6 -c tessedit_char_whitelist=123456789'
    boxes = pytesseract.image_to_boxes(atlas, config=config, timeout=min(5, remaining))
    for line in boxes.splitlines():
        fields = line.split()
        if len(fields) < 5 or len(fields[0]) != 1 or fields[0] not in '123456789':
            continue
        x0, y0, x1, y1 = map(int, fields[1:5])
        # Use each character's centre to find its atlas slot.
        column = int((x0 + x1) / 2) // stride
        # Tesseract measures y from the bottom, while image rows start at the top.
        row = int(len(atlas) - (y0 + y1) / 2) // stride
        slot = row * 9 + column
        if 0 <= column < 9 and 0 <= slot < len(filled_indices):
            # Convert the packed atlas slot back to the original puzzle cell.
            index = filled_indices[slot]
            grid[index // 9][index % 9] = int(fields[0])

    unread_cells = []
    for index in filled_indices:
        row, column = index // 9, index % 9
        # Retry unread digits with single-character PSM 10 while time remains.
        if grid[row][column] == 0 and deadline - perf_counter() >= 0.5:
            text = pytesseract.image_to_string(
                tiles[index], config='--psm 10 -c tessedit_char_whitelist=123456789',
                timeout=min(1.5, deadline - perf_counter()),
            ).strip()
            if len(text) == 1 and text in '123456789':
                grid[row][column] = int(text)
        # Record filled but unread cells separately from actual empty cells.
        if grid[row][column] == 0:
            unread_cells.append([row, column])
    return {'grid': grid, 'empty_mask': empty_mask, 'unread_cells': unread_cells}
