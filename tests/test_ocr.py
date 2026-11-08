import unittest
from unittest.mock import patch

import numpy as np

from ocr import extract_cells, prepare_cell, recognise_cells


class OCRTests(unittest.TestCase):
    def test_cells_exclude_ten_percent_margin(self):
        cells = extract_cells(np.full((450, 450), 255, np.uint8))
        self.assertEqual(len(cells), 81)
        self.assertEqual(cells[0].shape, (40, 40))

    def test_noise_and_border_fragments_are_empty(self):
        noise = np.random.default_rng(12).integers(240, 250, size=(40, 40), dtype=np.uint8)
        self.assertTrue(prepare_cell(noise)[0])
        noise[:, :3] = 0
        self.assertTrue(prepare_cell(noise)[0])
        digit = np.full((40, 40), 255, np.uint8)
        digit[10:30, 18:22] = 0
        empty, tile = prepare_cell(digit)
        self.assertFalse(empty)
        self.assertEqual(tile.shape, (96, 96))

    def test_empty_cells_never_reach_ocr(self):
        cells = [np.full((40, 40), 255, np.uint8)] * 81
        with patch('ocr.pytesseract.image_to_boxes') as engine:
            result = recognise_cells(cells)
        engine.assert_not_called()
        self.assertEqual(result['grid'], [[0] * 9 for _ in range(9)])
