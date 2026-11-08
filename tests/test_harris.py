import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from corners import detect_candidates, find_corners, select_corners
from preprocessing import preprocess
from helpers import puzzle_image


class HarrisTests(unittest.TestCase):
    def test_selected_corners_are_refined_harris_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'grid.png'
            cv2.imwrite(str(path), puzzle_image())
            prepared = preprocess(path)
            selected, candidates = find_corners(prepared)
            self.assertIsNotNone(selected)
            peaks = candidates['points'][selected['indices']]
            self.assertLess(np.linalg.norm(selected['corners'] - peaks, axis=1).max(), 6.01)
            expected = np.array([[75, 75], [525, 75], [525, 525], [75, 525]])
            self.assertLess(np.linalg.norm(selected['corners'] - expected, axis=1).max(), 5)

    def test_plain_rectangle_is_not_a_sudoku_grid(self):
        thresh = np.zeros((300, 300), np.uint8)
        cv2.rectangle(thresh, (30, 30), (270, 270), 255, 3)
        candidates = detect_candidates(thresh, thresh)
        self.assertIsNone(select_corners(thresh, thresh, candidates))
