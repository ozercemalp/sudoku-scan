import unittest

import numpy as np

from preprocessing import map_points
from rectification import rectify


class RectificationTests(unittest.TestCase):
    def test_a4_is_least_squares_and_h4_maps_all_four_corners(self):
        corners = np.float32([[30, 20], [320, 45], [250, 280], [70, 300]])
        image = np.full((340, 350, 3), 255, np.uint8)
        images, matrices, errors = rectify(image, corners)
        target = np.array([[0, 0], [450, 0], [450, 450], [0, 450]])
        design = np.column_stack([corners, np.ones(4)])
        expected_a4 = np.linalg.lstsq(design, target, rcond=None)[0].T
        np.testing.assert_allclose(matrices['A4'], expected_a4, atol=1e-4)
        np.testing.assert_allclose(map_points(corners, matrices['H4']), target, atol=1e-4)
        self.assertGreater(errors['a3_corner_err_px'], 1)
        self.assertGreater(errors['a4_rms_px'], 1)
        self.assertLess(errors['h4_max_corner_err_px'], 1e-4)
        for warped in images.values():
            self.assertEqual(warped.shape, (450, 450, 3))
