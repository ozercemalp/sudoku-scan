import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from preprocessing import preprocess, map_points


class PreprocessingTests(unittest.TestCase):
    def test_resize_maps_back_to_original_pixel_centres(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'photo.png'
            Image.new('RGB', (800, 600), 'white').save(path)
            prepared = preprocess(path, max_side=400)
            self.assertEqual(prepared['gray'].shape, (300, 400))
            points = map_points([[0, 0], [399, 299]], prepared['to_original'])
            np.testing.assert_allclose(points, [[0.5, 0.5], [798.5, 598.5]])
            self.assertFalse(prepared['thresh'].any())

    def test_exif_rotation_and_resize_share_one_coordinate_map(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rotated.jpg'
            exif = Image.Exif()
            exif[274] = 8
            Image.new('RGB', (300, 240), 'white').save(path, exif=exif)
            prepared = preprocess(path, max_side=150)
            self.assertEqual(prepared['gray'].shape, (150, 120))
            np.testing.assert_allclose(map_points([[20, 15]], prepared['to_original']), [[268.5, 40.5]])
