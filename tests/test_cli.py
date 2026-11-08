import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from helpers import puzzle_image
from sudoku_scan import process_image
from validation import validate_grid


ROOT = Path(__file__).resolve().parents[1]


class ValidationAndCLITests(unittest.TestCase):
    def test_all_conflicting_cells_are_reported(self):
        grid = [[0] * 9 for _ in range(9)]
        for row, column, value in ((0, 0, 5), (1, 1, 5), (3, 0, 5), (8, 2, 7), (8, 8, 7)):
            grid[row][column] = value
        self.assertEqual(validate_grid(grid), ('invalid', [[0, 0], [1, 1], [3, 0], [8, 2], [8, 8]]))
        self.assertEqual(validate_grid([[0] * 9 for _ in range(9)]), ('valid', []))

    def test_batch_records_corrupt_images_and_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'
            images.mkdir()
            (images / '01_bad.jpg').write_bytes(b'broken image')
            cv2.imwrite(str(images / '02_blank.png'), np.full((300, 300, 3), 255, np.uint8))
            output = root / 'out'
            run = subprocess.run([sys.executable, str(ROOT / 'sudoku_scan.py'), '--batch', str(images),
                                  '--outdir', str(output)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            self.assertEqual(json.loads((output / '01_bad_result.json').read_text())['verdict'], 'error')
            self.assertEqual(json.loads((output / '02_blank_result.json').read_text())['verdict'], 'not_found')

    @unittest.skipUnless(shutil.which('tesseract'), 'Tesseract is not installed')
    def test_required_outputs_and_stale_prediction_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, output = root / 'sample.png', root / 'out'
            cv2.imwrite(str(path), puzzle_image())
            result = process_image(path, output)
            self.assertIn(result['verdict'], ('valid', 'invalid'), result)
            self.assertEqual(np.asarray(result['grid']).shape, (9, 9))
            self.assertEqual(np.sum(~np.asarray(result['empty_mask'])), 4)
            for model in ('A3', 'A4', 'H4'):
                self.assertEqual(cv2.imread(str(output / f'sample_rect_{model}.png')).shape, (450, 450, 3))
            rows = (output / 'sample_grid.txt').read_text().splitlines()
            self.assertEqual(rows, [''.join(map(str, row)) for row in result['grid']])
            self.assertTrue((output / 'sample_harris.png').is_file())
            self.assertTrue((output / 'sample_debug.png').is_file())
            cv2.imwrite(str(path), np.full((600, 600, 3), 255, np.uint8))
            self.assertEqual(process_image(path, output)['verdict'], 'not_found')
            self.assertFalse((output / 'sample_grid.txt').exists())

    def test_ambiguous_arguments_are_rejected(self):
        run = subprocess.run([sys.executable, str(ROOT / 'sudoku_scan.py'), '--input', 'one.jpg',
                              '--batch', '.', '--outdir', 'out'], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
