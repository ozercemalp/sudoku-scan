import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from evaluate import evaluate, line_measurements


class EvaluationTests(unittest.TestCase):
    def test_missing_results_count_as_zero_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            truth_dir, results = root / 'truth', root / 'results'
            truth_dir.mkdir()
            results.mkdir()
            truth = {'conditions': ['C1'], 'corners': [[0, 0], [450, 0], [450, 450], [0, 450]],
                     'grid': [[0] * 9 for _ in range(9)]}
            truth['grid'][0][0] = 5
            for name in ('one', 'two'):
                (truth_dir / (name + '.json')).write_text(json.dumps(dict(truth, image=name + '.jpg')))
            predicted = dict(truth, image='one.jpg', verdict='valid', runtime_s=1)
            (results / 'one_result.json').write_text(json.dumps(predicted))
            with patch('evaluate.save_plots'), patch('evaluate.save_confusions'):
                summary = evaluate(truth_dir, results, root / 'metrics')
            self.assertEqual(summary['cell_accuracy'], 0.5)
            self.assertEqual(summary['grid_exact_match'], 0.5)
            self.assertEqual(summary['robustness'], 0.5)
            self.assertAlmostEqual(summary['empty_cell_f1'], 2 / 3)

    def test_line_measurements_on_an_ideal_grid(self):
        import cv2
        image = np.full((450, 450, 3), 255, np.uint8)
        for position in range(0, 451, 50):
            position = min(position, 449)
            cv2.line(image, (0, position), (449, position), (0, 0, 0), 1)
            cv2.line(image, (position, 0), (position, 449), (0, 0, 0), 1)
        measurements = line_measurements(image)
        for axis in measurements.values():
            self.assertLess(axis['position_rmse_px'], 1)
            self.assertLess(axis['spacing_rmse_px'], 1)
