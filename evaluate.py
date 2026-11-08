"""Compare saved predictions with ground truth after scanning."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from ocr import extract_cells
from sudoku_scan import write_json


def line_measurements(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 9,
    )
    measurements = {}
    size = len(gray)
    cell_size = size / 9
    for name, axis in (('vertical', 0), ('horizontal', 1)):
        # row and column averages make grid lines show up as peaks
        profile = thresh.mean(axis=axis) / 255
        positions, errors, straightness = [], [], []
        # look near the 10 expected lines every 50 pixels on a 450px grid
        for ideal in np.linspace(0, size, 10):
            left = max(0, int(ideal - cell_size * 0.3))
            right = min(size, int(ideal + cell_size * 0.3) + 1)
            search_profile = profile[left:right]
            peak = left + int(np.argmax(search_profile))
            # below 25% ink coverage count the line as missing with a half cell penalty
            if profile[peak] < 0.25:
                positions.append(None)
                errors.append(cell_size / 2)
                continue
            positions.append(peak)
            errors.append(peak - ideal)
            # check peaks in 6 sections to see how much the line position changes
            for segment in np.array_split(thresh, 6, axis=axis):
                segment_profile = segment.mean(axis=axis)
                local_peak = left + int(np.argmax(segment_profile[left:right]))
                straightness.append(local_peak - peak)
        spacing_errors = []
        for index in range(9):
            if positions[index] is None or positions[index + 1] is None:
                spacing_errors.append(cell_size / 2)
            else:
                measured_spacing = positions[index + 1] - positions[index]
                spacing_errors.append(measured_spacing - cell_size)

        position_rmse = float(np.sqrt(np.mean(np.square(errors))))
        spacing_rmse = float(np.sqrt(np.mean(np.square(spacing_errors))))
        straightness_rms = None
        if straightness:
            straightness_rms = float(np.sqrt(np.mean(np.square(straightness))))
        measurements[name] = {
            'positions': positions,
            'position_rmse_px': position_rmse,
            'spacing_rmse_px': spacing_rmse,
            'straightness_rms_px': straightness_rms,
        }
    return measurements


def summarize(rows):
    count = len(rows)
    true_positive = false_positive = false_negative = 0
    detected_count = exact_count = robust_count = 0
    cell_accuracies = []
    corner_errors = []
    runtimes = []
    for row in rows:
        true_positive += row['empty_tp']
        false_positive += row['empty_fp']
        false_negative += row['empty_fn']
        detected_count += row['detected']
        cell_accuracies.append(row['cell_accuracy'])
        exact_count += row['exact_match']
        robust_count += row['robust']
        if row['detected']:
            corner_errors.append(row['corner_error_pct'])
        if row['runtime_s'] is not None:
            runtimes.append(row['runtime_s'])

    # missing detections get zero accuracy but only detected corners have an error
    total_accuracy = sum(cell_accuracies)
    denominator = 2 * true_positive + false_positive + false_negative
    return {
        'images': count,
        'detected_images': detected_count,
        'cell_accuracy': total_accuracy / count if count else None,
        'empty_cell_f1': 2 * true_positive / denominator if denominator else 1.0,
        'grid_exact_match': exact_count / count if count else None,
        'robustness': robust_count / count if count else None,
        'mean_corner_error_pct_detected': float(np.mean(corner_errors)) if corner_errors else None,
        'max_runtime_s': max(runtimes) if runtimes else None,
        'mean_runtime_s': float(np.mean(runtimes)) if runtimes else None,
    }


def evaluate(annotations, results, outdir):
    annotations = Path(annotations)
    results = Path(results)
    outdir = Path(outdir)
    files = sorted(annotations.glob('*.json'))
    if not files:
        raise ValueError('no annotations found')
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    gallery = []
    confusions = Counter()
    for path in files:
        truth = json.loads(path.read_text())
        name = Path(truth['image']).stem
        result_path = results / (name + '_result.json')
        prediction = {}
        if result_path.is_file():
            prediction = json.loads(result_path.read_text())
            if prediction.get('image') != truth['image']:
                raise ValueError(f'image name mismatch in {result_path}')
        grid = np.array(prediction.get('grid', []))
        target = np.array(truth['grid'])
        detected = prediction.get('verdict') in ('valid', 'invalid') and grid.shape == (9, 9)
        runtime = prediction.get('runtime_s')
        if type(runtime) not in (int, float) or not np.isfinite(runtime) or runtime < 0:
            runtime = None
        completed = prediction.get('verdict') in ('valid', 'invalid', 'not_found')
        robust = completed and runtime is not None and runtime < 20
        row = {
            'image': truth['image'], 'conditions': truth.get('conditions', []),
            'verdict': prediction.get('verdict', 'missing'), 'detected': detected,
            'runtime_s': runtime, 'robust': robust,
            'cell_accuracy': 0.0, 'exact_match': False, 'corner_error_pct': None,
            'empty_tp': 0, 'empty_fp': 0, 'empty_fn': int(np.sum(target == 0)),
            'models': {},
        }
        if detected:
            row['cell_accuracy'] = float(np.mean(grid == target))
            row['exact_match'] = bool(np.array_equal(grid, target))
            # empty is the positive class for this F1 score
            # use the final OCR grid here so unread digits count as empty
            true_empty = target == 0
            predicted_empty = grid == 0
            row['empty_tp'] = int(np.sum(true_empty & predicted_empty))
            row['empty_fp'] = int(np.sum(~true_empty & predicted_empty))
            row['empty_fn'] = int(np.sum(true_empty & ~predicted_empty))
            corners = np.array(prediction['corners'])
            true_corners = np.array(truth['corners'])
            true_edges = true_corners - np.roll(true_corners, 1, axis=0)
            mean_side_length = np.linalg.norm(true_edges, axis=1).mean()
            corner_distances = np.linalg.norm(corners - true_corners, axis=1)
            row['corner_error_pct'] = float(corner_distances.mean() / mean_side_length * 100)

            # use opposite side ratios for obliqueness as suggested in the PDF
            # roll pairs each corner with the previous one to get the edges
            edges = corners - np.roll(corners, 1, axis=0)
            sides = np.linalg.norm(edges, axis=1)
            left_to_right = sides[0] / sides[2]
            right_to_left = sides[2] / sides[0]
            top_to_bottom = sides[1] / sides[3]
            bottom_to_top = sides[3] / sides[1]
            # use both directions so the ratio starts at 1 and grows with unequal sides
            opposite_side_ratios = (left_to_right, right_to_left, top_to_bottom, bottom_to_top)
            row['obliqueness'] = float(max(opposite_side_ratios))
            for key in ('a3_corner_err_px', 'a4_rms_px', 'h4_max_corner_err_px'):
                row[key] = prediction.get(key)
            for model in ('A3', 'A4', 'H4'):
                model_prediction = prediction.get('models', {}).get(model, {})
                predicted_grid = np.array(model_prediction.get('grid', []))
                if model == 'H4':
                    predicted_grid = grid
                accuracy = None
                if predicted_grid.shape == (9, 9):
                    accuracy = float(np.mean(predicted_grid == target))
                image_path = results / f'{name}_rect_{model}.png'
                image = cv2.imread(str(image_path)) if image_path.is_file() else None
                lines = line_measurements(image) if image is not None else {}
                row['models'][model] = {'cell_accuracy': accuracy, 'grid_lines': lines}
            if 'ocr_without_preparation' in prediction:
                raw_grid = np.array(prediction['ocr_without_preparation'])
                row['raw_cell_accuracy'] = float(np.mean(raw_grid == target))
            image_path = results / (name + '_rect_H4.png')
            cell_images = None
            if image_path.is_file():
                rectified_image = cv2.imread(str(image_path))
                cell_images = extract_cells(rectified_image)
            for row_index, column_index in np.argwhere(grid != target):
                pair = (int(target[row_index, column_index]), int(grid[row_index, column_index]))
                confusions[pair] += 1
                if cell_images is not None and len(gallery) < 36:
                    cell_index = row_index * 9 + column_index
                    cell_image = cell_images[cell_index]
                    gallery.append((name, int(row_index), int(column_index), pair, cell_image))
        rows.append(row)
    summary = summarize(rows)
    summary['by_condition'] = {}
    for condition in ('C1', 'C2', 'C3', 'C4', 'C5'):
        condition_rows = []
        for row in rows:
            if condition in row['conditions']:
                condition_rows.append(row)
        summary['by_condition'][condition] = summarize(condition_rows)
    summary['models'] = {}
    # missing model predictions still count as zero in the all image average
    for model in ('A3', 'A4', 'H4'):
        accuracies = []
        for row in rows:
            model_metrics = row['models'].get(model, {})
            accuracy = model_metrics.get('cell_accuracy')
            accuracies.append(accuracy if accuracy is not None else 0.0)
        summary['models'][model] = {'cell_accuracy_all_images': float(np.mean(accuracies))}
    # compare preparation only on images that have both results
    raw_accuracies = []
    prepared_accuracies = []
    for row in rows:
        if 'raw_cell_accuracy' in row:
            raw_accuracies.append(row['raw_cell_accuracy'])
            prepared_accuracies.append(row['cell_accuracy'])
    summary['ocr_preparation_ablation'] = {
        'paired_images': len(raw_accuracies),
        'without_preparation': float(np.mean(raw_accuracies)) if raw_accuracies else None,
        'with_preparation': float(np.mean(prepared_accuracies)) if prepared_accuracies else None,
    }
    summary['confusions_detected_images'] = []
    for pair, count in confusions.most_common():
        confusion = {'true': pair[0], 'predicted': pair[1], 'count': count}
        summary['confusions_detected_images'].append(confusion)
    summary['corner_error_note'] = 'Corner error averages detected images only. Missing results score zero cell accuracy.'
    write_json(outdir / 'summary.json', summary)
    write_json(outdir / 'per_image.json', rows)
    columns = ['image', 'verdict', 'runtime_s', 'cell_accuracy', 'exact_match', 'corner_error_pct', 'obliqueness']
    with (outdir / 'per_image.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    save_plots(rows, results, outdir)
    save_confusions(gallery, outdir)
    return summary


def save_plots(rows, results, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name in ('affine_vs_homography.png', 'most_oblique.png'):
        (outdir / name).unlink(missing_ok=True)
    detected = []
    for row in rows:
        if row['detected']:
            detected.append(row)
    detected.sort(key=lambda row: row['obliqueness'])
    if not detected:
        return
    obliqueness_values = []
    for row in detected:
        obliqueness_values.append(row['obliqueness'])
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    model_error_keys = (
        ('A3', 'a3_corner_err_px'),
        ('A4', 'a4_rms_px'),
        ('H4', 'h4_max_corner_err_px'),
    )
    for model, error_key in model_error_keys:
        corner_errors = []
        line_errors = []
        accuracies = []
        for row in detected:
            corner_errors.append(row.get(error_key))
            measurements = row['models'][model]
            position_errors = []
            for direction in measurements['grid_lines'].values():
                position_errors.append(direction['position_rmse_px'])
            line_errors.append(float(np.mean(position_errors)) if position_errors else np.nan)
            accuracy = measurements['cell_accuracy']
            accuracies.append(accuracy * 100 if accuracy is not None else np.nan)
        # left shows A3 BR error A4 RMS error and the largest H4 corner error
        axes[0].plot(obliqueness_values, corner_errors, 'o-', label=model)
        # middle averages horizontal and vertical grid position RMSE
        # spacing and segment straightness are saved separately in the JSON
        axes[1].plot(obliqueness_values, line_errors, 'o-', label=model)
        # right shows OCR cell accuracy for each warp against the same truth grid
        axes[2].plot(obliqueness_values, accuracies, 'o-', label=model)
    titles = ('Corner residual (px)', 'Grid position RMSE (px)', 'Cell accuracy (%)')
    for axis, title in zip(axes, titles):
        axis.set(title=title, xlabel='Opposite-side length ratio')
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(outdir / 'affine_vs_homography.png', dpi=150)
    plt.close(figure)
    # use the photo with the largest side ratio for the report comparison
    name = Path(detected[-1]['image']).stem
    panels = []
    for model in ('A3', 'A4', 'H4'):
        path = results / f'{name}_rect_{model}.png'
        image = cv2.imread(str(path)) if path.is_file() else None
        if image is not None:
            panel = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
            cv2.putText(panel, model + ': ' + name, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            panels.append(panel)
    if panels:
        cv2.imwrite(str(outdir / 'most_oblique.png'), np.hstack(panels))


def save_confusions(gallery, outdir):
    path = outdir / 'confusion_crops.png'
    path.unlink(missing_ok=True)
    if not gallery:
        return
    gallery_rows = (len(gallery) + 5) // 6
    canvas_height = gallery_rows * 130
    canvas = np.full((canvas_height, 900, 3), 255, np.uint8)
    for index, (name, row, column, pair, cell_img) in enumerate(gallery):
        x = (index % 6) * 150
        y = (index // 6) * 130
        resized_cell = cv2.resize(cell_img, (80, 80))
        tile = cv2.cvtColor(resized_cell, cv2.COLOR_GRAY2BGR)
        canvas[y:y + 80, x:x + 80] = tile
        label = f'{pair[0]} -> {pair[1]} r{row + 1}c{column + 1}'
        cv2.putText(canvas, label, (x, y + 97), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(canvas, name, (x, y + 117), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
    cv2.imwrite(str(path), canvas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    try:
        summary = evaluate(args.annotations, args.results, args.outdir)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, str(error) + '\n')
    print(f"Detected {summary['detected_images']}/{summary['images']}; "
          f"cell accuracy {summary['cell_accuracy']:.2%}; exact grids {summary['grid_exact_match']:.2%}; "
          f"empty F1 {summary['empty_cell_f1']:.4f}; robustness {summary['robustness']:.2%}")
