"""Measure Harris parameter effects on six annotated photographs."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from corners import detect_candidates, select_corners
from preprocessing import preprocess, map_points


def run_study(images_dir, annotations_dir, outdir, limit=6):
    # Six images meet the project's minimum sorting keeps the selection repeatable
    paths = sorted(Path(images_dir).glob('*.jpg'))[:limit]
    if not paths:
        raise ValueError('no photographs found')
    # Try small odd neighbourhoods and Sobel kernels to compare smoothing scales
    # k values around 0.04 vary the edge penalty in the Harris response
    # Thresholds keep peaks above 0.3%, 1%, or 3% of the strongest response
    variants = {'block_size': [3, 5, 7], 'ksize': [3, 5, 7],
                'k': [0.02, 0.04, 0.06], 'threshold': [0.003, 0.01, 0.03]}
    rows = []
    for path in paths:
        prepared = preprocess(path)
        truth = json.loads((Path(annotations_dir) / (path.stem + '.json')).read_text())
        true_corners = np.array(truth['corners'])
        # Roll pairs consecutive corners, including the last-to-first edge.
        # Average the four edge lengths to get a reference puzzle size.
        side_length = np.linalg.norm(true_corners - np.roll(true_corners, 1, axis=0), axis=1).mean()
        # Match the detector's default: 0.4% of the shorter side, at least 3 and odd.
        block_size = max(3, round(min(prepared['gray'].shape) * 0.004))
        if block_size % 2 == 0:
            block_size += 1
        defaults = {'block_size': block_size, 'ksize': 3, 'k': 0.04, 'threshold': 0.01}
        for parameter, values in variants.items():
            for value in values:
                # Change one parameter at a time so its effect can be compared.
                settings = dict(defaults)
                settings[parameter] = value
                candidates = detect_candidates(prepared['gray'], prepared['thresh'], **settings)
                selected = select_corners(prepared['gray'], prepared['thresh'], candidates)
                # A failed detection has no corner error, rather than zero error.
                error = None
                if selected is not None:
                    # Compare predictions and annotations in original-image pixels.
                    corners = map_points(selected['corners'], prepared['to_original'])
                    # Mean corner distance as a percentage of the mean puzzle side.
                    error = float(np.linalg.norm(corners - true_corners, axis=1).mean() / side_length * 100)
                rows.append({'image': path.name, 'parameter': parameter, 'value': value,
                             'candidates': len(candidates['points']), 'corner_error_pct': error})
        print(path.name, 'parameter study complete', flush=True)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / 'harris_study.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    # Save plots without needing a display window.
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # One column per parameter. Candidate counts above, corner errors below.
    figure, axes = plt.subplots(2, 4, figsize=(15, 7))
    for column, parameter in enumerate(variants):
        for path in paths:
            samples = [row for row in rows if row['image'] == path.name and row['parameter'] == parameter]
            x = [row['value'] for row in samples]
            axes[0, column].plot(x, [row['candidates'] for row in samples], 'o-', label=path.stem)
            # NaN leaves gaps where detection failed, instead of plotting false zeros.
            errors = [row['corner_error_pct'] if row['corner_error_pct'] is not None else np.nan for row in samples]
            axes[1, column].plot(x, errors, 'o-')
        axes[0, column].set(title=parameter, ylabel='Candidate count')
        axes[1, column].set(xlabel=parameter, ylabel='Corner error (% side)')
    axes[0, 0].legend(fontsize=7)
    figure.suptitle('Missing error points indicate detection failure')
    figure.tight_layout()
    figure.savefig(outdir / 'harris_study.png', dpi=150)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=Path('data/personal/images'))
    parser.add_argument('--annotations', type=Path, default=Path('data/personal/annotations'))
    parser.add_argument('--outdir', type=Path, default=Path('results/student_dev/harris_study'))
    parser.add_argument('--limit', type=int, default=6)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    run_study(args.images, args.annotations, args.outdir, args.limit)
