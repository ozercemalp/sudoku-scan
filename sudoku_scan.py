"""Run the Sudoku photograph pipeline and save the required outputs."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import cv2

from preprocessing import preprocess, map_points
from corners import find_corners, save_harris
from rectification import rectify, save_rectifications
from ocr import extract_cells, recognise_cells
from validation import validate_grid


def format_json(value, level=0):
    """Keep coordinate pairs and grid rows together on single lines."""
    indent = '  ' * level
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            lines.append('  ' * (level + 1) + json.dumps(key) + ': ' + format_json(item, level + 1))
        return '{\n' + ',\n'.join(lines) + '\n' + indent + '}'
    if isinstance(value, list) and any(isinstance(item, (list, dict)) for item in value):
        lines = []
        for item in value:
            lines.append('  ' * (level + 1) + format_json(item, level + 1))
        return '[\n' + ',\n'.join(lines) + '\n' + indent + ']'
    return json.dumps(value, allow_nan=False)


def write_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(format_json(result) + '\n', encoding='utf-8')


def remove_predictions(outdir, name):
    # clear old predictions if this run fails
    for suffix in ('grid.txt', 'rect_A3.png', 'rect_A4.png', 'rect_H4.png', 'debug.png'):
        (outdir / f'{name}_{suffix}').unlink(missing_ok=True)


def save_debug(path, image, prediction, conflicts):
    debug = image.copy()
    height, width = debug.shape[:2]
    for row in range(9):
        for column in range(9):
            x = round(column * width / 9)
            y = round(row * height / 9)
            right = round((column + 1) * width / 9) - 1
            bottom = round((row + 1) * height / 9) - 1
            # BGR colors green for normal red for conflicts orange for unread
            color = (0, 180, 0)
            if [row, column] in conflicts:
                color = (0, 0, 255)
            if [row, column] in prediction['unread_cells']:
                color = (0, 170, 255)
            cv2.rectangle(debug, (x, y), (right, bottom), color, 1)
            digit = prediction['grid'][row][column]
            if digit:
                cv2.putText(debug, str(digit), (x + 3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            elif [row, column] in prediction['unread_cells']:
                # use ? for unread digits so they dont look like empty cells
                cv2.putText(debug, '?', (x + 3, y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    if not cv2.imwrite(str(path), debug):
        raise OSError(f'could not write {path}')


def process_image(path, outdir, compare=False):
    start = perf_counter()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    result = {'image': path.name, 'conditions': []}
    try:
        prepared = preprocess(path)
        selected, candidates = find_corners(prepared)
        selected_points = None if selected is None else selected['corners']
        # save Harris output even if no puzzle is found
        save_harris(outdir / (path.stem + '_harris.png'), candidates, selected_points)
        if selected is None:
            result['verdict'] = 'not_found'
            remove_predictions(outdir, path.stem)
        else:
            # map corners back to the original photo size before warping
            corners = map_points(selected['corners'], prepared['to_original'])
            images, matrices, errors = rectify(prepared['image'], corners)
            # use H4 for the main grid and give OCR up to 5 seconds
            # cap its deadline at 18 seconds from start to leave time for saving
            prediction = recognise_cells(extract_cells(images['H4']), deadline=min(start + 18, perf_counter() + 5))
            verdict, conflicts = validate_grid(prediction['grid'])
            result['corners'] = corners.tolist()
            for model, matrix in matrices.items():
                result[model] = matrix.tolist()
            result.update(errors)
            result.update(prediction)
            result['verdict'] = verdict
            result['conflicting_cells'] = conflicts
            harris_points = candidates['points'][selected['indices']]
            result['harris_points_before_refinement'] = map_points(harris_points, prepared['to_original']).tolist()
            result['grid_score'] = selected['score']
            result['threshold_c'] = selected['threshold_c']
            result['models'] = {'H4': prediction}
            if compare:
                # also try OCR on affine warps but keep H4 as the main result
                for model in ('A3', 'A4'):
                    # skip extra OCR after 16 seconds to leave some time for output
                    if perf_counter() - start > 16:
                        result['models'][model] = {'error': 'OCR time budget exhausted'}
                        continue
                    try:
                        result['models'][model] = recognise_cells(
                            extract_cells(images[model]), deadline=min(start + 18, perf_counter() + 5),
                        )
                    except (RuntimeError, TimeoutError) as error:
                        result['models'][model] = {'error': str(error)}
                if perf_counter() - start < 16:
                    try:
                        # compare raw H4 cells with the prepared ones
                        raw = recognise_cells(extract_cells(images['H4']), prepare=False,
                                              deadline=min(start + 19, perf_counter() + 4))
                        result['ocr_without_preparation'] = raw['grid']
                    except (RuntimeError, TimeoutError) as error:
                        result['ocr_without_preparation_error'] = str(error)
            save_rectifications(outdir, path.stem, images)
            rows = []
            # save 9 digits per row and use 0 for empty cells
            for row in prediction['grid']:
                rows.append(''.join(map(str, row)))
            (outdir / (path.stem + '_grid.txt')).write_text('\n'.join(rows) + '\n', encoding='utf-8')
            save_debug(outdir / (path.stem + '_debug.png'), images['H4'], prediction, conflicts)
    except Exception as error:
        # one bad file shouldnt stop the batch
        # keep processing errors separate from no puzzle found
        result = {'image': path.name, 'conditions': [], 'verdict': 'error',
                  'error': f'{type(error).__name__}: {error}'}
        remove_predictions(outdir, path.stem)
    result['runtime_s'] = perf_counter() - start
    result_path = outdir / (path.stem + '_result.json')
    write_json(result_path, result)
    # include the first JSON write in runtime too
    result['runtime_s'] = perf_counter() - start
    write_json(result_path, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    # choose either one photo or a batch folder
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', type=Path)
    source.add_argument('--batch', type=Path)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--compare', action='store_true', help='also run A3/A4 OCR and the raw-cell comparison')
    args = parser.parse_args(argv)
    if args.outdir.exists() and not args.outdir.is_dir():
        parser.error('--outdir must be a directory')
    if args.input:
        if not args.input.is_file():
            parser.error('--input must name an existing photograph')
        images = [args.input]
    else:
        if not args.batch.is_dir():
            parser.error('--batch must name an existing directory')
        # sort files so each batch runs in the same order
        images = sorted(path for path in args.batch.iterdir()
                        if path.is_file() and path.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        if not images:
            parser.error('no JPEG or PNG photographs found')
    # same filename stem would overwrite outputs even with different extensions
    stems = [path.stem.casefold() for path in images]
    if len(stems) != len(set(stems)):
        parser.error('image filename stems must be unique')
    failed = False
    for path in images:
        try:
            result = process_image(path, args.outdir, args.compare)
            print(f"{path.name}: {result['verdict']} ({result['runtime_s']:.2f}s)", flush=True)
            if result['verdict'] == 'error':
                print(result['error'], file=sys.stderr)
                failed = True
        except OSError as error:
            print(f'{path.name}: {error}', file=sys.stderr)
            failed = True
    # conflicts are normal results only processing failures give exit code 1
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
