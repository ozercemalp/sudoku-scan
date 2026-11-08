"""Find Harris peaks and choose the quadrilateral containing a Sudoku grid."""

from pathlib import Path

import cv2
import numpy as np

from preprocessing import make_threshold, map_points, preprocess


def harris_peaks(gray, block_size=None, ksize=3, k=0.04, threshold=0.01):
    if block_size is None:
        # Scale the corner neighbourhood to 0.4% of the shorter image side.
        block_size = max(3, round(min(gray.shape) * 0.004))
        if block_size % 2 == 0:
            block_size += 1
    response = cv2.cornerHarris(gray.astype(np.float32) / 255.0, block_size, ksize, k)
    maximum = float(response.max())
    if maximum <= 0:
        return response, np.empty((0, 2))
    # Merge nearby peaks within 0.6% of the image size to avoid duplicate corners.
    radius = max(2, round(min(gray.shape) * 0.006))
    local_max = cv2.dilate(response, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    # Keep local peaks above 1% of the strongest response by default.
    ys, xs = np.nonzero((response > threshold * maximum) & (response == local_max))
    order = np.argsort(response[ys, xs])[::-1]
    suppressed = np.zeros(gray.shape, bool)
    points = []
    for index in order:
        x, y = int(xs[index]), int(ys[index])
        if suppressed[y, x]:
            continue
        points.append([x, y])
        suppressed[max(0, y - radius):y + radius + 1,
                   max(0, x - radius):x + radius + 1] = True
    return response, np.array(points, dtype=float).reshape(-1, 2)


def detect_candidates(gray, thresh, **parameters):
    response_gray, points_gray = harris_peaks(gray, **parameters)
    response_binary, points_binary = harris_peaks(thresh, **parameters)
    radius = max(2, round(min(gray.shape) * 0.006))
    occupied = np.zeros(gray.shape, bool)
    points = points_gray.tolist()
    for x, y in points_gray.astype(int):
        occupied[max(0, y - radius):y + radius + 1,
                 max(0, x - radius):x + radius + 1] = True
    # Binary Harris recovers faint printed corners; retain gray peaks when close.
    for x, y in points_binary.astype(int):
        if not occupied[y, x]:
            points.append([x, y])
    if response_gray.max() > 0:
        response_gray = response_gray / response_gray.max()
    if response_binary.max() > 0:
        response_binary = response_binary / response_binary.max()
    return {'response': np.maximum(response_gray, response_binary),
            'points': np.array(points, dtype=float).reshape(-1, 2)}


def order_corners(points):
    points = np.array(points, dtype=np.float32).reshape(4, 2)
    centre = points.mean(axis=0)
    # Sort by angle around the centre so the corners follow the boundary.
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    points = points[np.argsort(angles)]
    # The smallest x+y estimates the top-left corner; roll puts it first.
    start = int(np.argmin(points[:, 0] + points[:, 1]))
    return np.roll(points, -start, axis=0)


def grid_score(thresh, corners):
    # A 360-pixel square gives nine 40-pixel cells for checking grid spacing.
    target = np.float32([[0, 0], [360, 0], [360, 360], [0, 360]])
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
    square = cv2.warpPerspective(thresh, matrix, (360, 360), flags=cv2.INTER_NEAREST)
    strengths = []
    for axis in (0, 1):
        # Row and column averages show how much white grid ink crosses each line.
        profile = square.mean(axis=axis) / 255
        peaks = []
        # Look within 9 pixels of each expected internal line to allow small errors.
        for position in range(40, 360, 40):
            peaks.append(float(profile[position - 9:position + 10].max()))
        strengths.append(np.array(peaks))
    horizontal, vertical = strengths
    average = (horizontal.mean() + vertical.mean()) / 2
    # Count lines with over 40% ink coverage, using the weaker grid direction.
    coverage = min(np.mean(horizontal > 0.40), np.mean(vertical > 0.40))
    # The lower quartile checks that the score is not carried by a few strong lines.
    weak_lines = min(np.quantile(horizontal, 0.25), np.quantile(vertical, 0.25))
    # Reward average line strength, then coverage and support from weaker lines.
    return float(average + 0.5 * coverage + 0.25 * weak_lines)


def select_corners(gray, thresh, candidates):
    points = candidates['points']
    if len(points) < 4:
        return None
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    more_contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(list(contours) + list(more_contours), key=cv2.contourArea, reverse=True)[:180]
    best_corners = None
    best_indices = None
    # Require a grid score above 0.85 to reject weak matches; it is not a probability.
    best_score = 0.85
    seen = set()
    for contour in contours:
        # Ignore regions below 1.2% of the image area, such as small text or digits.
        if cv2.contourArea(contour) < thresh.size * 0.012:
            continue
        perimeter = cv2.arcLength(contour, True)
        # Try several simplification tolerances relative to the contour perimeter.
        for fraction in (0.015, 0.025, 0.045):
            polygon = cv2.approxPolyDP(contour, fraction * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            proposal = order_corners(polygon)
            # Roll pairs each corner with the previous one, including the closing edge.
            # The norm of each difference gives the four edge lengths.
            sides = np.linalg.norm(proposal - np.roll(proposal, 1, axis=0), axis=1)
            # Reject edges below 8% of the image size and very thin quadrilaterals.
            # The 0.18 shortest/longest ratio still allows strong perspective distortion.
            if sides.min() < min(thresh.shape) * 0.08 or sides.min() / sides.max() < 0.18:
                continue
            indices = []
            distances = []
            for vertex in proposal:
                # Measure each Harris point's distance to this proposed corner.
                distance = np.linalg.norm(points - vertex, axis=1)
                index = int(np.argmin(distance))
                indices.append(index)
                distances.append(distance[index])
            # Use four different peaks, each within 3.5% of the mean edge length.
            # A 5-pixel minimum tolerance allows small contour offsets on small grids.
            if len(set(indices)) != 4 or max(distances) > max(5, sides.mean() * 0.035):
                continue
            if tuple(indices) in seen:
                continue
            seen.add(tuple(indices))
            # The contour suggests where to look, use Harris points for the corners.
            snapped = points[indices].astype(np.float32)
            if not cv2.isContourConvex(snapped.reshape(4, 1, 2)):
                continue
            score = grid_score(thresh, snapped)
            if score > best_score:
                best_corners, best_indices, best_score = snapped, indices, score
    if best_corners is None:
        return None
    refined = cv2.cornerSubPix(
        gray, best_corners.reshape(4, 1, 2).copy(), (4, 4), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
    ).reshape(4, 2)
    for index in range(4):
        # Keep refinement close to its Harris peak rather than a neighbouring feature.
        if np.linalg.norm(refined[index] - best_corners[index]) <= 6:
            best_corners[index] = refined[index]
    return {'corners': best_corners, 'indices': best_indices, 'score': best_score}


def find_corners(prepared):
    for constant in (7, 1):
        thresh = make_threshold(prepared['gray'], prepared['block_size'], constant)
        candidates = detect_candidates(prepared['gray'], thresh)
        selected = select_corners(prepared['gray'], thresh, candidates)
        if selected is not None:
            selected['threshold_c'] = constant
            return selected, candidates
    return None, candidates


def save_harris(path, candidates, corners=None):
    response = np.maximum(candidates['response'], 0)
    if response.max() > 0:
        response = response / response.max()
    image = cv2.applyColorMap(np.uint8(np.sqrt(response) * 255), cv2.COLORMAP_INFERNO)
    for x, y in candidates['points']:
        cv2.circle(image, (round(x), round(y)), 3, (0, 255, 0), 1)
    if corners is not None:
        cv2.polylines(image, [np.rint(corners).astype(np.int32)], True, (255, 255, 255), 2)
        for name, point in zip(('TL', 'TR', 'BR', 'BL'), corners):
            position = tuple(np.rint(point).astype(int))
            cv2.circle(image, position, 7, (255, 255, 255), 2)
            cv2.putText(image, name, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    if not cv2.imwrite(str(path), image):
        raise OSError(f'could not save {path}')


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    prepared = preprocess(args.input)
    selected, candidates = find_corners(prepared)
    corners = None if selected is None else selected['corners']
    args.outdir.mkdir(parents=True, exist_ok=True)
    save_harris(args.outdir / (args.input.stem + '_harris.png'), candidates, corners)
    original = None if corners is None else map_points(corners, prepared['to_original']).tolist()
    print(json.dumps({'image': args.input.name, 'corners': original}))
