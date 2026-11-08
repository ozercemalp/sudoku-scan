"""Find Harris peaks and choose the quadrilateral containing a Sudoku grid."""

from pathlib import Path

import cv2
import numpy as np

from preprocessing import make_threshold, map_points, preprocess


def harris_peaks(gray, block_size=None, ksize=3, k=0.04, threshold=0.01):
    if block_size is None:
        # use 0.4% of the short side so the corner window scales with the photo
        block_size = max(3, round(min(gray.shape) * 0.004))
        if block_size % 2 == 0:
            block_size += 1
    response = cv2.cornerHarris(gray.astype(np.float32) / 255.0, block_size, ksize, k)
    maximum = float(response.max())
    if maximum <= 0:
        return response, np.empty((0, 2))
    # suppress nearby peaks within 0.6% of the image size to avoid duplicates
    radius = max(2, round(min(gray.shape) * 0.006))
    local_max = cv2.dilate(response, np.ones((2 * radius + 1, 2 * radius + 1), np.uint8))
    # keep peaks above 1% of the strongest response by default
    ys, xs = np.nonzero((response > threshold * maximum) & (response == local_max))
    # check strongest peaks first so nearby weaker ones get dropped
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
    # binary Harris helps with faint corners but prefer gray peaks when nearby
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
    # sort around the centre so the corners follow the boundary
    angles = np.arctan2(points[:, 1] - centre[1], points[:, 0] - centre[0])
    points = points[np.argsort(angles)]
    # smallest x+y is our top left estimate then roll moves it to the start
    start = int(np.argmin(points[:, 0] + points[:, 1]))
    return np.roll(points, -start, axis=0)


def grid_score(thresh, corners):
    # 360 pixels gives 40 per cell for checking grid spacing
    target = np.float32([[0, 0], [360, 0], [360, 360], [0, 360]])
    matrix = cv2.getPerspectiveTransform(corners.astype(np.float32), target)
    square = cv2.warpPerspective(thresh, matrix, (360, 360), flags=cv2.INTER_NEAREST)
    strengths = []
    for axis in (0, 1):
        # row and column averages make grid lines show up as peaks
        profile = square.mean(axis=axis) / 255
        peaks = []
        # allow 9 pixels either way since the proposed corners wont be exact
        for position in range(40, 360, 40):
            peaks.append(float(profile[position - 9:position + 10].max()))
        strengths.append(np.array(peaks))
    horizontal, vertical = strengths
    average = (horizontal.mean() + vertical.mean()) / 2
    # count lines above 40% ink coverage and use the weaker direction
    coverage = min(np.mean(horizontal > 0.40), np.mean(vertical > 0.40))
    # check the weaker quarter of lines so a few strong ones dont dominate
    weak_lines = min(np.quantile(horizontal, 0.25), np.quantile(vertical, 0.25))
    return float(average + 0.5 * coverage + 0.25 * weak_lines)


def select_corners(gray, thresh, candidates):
    points = candidates['points']
    if len(points) < 4:
        return None
    # closing joins small gaps but we also check contours from the original threshold
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    more_contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(list(contours) + list(more_contours), key=cv2.contourArea, reverse=True)[:180]
    best_corners = None
    best_indices = None
    # require a score above 0.85 to reject weak grid matches
    best_score = 0.85
    seen = set()
    for contour in contours:
        # skip regions below 1.2% of image area to avoid small text and digits
        if cv2.contourArea(contour) < thresh.size * 0.012:
            continue
        perimeter = cv2.arcLength(contour, True)
        # try a few simplification amounts based on the contour perimeter
        for fraction in (0.015, 0.025, 0.045):
            polygon = cv2.approxPolyDP(contour, fraction * perimeter, True)
            if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                continue
            proposal = order_corners(polygon)
            # roll pairs each corner with the previous one including the closing edge
            sides = np.linalg.norm(proposal - np.roll(proposal, 1, axis=0), axis=1)
            # skip edges shorter than 8% of the image size
            # the 0.18 side ratio rejects very thin shapes but still allows perspective
            if sides.min() < min(thresh.shape) * 0.08 or sides.min() / sides.max() < 0.18:
                continue
            indices = []
            distances = []
            for vertex in proposal:
                distance = np.linalg.norm(points - vertex, axis=1)
                index = int(np.argmin(distance))
                indices.append(index)
                distances.append(distance[index])
            # need 4 different peaks within 3.5% of the mean edge length
            # allow at least 5 pixels for contour offsets on small grids
            if len(set(indices)) != 4 or max(distances) > max(5, sides.mean() * 0.035):
                continue
            if tuple(indices) in seen:
                continue
            seen.add(tuple(indices))
            # contours tell us where to look then we take the corners from Harris
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
        # ignore refinements more than 6 pixels from the Harris peak
        if np.linalg.norm(refined[index] - best_corners[index]) <= 6:
            best_corners[index] = refined[index]
    return {'corners': best_corners, 'indices': best_indices, 'score': best_score}


def find_corners(prepared):
    # retry with 1 if 7 misses a faint grid border
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
    # square root makes weak Harris responses easier to see
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
