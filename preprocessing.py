"""Read phone photos and prepare them for corner detection."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def map_points(points, matrix):
    mapped = []
    for x, y in points:
        point = matrix @ np.array([x, y, 1.0])
        mapped.append([point[0] / point[2], point[1] / point[2]])
    return np.array(mapped, dtype=float).reshape(-1, 2)


def orientation_matrix(orientation, width, height):
    # These maps undo the EXIF display transformation, including reflections.
    transforms = {
        1: [[1, 0, 0], [0, 1, 0]],
        2: [[-1, 0, width - 1], [0, 1, 0]],
        3: [[-1, 0, width - 1], [0, -1, height - 1]],
        4: [[1, 0, 0], [0, -1, height - 1]],
        5: [[0, 1, 0], [1, 0, 0]],
        6: [[0, 1, 0], [-1, 0, height - 1]],
        7: [[0, -1, width - 1], [-1, 0, height - 1]],
        8: [[0, -1, width - 1], [1, 0, 0]],
    }
    rows = transforms.get(orientation, transforms[1])
    return np.array(rows + [[0, 0, 1]], dtype=float)


def make_threshold(gray, block_size, constant=7):
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, constant,
    )


def preprocess(path, max_side=1600):
    if max_side < 32:
        raise ValueError('max_side must be at least 32')
    with Image.open(path) as photo:
        width, height = photo.size
        orientation = photo.getexif().get(274, 1)
        original = cv2.cvtColor(np.array(photo.convert('RGB')), cv2.COLOR_RGB2BGR)
        upright = cv2.cvtColor(np.array(ImageOps.exif_transpose(photo).convert('RGB')), cv2.COLOR_RGB2BGR)
    if min(width, height) < 32:
        raise ValueError('photo is too small')

    upright_height, upright_width = upright.shape[:2]
    scale = min(1.0, max_side / max(upright_width, upright_height))
    new_width = max(3, round(upright_width * scale))
    new_height = max(3, round(upright_height * scale))
    resized = cv2.resize(upright, (new_width, new_height), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    short_side = min(gray.shape)
    blur_size = max(3, round(short_side * 0.002))
    if blur_size % 2 == 0:
        blur_size += 1
    block_size = max(3, round(short_side * 0.035))
    if block_size % 2 == 0:
        block_size += 1
    block_size = min(block_size, short_side if short_side % 2 else short_side - 1)
    gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    sigma = max(1.0, short_side * 0.03)
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    gray = cv2.divide(gray, background, scale=255)
    thresh = make_threshold(gray, block_size)

    # OpenCV resize maps pixel centres, so the inverse includes a half-pixel shift.
    scale_x = upright_width / new_width
    scale_y = upright_height / new_height
    resize_matrix = np.array([
        [scale_x, 0, (scale_x - 1) / 2],
        [0, scale_y, (scale_y - 1) / 2],
        [0, 0, 1],
    ])
    to_original = orientation_matrix(orientation, width, height) @ resize_matrix
    return {'image': original, 'gray': gray, 'thresh': thresh,
            'to_original': to_original, 'block_size': block_size}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    prepared = preprocess(args.input)
    args.outdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.outdir / (args.input.stem + '_gray.png')), prepared['gray'])
    cv2.imwrite(str(args.outdir / (args.input.stem + '_thresh.png')), prepared['thresh'])
