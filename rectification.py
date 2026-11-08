"""Compare three-point affine, four-point affine, and projective warps."""

from pathlib import Path

import cv2
import numpy as np

from preprocessing import map_points


# A 450x450 image gives 50x50 pixels per cell in a 9x9 grid.
TARGET_SIZE = 450


def rectify(image, corners):
    # Both corner lists use the order TL, TR, BR, BL.
    corners = np.array(corners, dtype=np.float32).reshape(4, 2)
    size = TARGET_SIZE
    target = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    # A3 fits TL, TR and BL exactly, leaving BR as a check on the affine model.
    # Affine maps preserve parallel lines, so they cannot fully undo perspective.
    matrix_a3 = cv2.getAffineTransform(corners[[0, 1, 3]], target[[0, 1, 3]])
    # A4 fits all four corners by least squares, sharing the error between them.
    # Keep every observation. RANSAC's usual threshold can discard the fourth
    # corner even though the purpose here is to measure the affine fit error.
    # The large threshold keeps all four points for the least-squares refinement.
    matrix_a4, inliers = cv2.estimateAffine2D(
        corners, target, method=cv2.RANSAC, ransacReprojThreshold=1e8,
        maxIters=1, confidence=0.99, refineIters=10,
    )
    if matrix_a4 is None or not np.all(inliers):
        raise ValueError('A4 could not fit all four corners')
    # H4 uses four pairs to fit a homography, which can undo perspective distortion.
    # Compare it with A3 and A4 to see when an affine approximation is enough.
    matrix_h4 = cv2.getPerspectiveTransform(corners, target)
    matrices = {'A3': matrix_a3, 'A4': matrix_a4, 'H4': matrix_h4}
    images = {}
    for name, matrix in matrices.items():
        # Homographies need a projective warp; white fills pixels outside the photo.
        if name == 'H4':
            images[name] = cv2.warpPerspective(image, matrix, (size, size), borderValue=(255, 255, 255))
        else:
            images[name] = cv2.warpAffine(image, matrix, (size, size), borderValue=(255, 255, 255))

    # Add a homogeneous row so map_points can also use the 2x3 affine matrices.
    affine_a3 = np.vstack([matrix_a3, [0, 0, 1]])
    affine_a4 = np.vstack([matrix_a4, [0, 0, 1]])
    mapped_a3 = map_points(corners, affine_a3)
    mapped_a4 = map_points(corners, affine_a4)
    mapped_h4 = map_points(corners, matrix_h4)
    # All residuals below measure distances in the rectified image's pixels.
    residuals = {
        # Where A3 sends the unused BR corner, ideally (450, 450).
        'a3_br_target': mapped_a3[2].tolist(),
        # Distance from that BR prediction to its ideal target position.
        'a3_corner_err_px': float(np.linalg.norm(mapped_a3[2] - target[2])),
        # Root mean squared distance over all four corners for the A4 fit.
        'a4_rms_px': float(np.sqrt(np.mean(np.sum((mapped_a4 - target) ** 2, axis=1)))),
        # Largest H4 corner error; it should be near zero apart from rounding.
        'h4_max_corner_err_px': float(np.linalg.norm(mapped_h4 - target, axis=1).max()),
    }
    return images, matrices, residuals


def save_rectifications(outdir, name, images):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for model, image in images.items():
        path = outdir / f'{name}_rect_{model}.png'
        if not cv2.imwrite(str(path), image):
            raise OSError(f'could not write {path}')


if __name__ == '__main__':
    import argparse
    from corners import find_corners
    from preprocessing import preprocess
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    prepared = preprocess(args.input)
    selected, candidates = find_corners(prepared)
    if selected is None:
        parser.exit(1, 'no puzzle found\n')
    # Map detected corners back before warping the original-resolution photo.
    corners = map_points(selected['corners'], prepared['to_original'])
    images, matrices, residuals = rectify(prepared['image'], corners)
    save_rectifications(args.outdir, args.input.stem, images)
    print(residuals)
