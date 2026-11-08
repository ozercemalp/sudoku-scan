"""Compare three-point affine, four-point affine, and projective warps."""

from pathlib import Path

import cv2
import numpy as np

from preprocessing import map_points


# 450x450 gives 50x50 pixels per cell for the 9x9 grid
TARGET_SIZE = 450


def rectify(image, corners):
    # both lists use TL TR BR BL order
    corners = np.array(corners, dtype=np.float32).reshape(4, 2)
    size = TARGET_SIZE
    # corner targets use 450 while the saved pixels run from 0 to 449
    target = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    # A3 fits TL TR and BL exactly then we check where BR ends up
    # affine maps keep lines parallel so they cant fully undo perspective
    matrix_a3 = cv2.getAffineTransform(corners[[0, 1, 3]], target[[0, 1, 3]])
    # A4 shares the fitting error across all 4 corners using least squares
    # huge RANSAC threshold keeps all 4 points for the least squares refinement
    matrix_a4, inliers = cv2.estimateAffine2D(
        corners, target, method=cv2.RANSAC, ransacReprojThreshold=1e8,
        maxIters=1, confidence=0.99, refineIters=10,
    )
    if matrix_a4 is None or not np.all(inliers):
        raise ValueError('A4 could not fit all four corners')
    # H4 fits all 4 corners and can undo perspective too
    matrix_h4 = cv2.getPerspectiveTransform(corners, target)
    matrices = {'A3': matrix_a3, 'A4': matrix_a4, 'H4': matrix_h4}
    images = {}
    for name, matrix in matrices.items():
        if name == 'H4':
            images[name] = cv2.warpPerspective(image, matrix, (size, size), borderValue=(255, 255, 255))
        else:
            images[name] = cv2.warpAffine(image, matrix, (size, size), borderValue=(255, 255, 255))

    # extra row lets map_points work with the affine matrices too
    affine_a3 = np.vstack([matrix_a3, [0, 0, 1]])
    affine_a4 = np.vstack([matrix_a4, [0, 0, 1]])
    mapped_a3 = map_points(corners, affine_a3)
    mapped_a4 = map_points(corners, affine_a4)
    mapped_h4 = map_points(corners, matrix_h4)
    # errors are measured in target image pixels
    residuals = {
        # where A3 places BR which wasnt used in the fit
        'a3_br_target': mapped_a3[2].tolist(),
        # distance from BR to its target at 450 450
        'a3_corner_err_px': float(np.linalg.norm(mapped_a3[2] - target[2])),
        # RMS distance of all 4 corners from their targets
        'a4_rms_px': float(np.sqrt(np.mean(np.sum((mapped_a4 - target) ** 2, axis=1)))),
        # largest H4 corner error should be almost zero
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
    # map corners back before warping the original photo
    corners = map_points(selected['corners'], prepared['to_original'])
    images, matrices, residuals = rectify(prepared['image'], corners)
    save_rectifications(args.outdir, args.input.stem, images)
    print(residuals)
