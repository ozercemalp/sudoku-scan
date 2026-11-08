import cv2
import numpy as np


def puzzle_image():
    image = np.full((600, 600, 3), 245, np.uint8)
    for index in range(10):
        position = 75 + index * 50
        thickness = 3 if index % 3 == 0 else 1
        cv2.line(image, (75, position), (525, position), (0, 0, 0), thickness)
        cv2.line(image, (position, 75), (position, 525), (0, 0, 0), thickness)
    for row, column, digit in ((0, 0, 5), (1, 3, 3), (4, 4, 7), (8, 8, 2)):
        cv2.putText(image, str(digit), (89 + column * 50, 111 + row * 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    return image
