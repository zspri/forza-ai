from PIL import Image
from . import datatypes
import cv2
import numpy as np

# based off of https://towardsdatascience.com/tutorial-build-a-lane-detector-679fd8953132


def do_canny(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)

    return canny


def do_segment(frame: np.ndarray) -> np.ndarray:
    height = frame.shape[0]
    # Creates a triangular polygon for the mask defined by 3 (x, y) coordinates
    polygons = np.array([
        [(160, height - 300), (1920 - 160, height - 300), (960, 420)]
    ])
    # Creates an image filled with zero intensities with the same dimensions as the frame
    mask = np.zeros_like(frame)
    # Allows the mask to be filled with values of 1 and the other areas to be filled with values of 0
    cv2.fillPoly(mask, polygons, 255)
    # A bitwise and operation between the mask and frame keeps only the triangular area of the frame
    segment = cv2.bitwise_and(frame, mask)
    return segment


def calculate_lines(frame, lines):
    # Empty arrays to store the coordinates of the left and right lines
    left = []
    right = []
    # Loops through every detected line
    for line in lines:
        # Reshapes line from 2D array to 1D array
        x1, y1, x2, y2 = line.reshape(4)
        # print(f'{x1=} {y1=} {x2=} {y2=}')
        # Fits a linear polynomial to the x and y coordinates and returns a vector of coefficients which describe the slope and y-intercept
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        # If slope is negative, the line is to the left of the lane, and otherwise, the line is to the right of the lane
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    # Averages out all the values for left and right into a single slope and y-intercept value for each line
    left_avg = np.average(left, axis=0)
    right_avg = np.average(right, axis=0)
    # Calculates the x1, y1, x2, y2 coordinates for the left and right lines
    left_line = calculate_coordinates(frame, left_avg)
    right_line = calculate_coordinates(frame, right_avg)
    return np.array([left_line, right_line])


def calculate_coordinates(frame, parameters):
    slope, intercept = parameters

    # Sets initial y-coordinate as height from top down (bottom of the frame)
    y1 = 810  # frame.shape[0]
    # Sets final y-coordinate as 150 above the bottom of the frame
    y2 = 450  # int(y1 - 150)
    # Sets initial x-coordinate as (y1 - b) / m since y1 = mx1 + b
    x1 = int((y1 - intercept) / slope)
    # Sets final x-coordinate as (y2 - b) / m since y2 = mx2 + b
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def canny_all(frame: Image) -> datatypes.PredictionData:
    # TODO: separate visualization from line detection

    cv_img = np.array(frame.convert('RGB'))
    cv_img = cv_img[:, :, ::-1].copy()

    cn = do_canny(cv_img)
    seg = do_segment(cn)
    hough = cv2.HoughLinesP(seg, 2, np.pi / 180, 100, np.array([]), minLineLength=100, maxLineGap=10)

    lines = calculate_lines(cv_img, hough)
    # lines_visualize = visualize_lines(cv_img, lines)
    # output = cv2.addWeighted(cv_img, 0.9, lines_visualize, 1, 1)

    """
    seg_img = Image.fromarray(seg)
    seg_img = np.array(seg_img.convert('RGB'))
    seg_img = seg_img[:, :, ::-1].copy()
    if gui_mode == 0:  # lane detection w/ overlay
        cv_overlay = cv2.addWeighted(cv_img, 1, lines_visualize, 0.4, 0)
        output: np.ndarray = cv2.addWeighted(cv_overlay, 1, seg_img, 0.4, 0)
    elif gui_mode == 1:  # lane detection w/ segmentation
        output: np.ndarray = cv2.addWeighted(seg_img, 1, lines_visualize, 0.4, 0)
    else:
        output = np.ndarray([])
    """

    return datatypes.PredictionData(
        lines=lines,
        original=frame,
        segmentation=seg)
