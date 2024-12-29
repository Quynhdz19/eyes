import cv2
import numpy as np


def create_landolt_c_image(radius, thickness, orientation, canvas_size=(300, 300)):
    canvas = np.ones((canvas_size[1], canvas_size[0], 3), dtype=np.uint8) * 255
    center = (canvas_size[0] // 2, canvas_size[1] // 2)

    if orientation == 'right':
        start_angle, end_angle = 45, 315
    elif orientation == 'left':
        start_angle, end_angle = 225, 495
    elif orientation == 'up':
        start_angle, end_angle = -45, 225
    elif orientation == 'down':
        start_angle, end_angle = 135, 405

    cv2.ellipse(canvas, center, (radius, radius), 0, start_angle, end_angle, (0, 0, 0), thickness)
    return canvas


def calculate_hand_direction(wrist, index_finger_tip):
    import math

    dx = index_finger_tip.x - wrist.x
    dy = index_finger_tip.y - wrist.y
    angle = math.degrees(math.atan2(dy, dx))

    if -45 <= angle <= 45:
        return 'right'
    elif 135 < angle or angle < -135:
        return 'left'
    elif 45 < angle <= 135:
        return 'down'
    elif -135 <= angle < -45:
        return 'up'