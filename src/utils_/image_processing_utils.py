#!/usr/bin/env python3

import cv2

def draw_line(image, position, orientation):
    h,w,c = image.shape
    color = (255,0,0)
    thickness = 2
    
    if orientation == 'vertical':
        start_point = (position[0],0)
        end_point = (position[0], h)

    elif orientation == 'horizontal':
        start_point = (0, position[1])
        end_point = (w, position[1])

    image = cv2.line(image, start_point, end_point, color, thickness)
    return image


def crop_center_square(image):
    h, w, _ = image.shape
    size = min(h, w)
    x_start = (w - size) // 2
    x_end = x_start + size
    y_start = (h - size) // 2
    y_end = y_start + size
    cropped = image[y_start:y_end, x_start:x_end]
    return cropped


def write_text(img, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = position
    fontScale = 2
    color = (255, 0, 255)
    thickness = 3
    return cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    