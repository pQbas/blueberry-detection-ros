#!/usr/bin/env python3

import cv2

import numpy as np

def attach_information_zone(img):
    H,W,_= img.shape
    black = np.ones((H,W//3), dtype=np.uint8)*100
    grey_3_channel = cv2.cvtColor(black, cv2.COLOR_GRAY2BGR)
    img = np.concatenate((grey_3_channel, img), axis=1)
    return img

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
    x_start = 0 #(w - size) // 2
    x_end = x_start + size
    y_start = 0 #(h - size) // 2
    y_end = y_start + size
    cropped = image[y_start:y_end, x_start:x_end]
    return cropped


def write_text(img, text, position, scale_font, thick, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = position
    fontScale = scale_font
    color_ = color
    thickness = thick
    return cv2.putText(img, text, org, font, fontScale, color_, thickness, cv2.LINE_AA)
    
import torch

class counter:
    def __init__(self, count_mode, threshold_track, direction):
        self.LIST_0 = []
        self.LIST_1 = []
        self.count_mode = count_mode
        self.threshold_track = threshold_track
        self.direction = direction

        if self.direction == 'top2down':
            self.count_condition = lambda y: y > self.threshold_track
        elif self.direction == 'down2top':
            self.count_condition = lambda y: y < self.threshold_track 
        elif self.direction == 'right2left':
            self.count_condition = lambda x: x < self.threshold_track
        elif self.direction == 'left2right':
            self.count_condition = lambda x: x > self.threshold_track
        

    def update_count(self, prediction=None):
        
        if prediction[0] is not None and prediction is not None and prediction[0].boxes.shape[0] > 2:
            boxes = prediction[0].boxes.xywh.cpu()
            centers = boxes[:,:2]
            
            if prediction[0].boxes.id is not None:
                
                track_ids = prediction[0].boxes.id.int().cpu()
                track_ids = track_ids.reshape(track_ids.shape[0], 1)

                to_count = torch.cat((track_ids, centers),1)

                set_0 = set(self.LIST_0)
                set_1 = set(self.LIST_1)
                
                for (id, x, y) in to_count:
                    id = id.item()
        
                    if self.count_condition(x.item()) and self.count_mode == 'horizontal':
                        set_0.add(id)       # Adds the id if not already present                        
                        set_1.discard(id)   # Removes the id if present

                    if self.count_condition(y.item())  and self.count_mode == 'vertical':    
                        set_0.add(id)       # Adds the id if not already present                        
                        set_1.discard(id)   # Removes the id if present
                    
                    elif id in set_0:
                        set_1.add(id)

                self.LIST_0 = list(set_0)
                self.LIST_1 = list(set_1)
    
        return

    def get_number_counted(self):
        return {
            'counted': len(self.LIST_1)
        }
    
    def plot_line_threshold(self, img_pred):
        if self.count_mode == 'vertical':
            cv2.line(img_pred, (0,self.threshold_track), (self.threshold_track,self.threshold_track), (0,255,0), 2)
        if self.count_mode == 'horizontal':
            cv2.line(img_pred, (self.threshold_track,0), (self.threshold_track,self.threshold_track), (0,255,0), 2)
        return img_pred
