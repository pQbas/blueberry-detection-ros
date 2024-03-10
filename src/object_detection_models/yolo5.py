

import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
import numpy as np
import os
import time
import tkinter
import matplotlib
#matplotlib.use('TkAgg')
sys.path.append('/home/labinm-jetson/catkin_ws/src/blueberry-detection-ros/src/object_detection_models')
sys.path.append('/home/pqbas/catkin_ws/src/blueberry-detection-ros/src/object_detection_models')

#from yolov5.utils.general import scale_coords, xyxy2xywh
from common_functions_ import scale_coords

sys.path.append('/home/labinm-jetson/catkin_ws/src/blueberry-detection-ros/src/object_detection_models/yolov5')
sys.path.append('/home/pqbas/catkin_ws/src/blueberry-detection-ros/src/object_detection_models/yolov5')

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, PassImage
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh) #scale_segments, 
from yolov5.utils.segment.general import process_mask,  masks2segments #scale_masks,
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from yolov5.utils.dataloaders import letterbox


class Yolo5:
    def __init__(self, weights, data, device):
        # parameters of the model (definition, weights, device)
        self.weights = weights
        self.data = data
        self.device = select_device(device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=self.data, fp16=False)
        self.bs = 1
        _ , self.extension = os.path.splitext(self.weights)
        self.stride = self.model.stride
        self.pt = self.model.pt
        # configuration of inferences
        self.max_det = 1000
        self.augment = False
        self.visualize = False
        self.classes = None
        self.agnostic_nms = False
        self.imgsz = [640, 640]
        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))  # warmup


    def predict(self, image, conf_thres=0.1, enable_tracking=False):
        prediction = self.predict_(image, verbose=False)
        det = prediction
        if det is not None:
            return det
        return None
    
    def plot_prediction(self, img0, results):
        if results is None: return None
        for xyxy in results:        
            x1, y1, x2, y2 = torch.tensor(xyxy).view(1,4).view(-1).tolist()
            cv2.circle(img0, 
                    center = (int(x1+x2)//2,int(y1+y2)//2), 
                    radius = int(np.sqrt((x2-x1)**2 + (y2-y1)**2)/3), 
                    color= (255,0,0), 
                    thickness = 2)
        return img0

    @smart_inference_mode()
    def predict_(self, img_source, conf_thres=0.5, iou_thres=0.5, verbose=True):      

        img_letterbox, ratio, pad = letterbox(img_source, (640, 640), auto=False)
        dataset = PassImage(img_letterbox, img_size = self.imgsz, stride = self.stride, auto = self.pt)
        dt = (Profile(), Profile(), Profile())

        for idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):

            with dt[0]:
                im = torch.from_numpy(im).to(self.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                pred = self.model(im, augment=False, visualize=False)         
                if self.extension == 'engine':
                    proto = pred
                elif self.extension == 'pt':
                    proto = pred[1]
                
            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=1) #nm=32
                
            for i, det in enumerate(pred):  # per image
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                
                if len(det):
                    return scale_boxes(im.shape[2:], det[:, :4], img_source.shape).round()
        return None

if __name__ =='__main__':

    detector = Yolo5(weights='/home/pqbas/catkin_ws/src/blueberry-detection-ros/weights/best.pt', 
                    data='', 
                    device='cuda:0')
        
    img = cv2.imread('/home/pqbas/catkin_ws/src/blueberry-detection-ros/src/gallery/15dec23/img373.png')
    predictions = detector.predict(img)

    cv2.imshow('Prediction',detector.plot_prediction(img, predictions))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
