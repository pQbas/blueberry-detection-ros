import numpy as np
import sys
import cv2

sys.path.append('/home/pqbas/miniconda3/envs/dl/lib/python3.8/site-packages')
import torch


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def img_preprocess(img, device, half, net_size):
    net_image, ratio, pad = letterbox(img[:, :, :3], net_size, auto=False)
    net_image = net_image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    net_image = np.ascontiguousarray(net_image)

    img = torch.from_numpy(net_image).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img, ratio, pad


def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2]) * im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) * im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) * im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) * im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_min
    output[2][1] = y_max

    output[3][0] = x_max
    output[3][1] = y_max
    return output


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


# ..........................................

import rospy
from std_msgs.msg import String
from sensor_msgs.msg import Imu
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from vision_msgs.msg import Detection2D, BoundingBox2D, ObjectHypothesisWithPose, Detection2DArray
from geometry_msgs.msg import Pose2D, PoseWithCovariance, Pose
import rospy
import pyzed.sl as sl
import cv2
import numpy as np
import torch
from pathlib import Path
from object_detection_models.yolo5 import Yolo5
from common_functions import clip_coords, scale_coords, img_preprocess, xywh2abcd, letterbox, xyxy2xywh
from boxmot import DeepOCSORT
from callbacks import CompresedImageCallBack, ImageCallBack


def write_text(img, text, position):
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = position
    fontScale = 2
    color = (255, 0, 255)
    thickness = 3
    return cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    

def publish_bounding_boxes():
    object_detected_array = Detection2DArray()
    pose = Pose()
    PoseWithCovarance = PoseWithCovariance()
    PoseWithCovarance.pose = pose
    PoseWithCovarance.covariance = np.linspace(0,15,36).tolist()
    hypothesis = ObjectHypothesisWithPose()
    hypothesis.id = object.id
    hypothesis.score = object.confidence
    hypothesis.pose = PoseWithCovarance
    bbox = BoundingBox2D()
    bbox.size_x, bbox.size_y = abs(x1-x2), abs(y1-y2)
    center = Pose2D()
    center.x, center.y = xc, yc
    bbox.center = center
    object_detected.results = [hypothesis]
    object_detected.bbox = bbox
    object_detected_array.detections.append(object_detected)
    pub.publish(object_detected_array)
    rate.sleep()
    return

    # if prediction is None:
    #     return
    
    #publishing the detected objects
    # object_detected_array = Detection2DArray()
    
    # for *xyxy, id, conf, cls in tracker_outputs:
    #     x1, y1, x2, y2 = torch.tensor(xyxy).view(1,4).view(-1).tolist()
    #     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    #     pose = Pose()
    #     PoseWithCovarance = PoseWithCovariance()
    #     PoseWithCovarance.pose = pose
    #     PoseWithCovarance.covariance = np.linspace(0,15,36).tolist() # this doesn't have any meaning

    #     hypothesis = ObjectHypothesisWithPose()
    #     hypothesis.id = id
    #     hypothesis.score = conf
    #     hypothesis.pose = PoseWithCovarance

    #     bbox = BoundingBox2D()
    #     bbox.size_x, bbox.size_y = int(abs(x1-x2)), int(abs(y1-y2))
    #     center = Pose2D()
    #     center.x, center.y = int((x1 + x2)/2) , int((y1 + y2)/2)
    #     bbox.center = center

    #     object_detected.results = [hypothesis]
    #     object_detected.bbox = bbox        
    #     object_detected_array.detections.append(object_detected)        
    
    # pub.publish(object_detected_array)

    # OBJECT DETECTION ---------------

    # for id, object in enumerate(prediction):
    #     object_ = object.to('cpu')
    #     object_[:4] = object_[:4] + desplazamiento
    #     x1, y1, x2, y2 = object_[0], object_[1], object_[2], object_[3]
        
    #     pose = Pose()
    #     PoseWithCovarance = PoseWithCovariance()
    #     PoseWithCovarance.pose = pose
    #     PoseWithCovarance.covariance = np.linspace(0,15,36).tolist()

    #     hypothesis = ObjectHypothesisWithPose()
    #     hypothesis.id = id
    #     hypothesis.score = object_[4]
    #     hypothesis.pose = PoseWithCovarance

    #     bbox = BoundingBox2D()
    #     bbox.size_x, bbox.size_y = int(abs(x1-x2)), int(abs(y1-y2))
    #     center = Pose2D()
    #     center.x, center.y = int((x1 + x2)/2) , int((y1 + y2)/2)
    #     bbox.center = center
        
    #     object_detected.results = [hypothesis]
    #     object_detected.bbox = bbox
    #     object_detected_array.detections.append(object_detected)      

    # pub.publish(object_detected_array)
