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
sys.path.append('/home/pqbas/catkin_ws/src/blueberry-detection-ros/src/detection/object_detection_models')
sys.path.append('/home/labinm-jetson/catkin_ws/src/blueberry-detection-ros/src/object_detection_models')

#from yolov5.utils.general import scale_coords, xyxy2xywh
from common_functions_ import scale_coords


sys.path.append('/home/pqbas/catkin_ws/src/blueberry-detection-ros/src/detection/object_detection_models/yolov5')
sys.path.append('/home/labinm-jetson/catkin_ws/src/blueberry-detection-ros/src/object_detection_models')


from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from yolov5.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams, PassImage
from yolov5.utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh) #scale_segments, 
from yolov5.utils.segment.general import process_mask,  masks2segments #scale_masks,
from yolov5.utils.torch_utils import select_device, smart_inference_mode
from yolov5.utils.dataloaders import letterbox

class my_object:
    def __init__(self, mask=None, bbox=None, area=None, id=None, cls=None, conf=None):
        self.mask = mask
        self.bbox = bbox
        self.area = area
        self.id = id
        self.cls = cls
        self.conf = conf


class Yolo5:
    def __init__(self, weights, data, device):
        # parameters of the model (definition, weights, device)
        self.weights = weights
        self.data = data
        self.device = select_device(device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=False, data=self.data, fp16=False)
        self.bs = 1

        _ , self.extension = os.path.splitext(self.weights)
        
        # properties of the model (stride, names, pt)
        self.stride = self.model.stride
        #self.names = self.model.namesclip_coords, scale_coords, img_preprocess,
        self.pt = self.model.pt

        # configuration of inferences
        self.max_det = 1000
        self.augment = False
        self.visualize = False
        self.classes = None
        self.agnostic_nms = False
        self.imgsz = [640, 640]
        self.model.warmup(imgsz=(1 if self.pt else self.bs, 3, *self.imgsz))  # warmup


    # def yolo7_prediction_to_objects(self, mask, det, conf, cls):
    #     mask = mask.cpu().numpy().astype(np.uint8)
    #     area = mask.sum()
    #     bbox = det[0:4].cpu().detach().numpy()
    #     conf = conf.cpu().detach().numpy()
    #     cls = cls.cpu().detach().numpy()
    #     return my_object(mask=mask, area=area, bbox=bbox, conf=conf, cls=cls)

    def predict(self, image, conf_thres, enable_tracking=False):
        h,w,c = image.shape
        image_on_letterbox, ratio, pad = letterbox(image, (640, 640), auto=False)
        prediction = self.predict_(image_on_letterbox, verbose=False, with_preprocessing=True)

        det = prediction

        if det is not None:
            n_arandanos = det.shape[0]
            for det in [det]:
                det = det[:,:6].clone()
                if len(det):
                    det[:, :4] = scale_coords((640,640), det[:, :4], (h,w)).round()

                    return det
        
        return None
    
    def plot_prediction(self, img0, results):
        
        det = results
        n_arandanos = 0
        h,w,c = img0.shape
        tracking_enabled = False

        #if desplazamiento is None:
        desplazamiento = torch.tensor([[0,0,0,0]])
        
        if det is not None:
            n_arandanos = det.shape[0]
            for det in [det]:
                det = det[:,:6].clone()
                if len(det):
                    # det[:, :4] = scale_coords((640,640), det[:, :4], (h,w)).round()
                    
                    # cantidad_objetos = det.shape[0]
                    # desplazamiento_repetido = desplazamiento.repeat(cantidad_objetos, 1).to('cuda:0')
                    # det[:, :4] = det[:, :4] + desplazamiento_repetido
                    
                    # if tracking_enabled == True:                    
                    #     for *xyxy, id, conf, cls in tracker_outputs:
                    #         x1, y1, x2, y2 = torch.tensor(xyxy).view(1,4).view(-1).tolist()
                    #         x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    #         cv2.circle(img0, 
                    #                 center = (int(x1+x2)//2,int(y1+y2)//2), 
                    #                 radius = int(np.sqrt((x2-x1)**2 + (y2-y1)**2)/3), 
                    #                 color= (255,0,0), 
                    #                 thickness = 2)
                            
                    #         # cv2.rectangle(img0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #         label = f'{cls}: {conf:.2f}: {id}'
                    #         cv2.putText(img0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    
                    
                    # else:
                    for *xyxy, conf,cls in det:
                        
                        if cls.item() == 0 and conf.item() >= 0.50:                           
                            x1, y1, x2, y2 = torch.tensor(xyxy).view(1,4).view(-1).tolist()
                            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
                            img0 = cv2.cvtColor(img0, cv2.COLOR_RGB2BGR)
                            #label = f'{cls}: {conf:.2f}'
                            cv2.circle(img0, center = (int(x1+x2)//2,int(y1+y2)//2), radius = int(np.sqrt((x2-x1)**2 + (y2-y1)**2)/3), color= (255,0,0), 
                                    thickness = 2)
                            #cv2.rectangle(img0, (int(x1),int(y1)), (int(x2),int(y2)), (255,0,0), 2)
                            #cv2.putText(img0, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return img0



    @smart_inference_mode()
    def robot_mobil_inference(self, img_source, conf_thres=0.5, iou_thres=0.5):
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        with dt[1]:
            pred = self.model(img_source, augment=False, visualize=False)
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=10) #nm=32
        for i, det in enumerate(pred):
            if len(det):
                predictions = det
            return predictions


    @smart_inference_mode()
    def predict_(self, img_source, conf_thres=0.5, iou_thres=0.5, verbose=True, with_preprocessing=True):
        
        # if with_preprocessing == False:
        #     seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

        #     # Inference
        #     with dt[1]:
        #        #print(img_source.shape)
        #        pred = self.model(img_source, augment=False, visualize=False)
        #        proto = pred[1]

        #     # NMS
        #     with dt[2]:
        #        pred = non_max_suppression(pred, conf_thres, iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det, nm=10) #nm=32
            
        #     for i, det in enumerate(pred):  # per image
        #         p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                
        #         if len(det):        
        #             if verbose == True:
        #                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
        #                conf = pred[0][:,4]
        #                cls = pred[0][:,5]
        #                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
        #                predictions = list(map(self.yolo7_prediction_to_objects, masks, det, conf, cls))
        #             else:
        #                predictions = det
                    
        #             return predictions
        #     return None
        
        
        if with_preprocessing == True:
            dataset = PassImage(img_source, img_size = self.imgsz, stride = self.stride, auto = self.pt)
            seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
            image = []

            for idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
                height_img,width_img,ch_img = im0s.shape
                line_pt_1 = (0, height_img // 2)
                line_pt_2 = (width_img, height_img // 2)

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
                        if verbose == True:
                           det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                           conf = pred[0][:,4]
                           cls = pred[0][:,5]
                           masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                           predictions = list(map(self.yolo7_prediction_to_objects, masks, det, conf, cls))
                        else:
                           predictions = det                        
                        return predictions
            return None

if __name__ =='__main__':

    detector = Yolo5(weights='./yolov5/yolov5s-seg.engine', 
                    data='./yolov5/data/custom.yaml', 
                    device='cuda:0')
    
    img = cv2.imread('./yolov5/data/images/zidane.jpg')
    predictions = detector.predict(img)

    for pred in predictions:
        x, y, w, h = pred.bbox
        cv2.rectangle(img, (int(x), int(y)), (int(w), int(h)), (0,255,0), 2)
        

    cv2.imshow('Prediction',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
