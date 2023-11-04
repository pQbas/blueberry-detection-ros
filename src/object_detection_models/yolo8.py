import sys
sys.path.append('/home/pqbas/miniconda3/envs/dl/lib/python3.8/site-packages')
sys.path.append('/home/pqbas/catkin_ws/src/blueberry/src/detection/object_detection_models/yolov5')

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
import numpy as np
import cv2

class Yolo8:
    def __init__(self, weights, device):
        self.weights = weights
        self.device = 'cuda:0'
        self.model = YOLO(self.weights)
        self.letterbox = LetterBox()

    def predict(self, img, conf_thres=0.5, enable_tracking = False):
        #img = self.letterbox(labels=None, image=img)
        if enable_tracking != False:
            results = self.model.track(img, conf=conf_thres, persist=True)
            #results = self.model.track(img, conf=conf_thres, persist=True, tracker="bytetrack.yaml")
            return results
        results = self.model(img, conf=conf_thres)
        return results
    

    def plot_prediction(self, img, results):
        
        if results[0].boxes.data.shape[0] < 1:
            return img
        
        img = results[0].plot(conf=False, masks=False, labels=True, font_size = 0.1)
        return img

        # print(results[0].boxes.xywh[0].shape[0])
        for i in range(results[0].boxes.xywh[0].shape[0]):
            x = results[0].boxes.xywh[i][0].item()
            y = results[0].boxes.xywh[i][1].item()
            w = results[0].boxes.xywh[i][2].item()
            h = results[0].boxes.xywh[i][3].item()
            cv2.circle( img, 
                        center = (int(x),int(y)), 
                        radius = int(np.sqrt((w)**2 + (h)**2)/3), 
                        color= (255,0,0), 
                        thickness = 2)
        return img


if __name__ == '__main__':
    
    weights_path = "/home/pqbas/catkin_ws/src/blueberry/src/detection/weights/yolov8m_1kepochs/weights/best.pt"
    weights_path = "yolov8n-seg.pt"

    detector = Yolo8(weights= weights_path,
                    device='cuda:0')

    cap = cv2.VideoCapture(0)
    key = ''
    fotogram = 0
    print("  Save the current image:     s")
    print("  Quit the video reading:     q\n")

    while key != 113:
        ### Image cropping ---------------------------------------------------
        ret, img0 = cap.read()
        results0 = detector.predict(img0, enable_tracking=True)
        img0 = detector.plot_prediction(img=img0, results=results0)
        cv2.imshow('awd',img0)
        cv2.waitKey(1)




