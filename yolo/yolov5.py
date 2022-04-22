import os
import sys
sys.path.append(os.path.dirname(__file__))

import cv2
import torch
import time
import numpy as np

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors


class YOLOv5(object):

    def __init__(self,
                 weights='models/yolov5m.pt',  # model.pt path(s)
                 imgsz=(640, 640),  # inference size (height, width)
                 conf_thres=0.25,  # confidence threshold
                 iou_thres=0.45,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 device='cuda:0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 ):

        self.model = DetectMultiBackend(weights, device=device)
        self.model.model.float()
        self.model.warmup()
        
        self.imgsz = imgsz
        self.device = device
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
    
    def letterbox(self, im, color=(114, 114, 114)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]
        stride = self.model.stride

        # Scale ratio (new / old)
        r = min(self.imgsz[0] / shape[0], self.imgsz[1] / shape[1])

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = self.imgsz[1] - new_unpad[0], self.imgsz[0] - new_unpad[1]  # wh padding
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)
    
    
    def preprocess_image(self, img0):
        img = self.letterbox(img0)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255
        if len(img.shape) == 3:
            img = img[None]
        return img
    
    def detect(self, img0):
        t0 = time.time()
        img = self.preprocess_image(img0)
        t1 = time.time()
        pred = self.model(img, augment=False, visualize=False)
        t2 = time.time()
        det = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=self.max_det)[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        t3 = time.time()
        print("[Elapse] preprocess: %.4fsec | predict: %.4fsec | NMS: %.4fsec | total: %.4fsec" % (
            t1 - t0, t2 - t1, t3 - t2, t3 - t0
        ))
        
        bboxs = det[:, :4].cpu().numpy().astype(np.int)
        scores = det[:, 4].cpu().numpy().astype(np.float)
        cls = det[:, 5].cpu().numpy().astype(np.int)
        
        return bboxs, scores, cls

    def draw_result(self, img, bboxs, scores, cls):
        ndet = len(scores)
        annotator = Annotator(img, line_width=3, example=str(self.model.names))
        for i in range(ndet):
            c = self.model.names[cls[i]]
            label = '%s %.2f' % (c, scores[i])
            annotator.box_label(bboxs[i, :], label, color=colors(cls[i], True))
        return annotator.result()
