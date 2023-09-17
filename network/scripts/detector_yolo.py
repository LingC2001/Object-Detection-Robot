import os 
import time
import sys
sys.path.append('Week06-07/yolov7_fruits/yolov7')
sys.path.append('yolov7_fruits/yolov7')

import numpy as np
import torch
import cv2
import random 

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class Detector:
    def __init__(self, use_gpu=False):
        self.opt = {
            "weights_sim": "yolov7_fruits/yolov7/best_sim.pt",
            "weights_real": "yolov7_fruits/yolov7/best_real.pt",
            # "weights": "yolov7_fruits/yolov7/best-3.pt",
            # "yaml": "yolov7/YOLOv7_fruits-1/data.yaml",
            "img-size": 640,
            "conf-thres": 0.55,
            "iou-thres": 0.45,
            "device" : '0',
            "classes" : None
        }
        if use_gpu == False:
            self.opt["device"] = 'cpu'
        self.model = None
        self.half = False
        self.device = 0
        self.stride = 0
        self.imgsz = None
        self.names = None
        self.colors = None

    def load_weights(self, sim):
        with torch.no_grad():
            if sim:
                weights, imgsz = self.opt['weights_sim'], self.opt['img-size']
                print("Loading simulation model weights!")
                # print(weights)
            else:
                weights, imgsz = self.opt['weights_real'], self.opt['img-size']
                print("Loading real world model weights!")
            set_logging()
            self.device = select_device(self.opt['device'])
            self.half = self.device.type != 'cpu'
            model = attempt_load(weights, map_location=self.device)  # load FP32 model
            self.stride = int(model.stride.max())  # model stride
            self.imgsz = check_img_size(imgsz, s=self.stride)  # check img_size
            if self.half:
                model.half()

            self.names = model.module.names if hasattr(model, 'module') else model.names
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
            if self.device.type != 'cpu':
                model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(model.parameters())))
            self.model = model

    def detect_single_image(self, img0):
        with torch.no_grad():
            # img0 = cv2.imread(source_image_path)
            img = self.letterbox(img0, self.imgsz, stride=self.stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = self.model(img, augment= False)[0]
            # print(pred)

            # Apply NMS
            classes = None
            if self.opt['classes']:
                classes = []
                for class_name in self.opt['classes']:
                    classes.append(self.opt['classes'].index(class_name))


            pred = non_max_suppression(pred, self.opt['conf-thres'], self.opt['iou-thres'], classes= classes, agnostic= False)
            # print(pred) # Print all output bounding boxes above threshoold
            t2 = time_synchronized()
            for i, det in enumerate(pred):
                s = ''
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    
                    for *xyxy, conf, cls in reversed(det):

                        # Drawing the bound box labels
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, img0, label=label, color=self.colors[int(cls)], line_thickness=1)
        return pred, img0

    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better test mAP)
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
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return img, ratio, (dw, dh)

