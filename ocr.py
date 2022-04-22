from glob import glob
import json
import cv2
import numpy as np
from tqdm import tqdm
from ocr import predict_cls, predict_det
from ocr import utility


args = utility.parse_args()
args.det_model_dir = "/Users/chenyifan/data/weights/paddleocr/ch_PP-OCRv2_det_slim_quant_infer"
args.use_gpu = False
args.det_limit_side_len = 720
print(args)
model = predict_det.TextDetector(args)
img = cv2.imread("IMG_1102.jpg")
img = cv2.resize(img, None, fx=0.5, fy=0.5)

for _ in range(10):
    dt_boxs, elapse = model(img)
    print(dt_boxs.shape[0])