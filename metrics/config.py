"""Конфиг для рассчета метрик"""

import torch

REID_DATA_PATH = 'drive/MyDrive/reid_markup/clean'
REID_MODELS_LINKS = ['osnet_x1_0', 'osnet_ain_x1_0', 'osnet_ibn_x1_0', 'resnet50', 'shufflenet', 'mobilenetv2_x1_4']
DETECTION_MODEL_LINKS = ['yolo11s.pt', 'yolo11x.pt', 'yolov10s.pt', 'yolov9s.pt', 'yolov8s.pt', 'yolov5s.pt']
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
