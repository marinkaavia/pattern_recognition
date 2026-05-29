"""Файл с конфигом"""

import torch
import datetime

MODE = 'DEBUG'  # FINAL
CUDA = 'cuda' if torch.cuda.is_available() else 'cpu'
YOLO_MODEL_PATH = "yolox_x.pth"
YOLO_MODEL_TYPE = "yolox-x"
TEST_VIDEO_PATH = "data/test1.mp4"
LOG_FILE = 'MCMOT'
LOG_LEVEL = 10
FRAME_RATE = 15
TRACKER_PARAMS = {
        "track_thresh": 0.5,
        "track_buffer": 30,
        "match_thresh": 0.8,
        "frame_rate": FRAME_RATE
    }
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
#VIDEO_PATHS = ['pipeline_test0.mp4', 'pipeline_test1.mp4', 'pipeline_test2.mp4', 'pipeline_test3.mp4']
VIDEO_PATHS = ['pipeline_test01.mp4', 'pipeline_test11.mp4', 'pipeline_test21.mp4', 'pipeline_test31.mp4']
PROCESS_FRAMES = 275 if MODE == 'DEBUG' else 10000
REID_THRESHOLD = 0.75
DATABASE_PATH = 'tracking_large.db'
START_TIME = datetime.datetime(2024, 10, 19, 15, 00, 00)
# PERSON_IMAGES_DIR = 'person_images_test0'
PERSON_IMAGES_DIR = 'person_images_test2'
SAVE_TRACKED_IMAGES = True