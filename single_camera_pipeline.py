"""Процессинг данных с одной камеры"""
from argparse import Namespace

from collections import defaultdict
import numpy as np
import torch
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from yolox.tracker.byte_tracker import BYTETracker
from config import cuda, YOLO_MODEL_PATH, TEST_VIDEO_PATH, YOLO_MODEL_TYPE, TRACKER_PARAMS, IMAGENET_MEAN, IMAGENET_STD
from db import init_db, save_to_db
from log.logger_base import selector_logger
from config import LOG_FILE, LOG_LEVEL
from utils import generate_colors
from yolov8 import make_yolov8s_preds

logger = selector_logger(LOG_FILE, LOG_LEVEL)


def process_one_video(video_path: str = TEST_VIDEO_PATH, camera_id: int = 0):
    """Тестовая функция для прогона и получения bounding box для объектов на видео"""
    model = YOLO('yolov8s.pt')

    logger.info('Инициализация базы данных')
    db_conn = init_db()

    logger.info('Инициализация трекера ByteTrack')
    tracker = BYTETracker(args=Namespace(**TRACKER_PARAMS, mot20=False))

    logger.info('Загрузка видео')
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('tracked_test.mp4', fourcc, fps, (frame_width, frame_height))

    frame_cnt = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f'Всего кадров: {total_frames}')
    color_palette = {}
    track_history = defaultdict(list)

    while True:
        frame_cnt += 1
        print(f'Завершено {frame_cnt} из {total_frames}')
        logger.info(f'Обработка кадра номер {frame_cnt}')
        ret, frame = cap.read()
        if not ret:
            break

        logger.info(f'Старт детекции')
        processed_frame, detections = make_yolov8s_preds(model, np.ascontiguousarray(frame))

        logger.info(f'Старт трекинга')
        if detections is not None:
            online_targets = tracker.update(detections, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]])
            for t in online_targets:
                track_id = int(t.track_id)
                if track_id not in color_palette:
                    color_palette[track_id] = generate_colors(1)[0]
                cv2.rectangle(frame, (int(t.tlbr[0]), int(t.tlbr[1])), (int(t.tlbr[2]), int(t.tlbr[3])), color_palette[track_id], 2)
                cv2.putText(frame, f"ID: {t.track_id}", (int(t.tlbr[0]), int(t.tlbr[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_palette[track_id], 2)
                for i in range(1, len(track_history[track_id])):
                    cv2.line(frame, track_history[track_id][i-1], track_history[track_id][i], color_palette[track_id], 2)
                #save_to_db(db_conn,
                #           track_id=t.track_id,
                #           frame_num=frame_cnt,
                #           bbox=(t.tlbr[0], t.tlbr[1], t.tlbr[2], t.tlbr[3]),
                #           score=t.score,
                #           class_id=0,
                #           camera_id=camera_id)
        logger.info('Окончание трекинга')
        out.write(frame)

    db_conn.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
