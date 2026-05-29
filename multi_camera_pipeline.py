"""Пайплайн для процессинга сразу 4 камер"""

from argparse import Namespace
import os
from IPython.core.magics import ExtensionMagics

import cv2
import torch
import numpy as np
from torchreid import models
from torchreid.utils import FeatureExtractor
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker

from config import LOG_FILE, LOG_LEVEL, CUDA, VIDEO_PATHS, PROCESS_FRAMES, TRACKER_PARAMS, \
    REID_THRESHOLD, PERSON_IMAGES_DIR, SAVE_TRACKED_IMAGES
from db import init_db, save_to_db_detected, save_to_db_person, save_to_db_reid
from log.logger_base import selector_logger
from yolov8 import make_yolov8s_preds
from utils import generate_colors


class Tracker:
    """Трекер объектов по 4 камерам."""

    def __init__(self):
        self.logger = selector_logger(LOG_FILE, LOG_LEVEL)
        self.cuda = CUDA

        self.detector = YOLO('yolov8s.pt')
        self.detector.to(self.cuda)
        self.detector.eval()

        self.reid_model = FeatureExtractor(
              model_name='osnet_ain_x1_0',
              device=self.cuda,
        )
        self.db = init_db()
        self.video_paths = VIDEO_PATHS

        self.local_global_track_map = dict()  # храним пары (camera_id, track_id) -> global_id
        self.global_id_features = None
        self.color_palette = dict()  # храним global_id -> color
        self.patience_map = dict()  # храним (camera_id, track_id) -> int


    def get_features_reid(self, image: np.ndarray) -> np.array:
        """
        Рассчитывает feature vector для человека из глобального трека

        :param image: Изображение человека.
        :return: Вектор извлеченных фичей.
        """
        image = cv2.resize(image, (128, 256))
        features = self.reid_model(image)
        return features.numpy().flatten()

    def get_global_id(self,
                      local_track_id: int,
                      camera_id: int,
                      image: np.ndarray,
                      threshold: float = REID_THRESHOLD) -> int:
        """
        Определяет является ли объект новым в трекинге, или он уже появлялся в зоне видимости камер.

        :param local_track_id: ID локального трека из bytetrack.
        :param camera_id: ID камеры откуда получено изображение.
        :param image: Кроп изображения человека.
        :param threshold: Порог при котором считаем, что это один и тот же человек.
        :return: Глобальный ID.
        """
        # если мы уже локально трекаем этот объект - то присваиваем ему тот же глобальный Id
        if (camera_id, local_track_id) in self.local_global_track_map:
            return self.local_global_track_map[(camera_id, local_track_id)]
        else:
            features = self.get_features_reid(image)
            features = features / np.linalg.norm(features)
            if self.global_id_features is None:
                #print('Инициализация хранилища global_id features')
                self.global_id_features = features
                global_id = 0
            else:
                # print(self.global_id_features.shape, features.shape)
                scores = self.global_id_features @ features
                closest_id = np.argmax(scores)
                closest_score = np.max(scores)
                #print(f'Скоры реидентификации: {scores}')
                if closest_score > threshold:
                    # считаем, что реидентифицировали человека
                    global_id = closest_id
                else:
                    # считаем, что появился ранее не наблюдаемый объект
                    self.global_id_features = np.vstack((self.global_id_features, features))
                    global_id = self.global_id_features.shape[0]
                    #save_to_db_person(self.db, global_id, image)
                self.local_global_track_map[(camera_id, local_track_id)] = global_id
                #save_to_db_reid(self.db, closest_id, image, closest_score)
            return global_id


    def track_pipeline(self, save_tracked_images: bool = SAVE_TRACKED_IMAGES) -> None:
        """
        Делает трекинг объектов по предоставленным в конфиге файлам с камер видеонаблюдения и проводит MCMOT.

        :param save_tracked_images: Сохранять или нет изображения трекаемых людей (для сбора разметки).
        :return: None.
        """

        # Инициализация видео и локальных трекеров
        video_caps = list()
        video_outs = list()
        video_trackers = list()
        for i, video_path in enumerate(self.video_paths):
            print(f'Подготовка для камеры {i}')
            cap = cv2.VideoCapture(video_path)
            out = cv2.VideoWriter(f'tracked_test_{i}.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            cap.get(cv2.CAP_PROP_FPS),
                            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            tracker = BYTETracker(args=Namespace(**TRACKER_PARAMS, mot20=False))

            video_caps.append(cap)
            video_outs.append(out)
            video_trackers.append(tracker)

        if save_tracked_images:
            os.makedirs(PERSON_IMAGES_DIR, exist_ok=True)
            for camera_id in range(len(video_caps)):
                os.makedirs(PERSON_IMAGES_DIR + f'/camera_{camera_id}', exist_ok=True)

        # Процессинг кадров со всех камер
        for i in range(PROCESS_FRAMES):
            print(f'Начат {i} кадр из {PROCESS_FRAMES}')
            for camera_id in range(len(video_caps)):
                cap = video_caps[camera_id]
                ret, frame = cap.read()
                if not ret:
                    break
                # Детекция
                processed_frame, detections = make_yolov8s_preds(self.detector, self.cuda, np.ascontiguousarray(frame))
                if detections is not None:
                    # Делаем локальный трекинг
                    tracker = video_trackers[camera_id]
                    online_targets = tracker.update(detections,
                                                    [frame.shape[0], frame.shape[1]],
                                                    [frame.shape[0], frame.shape[1]])
                    self.logger.info(f'На кадре {i} камеры {camera_id} трекается {len(online_targets)} объектов')
                    for local_track in online_targets:
                        x1, y1, x2, y2 = map(int, local_track.tlbr)
                        if save_tracked_images:
                            if (camera_id, local_track) in self.patience_map:
                                if self.patience_map[(camera_id, local_track)] == 0:
                                    local_path = PERSON_IMAGES_DIR + f'/camera_{camera_id}/{local_track.track_id}'
                                    os.makedirs(local_path, exist_ok=True)
                                    try:
                                      cv2.imwrite(f'{local_path}/{len(os.listdir(local_path))}.jpg', cv2.resize(frame[y1:y2, x1:x2], (256, 128)))
                                    except Exception as e:
                                      print(f'Ошибка при сохранении изображения {e}')
                                    self.patience_map[(camera_id, local_track)] = 15
                                self.patience_map[(camera_id, local_track)] -= 1
                            else:
                              self.patience_map[(camera_id, local_track)] = 15
                        # присваиваем глобальный ID
                        global_id = self.get_global_id(local_track.track_id, camera_id, frame[y1:y2, x1:x2])
                        if global_id not in self.color_palette:
                            self.color_palette[global_id] = generate_colors(1)[0]
                        # делаем отметку на видео
                        cv2.rectangle(frame,
                                      (int(local_track.tlbr[0]), int(local_track.tlbr[1])),
                                      (int(local_track.tlbr[2]), int(local_track.tlbr[3])),
                                      self.color_palette[global_id], 2)
                        cv2.putText(frame,
                                    f"local_id: {local_track.track_id}, global_id: {global_id}",
                                    (int(local_track.tlbr[0]), int(local_track.tlbr[1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    self.color_palette[global_id], 2)
                        # делаем запись в БД
                        save_to_db_detected(self.db,
                                   local_track_id=local_track.track_id,
                                   global_track_id=global_id,
                                   frame_num=i,
                                   bbox=(local_track.tlbr[0], local_track.tlbr[1], local_track.tlbr[2], local_track.tlbr[3]),
                                   score=local_track.score,
                                   camera_id=camera_id)
                out = video_outs[camera_id]
                out.write(frame)

        self.db.close()
        for cap in video_caps:
            cap.release()
        for out in video_outs:
            out.release()
