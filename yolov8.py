"""Модель yolov8s"""

import torch
import numpy as np
import cv2
from ultralytics.utils.plotting import Annotator

def make_yolov8s_preds(model, device, frame=None, debug=False, img_path: str = 'test3.jpg'):
    if debug:
        image = cv2.imread(img_path)
    else:
        image = frame.copy()
    results = model(image, verbose=False, classes=[0], conf=0.2)
    results = results
    annotator = Annotator(image, line_width=2)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            conf = box.conf.item()
            if (b[3] - b[1]) < 0.05 * image.shape[0]:
                continue

            label = f"person {conf:.2f}"
            annotator.box_label(b, label, color=(0, 255, 0))
            if debug:
                cv2.imwrite("test_result.jpg", annotator.result())
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf.item()
        detections.append([x1, y1, x2, y2, conf])
    detections = np.array(detections) if detections else np.empty((0, 5))
    # print(f'Результаты детекции кадра: {detections}')
    return annotator.result(), detections