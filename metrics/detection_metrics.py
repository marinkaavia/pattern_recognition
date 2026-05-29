"""Вспомогательный скрипт для рассчета метрик детекции"""

from ultralytics import YOLO

from config import DETECTION_MODEL_LINKS

def score_models():
    """"""
    for model in DETECTION_MODEL_LINKS:
        model = YOLO(model)
        results = model.val(
            data='data.yaml',
            split='test',
            iou=0.5,
            conf=0.5,
            save_json=True
        )

score_models()
