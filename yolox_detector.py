"""Детекция с помощью yolox"""

import numpy as np
import torch
import cv2

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import postprocess, vis

from config import YOLO_MODEL_TYPE,YOLO_MODEL_PATH, IMAGENET_MEAN, IMAGENET_STD


def get_yolox_detector():
    """Загружает yolox модель"""

    exp = get_exp(None, YOLO_MODEL_TYPE)
    exp.class_names = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
        "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
        "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
        "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
        "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv",
        "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    model = exp.get_model()
    ckpt = torch.load(YOLO_MODEL_PATH, map_location="cpu")
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def make_yolox_preds(exp, model, img_path: str = 'test3.jpg'):
    image = cv2.imread(img_path)
    orig_h, orig_w = image.shape[:2]

    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img, ratio = preproc(rgb_img, (640, 640), IMAGENET_MEAN, IMAGENET_STD)

    with torch.no_grad():
        img = torch.from_numpy(img).float()
        if img.ndim == 3:
            img = img.unsqueeze(0)
        outputs = model(img)
        outputs = postprocess(
            outputs,
            num_classes=exp.num_classes,
            conf_thre=0.01,
            nms_thre=0.01
        )
    if outputs[0] is not None:
        bboxes = outputs[0][:, :4]
        bboxes /= ratio
        bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, orig_w)
        bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, orig_h)
        scores = outputs[0][:, 4] * outputs[0][:, 5]
        cls_ids = outputs[0][:, 6]

        result_img = vis(image, bboxes, scores, cls_ids, conf=0.01, class_names=exp.class_names)

        cv2.imwrite("test_result.jpg", result_img)
        print(f"Обнаружено {len(bboxes)} объектов")
        for i, (bbox, score, cls_id) in enumerate(zip(bboxes, scores, cls_ids)):
            print(f"{i + 1}. {exp.class_names[int(cls_id)]} ({score:.2f}): {bbox}")
        return zip(bboxes, scores, cls_ids)
    else:
        print("Объекты не обнаружены!")
        return None