"""Рассчет person REid метрик"""

import torchreid
from torchreid import models
import torch
import numpy as np

from torchreid.utils import FeatureExtractor
from torchreid.metrics.rank import evaluate_rank

from config import REID_DATA_PATH, REID_MODELS_LINKS, DEVICE
from reid_dataset import ReIDDataset


def calc_metrics(mode: str = 'all') -> None:
     """
     Выполняет рассчет метрик для разных моделей ReID

     :param mode: all или model_path. Если передан model_path - проводит валидацию метрик по весам переданной модели.
     :return: None
     """
     reid_dataset = ReIDDataset
     try:
         torchreid.data.register_image_dataset('dataset1', reid_dataset)
     except Exception as e:
      print(e)
     datamanager = torchreid.data.ImageDataManager(
          root='dataset1',
          sources=['dataset1'],
          height=256,
          width=128,
          batch_size_train=128,
          batch_size_test=100,
          transforms=['random_flip', 'random_crop']
     )
     dataset = ReIDDataset()
     if mode == 'all':
          model_list = REID_MODELS_LINKS
     else:
          model_list = [mode]
     for model in model_list:
          print(f'Старт рассчета для модели: {model}')
          if mode != 'all':
              extractor = FeatureExtractor(
                  model_name='osnet_x1_0',
                  model_path=mode,
                  device=DEVICE,
              )
          else:
              extractor = FeatureExtractor(
                  model_name=model,
                  device=DEVICE
              )
          query_features = extractor([x[0] for x in dataset.query])
          gallery_features = extractor([x[0] for x in dataset.gallery])

          distmat = torchreid.metrics.compute_distance_matrix(
               query_features, gallery_features, metric="euclidean"
          )
          print([x[1] for x in dataset.query])
          print([x[1] for x in dataset.gallery])
          cmc, mAP = evaluate_rank(
               distmat,
               [x[1] for x in dataset.query],
               [x[1] for x in dataset.gallery],
               [x[2] for x in dataset.query],
               [x[2] for x in dataset.gallery],
               use_metric_cuhk03=False,
          )
          print(f"Model: {mode}, Rank-1: {cmc[0]:.2%} | mAP: {mAP:.2%}")

#calc_metrics()