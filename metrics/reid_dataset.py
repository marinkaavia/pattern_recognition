"""Датасет для ReID"""

import os


from sklearn.model_selection import train_test_split
from torchreid.data.datasets import ImageDataset

from config import REID_DATA_PATH

class ReIDDataset(ImageDataset):
    """Реализует рассчет метрик для разных моделей детекции"""

    def __init__(self, root='data', **kwargs):
        self.data_path = REID_DATA_PATH
        self.person_list = os.listdir(self.data_path)

        print('Старт создания датасета')
        train = list()
        query = list()
        gallery = list()

        train_links, test_links = train_test_split(self.person_list, test_size=0.3, random_state=11)

        # создаем трейн
        for i, link in enumerate(train_links):
            for j, image_path in enumerate(os.listdir(self.data_path + f'/{link}')):
                train.append((self.data_path + f'/{link}/{image_path}', i, 0))

        # создаем валидацию
        for i, link in enumerate(test_links):
            query_links, gallery_links = train_test_split(os.listdir(f'{REID_DATA_PATH}/{link}'), test_size=0.66,
                                                          random_state=11)
            # добавляем в запросы
            for j, query_link in enumerate(query_links):
                query.append((self.data_path + f'/{link}/{query_link}', i, 0))
            # добавляем в галерею
            for j, gallery_link in enumerate(gallery_links):
                gallery.append((self.data_path + f'/{link}/{gallery_link}', i, 1))

        self._check_before_run(query, gallery)

        super(ReIDDataset, self).__init__(train, query, gallery, **kwargs)

        self.train = train
        self.query = query
        self.gallery = gallery


    @staticmethod
    def _check_before_run(query, gallery):
        query_pids = {pid for _, pid, _ in query}
        gallery_pids = {pid for _, pid, _ in gallery}

        if not query_pids.issubset(gallery_pids):
            raise RuntimeError("Некоторые ID query отсутствуют в gallery!")