"""Загрузка данных для аналитики из базы данных"""

import sqlite3
import pandas as pd
from datetime import timedelta

from config import DATABASE_PATH, START_TIME, FRAME_RATE


def filter_static(df):
    """Фильтрует статичных манекенов"""
    good_tracks = list()
    for local_track_id in df['local_track_id'].unique():
        x_min = df[df['local_track_id'] == local_track_id]['x2'].min()
        x_max = df[df['local_track_id'] == local_track_id]['x2'].max()
        y_min = df[df['local_track_id'] == local_track_id]['y2'].min()
        y_max = df[df['local_track_id'] == local_track_id]['y2'].max()
        if abs(x_min - x_max) > 150 and abs(y_min - y_max) > 150:
            good_tracks.append(local_track_id)
    return df[df['local_track_id'].isin(good_tracks)]


def load_main_dataset(db_path: str = DATABASE_PATH) -> pd.DataFrame:
    """
    Загружает данные по трекингу объектов

    :param db_path: Путь до файла с базой данных.
    :return: Датафрейм с данными
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM tracked_objects"
    df = pd.read_sql_query(query, conn)
    df['timestamp'] = df['frame_num'].apply(lambda x: START_TIME + timedelta(milliseconds=1000 * x / FRAME_RATE))
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    conn.close()
    return filter_static(df)
