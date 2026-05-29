"""Работа с базой данных"""

import cv2
import numpy as np
import sqlite3
from datetime import datetime

from config import DATABASE_PATH


def init_db(db_path: str = DATABASE_PATH):
    """
    Инициализация базы данных

    :param db_path: Ссылка на файл с базой данных.
    :return: Подключение к бд.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS tracked_objects
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      local_track_id INTEGER,
                      global_track_id INTEGER,
                      frame_num INTEGER,
                      timestamp DATETIME,
                      x1 REAL,
                      y1 REAL,
                      x2 REAL,
                      y2 REAL,
                      score REAL,
                      camera_id INTEGER)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS global_people
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                          global_track_id INTEGER,
                          person_crop BLOB)''')

    cursor.execute('''CREATE TABLE IF NOT EXISTS reid_logs
                             (id INTEGER PRIMARY KEY AUTOINCREMENT,
                              global_track_id INTEGER,
                              sim_score REAL,
                              candidate BLOB)''')

    conn.commit()
    return conn


def save_to_db_person(conn, global_id: int, image_crop: np.ndarray) -> None:
    """
    Сохраняет информацию о новом человеке в БД.

    :param conn: Подключение к БД.
    :param global_id: Глобальный ID.
    :param image_crop: Картинка человека
    :return: None
    """
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO global_people (global_track_id, person_crop)
        VALUES (?, ?)
    ''', (global_id, cv2.imencode('.jpg', image_crop)[1].tobytes()))
    conn.commit()
    return

def save_to_db_reid(conn, global_id: int, image_crop: np.ndarray, sim_score: float) -> None:
    """
    Сохраняет информацию о сравнение в реидентификации.

    :param conn: Подключение к БД.
    :param global_id: Глобальный ID.
    :param image_crop: Картинка человека.
    :param sim_score: Скор близости.
    :return: None
    """
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO reid_logs (global_track_id, sim_score, candidate)
        VALUES (?, ?, ?)
    ''', (global_id, sim_score, cv2.imencode('.jpg', image_crop)[1].tobytes()))
    conn.commit()
    return


def save_to_db_detected(conn,
                        local_track_id: int,
                        global_track_id: int,
                        frame_num: int,
                        bbox: tuple,
                        score: float,
                        camera_id: int) -> None:
    """
    Сохранение данных о детектированном объекте в БД

    :param conn: Подключение к бд.
    :param local_track_id: Локальный ID из bytetrack.
    :param global_track_id: Глобальный ID.
    :param frame_num: Номер кадра.
    :param bbox: Границы объекта.
    :param score: Скор вероятности.
    :param camera_id: ID камеры.
    :return: None
    """
    cursor = conn.cursor()
    x1, y1, x2, y2 = bbox
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute('''INSERT INTO tracked_objects 
                     (local_track_id, global_track_id, frame_num, timestamp, x1, y1, x2, y2, score, camera_id)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                   (local_track_id, global_track_id, frame_num, timestamp, x1, y1, x2, y2, score, camera_id))
    conn.commit()
    return
