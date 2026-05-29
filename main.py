"""Точка входа для проекта"""

from multi_camera_pipeline import Tracker


if __name__ == "__main__":
    tracker = Tracker()
    tracker.track_pipeline()

