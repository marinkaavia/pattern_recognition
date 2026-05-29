"""Проецирует и отрисовывает точки на 2D карту"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cv2
import os
from io import BytesIO

src_points = {0: np.array([[1100, 260], [720, 490], [520, 200], [110, 160]], dtype=float),
              1: np.array([[320, 520], [880, 480], [510, 100], [630, 50]], dtype=float),
              2: np.array([[270, 667], [940, 320], [1115, 125], [1205, 170]], dtype=float),
              3: np.array([[800, 630], [140, 300], [630, 125], [1115, 200]], dtype=float),
              }

tgt_points = {0: np.array([[900, 80], [540, 410], [460, 370], [20, 650]], dtype=float),
              1: np.array([[460, 480], [450, 630], [1310, 475], [1550, 650]], dtype=float),
              2: np.array([[1000, 470], [1370, 490], [1575, 460], [1600, 660]], dtype=float),
              3: np.array([[470, 480], [380, 230], [750, 230], [1000, 470]], dtype=float),
              }


matrices = list()
for i in range(len(src_points)):
    H, _ = cv2.findHomography(src_points[i], tgt_points[i])
    matrices.append(H)


def project(x1: float, y1: float, x2: float, y2: float, camera_id: int) -> tuple:
    """
    Процецирует bbox объекта на карту магазина.

    :param x1: координаты
    :param y1: координаты
    :param x2: координаты
    :param y2: координаты
    :param camera_id: id камеры.
    :return: (x, y) - координаты объекта на карте.
    """
    x_cent = (x1 + x2) / 2
    y_bot = y2
    projected = cv2.perspectiveTransform(np.array([[[x_cent, y_bot]]], dtype=np.float32), matrices[camera_id])
    return projected[0, 0, 0], projected[0, 0, 1]


def visualize_on_plan(df, plan_img_path = 'plan.jpg'):
    """Отрисовывает точки на карте"""

    plan = cv2.imread(plan_img_path)
    plan = cv2.cvtColor(plan, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 8))
    plt.imshow(plan)

    for i in range(len(df)):
        x, y = project(df['x1'].iloc[i], df['y1'].iloc[i], df['x2'].iloc[i], df['y2'].iloc[i], df['camera_id'].iloc[i])
        plt.plot(x, y, 'ro', markersize=5)
    #plt.savefig("points.png", dpi=300)
    plt.show()


def draw_heatmap(df, plan_img_path = 'plan.jpg', bins=(25, 25)):
    """Отрисовывает хитмапу"""

    if not os.path.exists(plan_img_path):
        raise FileNotFoundError(f"Изображение не найдено по пути {plan_img_path}")

    plan = cv2.imread(plan_img_path)
    plan = cv2.cvtColor(plan, cv2.COLOR_BGR2RGB)
    height, width = plan.shape[:2]

    heatmap_points = []

    for i in range(len(df)):
        x, y = project(df['x1'].iloc[i], df['y1'].iloc[i], df['x2'].iloc[i], df['y2'].iloc[i], df['camera_id'].iloc[i])
        heatmap_points.append([x, y])

    heatmap_points = np.array(heatmap_points)
    heatmap, xedges, yedges = np.histogram2d(heatmap_points[:, 1], heatmap_points[:, 0], bins=bins, range=[[0, height], [0, width]])

    heatmap = np.clip(heatmap * 100, 0, None)
    heatmap = heatmap / np.max(heatmap) * 255.0
    heatmap = heatmap.astype(np.uint8)
    heatmap = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    overlay = cv2.addWeighted(plan, 0.4, heatmap_color, 0.6, 0)

    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    plt.axis('off')
    plt.tight_layout()

    #plt.savefig("points.png", dpi=300)
    #plt.show()

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=300, bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    return buf