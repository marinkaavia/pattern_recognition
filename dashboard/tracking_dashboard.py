"""Построение аналитики собранной с трекинга"""

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from proector import draw_heatmap
from dataloader import load_main_dataset

st.set_page_config(page_title="Аналитика трекинга", layout="wide")
st.title("📹 Аналитика по трекингу людей с камер")

df = load_main_dataset()

min_time, max_time = df['timestamp'].min(), df['timestamp'].max()
start_time, end_time = st.slider("Выберите диапазон времени",
                                 min_value=min_time.to_pydatetime(),
                                 max_value=max_time.to_pydatetime(),
                                 value=(min_time.to_pydatetime(), max_time.to_pydatetime()),
                                 step=pd.Timedelta(minutes=15),
                                 key="time_slider")

df = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]


freq = st.selectbox("Частота агрегации по времени", ['1Min', '5Min', '15Min'], index=1)


st.subheader("👥 Количество уникальных людей по времени (все камеры)")

grouped_all = df.groupby([pd.Grouper(key='timestamp', freq=freq), 'camera_id'])['local_track_id'].nunique().reset_index()
grouped_all.columns = ['timestamp', 'camera_id', 'unique_people']

fig_all = px.line(grouped_all, x='timestamp', y='unique_people', color='camera_id',
                  title='Активность по всем камерам', markers=True)

st.plotly_chart(fig_all, use_container_width=True)


st.subheader("📍 Отображение heatmap активности")

df_filtered = df[(df['timestamp'] >= start_time) & (df['timestamp'] <= end_time)]
heatmap_buf = draw_heatmap(df[df_filtered['camera_id'] == 3], plan_img_path='plan.jpg')
st.image(heatmap_buf, caption="Heatmap активности", use_container_width=True)


st.subheader("⏱ Среднее время, проведённое покупателями")

df_filtered['time_window'] = df_filtered['timestamp'].dt.floor(freq)
track_windows = df_filtered.groupby(['time_window', 'camera_id', 'local_track_id']).agg(
    start_time=('timestamp', 'min'),
    end_time=('timestamp', 'max')
).reset_index()
track_windows['duration'] = (track_windows['end_time'] - track_windows['start_time']).dt.total_seconds()
avg_durations = track_windows.groupby(['time_window', 'camera_id'])['duration'].mean().reset_index()
avg_durations.columns = ['timestamp', 'camera_id', 'avg_duration']
fig_dur = px.line(avg_durations, x='timestamp', y='avg_duration', color='camera_id',
                  title='Среднее время пребывания по окнам времени',
                  labels={'avg_duration': 'Среднее время'}, markers=True)

st.plotly_chart(fig_dur, use_container_width=True)

st.subheader("🚪 Отношение количества людей: внутри / снаружи")
df_filtered['time_window'] = df_filtered['timestamp'].dt.floor(freq)
inside = df_filtered[df_filtered['camera_id'].isin([0, 3])]
outside = df_filtered[df_filtered['camera_id'].isin([1, 2])]
inside_count = inside.groupby('time_window')['local_track_id'].nunique().reset_index(name='inside_count')
outside_count = outside.groupby('time_window')['local_track_id'].nunique().reset_index(name='outside_count')
ratio_df = pd.merge(inside_count, outside_count, on='time_window', how='outer').fillna(0)
ratio_df['ratio'] = ratio_df['inside_count'] / ratio_df['outside_count'].replace(0, np.nan)
ratio_df['ratio'] = ratio_df['ratio'].replace([np.inf, -np.inf], np.nan).fillna(0)
fig_ratio = px.line(ratio_df, x='time_window', y='ratio',
                    title='Отношение людей внутри / снаружи по времени',
                    labels={'time_window': 'Время', 'ratio': 'Отношение (внутри/снаружи)'},
                    markers=True)

st.plotly_chart(fig_ratio, use_container_width=True)

st.subheader("🔍 Анализ по конкретной камере")

selected_cam = st.selectbox("Выберите камеру", sorted(df['camera_id'].unique()))
df_cam = df[df['camera_id'] == selected_cam]

grouped_cam = df_cam.groupby(pd.Grouper(key='timestamp', freq=freq))['local_track_id'].nunique().reset_index()
grouped_cam.columns = ['timestamp', 'unique_people']

fig_cam = px.area(grouped_cam, x='timestamp', y='unique_people',
                  title=f'Активность по времени — Камера {selected_cam}',
                  labels={'unique_people': 'Кол-во людей'})

st.plotly_chart(fig_cam, use_container_width=True)
