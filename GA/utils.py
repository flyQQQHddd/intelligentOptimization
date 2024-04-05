import os
import time
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] =False


def draw_poi_figure(points: pd.DataFrame, filename: str) -> None:

    # 根据 path 列进行排序
    path = np.zeros(len(points), np.int32)
    for i in range(len(points)):
        path[path[i]] = i
    points['path'] = path
    points = points.sort_values(by='path', ascending=True)


    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(True)

    # 绘制闭合路径
    ax.fill(
        points['x'],
        points['y'],
        edgecolor='black',
        linestyle='--',
        linewidth=1, 
        fill = False
    )

    # 绘制关键点
    ax.scatter(
        points['x'],
        points['y'],
        c='red'
    )

    ax.set_xlabel('longitude', fontsize = 10)
    ax.set_ylabel('latitude', fontsize = 10)
    ax.set_title('求解路径示意图', fontsize = 14)

    ax.spines[:].set_linewidth(1.5)
    ax.tick_params(direction='in')

    fig.tight_layout()
    fig.savefig(filename, dpi=500)


def draw_training_figure(fitness_record, filename: str) -> None:

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.grid(True)

    ax.plot(
        fitness_record,
        color='blue',
        linewidth=3,
        label='种群平均适应度'
    )

    ax.set_xlabel('迭代次数', fontsize = 10)
    ax.set_ylabel('种群平均适应度', fontsize = 10)
    ax.set_title('训练中种群情况变化曲线', fontsize = 14)

    ax.spines[:].set_linewidth(1.5)
    ax.tick_params(direction='in')
    ax.legend()

    fig.tight_layout()
    fig.savefig(filename, dpi=500)
