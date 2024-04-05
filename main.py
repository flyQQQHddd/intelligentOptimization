from multiprocessing import Pool
import pandas as pd
from GA.utils import *
from GA.TSP_GA import *


if __name__ == "__main__":

    points = pd.read_csv('./dataset/dataset30.csv', index_col=0)


    # --------------------------
    # 多进程进行多个种群进化
    # p = Pool()
    # results = []
    # for i in range(100):
    #     result = p.apply_async(run, points)
    #     results.append(result)
    # p.close()
    # p.join()

    # # 取进化效果最好的种群
    # results = [result.get() for result in results]
    # best = max(results, key=lambda x: x[1])
    # path, fitness, record = best[0], best[1], best[2]

    # --------------------------
    # 单进程版本
    begin_time = time.perf_counter()
    path, fitness, record = run(points)
    end_time = time.perf_counter()
    print(f'总时间：{round(end_time - begin_time, 5)}s')
    print(f'求解路径：{path}')
    print(f'求解距离：{1 / fitness}')


    # --------------------------
    draw_poi_figure(points, './output/path.png')
    draw_training_figure(record, './output/train.png')