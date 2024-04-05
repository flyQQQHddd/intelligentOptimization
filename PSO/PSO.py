import numpy as np
import time
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

class PSO:

    def __init__(self, D, N, M, p_low, p_up, v_low, v_high, fitness, w_range=(0.5, 1.0), c1 = 2., c2 = 2.):
        """粒子群算法
        :param D: 粒子维度
        :param N: 粒子群规模，初始化种群个数
        :param M: 最大迭代次数
        :param p_low: 粒子位置的约束范围
        :param p_up: 粒子位置的约束范围
        :param v_low: 粒子速度的约束范围
        :param v_high: 粒子速度的约束范围
        :param fitness: 适应度函数
        :param w_range: 线性递减惯性权值范围
        :param c1: 个体学习因子
        :param c2: 群体学习因子
        """

        self.fitness = fitness # 适应度函数
        self.w_range = w_range # 线性惯性权值范围
        self.w = self.w_range[1]  # 惯性权值
        self.c1 = c1  # 个体学习因子
        self.c2 = c2  # 群体学习因子
        self.D = D  # 粒子维度
        self.N = N  # 粒子群规模
        self.M = M  # 最大迭代次数
        self.p_range = [p_low, p_up]  # 粒子位置的约束范围
        self.v_range = [v_low, v_high]  # 粒子速度的约束范围
        self.x = np.zeros((self.N, self.D))  # 所有粒子的位置，N行D列
        self.v = np.zeros((self.N, self.D))  # 所有粒子的速度，N行D列
        self.p_best = np.zeros((self.N, self.D))  # 每个粒子的最优位置
        self.g_best = np.zeros((1, self.D))[0]  # 种群（全局）的最优位置
        self.p_bestFit = np.zeros(self.N)  # 每个粒子的最优适应值
        self.g_bestFit = float('Inf')  # 始化种群（全局）的最优适应值

        # 初始化所有个体和全局信息
        for i in range(self.N):
            for j in range(self.D):
                self.x[i][j] = random.uniform(self.p_range[0][j], self.p_range[1][j])
                self.v[i][j] = random.uniform(self.v_range[0], self.v_range[1])
            self.p_best[i] = self.x[i]  # 保存个体历史最优位置，初始默认第0代为最优
            fit = self.fitness(self.p_best[i])
            self.p_bestFit[i] = fit  # 保存个体历史最优适应值
            if fit < self.g_bestFit:  # 寻找并保存全局最优位置和适应值
                self.g_best = self.p_best[i]
                self.g_bestFit = fit

    def update(self):
        for i in range(self.N):
            # 更新速度(核心公式)
            self.v[i] = self.w * self.v[i] + self.c1 * random.uniform(0, 1) * (
                    self.p_best[i] - self.x[i]) + self.c2 * random.uniform(0, 1) * (self.g_best - self.x[i])
            # 速度限制
            for j in range(self.D):
                if self.v[i][j] < self.v_range[0]:
                    self.v[i][j] = self.v_range[0]
                if self.v[i][j] > self.v_range[1]:
                    self.v[i][j] = self.v_range[1]
            # 更新位置
            self.x[i] = self.x[i] + self.v[i]
            # 位置限制
            for j in range(self.D):
                if self.x[i][j] < self.p_range[0][j]:
                    self.x[i][j] = self.p_range[0][j]
                if self.x[i][j] > self.p_range[1][j]:
                    self.x[i][j] = self.p_range[1][j]
            # 更新个体和全局历史最优位置及适应值
            _fit = self.fitness(self.x[i])
            if _fit < self.p_bestFit[i]:
                self.p_best[i] = self.x[i]
                self.p_bestFit[i] = _fit
            if _fit < self.g_bestFit:
                self.g_best = self.x[i]
                self.g_bestFit = _fit

    def run(self):

        best_fittness_record = []  # 记录每轮迭代的最佳适应度，用于绘图


        for _ in tqdm(range(self.M)):
            self.update()  # 更新主要参数和信息
            self.w -= (self.w_range[1] - self.w_range[0]) / self.M  # 惯性权重线性递减
            best_fittness_record.append(self.g_bestFit.copy()) # 记录每轮迭代的最佳适应度

        return self.g_best, self.g_bestFit, best_fittness_record


def rastrigin_fitness(x):
    """
    rastrigin测试函数
    全局最优值在原点处得到
    函数值最小为0
    """
    A = 10
    n = len(x)
    sum_sq = np.sum(x**2 - A*np.cos(2*np.pi*x))
    return n*A + sum_sq

def griewank_fitness(x):
    """
    griewank 测试函数
    全局最优值在原点处得到
    函数值最小为0
    """
    product = 1
    sum_part = 0
    for i in range(len(x)):
        sum_part += x[i]**2 / 4000
        product *= np.cos(x[i] / np.sqrt(i+1))
    return 1 + sum_part - product


if __name__ == '__main__':

    low = [-100] * 5
    up = [100] * 5
    pso = PSO(5, 100, 500, low, up, -1, 1, griewank_fitness)

    # ---------------------------------

    time_start = time.time()  # 记录迭代寻优开始时间
    best_x, best_fit, best_fit_record = pso.run()
    time_end = time.time()  # 记录迭代寻优结束时间

    # ---------------------------------

    print(f'Best solution is {best_x}')
    print(f'Best fitness is {best_fit}')
    print(f'Algorithm takes {round(time_end - time_start, 5)} seconds')

    # ---------------------------------

    plt.figure()
    plt.plot([i for i in range(len(best_fit_record))], best_fit_record)
    plt.xlabel("iter")
    plt.ylabel("fitness")
    plt.title("Iter process")
    plt.show()

