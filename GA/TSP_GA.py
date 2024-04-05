from typing import List
import numpy as np
import random
from tqdm import tqdm


def distance_matrix(points: np.ndarray) -> np.ndarray:

    # 创建距离矩阵
    distances = np.zeros((len(points), len(points)))
    
    for idx_begin, begin in points.iterrows():

        for idx_end, end in points.iterrows():

            if idx_begin == idx_end:
                continue

            distance = pow(pow(begin['x'] - end['x'], 2) + pow(begin['y'] - end['y'], 2), 0.5)

            distances[idx_begin - 1, idx_end - 1] = distance * 100
        
    return distances


def _fitness(code:np.ndarray, dis_mat:np.ndarray):

    index = code * len(code) + np.roll(code, -1)
    return 1 / sum(dis_mat.flat[index])


def fitness(population: List[np.ndarray], dis_mat:np.ndarray):

    return [_fitness(code, dis_mat) for code in population]


def select(population: List[np.ndarray], fitness_value: List[float]):

    # 锦标赛法
    pop_with_fitness = list(zip(population, fitness_value))
    
    sel = []
    for _ in range(len(population)):
        # 随机选择4个个体，作为锦标赛小组
        tournament = random.choices(pop_with_fitness, k=4)
        # 锦标赛获胜者被选择
        winner = max(tournament, key=lambda x: x[-1])
        sel.append(winner[0])

    return sel


def intercross(parents: List[np.ndarray], pval):

    children =[]
    
    # 次序杂交法
    length_code = len(parents[0])
    for parent1, parent2 in list(zip(parents, parents[1:]+parents[:1])):

        if random.random() > pval:
            children.append(random.choice([parent1, parent2]))
            continue

        # 构造子代
        child = np.zeros(length_code, dtype=parent1.dtype)

        # 确定杂交位置
        pos = random.sample(range(length_code) ,2)
        beg_pos = min(pos)
        end_pos = max(pos)

        # parent1 提供基因
        gene_from_parent1 = parent1[beg_pos:end_pos]
        child[:(end_pos-beg_pos)] = gene_from_parent1

        # parent2 提供基因
        gene_from_parent1 = set(child[:(end_pos-beg_pos)])
        gene_from_parent2 = [gene for gene in parent2 if gene not in gene_from_parent1]
        child[end_pos-beg_pos:] = gene_from_parent2

        # 添加到子群
        children.append(child)

    return children


def mutate(population: List[np.ndarray], pval):

    # 互换变异
    _random = np.random.random(len(population))
    for idx, item in enumerate(population):
        # 依概率进行变异
        if _random[idx] > pval: continue
        # 生成随机交换位置
        pos = random.sample(range(len(population[0])), 2)
        # 交换
        item[[pos[0], pos[1]]] = item[[pos[1], pos[0]]]

    return population


def reverse(population: List[np.ndarray], dis_mat:np.ndarray):

    # 进化逆转
    length = len(population[0])
    _random = np.random.randint(1, length-1, size=len(population) * 2)

    for idx, item in enumerate(population):

        # # 确定逆转的位置
        # pos = random.sample(range(length), 2)
        # reverse_list = list(range(min(pos), max(pos) + 1))
        # # 执行逆转
        # new_item = item.copy()
        # new_item[list(reversed(reverse_list))] = new_item[reverse_list]
        # # 选择保留适应值大的个体
        # if _fitness(new_item, dis_mat) > _fitness(item, dis_mat):
        #     population[idx] = new_item
  
        # 优化版本，只计算交换片段接口距离
        # pos = sorted(random.sample(range(1, length-1), 2))
        pos = sorted([_random[idx * 2], _random[idx * 2 + 1]])
        old_d1 = dis_mat[item[pos[0]-1], item[pos[0]]]
        old_d2 = dis_mat[item[pos[1]], item[pos[1]+1]]
        new_d1 = dis_mat[item[pos[0]-1], item[pos[1]]]
        new_d2 = dis_mat[item[pos[0]], item[pos[1]+1]]
        if old_d1 + old_d2 > new_d1 + new_d2:
            reverse_list = list(range(pos[0], pos[1] + 1))
            population[idx][reverse_list[::-1]] = population[idx][reverse_list]
            
    return population


def run(
        points,
        max_time=200,
        popsize=100,
        pIntercross=0.95,
        pMutate=0.01):
    
    dis_mat = distance_matrix(points)

    # 随机生成初始种群
    population = [np.random.permutation(len(points)) for _ in range(popsize)]


    # 最优个体记录
    best = None
    best_fitness = 0
    fitness_record = []

    # 开始迭代
    for _ in range(max_time):

        # 计算适应度
        fitness_value = fitness(population, dis_mat)

        # 记录最优适应度，以及历史最优适应度
        if max(fitness_value) > best_fitness:
            best_idx = np.argmax(fitness_value)
            best_fitness = fitness_value[best_idx]
            best = population[best_idx]
        
        # 保存种群适应度的统计信息
        fitness_record.append(np.mean(fitness_value))

        # 父体选择
        parent = select(population, fitness_value)

        # 交叉
        population = intercross(parent, pIntercross)

        # 变异
        population = mutate(population, pMutate)

        # 进化逆转
        population = reverse(population, dis_mat)

    return best, best_fitness, fitness_record


