# 智能优化算法


## 遗传算法求解TSP问题

### 算法迭代流程
1. 计算适应度
2. 父体选择（锦标赛法）
3. 交叉（次序杂交发）
4. 变异（互换变异）
5. 进化逆转

### 性能优化
这里只列出几个显著提高性能的优化方法
1. 次序杂交法中查询重复基因时，先将待查询的基因序列转为`set`格式

```python
# parent1 提供基因
gene_from_parent1 = parent1[beg_pos:end_pos]
child[:(end_pos-beg_pos)] = gene_from_parent1

# parent2 提供基因
gene_from_parent1 = set(child[:(end_pos-beg_pos)])
gene_from_parent2 = [gene for gene in parent2 if gene not in gene_from_parent1]
child[end_pos-beg_pos:] = gene_from_parent2

```

2. 进化逆转时，只计算逆转片段前后接点的适应度差，而不计算整个片段的适应度差

```python
# 优化版本，只计算交换片段接口距离
pos = sorted([_random[idx * 2], _random[idx * 2 + 1]])
old_d1 = dis_mat[item[pos[0]-1], item[pos[0]]]
old_d2 = dis_mat[item[pos[1]], item[pos[1]+1]]
new_d1 = dis_mat[item[pos[0]-1], item[pos[1]]]
new_d2 = dis_mat[item[pos[0]], item[pos[1]+1]]
if old_d1 + old_d2 > new_d1 + new_d2:
    reverse_list = list(range(pos[0], pos[1] + 1))
    population[idx][reverse_list[::-1]] = population[idx][reverse_list]
```


### 性能测试

CPU: AMD Ryzen 7 4800H with Radeon Graphics

- MAX_TIME = 200
- NUMBER_OF_INDIVIDUAL = 100
- pIntercross = 0.95
- pMutate = 0.01

经过优化，可以在0.8s左右完成计算
