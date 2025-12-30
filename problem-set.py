import tsplib95
import numpy as np

berlin52 = tsplib95.load('data/berlin52.tsp')
nodes = list(berlin52.get_nodes())
n = berlin52.dimension

# Distance Matrix
D = np.zeros((n,n))
for i, ni in enumerate(nodes):
    for j, nj in enumerate(nodes):
        D[i,j] = berlin52.get_weight(ni, nj)

# In Graph (Dictionary)
G = {u: {} for u in nodes}

for i in nodes:
    for j in nodes:
        if i != j:
            G[i][j] = berlin52.get_weight(i,j)

