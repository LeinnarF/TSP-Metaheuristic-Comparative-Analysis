import tsplib95
import random
import numpy as np
import time 
from utils import tour_length


berlin52 = tsplib95.load('data/berlin52.tsp')

nodes = list(berlin52.get_nodes())
n = berlin52.dimension

# Distance matrix

D = np.zeros((n, n))
for i, ni in enumerate(nodes):
    for j, nj in enumerate(nodes):
        D[i, j] = berlin52.get_weight(ni, nj)

# Util: Tour length
def tour_length(tour, D):
    cost = 0.0
    for i in range(len(tour)):
        cost += D[tour[i], tour[(i + 1) % len(tour)]]
    return cost

# Initial solution
def random_tour(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour

# Neighborhood: 2-opt swap
def two_opt(tour):
    i, j = sorted(random.sample(range(len(tour)), 2))
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    return new_tour

# Simulated Annealing
def simulated_annealing(
        D,
        T0=1000.0,
        Tmin=1e-3,
        alpha=0.995,
        max_iter=1000
):
    n = len(D)
    current = random_tour(n)
    current_cost = tour_length(current, D)

    best = current[:]
    best_cost = current_cost

    T  = T0
    it = 0

    while T > Tmin and it < max_iter:
        candidate = two_opt(current)
        candidate_cost = tour_length(candidate, D)

        delta = candidate_cost - current_cost

        if delta < 0 or random.random() < np.exp(-delta / T):
            current = candidate
            current_cost = candidate_cost

            if current_cost < best_cost:
                best = current[:]
                best_cost = current_cost

        T *= alpha
        it += 1

    return best, best_cost

# Run single experiment

# Berlin52 solution: 7542

start = time.time()
best_tour, best_cost = simulated_annealing(D)
elapsed = time.time() - start
print("Size:", n)
print("Best tour length:", best_cost)
print("Elapsed time (s):", elapsed)