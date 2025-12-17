from utils import tour_length
import tsplib95
import random
import numpy as np
import time

berlin52 = tsplib95.load('data/berlin52.tsp')
nodes = list(berlin52.get_nodes())
n = berlin52.dimension

# Distance matrix
D = np.zeros((n, n))
for i, ni in enumerate(nodes):
    for j, nj in enumerate(nodes):
        D[i, j] = berlin52.get_weight(ni, nj)

# Initialization
def init_pheromone(n, tau0 = 1.0):
    return np.full((n, n), tau0)

# Construct a tour
def construct_tour(D, pheromone, alpha, beta):
    n = len(D)
    start = random.randrange(n)
    tour = [start]
    unvisited = set(range(n))
    unvisited.remove(start)

    while unvisited:
        i = tour[-1]
        probs = []

        for j in unvisited:
            tau = pheromone[i, j] ** alpha
            eta = (1.0 / D[i, j]) ** beta
            probs.append((j, tau * eta))
        
        total = sum(p for _, p in probs)
        r = random.random() * total
        acc = 0.0

        for j, p in probs:
            acc += p
            if acc >= r:
                tour.append(j)
                unvisited.remove(j)
                break
    return tour

# Update pheromone
def update_pheromone(pheromone, ants_tours, ants_costs, rho):
    pheromone *= (1 - rho)

    for tour, cost in zip(ants_tours, ants_costs):
        for i in range(len(tour)):
            a = tour[i]
            b = tour[(i + 1) % len(tour)]
            pheromone[a, b] += 1.0 / cost
            pheromone[b, a] += 1.0 / cost

# Ant Colony Optimization
def ant_colony_optimization(
        D,
        ants=20,
        iterations=100,
        alpha=1.0,
        beta=2.0,
        rho=0.1,
    ):
    n = len(D)
    pheromone = init_pheromone(n)

    best = None
    best_cost = float('inf')

    for _ in range(iterations):
        tours = []
        costs = []

        for _ in range(ants):
            tour = construct_tour(D, pheromone, alpha, beta)
            cost = tour_length(tour, D)

            tours.append(tour)
            costs.append(cost)

            if cost < best_cost:
                best = tour[:]
                best_cost = cost
        
        update_pheromone(pheromone, tours, costs, rho)

    return best, best_cost

# Run single experiment
start_time = time.time()
# Berlin52 solution: 7542
best_tour, best_cost = ant_colony_optimization(D)
end_time = time.time()
print(f"Best cost: {best_cost}")
print(f"Time taken: {end_time - start_time} seconds")