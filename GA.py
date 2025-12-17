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
def init_population(pop_size, n):
    return [random.sample(range(n), n) for _ in range(pop_size)]

# Selection: Tournament selection
def tournament_selection(pop, fitness, k=3):
    candidates = random.sample(range(len(pop)), k)
    best = min(candidates, key=lambda i : fitness[i])
    return pop[best]

# Crossover: Order Crossover (OX)
def order_crossover(p1, p2):
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [-1]*n

    child[a:b+1] = p1[a:b+1]

    ptr = 0
    for x in p2:
        if x not in child:
            while child[ptr] != -1:
                ptr += 1
            child[ptr] = x
    
    return child

# Mutation: 2-opt swap
def swap_mutation(tour, pm = 0.1):
    if random.random() < pm:
        i, j = random.sample(range(len(tour)), 2)
        tour[i], tour[j] = tour[j], tour[i]

    return tour


# Genetic Algorithm
def genetic_algorithm(
        D,
        pop_size=100,
        generations=500,
        pm = 0.1
):
    n = len(D)
    population = init_population(pop_size, n)

    best = None
    best_cost = float('inf')

    for _ in range(generations):
        fitness = [tour_length(ind, D) for ind in population]

        elite_idx = np.argmin(fitness)
        elite = population[elite_idx][:]

        if fitness[elite_idx] < best_cost:
            best = elite[:]
            best_cost = fitness[elite_idx]

        new_population = [elite]

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitness)
            parent2 = tournament_selection(population, fitness)

            child = order_crossover(parent1, parent2)
            child = swap_mutation(child, pm)

            new_population.append(child)
        
        population = new_population

    return best, best_cost

# Run single experiment
# Berlin52 solution: 7542
start_time = time.time()
best_tour, best_cost = genetic_algorithm(D)
end_time = time.time()
print(f"Best cost: {best_cost}")
print(f"Elapsed time: {end_time - start_time:.4f} seconds")

