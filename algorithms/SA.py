import random 
from functools import cache
import numpy as np


def simulated_annealing(G: dict[str,dict[str,float]], temperature: float, alpha: float, iterations: int) -> tuple[tuple, float]:

    nodes = list(G.keys())
    distances = {u: {v: G[u][v] for v in G[u] if u != v} for u in nodes}

    @cache
    def get_length(tour: tuple) -> float:
        tour_size = len(tour)
        return sum(
            distances[tour[i]][tour[(i+1) % tour_size]] for i in range(tour_size)
        ) 
    
    def random_neighbor(tour: tuple) -> tuple:
        neighbor = list(tour)
        i, j = random.sample(range(len(tour)), 2)
        neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        return tuple(neighbor)

    current_tour = tuple(random.sample(nodes, len(nodes)))
    current_distance = get_length(current_tour)
    best_tour = current_tour
    best_distance = current_distance

    for _ in range(iterations):
        next_tour = random_neighbor(current_tour)
        next_distance = get_length(next_tour)

        if next_distance < current_distance or random.random() < np.exp((current_distance - next_distance) / temperature):
            current_tour, current_distance = next_tour, next_distance

            if current_distance < best_distance:
                best_tour, best_distance = current_tour, current_distance

        temperature *= alpha

    return best_tour, best_distance
