import numpy as np
import random

def ant_colony(G: dict[str,dict[str,float]] , ants=50, epochs=200, alpha=1.0, beta=5.0, rho=0.1, q=100):
    nodes = list( G.keys())
    n = len(nodes)
    distance = np.zeros((n, n))
    for i, u in enumerate(nodes):
        for j, v in enumerate(nodes):
            if u != v:
                distance[i][j] = G[u][v]
            else:
                distance[i][j] = np.inf

    def get_length(tour: list) -> float:
        return sum(distance[tour[i]][tour[(i + 1) % n]] for i in range(n))

    # Initial pheromone levels 
    pheromone = np.ones((n, n))  

    best_tour = None
    best_length = float('inf')

    for _ in range(epochs):
        all_tours = []
        all_lengths = []
        for _ in range(ants):
            tour = []
            unvisited = set(range(n))
            current = random.choice(range(n))
            tour.append(current)
            unvisited.remove(current)

            # Generate a tour based on pheromone trail
            while unvisited:
                probabilities = []
                denom = sum((pheromone[current][j] ** alpha) * ((1 / distance[current][j]) ** beta) for j in unvisited)
                for j in unvisited:
                    p = (pheromone[current][j] ** alpha) * ((1 / distance[current][j]) ** beta) / denom
                    probabilities.append((j, p))
                next_city = random.choices([j for j, _ in probabilities], [p for _, p in probabilities])[0]
                tour.append(next_city)
                unvisited.remove(next_city)
                current = next_city
            
            tour_length = get_length(tour)
            all_tours.append(tour)
            all_lengths.append(tour_length)

            if tour_length < best_length:
                best_length = tour_length
                best_tour = tour

        # Evaporation of pheromones
        pheromone *= (1 - rho)

        # Deposit pheromones
        for tour, length in zip(all_tours, all_lengths):
            for i in range(n):
                u, v = tour[i], tour[(i + 1) % n]
                pheromone[u][v] += q / length
                pheromone[v][u] += q / length  

    best_path = tuple([nodes[i] for i in best_tour])
    return best_path, best_length



