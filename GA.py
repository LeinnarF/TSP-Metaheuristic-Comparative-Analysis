import random
from functools import cache
from typing import List, Tuple, Dict
from itertools import cycle

def random_population(nodes: list, size: int) -> List[Tuple]:
    return [tuple(random.sample(nodes, len(nodes))) for _ in range(size)]

def improved_partially_matched_crossover(parent1: Tuple, parent2: Tuple, nodes: List[str]) -> Tuple:
    """Order Crossover (OX) - preserves relative order better than your current method"""
    size = len(nodes)
    start, end = sorted(random.sample(range(size), 2))
    
    # Copy slice from parent1
    child = [None] * size
    child[start:end] = parent1[start:end]
    
    # Fill rest from parent2, preserving relative order
    remaining = [(node, i) for i, node in enumerate(parent2) if node not in child]
    remaining.sort(key=lambda x: x[1])  # Sort by original position in parent2
    
    # Fill remaining positions in order
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = remaining[idx][0]
            idx += 1
    
    return tuple(child)

def mutate_2opt(route: Tuple, mutation_rate: float, nodes: List[str]) -> Tuple:
    """2-opt mutation - much more effective for TSP"""
    route_list = list(route)
    size = len(route_list)
    
    if random.random() < mutation_rate:
        # 2-opt swap: reverse segment between i and j
        i, j = sorted(random.sample(range(size), 2))
        route_list[i:j+1] = reversed(route_list[i:j+1])
    
    return tuple(route_list)

def genetic_alg(G: Dict[str, Dict[str, float]], 
                pop_size: int = 100, 
                mutation_rate: float = 0.2, 
                generations: int = 500,
                elitism_ratio: float = 0.1,
                tournament_size: int = 5) -> Tuple[List[str], float]:
    
    nodes = list(G.keys())
    n = len(nodes)
    
    # Precompute distances (avoid self-loops)
    distances = {}
    for u in nodes:
        distances[u] = {v: G[u][v] for v in G[u] if u != v}
    
    def tour_length(tour: Tuple) -> float:
        """Calculate total tour length with caching"""
        return sum(distances[tour[i]][tour[(i+1) % n]] for i in range(n))
    
    @cache
    def fitness(tour: Tuple) -> float:
        return 1.0 / (tour_length(tour) + 1e-6)  # Avoid division by zero
    
    def tournament_selection(population: List[Tuple]) -> Tuple:
        """Larger tournament for better selection pressure"""
        candidates = random.sample(population, min(tournament_size, len(population)))
        return max(candidates, key=fitness)
    
    def local_search(tour: Tuple, max_iter: int = 100) -> Tuple:
        """2-opt local search to improve solutions"""
        best_tour = list(tour)
        improved = True
        
        iter_count = 0
        while improved and iter_count < max_iter:
            improved = False
            for i in range(n):
                for j in range(i + 2, n if i > 0 else n - 1):
                    # 2-opt swap
                    new_tour = best_tour[:i] + best_tour[i:j+1][::-1] + best_tour[j+1:]
                    new_tour_tuple = tuple(new_tour)
                    
                    if tour_length(new_tour_tuple) < tour_length(tuple(best_tour)):
                        best_tour = new_tour
                        improved = True
                        break
                if improved:
                    break
            iter_count += 1
        
        return tuple(best_tour)
    
    # Initialize population
    population = random_population(nodes, pop_size)
    
    # Track best solution
    best_tour = min(population, key=lambda r: tour_length(r))
    best_distance = tour_length(best_tour)
    
    for gen in range(generations):
        new_population = []
        
        # Elitism: keep best solutions
        elite_size = max(1, int(pop_size * elitism_ratio))
        elite = sorted(population, key=lambda x: fitness(x), reverse=True)[:elite_size]
        new_population.extend(elite)
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            # Crossover
            if random.random() < 0.9:  # Crossover probability
                child = improved_partially_matched_crossover(parent1, parent2, nodes)
            else:
                child = parent1 if random.random() < 0.5 else parent2
            
            # Mutation
            child = mutate_2opt(child, mutation_rate, nodes)
            
            # Local search (hybridization - crucial for small instances)
            child = local_search(child, max_iter=50)
            
            new_population.append(child)
        
        population = new_population[:pop_size]
        
        # Update best solution
        current_best = min(population, key=lambda r: tour_length(r))
        current_dist = tour_length(current_best)
        if current_dist < best_distance:
            best_distance = current_dist
            best_tour = current_best
            # print(f"Gen {gen}: Improved to {best_distance:.2f}")  # Progress tracking
    
    return list(best_tour), best_distance
