import random
import math
import copy
from typing import List, Tuple, Dict, Optional

# --- Type Aliases ---
Coordinates = Dict[str, Tuple[float, float]]  # Maps city name to (x, y)
DistanceDict = Dict[str, Dict[str, float]]    # Nested dictionary for distances
Tour = List[str]                              # List of city names

# --- Problem Definition: TSP Instance ---

def create_distance_dictionary(coordinates: Coordinates) -> DistanceDict:
    """
    Creates a nested dictionary where d[i][j] is the Euclidean 
    distance between city i and city j.
    """
    cities = list(coordinates.keys())
    dist_dict: DistanceDict = {city: {} for city in cities}
    
    for i in cities:
        for j in cities:
            if i != j:
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]
                dist = math.hypot(x2 - x1, y2 - y1)
                dist_dict[i][j] = dist
            else:
                dist_dict[i][j] = 0.0
    return dist_dict

def total_distance(tour: Tour, distance_dict: DistanceDict) -> float:
    dist = 0.0
    n = len(tour)
    for i in range(n):
        from_city = tour[i]
        to_city = tour[(i + 1) % n]
        # Access using nested keys
        dist += distance_dict[from_city][to_city]
    return dist


# --- EHO Components for TSP ---

class Elephant:
    def __init__(self, tour: Tour) -> None:
        self.tour: Tour = tour
        self.fitness: Optional[float] = None

    def evaluate(self, distance_dict: DistanceDict) -> float:
        self.fitness = total_distance(self.tour, distance_dict)
        return self.fitness

def initialize_population(pop_size: int, city_names: List[str]) -> List[Elephant]:
    population: List[Elephant] = []
    base_tour = city_names.copy()
    for _ in range(pop_size):
        tour = base_tour.copy()
        random.shuffle(tour)
        population.append(Elephant(tour))
    return population

def clan_update(elephant: Elephant, leader: Elephant) -> Tour:
    """
    Update an elephant's tour by using a simple operator that
    makes its permutation more like the leader's tour.
    """
    new_tour = elephant.tour.copy()
    n = len(new_tour)
    
    # Select random segment indices
    idx1, idx2 = sorted(random.sample(range(n), 2))
    
    # Extract segment from leader (list of strings)
    segment = leader.tour[idx1 : idx2 + 1]
    
    # Create tour excluding the segment cities
    new_tour = [city for city in new_tour if city not in segment]
    
    # Insert segment at random position
    insert_pos = random.randint(0, len(new_tour))
    new_tour[insert_pos:insert_pos] = segment
    
    return new_tour

def separating_operator(population: List[Elephant], 
                        distance_dict: DistanceDict, 
                        replace_frac: float = 0.1) -> List[Elephant]:
    """
    Replace the worst performing elephants with random new solutions.
    """
    population.sort(key=lambda e: e.fitness if e.fitness is not None else float('inf'))
    n_replace = max(1, int(len(population) * replace_frac))
    
    # Get the list of all cities from the first elephant to generate new random tours
    all_cities = population[0].tour.copy()
    
    for i in range(1, n_replace + 1):
        new_tour = all_cities.copy()
        random.shuffle(new_tour)
        population[-i] = Elephant(new_tour)
        population[-i].evaluate(distance_dict)
    
    return population

# --- Main EHO Algorithm for TSP ---

def elephant_herd_optimization(distance_dict: DistanceDict, 
                               pop_size: int = 50, 
                               clans: int = 5, 
                               max_iter: int = 500) -> Tuple[Elephant, float]:
    
    # Extract city names directly from the dictionary keys
    city_names = list(distance_dict.keys())
    
    population = initialize_population(pop_size, city_names)
    for elephant in population:
        elephant.evaluate(distance_dict)
    
    clan_size = pop_size // clans
    best_solution: Optional[Elephant] = None
    best_fitness = float('inf')
    
    for iteration in range(max_iter):
        random.shuffle(population)
        clans_list = [population[i*clan_size:(i+1)*clan_size] for i in range(clans)]
        
        for clan in clans_list:
            if not clan:
                continue
                
            clan.sort(key=lambda e: e.fitness if e.fitness is not None else float('inf'))
            leader = clan[0]
            
            # Update clan members based on leader
            for i in range(1, len(clan)):
                new_tour = clan_update(clan[i], leader)
                candidate = Elephant(new_tour)
                candidate.evaluate(distance_dict)
                
                # Greedy selection
                if candidate.fitness < clan[i].fitness: 
                    clan[i] = candidate
            
            # Update global best
            if leader.fitness < best_fitness: 
                best_fitness = leader.fitness 
                best_solution = copy.deepcopy(leader)
        
        # Flatten population and apply separating operator
        population = [elephant for clan in clans_list for elephant in clan]
        population = separating_operator(population, distance_dict)
        
    
    if best_solution is None:
        raise ValueError("Optimization failed to produce a solution.")

    return best_solution, best_fitness

# --- Example Usage ---

if __name__ == "__main__":
    n_cities = 20
    # Generate coordinates with string keys
    coords: Coordinates = {
        f"City_{i}": (random.uniform(0, 100), random.uniform(0, 100)) 
        for i in range(n_cities)
    }
    
    dist_matrix = create_distance_dictionary(coords)
    
    best_sol, best_fit = elephant_herd_optimization(dist_matrix, pop_size=50, clans=5, max_iter=500)
    
    print("\nBest tour found:")
    print(best_sol.tour)
    print(f"With total distance: {best_fit:.2f}")