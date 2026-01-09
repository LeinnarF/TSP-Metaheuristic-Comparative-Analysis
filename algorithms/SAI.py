import random 
import math
from functools import cache

# utility function 

def tour_cost(graph, path):
    total = 0
    return sum(graph[path[i]][path[(i + 1) % len(path)]] for i in range(len(path)))
 
 
def simulated_annealing_improved(graph, T_start=1000, T_end=1, cooling_rate=0.999):
    def get_cost(u, v):
        return graph[u][v] if v in graph[u] else math.inf

    nodes = list(graph.keys())
    n = len(nodes)

    iter_per_temp = int(n * 1.5)

    current_tour = nodes[:]
    random.shuffle(current_tour)
    current_cost = tour_cost(graph, current_tour)
    best_tour = current_tour[:]
    best_cost = current_cost
    T = T_start

    while T > T_end:
        for _ in range(iter_per_temp):
            a, b = random.sample(range(n), 2)
            if abs(a - b) <= 1 or abs(a - b) == n - 1:
                continue

            node_a_idx = a
            node_prev_idx = (a - 1) % n
            node_next_idx = (a + 1) % n
            node_b_idx = b
            node_c_idx = (b + 1) % n

            node_prev = current_tour[node_prev_idx]
            node_a = current_tour[node_a_idx]
            node_next = current_tour[node_next_idx]
            node_b = current_tour[node_b_idx]
            node_c = current_tour[node_c_idx]

            cost_removed = (
                get_cost(node_prev, node_a) +
                get_cost(node_a, node_next) +
                get_cost(node_b, node_c)
            )
            cost_added = (
                get_cost(node_prev, node_next) +
                get_cost(node_b, node_a) +
                get_cost(node_a, node_c)
            )
            delta = cost_added - cost_removed

            if delta < 0 or random.random() < math.exp(-delta / T):
                node = current_tour.pop(node_a_idx)
                if node_a_idx < node_b_idx:
                    current_tour.insert(node_b_idx, node)
                else:
                    current_tour.insert(node_b_idx + 1, node)
                current_cost += delta
                if current_cost < best_cost:
                    best_cost, best_tour = current_cost, current_tour[:]

        T *= cooling_rate

    return best_cost, best_tour

 