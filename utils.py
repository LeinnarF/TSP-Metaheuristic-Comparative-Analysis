import numpy as np
import random

def tour_length(tour, distance_matrix):
    cost = 0.0
    for i in range(len(tour)):
        cost += distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
    return cost




def random_tour(size):
    tour = list(range(size))
    random.shuffle(tour)
    return tour




def two_opt(tour):
    i, j = sorted(random.sample(range(len(tour)), 2))
    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
    return new_tour
