import numpy as np
import random

class GreyWolfTSP:
    def __init__(self, graph, num_wolves=15, max_iter=500):
        self.graph = graph
        self.node_list = list(graph.keys())
        self.num_nodes = len(self.node_list)

        self.num_wolves = num_wolves
        self.max_iter = max_iter

        self.wolves = np.random.rand(num_wolves, self.num_nodes)

        self.alpha_pos = None
        self.beta_pos = None
        self.delta_pos = None

        self.alpha_score = float("inf")
        self.beta_score = float("inf")
        self.delta_score = float("inf")

    def calculate_distance(self, tour_indices):
        total = 0
        for i in range(len(tour_indices)):
            u = tour_indices[i]
            v = tour_indices[(i + 1) % len(tour_indices)]
            a = self.node_list[u]
            b = self.node_list[v]
            total += self.graph[a][b]
        return total

    def decode(self, wolf):
        """Convert priority vector → permutation."""
        return np.argsort(wolf)

    def evaluate_wolf(self, wolf):
        tour = self.decode(wolf)
        return self.calculate_distance(tour)

    def optimize(self):
        for t in range(self.max_iter):

            # Evaluate wolves
            for i, wolf in enumerate(self.wolves):
                score = self.evaluate_wolf(wolf)

                # Update Alpha, Beta, Delta
                if score < self.alpha_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = None if self.beta_pos is None else self.beta_pos.copy()

                    self.beta_score = self.alpha_score
                    self.beta_pos = None if self.alpha_pos is None else self.alpha_pos.copy()

                    self.alpha_score = score
                    self.alpha_pos = wolf.copy()

                elif score < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = None if self.beta_pos is None else self.beta_pos.copy()

                    self.beta_score = score
                    self.beta_pos = wolf.copy()

                elif score < self.delta_score:
                    self.delta_score = score
                    self.delta_pos = wolf.copy()

            a = 2 - 2 * (t / self.max_iter)

            for i in range(self.num_wolves):
                for j in range(self.num_nodes):

                    # Wolves move relative to Alpha, Beta, Delta
                    def compute_component(leader_pos):
                        r1, r2 = random.random(), random.random()
                        A = 2 * a * r1 - a
                        C = 2 * r2
                        D = abs(C * leader_pos[j] - self.wolves[i][j])
                        return leader_pos[j] - A * D

                    X1 = compute_component(self.alpha_pos)
                    X2 = compute_component(self.beta_pos)
                    X3 = compute_component(self.delta_pos)

                    # Final updated position
                    self.wolves[i][j] = (X1 + X2 + X3) / 3

            # Keep wolves inside bounds (for random keys)
            self.wolves = np.clip(self.wolves, 0, 1)
            
        best_tour = self.decode(self.alpha_pos)
        return best_tour, self.alpha_score