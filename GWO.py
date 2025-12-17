import random
import numpy as np

class GreyWolfTSP:
    def __init__(self, graph, num_wolves=5, max_iter=3, use_local_search=True, random_seed=None):

        self.graph = graph
        if isinstance(graph, np.ndarray):
          self.num_nodes = graph.shape[0]
          self.node_list = list(range(self.num_nodes))
        elif isinstance(graph, dict):
          self.node_list = list(graph.keys())
          self.num_nodes = len(self.node_list)
        else:
          raise TypeError("Graph must be a numpy.ndarray (distance matrix) or a dictionary.")

        # Set random seeds
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        self.random_seed = random_seed

        # Validate graph is complete
        self._validate_graph()

        # Initialize parameters
        self.num_wolves = num_wolves
        self.max_iter = max_iter
        self.use_local_search = use_local_search

        # Initialize wolves with random priorities
        self.wolves = np.random.rand(num_wolves, self.num_nodes)

        # Initialize leaders
        self.alpha_pos = np.zeros(self.num_nodes)
        self.beta_pos = np.zeros(self.num_nodes)
        self.delta_pos = np.zeros(self.num_nodes)

        self.alpha_score = float("inf")
        self.beta_score = float("inf")
        self.delta_score = float("inf")

        # Pre-compute distance matrix
        self._build_distance_matrix()

        # Tracking
        self.convergence_history = []
        self.iterations_run = 0
        self.execution_time = 0

    def _validate_graph(self):
        if isinstance(self.graph, np.ndarray):
            # For a distance matrix, assume all connections exist if distances are finite
            for i in self.node_list:
                for j in self.node_list:
                    if i != j and not np.isfinite(self.graph[i][j]):
                        raise ValueError(f"Invalid distance (non-finite) from {i} to {j}")
        elif isinstance(self.graph, dict):
            for node in self.node_list:
                for other in self.node_list:
                    if other not in self.graph[node]:
                        raise ValueError(f"No connection from {node} to {other}")

    def _build_distance_matrix(self):
        """Build distance matrix from graph."""
        self.distance_matrix = np.zeros((self.num_nodes, self.num_nodes))
        for i, u in enumerate(self.node_list):
            for j, v in enumerate(self.node_list):
                self.distance_matrix[i, j] = self.graph[u][v]

    def get_random(self):
        """Get a random number."""
        return random.random()

    def calculate_distance(self, tour_indices):
        """Calculate total distance for a tour."""
        if len(tour_indices) <= 1:
            return float("inf")
        next_indices = np.roll(tour_indices, -1)
        return np.sum(self.distance_matrix[tour_indices, next_indices])

    def decode(self, wolf):
        """Convert priority vector to permutation."""
        return np.argsort(wolf)

    def two_opt_swap(self, tour_indices):
        """Perform 2-opt local search."""
        best_distance = self.calculate_distance(tour_indices)
        n = len(tour_indices)
        improved = True

        while improved:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    # Skip adjacent edges
                    if j - i == 1:
                        continue

                    # Create new tour by reversing segment i..j
                    new_tour = tour_indices.copy()
                    new_tour[i:j] = new_tour[i:j][::-1]
                    new_distance = self.calculate_distance(new_tour)

                    if new_distance < best_distance:
                        tour_indices = new_tour
                        best_distance = new_distance
                        improved = True
                        break
                if improved:
                    break
        return tour_indices

    def evaluate_wolf(self, wolf):
        """Evaluate a wolf's solution."""
        tour = self.decode(wolf)
        distance = self.calculate_distance(tour)

        if self.use_local_search:
            improved_tour = self.two_opt_swap(tour.copy())
            improved_distance = self.calculate_distance(improved_tour)

            if improved_distance < distance:
                # Convert improved tour back to random keys
                new_wolf = np.zeros_like(wolf)
                for pos, node_idx in enumerate(improved_tour):
                    # Assign values that will sort to this order
                    new_wolf[node_idx] = pos + np.random.uniform(-0.01, 0.01)
                return improved_distance, new_wolf

        return distance, wolf

    def optimize(self):

        # Initial evaluation
        scores = []
        for i in range(self.num_wolves):
            score, updated_wolf = self.evaluate_wolf(self.wolves[i])
            scores.append(score)
            self.wolves[i] = updated_wolf

        scores = np.array(scores)
        sorted_indices = np.argsort(scores)

        # Initialize leaders
        self.alpha_score = scores[sorted_indices[0]]
        self.alpha_pos = self.wolves[sorted_indices[0]].copy()
        self.beta_score = scores[sorted_indices[1]]
        self.beta_pos = self.wolves[sorted_indices[1]].copy()
        self.delta_score = scores[sorted_indices[2]]
        self.delta_pos = self.wolves[sorted_indices[2]].copy()

        # Main optimization loop
        for iteration in range(self.max_iter):
            self.iterations_run = iteration + 1

            # Linearly decreasing coefficient a
            a = 2 - 2 * (iteration / self.max_iter)

            # Update each wolf
            for i in range(self.num_wolves):
                # Skip if this wolf is a leader
                if i in [sorted_indices[0], sorted_indices[1], sorted_indices[2]]:
                    continue

                new_position = np.zeros(self.num_nodes)

                for j in range(self.num_nodes):
                    # Update based on alpha
                    r1, r2 = self.get_random(), self.get_random()
                    A_alpha = 2 * a * r1 - a
                    C_alpha = 2 * r2
                    D_alpha = abs(C_alpha * self.alpha_pos[j] - self.wolves[i][j])
                    X1 = self.alpha_pos[j] - A_alpha * D_alpha

                    # Update based on beta
                    r1, r2 = self.get_random(), self.get_random()
                    A_beta = 2 * a * r1 - a
                    C_beta = 2 * r2
                    D_beta = abs(C_beta * self.beta_pos[j] - self.wolves[i][j])
                    X2 = self.beta_pos[j] - A_beta * D_beta

                    # Update based on delta
                    r1, r2 = self.get_random(), self.get_random()
                    A_delta = 2 * a * r1 - a
                    C_delta = 2 * r2
                    D_delta = abs(C_delta * self.delta_pos[j] - self.wolves[i][j])
                    X3 = self.delta_pos[j] - A_delta * D_delta

                    new_position[j] = (X1 + X2 + X3) / 3

                # Add small mutation
                if self.get_random() < 0.05:
                    mutation = np.random.uniform(-0.05, 0.05, self.num_nodes)
                    new_position += mutation

                # Keep within bounds
                self.wolves[i] = np.clip(new_position, 0, 1)

            # Re-evaluate wolves
            scores = []
            for i in range(self.num_wolves):
                score, updated_wolf = self.evaluate_wolf(self.wolves[i])
                scores.append(score)
                self.wolves[i] = updated_wolf

            scores = np.array(scores)
            sorted_indices = np.argsort(scores)

            # Update leaders if better solutions found
            if scores[sorted_indices[0]] < self.alpha_score:
                self.alpha_score = scores[sorted_indices[0]]
                self.alpha_pos = self.wolves[sorted_indices[0]].copy()

            if scores[sorted_indices[1]] < self.beta_score:
                self.beta_score = scores[sorted_indices[1]]
                self.beta_pos = self.wolves[sorted_indices[1]].copy()

            if scores[sorted_indices[2]] < self.delta_score:
                self.delta_score = scores[sorted_indices[2]]
                self.delta_pos = self.wolves[sorted_indices[2]].copy()

            # Track convergence
            self.convergence_history.append(self.alpha_score)

        # Get final best solution
        best_tour_indices = self.decode(self.alpha_pos)
        best_tour_nodes = [self.node_list[idx] for idx in best_tour_indices]

        return best_tour_nodes, self.alpha_score