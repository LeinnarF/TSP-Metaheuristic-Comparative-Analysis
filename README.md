## Comparative Analysis TSP Metaheuristics for Thesis

This repository contains the code and documentation for a comparative analysis of various metaheuristic algorithms applied to the Traveling Salesman Problem (TSP). The study aims to evaluate the performance of different metaheuristics in terms of solution quality, computational time, and convergence behavior.

### Metaheuristics Implemented
- Genetic Algorithm (GA)
- Simulated Annealing (SA)
- Ant Colony Optimization (ACO)
- Particle Swarm Optimization (PSO)
- Elepahant Herding Optimization (EHO)
- Gray Wolf Optimization (GWO)
- Pied Kingfisher Optimization (PKO)

### Required packages
```bash
pip install tsplib95 networkx numpy matplotlib pandas
```
- tsplib95: For loading TSP instances
- networkx: For graph representation and manipulation
- numpy: For numerical computations
- matplotlib: For plotting results
- pandas: For data handling and analysis

### Download TSPLIB Instances
Get problem files from:

- https://www.math.uwaterloo.ca/tsp/world/countries.html
- or the original TSPLIB site

You will typically download files like:

```
berlin52.tsp
eil51.tsp
att48.tsp
```

Place them in your project directory, e.g.:
```
project/
 ├── data/
 │   └── berlin52.tsp
 └── main.py
```

### Load a TSPLIB problem in Python

```python

import tsplib95

problem = tsplib95.load('data/berlin52.tsp')

print(problem.name)
print(problem.dimension)
print(problem.edge_weight_type)
```









