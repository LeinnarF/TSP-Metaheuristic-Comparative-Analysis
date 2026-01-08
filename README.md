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
pip install tsplib95 numpy pandas scikit-learn
```
- tsplib95: For loading TSP instances
- numpy: For numerical computations
- pandas: For data handling and analysis
- scikit-learn: For data normalization and analysis

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
 ├── algorithms/
 │   ├── ga.py
 │   ├── sa.py
 │   ├── aco.py
 │   ├── pso.py
 │   ├── eho.py
 │   ├── gwo.py
 │   └── pko.py
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
## Usage 

load the `Solver` class with a TSP instance and call the `solve` method to run the metaheuristic algorithms and compare their performance.

```python
from Solver import Solver
solver = Solver('berlin52')
solver.solve()
```
To presents a comparison of the different metaheuristics on the specified TSP instance, `solve()` must be called before presenting results.
```python
solver.present_results()
```
The results include metrics such as best solution found, average solution quality, computational time. 

weighted sum of normalized metrics to rank the algorithms based on user-defined priorities. `present_results()` must be called to use this feature.`
```python
solver.weighted_score(weight_speed=0.5)
```










