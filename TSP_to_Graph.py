import tsplib95
import numpy as np

def turnToGraph(targetCity):
  targetCity = tsplib95.load('data/' + targetCity +'.tsp')
  nodes = list(targetCity.get_nodes())
  n = targetCity.dimension

  D = np.zeros((n, n))
  for i, ni in enumerate(nodes):
      for j, nj in enumerate(nodes):
         D[i, j] = targetCity.get_weight(ni, nj)

  return D