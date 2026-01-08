from TSPLIPLoader import TPSLIBLoader
from algorithms.HSFFPKO import HSFFPKO, universal_tsp_wrapper
from algorithms.EHO import elephant_herd_optimization
from algorithms.ACO import ant_colony
from algorithms.SA import simulated_annealing
from algorithms.SAI import simulated_annealing_improved
from algorithms.GWO import GreyWolfTSP
from algorithms.PSO import hybrid_pso_tsp
from algorithms.GA import genetic_alg 
from sklearn.preprocessing import MinMaxScaler
import time 
import numpy as np 
import pandas as pd 

class Solver:
    def __init__(self, instace_name):
        self.instance_name = instace_name
        
        # Lazy Loading
        self._is_loaded = False
        self.best_cost = None
        self.tsp_instance = None
        self.num_nodes = None
        self.graph = None
        self.matrix = None
        
        # Internal State for Results
        self.raw_data = None
        self.results_df = None

    def _load_data(self):
        if not self._is_loaded:
            print(f"Loading TSP instance: {self.instance_name}...")
            self.tsp_instance = TPSLIBLoader(self.instance_name)
            self.num_nodes = len(self.tsp_instance.nodes)
            self.graph = self.tsp_instance.to_graph()
            self.matrix = self.tsp_instance.to_matrix()
            self.best_cost = self.tsp_instance.get_solution
            self._is_loaded = True        
            print(f'Best known cost: {self.best_cost}')

    # Solve TSP using multiple algorithms and record performance
    def solve(self,iterations=10):
        
        self._load_data()

        data = {
            'HSFFPKO': [],
            'EHO': [],
            'ACO': [],
            'SA' : [],
            'SAI': [],
            'GWO': [],
            'PSO': [],
            'GA': []
        }

        N = iterations # Number of runs for averaging
        print(f"Starting TSP solving for {N} iterations...")
        for _ in range(N):

            # take the time for 1 iteration
            iter_time_start = time.perf_counter()

            start = time.perf_counter()
            output = HSFFPKO(
                Popsize=50,
                Maxiteration=500,
                LB=np.zeros(self.num_nodes),
                UB=np.ones(self.num_nodes) * 20,
                Dim=self.num_nodes,
                Fobj=universal_tsp_wrapper(self.matrix),  # MATRIX
                scouts_rate=0.15,
                num_flocks=8
            )[0] 
            end = time.perf_counter()
            data['HSFFPKO'].append([output,end - start])
            
            start = time.perf_counter()
            output = elephant_herd_optimization(self.graph)[1]
            end = time.perf_counter()
            data['EHO'].append([output,end - start])

            start = time.perf_counter()
            output = ant_colony(self.graph)[1]
            end = time.perf_counter()
            data['ACO'].append([output,end - start])

            start = time.perf_counter()
            output = simulated_annealing(self.graph, temperature=1000, alpha=0.995, iterations=1000)[1]
            end = time.perf_counter()
            data['SA'].append([output,end - start])

            start = time.perf_counter()
            output = simulated_annealing_improved(self.graph, iter_per_temp=100)[0]
            end = time.perf_counter()
            data['SAI'].append([output,end - start])

            gwo = GreyWolfTSP(self.graph)
            start = time.perf_counter()
            output = gwo.optimize()[1]
            end = time.perf_counter()
            data['GWO'].append([output,end - start])

            start = time.perf_counter()
            output = hybrid_pso_tsp(self.graph)[1]
            end = time.perf_counter()
            data['PSO'].append([output,end - start])

            start = time.perf_counter()
            output = genetic_alg(self.graph, pop_size=100, mutation_rate=0.01, generations=50)[1]
            end = time.perf_counter()
            data['GA'].append([output,end - start])

            iter_time_end = time.perf_counter()

            # get ETA
            elapsed = iter_time_end - iter_time_start
            eta = elapsed * (N - _ - 1)
            print(f'Progress: {(_ + 1) / N * 100:.0f}% | Elapsed Time: {elapsed:.2f}s | ETA: {eta/60:.2f} min')

        self.raw_data = data
        print("All algorithms have been executed.")
        return self 
    
    # Present results in a DataFrame
    def present_results(self):

        if self.raw_data is None:
            raise ValueError("No raw data available. Please run the solve() method first.")

        cost_dict = {k: [x[0] for x in v] for k, v in self.raw_data.items()}
        df_cost = pd.DataFrame(cost_dict)

        runtime_dict = {k: [x[1] for x in v] for k, v in self.raw_data.items()}
        df_runtime = pd.DataFrame(runtime_dict)

        # Get the relative error
        def relative_error(costs, best_cost):
            return (costs - best_cost) / best_cost

        rel_error_dict = {k: relative_error(np.array(v), self.best_cost) for k, v in cost_dict.items()}
        df_rel_error = pd.DataFrame(rel_error_dict)

        cols = ['HSFFPKO', 'EHO', 'ACO', 'SA', 'SAI', 'GWO', 'PSO', 'GA']

        metrics = [
            df_cost.mean(),
            df_runtime.mean(),
            df_cost.min(),
            df_rel_error.mean() * 100,
            df_rel_error.min() * 100
        ]

        index_names = [
            'Avg Cost',
            'Avg Runtime',
            'Min Cost',
            'Avg Rel Error',
            'Min Rel Error'
        ]

        df = pd.DataFrame(np.vstack(metrics),columns=cols,index=index_names).transpose()

        df.sort_values(by='Avg Cost', inplace=True)

        self.results_df = df
        return df
    
    # Weighted score calculation
    def weighted_score(self, weight_speed: float):
        
        if self.results_df is None:
            raise ValueError("No results DataFrame available. Please run the present_results() method first.")
        
        df = self.results_df.copy()

        scaler = MinMaxScaler()
        df[['Norm Rel Error', 'Norm Runtime']] = scaler.fit_transform(df[['Avg Rel Error', 'Avg Runtime']])

        w_speed = weight_speed
        w_accuracy = 1 - weight_speed

        df['Score'] = (df['Norm Runtime'] * w_speed) + (df['Norm Rel Error'] * w_accuracy)

        return df[['Score']].sort_values(by='Score')
        
