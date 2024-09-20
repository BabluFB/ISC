import numpy as np
import heapq
from typing import List, Tuple, Dict
import random
import time
import osqp
from scipy import sparse
import math
from mrg32k3a.mrg32k3a import MRG32k3a

class Solution:
    def __init__(self, position: List[int]):
        self.position = position
        self.cost = float('inf')
        self.is_redundant = False
        self.repetitions = 0
        self.len = len(position)
        self.random_state = None

    def evaluate(self, eval_function, n=10, random_state = 1234):
        if random_state is not None:
            self.random_state = random_state
        self.cost = eval_function(self.position, n, self.random_state)
        self.repetitions += 1

    def __lt__(self, other):
        return self.cost < other.cost

class Population:
    def __init__(self):
        self.solutions = []
        self.best_solution = None

    def add(self, solution: Solution):
        self.solutions.append(solution)
        if self.best_solution is None or solution < self.best_solution:
            self.best_solution = solution

    def get_positions(self) -> List[List[int]]:
        return [sol.position for sol in self.solutions]

    def update_best(self):
        self.best_solution = min(self.solutions, key=lambda x: x.cost)

class StochasticOptimizer:
    def __init__(self, eval_function, bounds, dimension):
        self.eval_function = eval_function
        self.bounds = bounds
        self.dimension = dimension
        self.gas = 0
        self.rng = MRG32k3a(s_ss_sss_index=[0, 0, 0])  # Main stream for create_solution and get_neighbors

    def create_solution(self) -> Solution:
        position = [self.rng.randint(self.bounds[0], self.bounds[1]) for _ in range(self.dimension)]
        return Solution(position)

    def get_neighbors(self, solution: Solution, n: int, step_size: float) -> List[Solution]:
        neighbors = []
        for _ in range(n):
            # Fix: Call normalvariate for each dimension separately
            reach = np.floor([step_size * self.rng.normalvariate(0, 1) for _ in range(self.dimension)])
            new_position = np.clip(np.array(solution.position) + reach, self.bounds[0], self.bounds[1])
            new_position = np.round(new_position).astype(int)
            neighbors.append(Solution(list(new_position)))
        return neighbors

class ISC(StochasticOptimizer):
    def __init__(self, eval_function, bounds, dimension, total_budget, alpha, no_of_clusters):
        super().__init__(eval_function, bounds, dimension)
        self.total_budget = total_budget
        self.alpha = alpha
        self.no_of_clusters = no_of_clusters
        self.GA_budget = total_budget * alpha
        self.compass_budget = total_budget * (1 - alpha)
        self.global_random_state = np.random.randint(0, 2**32 - 1)
        self.eval_rng = MRG32k3a(s_ss_sss_index=[1, 0, 0])  # Separate stream for evaluation
        self.evolution_rng = MRG32k3a(s_ss_sss_index=[2, 0, 0])  # Stream for evolution

    def SAR(self,k: int) -> int:
        n0 = 8
        return math.ceil(n0 * (np.log(k)**2))

    def initialize(self, mg: int) -> Tuple[Dict, int, float, Dict, Dict]:
        unique_set = {}
        while len(unique_set) < mg:
            temp_lower_bound = int((self.bounds[0]*3/4) + (self.bounds[1]*1/4))
            temp_upper_bound = int((self.bounds[0]*1/4) + (self.bounds[1]*3/4))
            genome = self.create_solution().position
            genome_tuple = tuple(genome)
            if genome_tuple in unique_set:
                unique_set[genome_tuple] = min(unique_set[genome_tuple], self.eval_function(genome, random_state=self.eval_rng))
            else:
                unique_set[genome_tuple] = self.eval_function(genome, random_state=self.eval_rng)
        
        sorted_items = sorted(unique_set.items(), key=lambda x: x[1])
        niche_centers = sorted_items[:self.no_of_clusters]
        sol_space = sorted_items[self.no_of_clusters:]
        
        centers = [list(center[0]) for center in niche_centers]
        clusters = {tuple(center): [] for center in centers}
        
        for sol in sol_space:
            closest_center = min(centers, key=lambda c: self.euclidean_distance(c, sol[0]))
            clusters[tuple(closest_center)].append(list(sol[0]))
        
        best_center = centers[0]
        farthest_point, min_radius = self.find_farthest_point(best_center, centers[1:])
        
        for center in centers[1:]:
            if self.euclidean_distance(center, best_center) < 0.5*min_radius:
                clusters[tuple(best_center)].extend(clusters[tuple(center)])
                clusters.pop(tuple(center))
        
        q = len(clusters)
        r = min(self.euclidean_distance(center, best_center) for center in centers[1:]) / 2
        
        sol_vals_dict = {tuple(sol): unique_set[tuple(sol)] for center in clusters for sol in clusters[tuple(center)]}
        center_vals = {tuple(center): self.eval_function(center, random_state=self.eval_rng) for center in centers}
        
        return clusters, q, r, sol_vals_dict, center_vals

    @staticmethod
    def euclidean_distance(a: List[int], b: List[int]) -> float:
        return np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def find_farthest_point(self, center: List[int], points: List[List[int]]) -> Tuple[List[int], float]:
        max_distance = 0
        farthest_point = None
        for point in points:
            distance = self.euclidean_distance(center, point)
            if distance > max_distance:
                max_distance = distance
                farthest_point = point
        return farthest_point, max_distance

    def closeset_centre(self,centers:List[List[int]],visited_solution:List[int]) -> List[int]:
        closest = [float('inf') for _ in centers]
        min_dist= float('inf')
        for center in centers:
            dist =self.euclidean_distance(center,visited_solution)
            if dist < min_dist:
                min_dist = dist
                closest = center
        return closest

    def evolution(self, clusters, sol_vals_dict, center_vals_dict):
        centers = [list(center) for center in center_vals_dict.keys()]
        center_vals = list(center_vals_dict.values())
        k =2
        while True:
            sorted_items = sorted(sol_vals_dict.items(), key=lambda x: x[1])
            other_genomes = [list(genome[0]) for genome in sorted_items]
            
            m = len(other_genomes)
            sol_val_new = {}
            sol_space = []
            unique_set = set()
            for _ in range(m):
                i = self.evolution_rng.randint(0, m-1)
                parent1 = other_genomes[i]
                parent2 = self.get_mate(parent1, sol_vals_dict)
                child1, child2 = self.single_point_crossover(parent1, parent2)
                cost_1 = self.eval_function(child1, random_state=self.eval_rng)
                self.gas += self.SAR(k)
                cost_2 = self.eval_function(child2, random_state=self.eval_rng)
                self.gas += self.SAR(k)
                add_1_to_sol = True
                add_2_to_sol = True
                unique_set.add(tuple(child1))
                unique_set.add(tuple(child2))
                for j in range(len(center_vals)):
                    if center_vals[j] > cost_1:
                        add_1_to_sol = False
                        centers[j] = child1
                        center_vals[j] = cost_1
                    if center_vals[j] > cost_2:
                        centers[j] = child2
                        center_vals[j] = cost_2
                        add_2_to_sol = False
                if add_1_to_sol:
                    sol_val_new[tuple(child1)] = cost_1
                    sol_space.append(child1)
                if add_2_to_sol:
                    sol_val_new[tuple(child2)] = cost_2
                    sol_space.append(child2)
            centers1 = [tuple(center) for center in centers]
            clusters ={center:[] for center in centers1}
            for sol in sol_space:
                clusters[tuple(self.closeset_centre(centers1,sol))].append(sol)
            best_center =  centers1[0]
            sol_space = clusters[best_center]
            farthest_point , min_radius = self.find_farthest_point(best_center,centers[1:])
            for center in centers[1:]:
                if self.euclidean_distance(center,best_center) < 0.5*min_radius:
                    if tuple(center) in clusters:
                        if len(clusters[tuple(center)]) >0:
                            if tuple(center) in clusters:
                                clusters[tuple(best_center)].append(clusters.pop(tuple(center))[0])
            k+=1
            sol_vals_dict = sol_val_new
            if len(centers) == 1:
                print("Only one center is present")
                break
            if self.gas > self.GA_budget:
                print('GA Budget over')
                break
            
            sol_vals_dict = sol_val_new
        print("\nClusters formed after evolution:")
        for i, (center, solutions) in enumerate(clusters.items()):
            print(f"Cluster {i + 1}:")
            print(f"  Center: {list(center)}")
            print(f"  Number of solutions: {len(solutions)}")
            if solutions:
                best_solution = min(solutions, key=lambda x: sol_vals_dict[tuple(x)])
                print(f"  Best solution: {best_solution}")
                print(f"  Best solution cost: {sol_vals_dict[tuple(best_solution)]}")
            print()

        return clusters

    def select_random_pair_in_range(self,dictionary: Dict[Tuple[int], int], lower_bound: int  , upper_bound: int ):
        eligible_pairs = [(key, value) for key, value in dictionary.items() 
                          if lower_bound <= value <= upper_bound]
        if not eligible_pairs:
            return None

        return eligible_pairs[self.evolution_rng.randint(0, len(eligible_pairs)-1)][0]

    def get_mate(self, genome, sol_val_dict):
        temp_dict = sol_val_dict.copy()
        cost_value = temp_dict[tuple(genome)]
        del temp_dict[tuple(genome)]
        beta = 0.1
        mate = self.select_random_pair_in_range(temp_dict, (1 - beta) * cost_value, (1 + beta) * cost_value)
        counter= 0 
        while not mate or list(mate) == genome:
            if counter > 10:
                mate = genome
                break
            beta += 0.1
            mate = self.select_random_pair_in_range(temp_dict, (1 - beta) * cost_value, (1 + beta) * cost_value)
            counter +=1

        return list(mate)

    def single_point_crossover(self, parent1, parent2):
            size = len(parent1)
            parent1 = tuple(parent1)
            parent2 = tuple(parent2)
            crossover_point = self.evolution_rng.randint(1, size-1)

            child1 = list(parent1[:crossover_point] + tuple([-1] * (size - crossover_point)))
            pointer1 = crossover_point
            for gene in parent2[crossover_point:] + parent2[:crossover_point]:
                if gene not in child1:
                    child1[pointer1] = gene
                    pointer1 += 1
                    if pointer1 == size:
                        pointer1 = 0

            child2 = list(parent2[:crossover_point] + tuple([-1] * (size - crossover_point)))
            pointer2 = crossover_point
            for gene in parent1[crossover_point:] + parent1[:crossover_point]:
                if gene not in child2:
                    child2[pointer2] = gene
                    pointer2 += 1
                    if pointer2 == size:
                        pointer2 = 0

            return child1, child2

    def run(self):
        print("Starting")
        clusters, q, r, sol_vals_dict, center_vals = self.initialize(50)
        final_cluster_set = self.evolution(clusters, sol_vals_dict, center_vals)
        print('GA Done')
        centers = [list(center) for center in final_cluster_set.keys()]
        final_clusters = []
        cost_vals = []

        for i, local_cluster in enumerate(final_cluster_set.values()):
            local_cluster = local_cluster[:2]
            local_cluster.append(centers[i])
            final_clusters.append(local_cluster)
            cost_vals.append(self.eval_function(centers[i], random_state=self.eval_rng))

        fitness_vals = self.fitness_function(cost_vals)

        outputs = []
        compass = COMPASS(self.eval_function, self.bounds, self.dimension)
        print("\nBest solutions after COMPASS optimization:")
        for i, cluster in enumerate(final_clusters):
            if fitness_vals[i] < 0.05:
                print('Skipping cluster due to low fitness value')
                continue
            compass_budget = self.compass_budget * fitness_vals[i]
            best_solution, best_cost = compass.simulate(cluster, compass_budget)
            outputs.append(compass.simulate(cluster, compass_budget))
            print(f"Cluster {i+1}:")
            print(f"  Best solution: {best_solution}")
            print(f"  Best cost: {best_cost}")
            print()
        overall_best = min(outputs, key=lambda x: x[1])
        print("\nOverall best solution:")
        print(f"  Solution: {overall_best[0]}")
        print(f"  Cost: {overall_best[1]}")


        return overall_best

    @staticmethod
    def fitness_function(values):
        raw_fitness = [1 / (value + 1e-6) for value in values]
        total_fitness = sum(raw_fitness)
        return [f / total_fitness for f in raw_fitness]

class COMPASS(StochasticOptimizer):
    def __init__(self, eval_function, bounds, dimension):
        super().__init__(eval_function, bounds, dimension)
        self.step_size = 10
        self.step_size_param = 10
        self.global_random_state = np.random.randint(0, 2**32 - 1)
        self.compass_rng = MRG32k3a(s_ss_sss_index=[3, 0, 0])  # Stream for COMPASS-specific operations
        self.eval_rng = MRG32k3a(s_ss_sss_index=[4, 0, 0])  # Stream for evaluation

    def simulate(self, population: List[List[int]], budget: float) -> Tuple[List[int], float]:
        pop = Population()
        for sol in population:
            solution = Solution(sol)
            solution.evaluate(self.eval_function, random_state=self.eval_rng)
            pop.add(solution)
        
        self.gas = 0
        k = 2
        
        while self.gas < budget:
            mp_area = self.get_most_promising_area(pop.best_solution, pop)
            V_k = [pop.best_solution] + random.sample(mp_area, min(40, len(mp_area)))
            
            for sol in V_k:
                sol.evaluate(self.eval_function, random_state=self.eval_rng)
                self.gas += self.SAR(k)
            
            pop.solutions = V_k
            pop.update_best()
            redundant_vars =self.check_redundancy(pop.best_solution,V_k)
            pop.solutions = [pop.best_solution] + [sol for sol, red in zip(pop.solutions[1:], redundant_vars) if red]
            
            k += 1
            
            if self.gas >= 0.8 * budget:
                self.step_size = 1
            elif self.gas >= 0.7 * budget:
                self.step_size = self.step_size_param // 2
            elif self.gas >= 0.6 * budget:
                self.step_size = (3 * self.step_size_param) // 4
        
        return pop.best_solution.position, pop.best_solution.cost

    def get_most_promising_area(self, x_star: Solution, population: Population) -> List[Solution]:
        neighbors = self.get_neighbors(x_star, 40, self.step_size)
        closest_keys = self.find_closest_keys(x_star, population.solutions, 10)
        
        mp_area = []
        for sol in neighbors:
            sol_distance = self.euclidean_distance(x_star.position, sol.position)
            is_closer = all(sol_distance < self.euclidean_distance(sol.position, key.position) for key in closest_keys)
            if is_closer:
                mp_area.append(sol)
        
        return mp_area

    @staticmethod
    def euclidean_distance(a: List[int], b: List[int]) -> float:
        return np.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

    def find_closest_keys(self, x_star: Solution, solutions: List[Solution], n: int) -> List[Solution]:
        distances = [(sol, self.euclidean_distance(x_star.position, sol.position)) for sol in solutions if sol != x_star]
        return [key for key, _ in sorted(distances, key=lambda x: x[1])[:n]]

    def check_redundancy_osqp(self, x_star: Solution, V_k: List[Solution]) -> List[bool]:
        x_star_k = np.array(x_star.position)
        redundancy_status = []
        
        for x_i in V_k:
            x_i = np.array(x_i.position)
            n = len(x_star_k)
            
            diff = (x_star_k - x_i)
            P = sparse.diags([1.0] * n)
            q = -diff @ (x_star_k + x_i) / 2
            
            A_data = []
            l_data = []
            
            for x_j in V_k:
                if not np.array_equal(np.array(x_j.position), x_i):
                    x_j = np.array(x_j.position)
                    midpoint_j = (x_star_k + x_j) / 2
                    constraint = (x_star_k - x_j)
                    A_data.append(constraint)
                    l_data.append(midpoint_j @ constraint)
            
            A = sparse.vstack([sparse.csc_matrix(np.array(A_data))])
            l = np.array(l_data)
            u = np.inf * np.ones_like(l)
            
            solver = osqp.OSQP()
            solver.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
            
            res = solver.solve()
            
            if res.info.status == 'solved':
                obj_value = res.info.obj_val
                redundancy_status.append(obj_value >= 0)
            else:
                redundancy_status.append(False)
        
        return redundancy_status
    def check_redundancy(self,x_star_k : List[int], V_k:List[List[int]]):
        from gurobipy import GRB
        import gurobipy as gp
        n = x_star_k.len
        x_star_k = np.array(x_star_k.position)
        redundancy_status = []
        for x_i in V_k:
            x_i = np.array(x_i.position)
            model = gp.Model()
            model.setParam('OutputFlag', 0)
            x = model.addMVar(shape=n, vtype=GRB.CONTINUOUS, name="x")

            midpoint_i = (x_star_k + x_i) / 2
            diff = x_star_k - x_i

            objective = gp.LinExpr()
            for j in range(n):
                objective += diff[j] * (x[j] - midpoint_i[j])
            model.setObjective(objective, GRB.MINIMIZE)

            for x_j in V_k:
                if not np.array_equal(np.array(x_j), x_i):

                    x_j = np.array(x_j.position)
                    midpoint_j = (x_star_k + x_j) / 2
                    constraint = gp.LinExpr()
                    for j in range(n):
                        constraint += (x_star_k[j] - x_j[j]) * (x[j] - midpoint_j[j])
                    model.addConstr(constraint >= 0)

            model.optimize()

            if model.status == GRB.OPTIMAL:
                obj_value = model.ObjVal
                redundancy_status.append(obj_value >= 0)
            else:
                redundancy_status.append(False)

        return redundancy_status

    @staticmethod
    def SAR(k: int) -> int:
        n0 = 8
        return math.ceil(n0 * (np.log(k)**2))

def stochastic_cost_function(x: List[int], n: int = 10, random_state = None) -> float:
    if isinstance(random_state, MRG32k3a):
        rng = random_state
    else:
        rng = MRG32k3a(s_ss_sss_index=[4, 0, 0])  # Default stream for cost function
    
    result = np.mean([stochastic_cost_function_helper(x, rng) for _ in range(n)])
    
    # Increment the substream
    rng.advance_subsubstream()
    
    return result

def stochastic_cost_function_helper(x: list[int], rng: MRG32k3a):
    x = np.array(x)
    x1, x2, x3, x4 = x
    result = (x1 + 10 * x2)**2 + 5 * (x3 - x4)**2 + (x2 - 2 * x3)**4 + 10 * (x1 - x4)**4 + 1
    noise = rng.normalvariate(0, np.sqrt(result))
    return result + noise

bounds = (-100, 100)
dimension = 4
total_budget = 6e5
alpha = 0.7
no_of_clusters = 2
isc = ISC(stochastic_cost_function, bounds, dimension, total_budget, alpha, no_of_clusters)
time_curr = time.time()
best_solution, best_cost = isc.run()
print(f'Time taken for ISC: {time.time()-time_curr}')