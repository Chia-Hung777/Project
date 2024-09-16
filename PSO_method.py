import numpy as np
import matplotlib.pyplot as plt
from pyswarm import pso
import time


global_best_fitness = []
main_iteration_counter = 0  
iteration_counter = 0  
convergence_iteration = None  
convergence_threshold = 1e-2  
convergence_patience = 50  
stagnation_counter = 0  

# 優化目標函數
def objective_function(pgc_matrix_flat, generator_powers, customer_demands, penalty_factor=6000):
    num_generators = generator_powers.shape[1]  # 發電機數量    
    num_customers = customer_demands.shape[1]   # 客戶數量
    num_scenarios = generator_powers.shape[0]   # 時段數量或數據點數量

    # 重塑 pgc_matrix_flat 為合適的形狀
    pgc_matrix = pgc_matrix_flat.reshape((num_generators, num_customers))

    total_objective_value = 0
    # 計算所有數據點(時段)的目標函數值
    for scenario in range(num_scenarios):
        gen_powers = generator_powers[scenario, :]
        cust_demands = customer_demands[scenario, :]
        power_distribution = np.dot(gen_powers, pgc_matrix)
        demand_diff = cust_demands - power_distribution
        total_objective_value += np.sum(np.abs(demand_diff))

    row_sums = np.sum(pgc_matrix, axis=1)
    penalty = penalty_factor * np.sum(np.abs(row_sums - 1))

    return total_objective_value + penalty

def objective_function_wrapper(pgc_matrix_flat, generator_powers, customer_demands, penalty_factor):
    global iteration_counter
    global main_iteration_counter
    global convergence_iteration
    global stagnation_counter

    fitness = objective_function(pgc_matrix_flat, generator_powers, customer_demands, penalty_factor)
    global_best_fitness.append(fitness)  
    iteration_counter += 1  # 記錄總迭代次數

    
    if iteration_counter % num_particles == 0:
        main_iteration_counter += 1

    
    if iteration_counter % 100 == 0:
        print(f"Iteration {iteration_counter}, Fitness: {fitness}")

    # 檢查是否收斂
    if convergence_iteration is None and len(global_best_fitness) > 1:
        if abs(global_best_fitness[-1] - global_best_fitness[-2]) < convergence_threshold:
            stagnation_counter += 1
            print(f"Stagnation count: {stagnation_counter} at iteration {iteration_counter}")
            if stagnation_counter >= convergence_patience:
                convergence_iteration = main_iteration_counter  # 修改為主要迭代次數
        else:
            stagnation_counter = 0

    return fitness


num_generators = 5
num_customers = 4
lb = np.zeros((num_generators, num_customers)).flatten()
ub = np.ones((num_generators, num_customers)).flatten()
num_particles = 100
max_iter = 500
generator_powers = np.array(Generator)  
customer_demands = np.array(Client) 


start_time = time.time()

# 執行PSO算法
pgc_matrix_optimal, fopt = pso(
    objective_function_wrapper, lb, ub,
    args=(generator_powers, customer_demands, 6000),
    swarmsize=num_particles, maxiter=max_iter, debug=True
)

# 將扁平化的結果轉換成矩陣形式
pgc_matrix_optimal = pgc_matrix_optimal.reshape((num_generators, num_customers))
end_time = time.time()

print("Optimal pGC matrix:")
print(pgc_matrix_optimal)
print("Objective value:", fopt)
print("Execution time: {:.2f} seconds".format(end_time - start_time))
print("Total iterations:", iteration_counter)
print("Main iterations:", main_iteration_counter)
print("Convergence iterations (main):", convergence_iteration if convergence_iteration is not None else "Not converged")




# 繪製迭代次數和適應度值的圖形
plt.figure(figsize=(10, 6))
plt.plot(global_best_fitness, label='Best Fitness Value')
# if convergence_iteration is not None:
#         plt.axvline(x=convergence_iteration * num_particles, color='r', linestyle='--', label=f'Converged at main iteration {convergence_iteration}')
plt.xlabel('Iteration')
plt.ylabel('Fitness Value')
plt.title('PSO Optimization Progress')
plt.legend()
plt.grid(True)
plt.show()
