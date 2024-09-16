# import numpy as np
# import csv
# import pandas as pd


# # In[2]:


# file1_path = r'C:\Users\user\Downloads\GDA7.csv'
# df1 = pd.read_csv(file1_path)
# print(df1)


# # In[3]:


# file2_path = r'C:\Users\user\Downloads\GGS7.csv'
# df2 = pd.read_csv(file2_path)
# print(df2)


# # In[4]:


# file3_path = r'C:\Users\user\Downloads\GTW7.csv'
# df3 = pd.read_csv(file3_path)
# print(df3)


# # In[5]:


# file4_path = r'C:\Users\user\Downloads\GDA8.csv'
# df4 = pd.read_csv(file4_path)
# print(df4)


# # In[6]:


# file5_path = r'C:\Users\user\Downloads\GGS8.csv'
# df5 = pd.read_csv(file5_path)
# print(df5)


# # In[7]:


# kwh1 = df1['kWh'].tolist()
# kwh2 = df2['kwh'].tolist()
# kwh3 = df3['kwh'].tolist()
# kwh4 = df4['kwh'].tolist()
# kwh5 = df5['kwh'].tolist()


# # In[8]:


# num_groups = min(len(kwh1), len(kwh2), len(kwh3), len(kwh4), len(kwh5))
# matrix = [[kwh1[i], kwh2[i], kwh3[i], kwh4[i], kwh5[i] ] for i in range(num_groups)]

# generator_df = pd.DataFrame(matrix, columns=['G1', 'G2', 'G3', 'G4', 'G5'])

# print(generator_df)


# # In[9]:


# file6_path = r'C:\Users\user\Downloads\CYT7.csv'
# df6 = pd.read_csv(file6_path)
# print(df6)


# # In[10]:


# file7_path = r'C:\Users\user\Downloads\CYB7.csv'
# df7 = pd.read_csv(file7_path)
# print(df7)


# # In[11]:


# file8_path = r'C:\Users\user\Downloads\CYT8.csv'
# df8 = pd.read_csv(file8_path)
# print(df8)


# # In[12]:


# file9_path = r'C:\Users\user\Downloads\CYB8.csv'
# df9 = pd.read_csv(file9_path)
# print(df9)


# # In[13]:


# kwh6 = df6['load'].tolist()
# kwh7 = df7['load'].tolist()
# kwh8 = df8['load'].tolist()
# kwh9 = df9['load'].tolist()
# num_groups = min(len(kwh4), len(kwh5))
# matrix = [(kwh6[i], kwh7[i], kwh8[i], kwh9[i]) for i in range(num_groups)]

# client_df = pd.DataFrame(matrix, columns=['C1', 'C2', 'C3', 'C4'])

# print(client_df)

# In[14]:


import numpy as np
import time
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt

# 將 DataFrame 轉換為 NumPy 數組
customer_demands = client_df.values
generator_powers = generator_df.values


def objective_function(individual, generator_powers, customer_demands, penalty_factor=6000):
    num_generators = generator_powers.shape[1]
    num_customers = customer_demands.shape[1]
    num_scenarios = generator_powers.shape[0]

    pgc_matrix_flat = np.array(individual)
    pgc_matrix = pgc_matrix_flat.reshape((num_generators, num_customers))

    total_objective_value = 0
    for scenario in range(num_scenarios):
        gen_powers = generator_powers[scenario, :]
        cust_demands = customer_demands[scenario, :]

        power_distribution = np.dot(gen_powers, pgc_matrix)
        demand_diff = cust_demands - power_distribution

        total_objective_value += np.sum(np.abs(demand_diff))

    row_sums = np.sum(pgc_matrix, axis=1)
    penalty = penalty_factor * np.sum(np.abs(row_sums - 1))

    return total_objective_value + penalty,


num_generators = generator_powers.shape[1]
num_customers = customer_demands.shape[1]


if hasattr(creator, "FitnessMin"):
    del creator.FitnessMin
if hasattr(creator, "Individual"):
    del creator.Individual


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.uniform, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, num_generators * num_customers)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", objective_function, generator_powers=generator_powers, customer_demands=customer_demands, penalty_factor=6000)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)

# GA的參數
num_particles = 70  
max_iter = 130  
cxpb = 0.8  
mutpb = 0.3  

# 收斂條件
convergence_threshold = 1e-1  
convergence_patience = 50  
stagnation_counter = 0  


fitness_values = []

def record_stats(population):
    fitnesses = [ind.fitness.values[0] for ind in population]
    fitness_values.append(min(fitnesses))

population = toolbox.population(n=num_particles)
start_time = time.time()
previous_best = None

for gen in range(max_iter):
    offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
    fits = toolbox.map(toolbox.evaluate, offspring)
    
    for fit, ind in zip(fits, offspring):
        ind.fitness.values = fit
    
    population = toolbox.select(offspring, k=len(population))
    
    
    record_stats(population)
    
    
    current_best = fitness_values[-1]
    if previous_best is not None and abs(previous_best - current_best) < convergence_threshold:
        stagnation_counter += 1
    else:
        stagnation_counter = 0

    if stagnation_counter >= convergence_patience:
        print(f"Algorithm converged after {gen+1} generations.")
        break

    previous_best = current_best

end_time = time.time()

# 繪製適應度值變化曲線
plt.plot(fitness_values)
plt.xlabel('Generation')
plt.ylabel('Fitness Value')
plt.title('Fitness Value over Generations')
plt.show()

# 找到最佳解
best_individual = tools.selBest(population, k=1)[0]
pgc_matrix_optimal = np.array(best_individual).reshape((num_generators, num_customers))

# 對最終顯示的最佳解矩陣進行正規化
pgc_matrix_optimal = np.clip(pgc_matrix_optimal, 0, None)  
row_sums = np.sum(pgc_matrix_optimal, axis=1, keepdims=True)
pgc_matrix_optimal = pgc_matrix_optimal / row_sums

print("Optimal pGC matrix :")
print(pgc_matrix_optimal)
print("Objective value:", objective_function(best_individual, generator_powers, customer_demands, 6000)[0])
print("Execution time: {:.2f} seconds".format(end_time - start_time))


