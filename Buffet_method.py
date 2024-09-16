# # In[1]:


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


import pyswarms as pso
import numpy as np
# 第一個DF包含列 'G1', 'G2', 'G3' 的發電機容量
generator_powers = generator_df[['G1', 'G2', 'G3', 'G4', 'G5']].values

# 第二個DF包含列 'C1', 'C2' 的客戶需求
customer_demands = client_df[['C1', 'C2', 'C3', 'C4']].values
import numpy as np
from scipy.optimize import minimize
import time

# 優化目標函數
def objective_function(pgc_matrix_flat, generator_powers, customer_demands, previous_pgc_matrix, penalty_factor=6000):
    num_generators = generator_powers.shape[0]  # 發電機數量    
    num_customers = customer_demands.shape[0]   # 客戶數量

    # 重塑 pgc_matrix_flat 為合適的形狀
    pgc_matrix = pgc_matrix_flat.reshape((num_generators, num_customers))

    # 計算電力分配，使用上一時段的分配矩陣
    if previous_pgc_matrix is not None:
        power_distribution = np.dot(generator_powers, previous_pgc_matrix)
    else:
        power_distribution = np.zeros(num_customers)

    
    demand_diff = customer_demands - power_distribution

    # 計算總差異的絕對值和
    total_objective_value = np.sum(np.abs(demand_diff))

    # 使用當前時段的分配矩陣計算連續性差異
    continuity_diff = np.dot(generator_powers, pgc_matrix) - power_distribution
    total_objective_value += np.sum(np.abs(continuity_diff))

    
    row_sums = np.sum(pgc_matrix, axis=1)
    penalty = penalty_factor * np.sum(np.abs(row_sums - 1))
    total_objective_value += penalty

    return total_objective_value


def objective_function_wrapper(pgc_matrix_flat, generator_powers, customer_demands, previous_pgc_matrix, penalty_factor):
    return objective_function(pgc_matrix_flat, generator_powers, customer_demands, previous_pgc_matrix, penalty_factor)


num_generators = 5
num_customers = 4

generator_powers = np.array(generator_powers)
customer_demands = np.array(customer_demands)


x0 = np.random.rand(num_generators * num_customers)
bounds = [(0, 1) for _ in range(num_generators * num_customers)]


previous_pgc_matrix = None  


optimal_pgcs = []


total_objective_value = 0


start_time = time.time()

# 逐時段優化
for t in range(generator_powers.shape[0]):
    print(f"Optimizing for period {t + 1}")

    period_start_time = time.time()  

    result = minimize(
        objective_function_wrapper, x0, 
        args=(generator_powers[t], customer_demands[t], previous_pgc_matrix, 6000),
        bounds=bounds, method='L-BFGS-B', options={'disp': True}
    )

    period_end_time = time.time()  
    period_duration = period_end_time - period_start_time
    print(f"Time taken for period {t + 1}: {period_duration:.4f} seconds")

    
    pgc_matrix_optimal = result.x.reshape((num_generators, num_customers))
    optimal_pgcs.append(pgc_matrix_optimal)

    # 使用上一時段的最佳分配矩陣計算當前時段的目標函數值並累加
    current_power_distribution = np.dot(generator_powers[t], pgc_matrix_optimal)
    current_objective_value = objective_function(result.x, generator_powers[t], customer_demands[t], previous_pgc_matrix, 6000)
    total_objective_value += current_objective_value

    
    previous_pgc_matrix = pgc_matrix_optimal

# 計算總時間
end_time = time.time()
total_duration = end_time - start_time


#for i, pgc in enumerate(optimal_pgcs):
    #print(f"Optimal pGC matrix for period {i + 1}:")
    #print(pgc)

print(f"Total objective value: {total_objective_value}")
print(f"Total time taken: {total_duration:.4f} seconds")
