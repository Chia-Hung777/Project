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
import time

# 優化目標函數
def objective_function(pgc_matrix, generator_powers, customer_demands):
    num_generators = generator_powers.shape[1]  # 發電機數量    
    num_customers = customer_demands.shape[1]   # 客戶數量
    num_scenarios = generator_powers.shape[0]   # 時段數量或數據點數量

    total_objective_value = 0
    # 計算所有數據點(時段)的目標函數值
    for scenario in range(num_scenarios):
        gen_powers = generator_powers[scenario, :]
        cust_demands = customer_demands[scenario, :]
        power_distribution = np.dot(gen_powers, pgc_matrix)  # 發電機功率與生成矩陣相乘
        demand_diff = cust_demands - power_distribution      # 計算需求差異
        total_objective_value += np.sum(np.abs(demand_diff)) # 累加絕對差值

    return total_objective_value

# 定義搜索範圍和問題參數
num_generators = 5
num_customers = 4

# 生成隨機矩陣
random_matrix = np.random.rand(num_generators, num_customers)


start_time = time.time()

# 計算目標函數值
objective_value = objective_function(random_matrix, generator_powers, customer_demands)


end_time = time.time()

# 輸出原始隨機矩陣及其目標函數值
print("Generated Random Matrix:")
print(random_matrix)
print("Objective value (without normalization):", objective_value)
print("Execution time: {:.2f} seconds".format(end_time - start_time))

# 對隨機矩陣進行正規化
row_sums = random_matrix.sum(axis=1)
normalized_matrix = random_matrix / row_sums[:, np.newaxis]

# 將正規化後的矩陣小數點限制到指定位數
decimal_places = 2
normalized_matrix = np.round(normalized_matrix, decimals=decimal_places)


print("Normalized Matrix (rounded to 2 decimal places):")
print(normalized_matrix)


normalized_objective_value = objective_function(normalized_matrix, generator_powers, customer_demands)

print("Objective value (with normalization):", normalized_objective_value)