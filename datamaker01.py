import random 
import math
import copy
import matplotlib.pyplot as plt

Kw = 10
Generator = []
Client = []
FixedTime1 = 2
FixedTime2 = 10
rise = 6 * 4
stop_rise = 9 * 4
fall = 17 * 4
stop_fall = 21 * 4
Power_Sum = []
Demand_Sum = []
Client_Diffrence = 3
Client_Diffrence_Noise_Percent = 10
Residual = []
Max_Monthly_limit = 300
Min_Monthly_limit = 200

for _ in range(96):
    Client.append([])

def Read_File():
    f = open("C:/Users/gyes9/Desktop/Information.txt", "r")

    NumberOfGenerator = None
    NumberOfClient = None

    for Lines in range(1,11):
        File = f.readline()

        if Lines == 1:
            NumberOfGenerator = int(File[23:])
        elif Lines == 2:
            NumberOfClient = int(File[20:])
                
        if Lines == 3:
            GeneratePgcAmount = int(File[22:])
        if Lines == 4:
            Cross_OverAmount = int(File[21:])
        if Lines == 5:
            MutationsAmount = int(File[20:])
        if Lines == 6:
            Generation_Amount = int(File[21:])
        if Lines == 9:
            mean = float(File[9:])
        elif Lines == 10:
            sigma = float(File[11:])
        else:
            continue
    
    f.close()
    return mean, sigma, Generator, Client, NumberOfGenerator, NumberOfClient, GeneratePgcAmount, Cross_OverAmount, MutationsAmount, Generation_Amount

mean, sigma, Generator, Client, NumberOfGenerator, NumberOfClient, GeneratePgcAmount, Cross_OverAmount, MutationsAmount, Generation_Amount = Read_File()

def Make_Data_Generator():
    gauss_List = []
    gauss_List_count_dict = {}

    for _ in range(96): 
        temp = random.gauss(mean, sigma) 
        gauss_List.append(math.floor(temp)) 
    
    for gauss in gauss_List:
        gauss_count = 0

        if str(gauss) in gauss_List_count_dict:
            continue

        for Pre_gauss in gauss_List:
            if Pre_gauss == gauss:
                gauss_count += 1
        
        gauss_List_count_dict[gauss] = gauss_count

    return gauss_List_count_dict

def Make_Monthly_limit():
    Monthly_limit = []

    for _ in range(NumberOfGenerator):
        Monthly_Limit_list = []

        for _ in range(NumberOfClient):
            Monthly_Limit_list.append(random.randint(Min_Monthly_limit, Max_Monthly_limit))

        Monthly_limit.append(Monthly_Limit_list)

    return Monthly_limit

def Save_To_File(Gen, Cli):
    if Gen != None:
        if Gen == []:
            for time in range(24):
                time_List = []
                if time in gauss_List_count_dict:
                    time_List.append(gauss_List_count_dict[time] * Kw) 
                else:
                    time_List.append(0)
                Gen.append(time_List)
        else:
            for time in range(24):
                if time in gauss_List_count_dict:
                    Gen[time].append(gauss_List_count_dict[time] * Kw)
                else:
                    Gen[time].append(0)

    if Cli != None:
        During_rise = False
        During_fall = True
        Steps_Of_rise_fall = 1

        for time in range(96):
            if time != rise - 1 and not During_rise and During_fall:
                Noise = FixedTime1 * random.randint(-1 * Client_Diffrence_Noise_Percent, Client_Diffrence_Noise_Percent + 1) / 100
                Cli[time].append(round(FixedTime1 + Noise, 1))
            if time != fall - 1 and During_rise and not During_fall:
                Noise = FixedTime2 * random.randint(-1 * Client_Diffrence_Noise_Percent, Client_Diffrence_Noise_Percent + 1) / 100
                Cli[time].append(round(FixedTime2 + Noise, 1))
            if time == rise - 1:
                Steps_Of_rise_fall = 1
                During_fall = False
                During_rise = False
            if time == fall - 1:
                Steps_Of_rise_fall = stop_fall - fall - 1 
                During_fall = True
                During_rise = True
            if not During_fall and not During_rise:
                temp = FixedTime1 + ((FixedTime2 - FixedTime1) * Steps_Of_rise_fall / (stop_rise - rise))
                Noise = temp * random.randint(-1 * Client_Diffrence_Noise_Percent, Client_Diffrence_Noise_Percent + 1) / 100
                Cli[time].append(round(round(temp * 100) / 100 + Noise, 1))
                Steps_Of_rise_fall = Steps_Of_rise_fall + 1
                if Steps_Of_rise_fall / (stop_rise - rise) == 1:
                    During_rise = True
                    During_fall = False
            if During_fall and During_rise:
                temp = FixedTime1 + ((FixedTime2 - FixedTime1) * Steps_Of_rise_fall / (stop_fall - fall))
                Noise = temp * random.randint(-1 * Client_Diffrence_Noise_Percent, Client_Diffrence_Noise_Percent + 1) / 100
                Cli[time].append(round(round(temp * 100) / 100 + Noise, 1))
                Steps_Of_rise_fall = Steps_Of_rise_fall - 1
                if Steps_Of_rise_fall == 1:
                    During_rise = False
                    During_fall = True

def Data_Sum(Data_List):
    data_sum_list =[]
    data_sum = 0

    for Data_Time_List in Data_List:
        data_sum = 0

        for Data in Data_Time_List:
            data_sum += Data
        
        data_sum_list.append(data_sum)

    return data_sum_list  

def Hour_To_15Mins(Data_List):
    Index_time = -3

    for _ in range(23):
        Index_time = Index_time + 4
        time_list = []
        Data_List.insert(Index_time, copy.deepcopy(time_list))
        Data_List.insert(Index_time + 1, copy.deepcopy(time_list))
        Data_List.insert(Index_time + 2, copy.deepcopy(time_list))

    for Index_gen in range(NumberOfGenerator):
        Index_time = -4

        for _ in range(23):
            Index_time = Index_time + 4

            if Data_List[Index_time][Index_gen] <= Data_List[Index_time + 4][Index_gen]:
                Data_List[Index_time + 1].append(((Data_List[Index_time + 4][Index_gen] - Data_List[Index_time][Index_gen]) * 1/4) + Data_List[Index_time][Index_gen])
                Data_List[Index_time + 2].append(((Data_List[Index_time + 4][Index_gen] - Data_List[Index_time][Index_gen]) * 2/4) + Data_List[Index_time][Index_gen])
                Data_List[Index_time + 3].append(((Data_List[Index_time + 4][Index_gen] - Data_List[Index_time][Index_gen]) * 3/4) + Data_List[Index_time][Index_gen])
            elif Data_List[Index_time][Index_gen] > Data_List[Index_time + 4][Index_gen]:
                Data_List[Index_time + 1].append(((Data_List[Index_time][Index_gen] - Data_List[Index_time + 4][Index_gen]) * 3/4) + Data_List[Index_time + 4][Index_gen])
                Data_List[Index_time + 2].append(((Data_List[Index_time][Index_gen] - Data_List[Index_time + 4][Index_gen]) * 2/4) + Data_List[Index_time + 4][Index_gen])
                Data_List[Index_time + 3].append(((Data_List[Index_time][Index_gen] - Data_List[Index_time  + 4][Index_gen]) * 1/4) + Data_List[Index_time  + 4][Index_gen])

    leftOver = []
    for _ in range(NumberOfGenerator):
        leftOver.append(0)
        
    for _ in range(3):
        Data_List.insert(0, copy.deepcopy(leftOver))

    return Data_List

Monthly_Limit = Make_Monthly_limit()

for _ in range(NumberOfGenerator):
    gauss_List_count_dict = Make_Data_Generator()
    Save_To_File(Generator, None)
    
for _ in range(NumberOfClient):
    Save_To_File(None, Client)
    FixedTime2 += Client_Diffrence

Final_Generator = Hour_To_15Mins(Generator)

Demand_Sum = Data_Sum(Client)
Power_Sum = Data_Sum(Generator)

f = open("C:/Users/gyes9/Desktop/Information.txt", "r")
data = f.readlines()
data[6] = "7-Generator = " + str(Final_Generator) + "\n"
data[7] = "8-Client = " + str(Client) + "\n"
data[10] = "11-Monthly_limit = " + str(Monthly_Limit) + "\n"
f.close()
        
f = open("C:/Users/gyes9/Desktop/Information.txt", "w")
f.writelines(data)
f.close()

X = []

for i in range(len(Generator)):
    if Power_Sum[i] - Demand_Sum[i] > 0:
        Residual.append(Power_Sum[i] - Demand_Sum[i])
    else:
        Residual.append((Power_Sum[i] - Demand_Sum[i]) * -1)
    X.append(i)

# Plotting both the curves simultaneously
plt.plot(X, Power_Sum, color='r', label='Power')
plt.plot(X, Demand_Sum, color='g', label='Demand')
plt.plot(X, Residual, color='b', label="Residual")

# Naming the x-axis, y-axis and the whole graph
plt.ylabel("Kilo Watt")
plt.xlabel("Time")
plt.title("Solar Power For One Day (96 Samples)")
  
# Adding legend, which helps us recognize the curve according to it's color
plt.legend()
  
# To load the display window
plt.show()


