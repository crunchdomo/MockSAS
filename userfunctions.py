from scipy.stats import truncnorm
import gym
from gym import spaces
import numpy as np
import ast
import os
import pandas as pd
import xml.etree.ElementTree as ET
from glob import glob


def myfunction(posarg, posarg2):
    print(posarg)
    return (posarg + posarg2) / 10

def truncated_normal(lower, upper, mean, stdev):
    a, b = (lower - mean) / stdev, (upper - mean) / stdev
    return truncnorm.rvs(a, b, loc=mean, scale=stdev)

def normal(mean, stdev):
    return np.random.normal(loc=mean, scale=stdev)

def uniform(lower, upper):
    return np.random.uniform(low=lower, high=upper)

def truncate(utility):
    bounds = (140, 300)
    lower_bound, upper_bound = bounds

    if utility > upper_bound:
        utility = upper_bound
    elif utility < lower_bound:
        utility = lower_bound

    range_ = upper_bound - lower_bound
    result = float((utility - lower_bound) / range_)
    return result

def utilitySWIM(arrival_rate, dimmer, avg_response_time, max_servers, servers):
    OPT_REVENUE = 1.5
    BASIC_REVENUE = 1
    SERVER_COST = 10
    RT_THRESH = 0.75

    ur = arrival_rate * ((1 - dimmer) * BASIC_REVENUE + dimmer * OPT_REVENUE)
    uc = SERVER_COST * (max_servers - servers)
    urt = 1 - ((avg_response_time - RT_THRESH) / RT_THRESH)

    UPPER_RT_THRESHOLD = RT_THRESH * 4
    delta_threshold = UPPER_RT_THRESHOLD - RT_THRESH
    UrtPosFct = (delta_threshold / RT_THRESH)

    urt_final = None
    if avg_response_time <= UPPER_RT_THRESHOLD:
        urt = ((RT_THRESH - avg_response_time) / RT_THRESH)
    else:
        urt = ((RT_THRESH - UPPER_RT_THRESHOLD) / RT_THRESH)

    if avg_response_time <= RT_THRESH:
        urt_final = urt * UrtPosFct
    else:
        urt_final = urt

    revenue_weight = 0.7
    server_weight = 0.3
    utility = urt_final * ((revenue_weight * ur) + (server_weight * uc))

    truncated_reward = truncate(utility)
    print(truncated_reward)
    return truncated_reward

#############################################################################################################
#EVERYTHING BELOW HERE BASIC DATA PROCESSING.

def xml_to_df(root, path):
    data = []
    for element in root.findall(path):
        row = {}
        for child in element:
            row[child.tag] = child.text
        for key, value in element.attrib.items():
            row[key] = value
        data.append(row)
    df = pd.DataFrame(data)
    return df

# Define the base directory
base_dir = "Final Runs" #Change this to your base directory! 

# Define the environments
environments = ["City", "Forest", "Plains"]

# Define the power settings
power_settings = [f"PS {i}" for i in range(1, 15)]

# Define the runs
runs = [f"Run {i}" for i in range(1, 4)]

# The DataFrame to hold all the data
all_data = pd.DataFrame()

# Iterate over environments, power settings, and runs
for environment in environments:
    for power_setting in power_settings:
        for run in runs:
            run_dir = os.path.join(base_dir, environment, power_setting, run)
            
            # Parse the main XML file
            main_file = os.path.join(run_dir, "main.xml")
            tree = ET.parse(main_file)
            root = tree.getroot()
            df_main = xml_to_df(root, './/mote[number="1"]/receivedTransmissions/receivedTransmission')
            
            # Parse the mote XML files
            mote_files = glob(os.path.join(run_dir, "mote_*.xml"))
            df_motes = pd.concat([xml_to_df(ET.parse(file).getroot(), './/transmissionPower') for file in mote_files])
            
            # Ensure 'moteIdentifier' and 'transmissionPower' are the same type (string)
            df_main[['moteIdentifier', 'transmissionPower']] = df_main[['moteIdentifier', 'transmissionPower']].astype(str)
            df_motes[['moteIdentifier', 'transmissionPower']] = df_motes[['moteIdentifier', 'transmissionPower']].astype(str)
            
            # Merge the data based on 'moteIdentifier' and 'transmissionPower'
            run_data = pd.merge(df_main, df_motes, on=['moteIdentifier', 'transmissionPower'])

            # Add information about the run directory
            run_data['run_directory'] = run_dir
            
            # Add the run data to the overall data
            all_data = pd.concat([all_data, run_data])

# Convert the string representation of tuples into actual tuples
all_data['powerSetting'] = all_data['powerSetting'].apply(ast.literal_eval)

# Split the powerSetting column into two new columns
all_data[['transmissionCount', 'powerSetting']] = pd.DataFrame(all_data['powerSetting'].tolist(), index=all_data.index)

# Ensure 'powerSetting' is integer type
all_data['powerSetting'] = all_data['powerSetting'].astype(int)

all_data['collision'] = all_data['collision'].map({'true': True, 'false': False})

all_data['received'] = all_data['received'].map({'true': True, 'false': False})

# Convert the 'energy' column to numeric, forcing non-numeric values to NaN
all_data['energy'] = pd.to_numeric(all_data['energy'], errors='coerce')


#############################################################################################################
#EVERYTHING BELOW HERE IS FOR DINGNET. THIS IS WHERE WE WILL BE WORKING! <3


def powerConsumed(powerSetting):
    # Filter the data for the given environment and power setting
    filtered_data = all_data[(all_data['powerSetting'] == powerSetting)]

    # Calculate the mean and std deviation of the energy consumption
    mean_power = filtered_data['energy'].mean()
    std_power = filtered_data['energy'].std()

    # Get the total number of trials (energy consumptions) for the given power setting and environment
    n = len(filtered_data)
    # print('Number of energy readings:', n)
    
    # number of repetitions
    num_repetitions = 1000

    # Generate multiple normal samples
    normal_samples = np.random.normal(mean_power, std_power, num_repetitions)

    return normal_samples.mean()



def _packet_success(environment, powerSetting):
    # Filter the data for the given environment and power setting
    filtered_data = all_data[(all_data['environment'] == environment) & (all_data['powerSetting'] == powerSetting)]
    
    # Calculate the packet loss rate
    _packet_success = len(filtered_data[filtered_data['received'] == True]) / len(filtered_data)
    
    # Get the total number of trials (packet transmissions) for the given power setting and environment
    n = len(filtered_data)
    # print("Legnth of data:", n)
    
    # number of repetitions
    num_repetitions = 1000

    # Generate multiple binomial samples
    binomial_samples = np.random.binomial(n, (1-_packet_success), num_repetitions)
    # print(f"Environment: {environment}, PowerSetting: {powerSetting}")

    return binomial_samples.mean()


# #Print Packet Success Rate
# for i in range(1, 15):
#    print("City",i,":",_packet_success("City", i))
   
# for i in range(1, 15):
#    print("Forest",i,":",_packet_success("Forest", i))

# for i in range(1, 15):
#    print("Plain",i,":",_packet_success("Plain", i))

# #Print Power Consumed
# for i in range(1, 15):
#    print("Power",i,":",powerConsumed(i))

# def normalize(value, min_value, max_value):
#     normalized_value = (value - min_value) / (max_value - min_value)
#     return normalized_value

#NEEDED TO MAKE UTILITY FUNCTION WORK! <3
def _packet_success_forest(powerSetting):
    return _packet_success("Forest", powerSetting)

def _packet_success_city(powerSetting):
    return _packet_success("City", powerSetting)

def _packet_success_plain(powerSetting):
    return _packet_success("Plain", powerSetting)

# Compute the min and max of packet success rate and power consumed across all settings and environments
packet_success_rates = [_packet_success(environment, powerSetting) for environment in ['Forest', 'City', 'Plain'] for powerSetting in range(1, 15)]
min_packet_success, max_packet_success = min(packet_success_rates), max(packet_success_rates)

power_consumptions = [powerConsumed(powerSetting) for powerSetting in range(1, 15)]
min_power_consumed, max_power_consumed = min(power_consumptions), max(power_consumptions)

def normalize(value, min_value, max_value):
    normalized_value = (value - min_value) / (max_value - min_value)
    return normalized_value

def _utilityDingNet(_packet_success_func, power_func, powerSetting, power_penalty_factor):
    try:
        _packet_success = normalize(_packet_success_func(powerSetting), min_packet_success, max_packet_success)
        power_consumed = normalize(power_func(powerSetting), min_power_consumed, max_power_consumed)
        utility = _packet_success - power_penalty_factor * power_consumed
        return utility
    except Exception as e:
        print(f"Error computing utility for power setting {powerSetting}: {e}")
        return None

def utilityDingNet_forest(powerSetting, power_penalty_factor= 0.08056640625):
    return _utilityDingNet(_packet_success_forest, powerConsumed, powerSetting, power_penalty_factor)

def utilityDingNet_city(powerSetting, power_penalty_factor=0.88037109375):
    return _utilityDingNet(_packet_success_city, powerConsumed, powerSetting, power_penalty_factor)

def utilityDingNet_plain(powerSetting, power_penalty_factor=0.35693359375):
    return _utilityDingNet(_packet_success_plain, powerConsumed, powerSetting, power_penalty_factor)



#determine optimal power_penalty_factor for Dingnet
def binary_search(func, low, high, tol=1e-3, max_iter=100):
    for _ in range(max_iter):
        mid = (low + high) / 2
        if high - low < tol:
            return mid
        if func(mid) > func(mid + tol):
            high = mid
        else:
            low = mid
    return mid


packet_success_funcs = {
    "Forest": _packet_success_forest,
    "City": _packet_success_city,
    "Plain": _packet_success_plain,
}

def total_utility_per_environment(environment, power_penalty_factor):
    total_utility = 0
    
    if environment not in packet_success_funcs:
        raise ValueError(f'Unknown environment: {environment}')
    
    for powerSetting in range(1, 15):
        total_utility += _utilityDingNet(packet_success_funcs[environment], powerConsumed, powerSetting, power_penalty_factor)
        print(f"Total Utility {powerSetting} for {environment}: {total_utility}")
            
    if total_utility == 0:
        raise ValueError('Total utility cannot be 0')
    
    print(f"Total Utility for {environment}: {total_utility}")
    return total_utility

environments = ['Forest', 'City', 'Plain']
optimal_power_penalty_factors = {}

for environment in environments:
    optimal_power_penalty_factors[environment] = binary_search(lambda ppf: total_utility_per_environment(environment, ppf), 0, 1)

for environment, optimal_ppf in optimal_power_penalty_factors.items():
    print(f'Optimal power penalty factor for {environment}: {optimal_ppf}')




#Epislon Greedy Search
def epsilon_greedy(utilities, epsilon):
    if np.random.rand() < epsilon:
        # Exploration: choose a random power setting
        powerSetting = np.random.choice(len(utilities))
    else:
        # Exploitation: choose the power setting with the highest estimated utility
        powerSetting = np.argmax(utilities)
    return powerSetting

# The utility functions for different environments
utility_funcs = {
    "Forest": utilityDingNet_forest,
    "City": utilityDingNet_city,
    "Plain": utilityDingNet_plain,
}

# Initialize estimated utilities for each environment
estimated_utilities = {environment: np.zeros(14) for environment in environments}

# Number of times each power setting has been chosen for each environment
counts = {environment: np.zeros(14) for environment in environments}

# Total number of trials
N = 1000

for environment in environments:
    for i in range(N):
        # Choose a power setting
        powerSetting = epsilon_greedy(estimated_utilities[environment], epsilon=0.1)

        # Get the utility of the chosen power setting
        utility = utility_funcs[environment](powerSetting+1, optimal_power_penalty_factors[environment])
        
        # If utility is None, skip the current iteration
        if utility is None:
            continue

        # Update the estimated utility of the chosen power setting
        counts[environment][powerSetting] += 1
        estimated_utilities[environment][powerSetting] = ((counts[environment][powerSetting] - 1) / counts[environment][powerSetting]) * estimated_utilities[environment][powerSetting] + (1 / counts[environment][powerSetting]) * utility

# Now, the best power setting for each environment can be found as follows:
best_power_settings = {environment: np.argmax(utilities) + 1 for environment, utilities in estimated_utilities.items()}

print(best_power_settings)




#EXHAUSTIVE SEARCH ALGORITHM BELOW 
def best_power_environment():
    best_power_settings = {}
    for powerSetting in range(1, 15):
        forest_utility = utilityDingNet_forest(powerSetting)
        city_utility = utilityDingNet_city(powerSetting)
        plain_utility = utilityDingNet_plain(powerSetting)
        
        if 'Forest' not in best_power_settings or forest_utility > best_power_settings['Forest'][1]:
            best_power_settings['Forest'] = (powerSetting, forest_utility)

        if 'City' not in best_power_settings or city_utility > best_power_settings['City'][1]:
            best_power_settings['City'] = (powerSetting, city_utility)

        if 'Plain' not in best_power_settings or plain_utility > best_power_settings['Plain'][1]:
            best_power_settings['Plain'] = (powerSetting, plain_utility)
            
    return best_power_settings

print(best_power_environment())


#BELOW IS THE OLD EXHAUSTIVE SEARCH ALGORITHMS. THE ONE ABOVE IS THE ONE WE ARE USING.
#############################################################################################################


# def exhaustive_search(n_trials):
#     # Define the environments and power settings
#     environments = ["Forest", "City", "Plain"]
#     power_settings = list(range(1, 15))
    
#     # Initialize a dictionary to store the total best power setting for each environment
#     total_best_power_settings = {environment: 0 for environment in environments}
    
#     for _ in range(n_trials):
#         for environment in environments:
#             best_utility = -float('inf')
#             best_power_setting = None
            
#             for power_setting in power_settings:
#                 if environment == "Forest":
#                     utility = utilityDingNet_forest(power_setting)
#                 elif environment == "City":
#                     utility = utilityDingNet_city(power_setting)
#                 elif environment == "Plain":
#                     utility = utilityDingNet_plain(power_setting)
                
#                 if utility > best_utility:
#                     best_utility = utility
#                     best_power_setting = power_setting
            
#             total_best_power_settings[environment] += best_power_setting
    
#     # Compute the average best power setting for each environment
#     average_best_power_settings = {environment: total / n_trials for environment, total in total_best_power_settings.items()}
    
#     return average_best_power_settings

# n_trials = 3  # Replace with the desired number of trials
# average_best_power_settings = exhaustive_search(n_trials)
# for environment, avg_best_power_setting in average_best_power_settings.items():
#     print(f"The average best power setting for {environment} over {n_trials} trials is {avg_best_power_setting}")


# def exhaustive_search():
#     # Define the environments and power settings
#     environments = ["Forest", "City", "Plain"]
#     power_settings = list(range(1, 15))
    
#     best_configs = {}
    
#     for environment in environments:
#         best_utility = -float('inf')
#         best_power_setting = None
        
#         for power_setting in power_settings:
#             if environment == "Forest":
#                 utility = utilityDingNet_forest(power_setting)
#             elif environment == "City":
#                 utility = utilityDingNet_city(power_setting)
#             elif environment == "Plain":
#                 utility = utilityDingNet_plain(power_setting)
            
#             if utility > best_utility:
#                 best_utility = utility
#                 best_power_setting = power_setting
        
#         best_configs[environment] = (best_power_setting, best_utility)
    
#     return best_configs

# best_configs = exhaustive_search()
# for environment, (best_power_setting, best_utility) in best_configs.items():
#     print(f"The best power setting for {environment} is {best_power_setting} with a utility of {best_utility}")
################################################################################################################################################

