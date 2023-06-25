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

def powerConsumed(powerSetting):
    # Filter the data for the given environment and power setting
    filtered_data = all_data[(all_data['powerSetting'] == powerSetting)]

    # Calculate the mean and std deviation of the energy consumption
    mean_power = filtered_data['energy'].mean()
    std_power = filtered_data['energy'].std()

    # Get the total number of trials (energy consumptions) for the given power setting and environment
    n = len(filtered_data)
    print('Number of energy readings:', n)
    
    # number of repetitions
    num_repetitions = 1000

    # Generate multiple normal samples
    normal_samples = np.random.normal(mean_power, std_power, num_repetitions)

    return normal_samples.mean()

def packet_loss(environment,powerSetting):
    # Filter the data for the given environment and power setting
    filtered_data = all_data[(all_data['environment'] == environment) & (all_data['powerSetting'] == powerSetting)]
    # print(filtered_data)
    # Calculate the packet loss rate
    packet_loss_rate = len(filtered_data[filtered_data['received'] == True]) / len(filtered_data)
    
    # Get the total number of trials (packet transmissions) for the given power setting and environment
    n = len(filtered_data)
    # print('fNumber of Transmissions:',n)
    
    # number of repetitions
    num_repetitions = 1000

    # Generate multiple binomial samples
    binomial_samples = np.random.binomial(n, (1-packet_loss_rate), num_repetitions)

    return binomial_samples.mean()


def utilityDingNet(environment, powerSetting):
    # Calculate the packet loss and power consumed for the given environment and power setting
    print(f"Environment: {environment}, PowerSetting: {powerSetting}")
    packet_loss_rate = packet_loss(environment, powerSetting)
    # print(packet_loss_rate)
    # power_consumed = powerConsumed(powerSetting)
    
    # Calculate the utility as the inverse of the packet loss rate multiplied by the inverse of the power consumed
    # utility = 1 / packet_loss_rate * 1 / power_consumed
    # truncated_reward = truncate(utility)
    
    return packet_loss_rate


# print(packet_loss("Forest", 1))
print(utilityDingNet("Forest", 1))