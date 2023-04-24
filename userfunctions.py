from scipy.stats import truncnorm
import numpy as np

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

def utilityDeltaIoT(packet_loss_rate, energy_consumption, connectivity, max_motes, active_motes):
    # Weights for the utility components but they're comnpletely random. Still figuring out how to calculate them
    mote_cost = uniform(0,50)
    packet_loss_weight = 0.3
    energy_weight = 0.75
    mote_weight = 0.5
    connectivity_weight = 0.25


    upl = packet_loss_rate
    uec = energy_consumption 
    uc = mote_cost * (max_motes - active_motes)
    conn = connectivity

    #the variables above are the utility components

    print(f"upl: {upl}, uec: {uec}, uc: {uc}, conn: {conn}")  # Debug - Print the values of the utility components

    utility = (packet_loss_weight * upl) + (energy_weight * uec) + (mote_weight * uc) + (connectivity_weight * conn)

    print(f"utility before truncation: {utility}")  # Utility is very fucked - why? - Print the utility before truncation 

    truncated_utility = truncate(utility)
    print(truncated_utility)
    return truncated_utility


