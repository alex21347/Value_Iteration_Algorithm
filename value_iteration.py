import numpy as np
from matplotlib import pyplot as plt



def value_iteration(transition, reward, discount_factor = 0.95):
    '''
    transition : a dict of arrays. each array represents the transition probabilities under a given action
    reward : a dict of arrays. each array represents the rewards under a given action
    '''
    
    for key in transition.keys():
        num_states = len(transition[key][0, :])
        break

    state_value_function = np.zeros((num_states,))
    next_state_value_function = np.zeros((num_states,))

    actions = transition.keys()
    num_actions = 0

    for key in transition.keys():
        num_actions += 1

    values = {}

    for key in transition.keys():
        values[key] = []

    delta = 1e10

    while delta > 1e-4:
        for i, state in enumerate(transition.keys()):
            v = state_value_function[i]
            sum_max = -1 * 1e10
            for action in actions:
                sum_action = 0
                for j in range(num_states):
                    sum_action += transition[action][i,j] * (reward[action][i,j] + discount_factor * state_value_function[j])
                if sum_action > sum_max:
                    sum_max = sum_action
                print(i, action, sum_action)
            next_state_value_function[i] = sum_max
            delta = np.max(np.array([np.abs(sum_max - v), 0]))
            values[state].append(sum_max)

        state_value_function = next_state_value_function

    return state_value_function, values



def generate_policy(transition, reward, state_value_function, discount_factor = 0.95):
    '''
    Given, transitions, rewards, and a state_value_function, generate the optimal policy
    '''
    for key in transition.keys():
        num_states = len(transition[key][0, :])
        break
    actions = transition.keys()
    policy = []

    for i in range(num_states):
        argmax = ''
        max_sum = -1 * 1e10
        for action in actions:
            sum_action = 0
            for j in range(num_states):
                sum_action += transition[action][i,j] * (reward[action][i,j] + discount_factor * state_value_function[j])
            if sum_action > max_sum:
                max_sum = sum_action
                argmax = action

        policy.append(argmax) 
    return policy

if __name__ == "__main__":
    transition = {}
    transition['no_advertising'] = np.array([[0.5 , 0.5] , [0.4 , 0.6]])
    transition['advertising'] = np.array([[0.8 , 0.2] , [0.7 , 0.3]])

    reward = {}
    reward['no_advertising'] = np.array([[9 , 3] , [3 , -7]])
    reward['advertising'] = np.array([[4 , 4] , [1 , -19]])


    state_value_function, values = value_iteration(transition, reward)
    print(state_value_function)
    policy = generate_policy(transition, reward, state_value_function)
    print(policy)


    plt.figure(figsize = (8,4))

    plt.scatter(range(len(np.array(values['no_advertising']))), np.array(values['no_advertising']), marker = 'o', alpha = 0.5, color = 'k', s = 6, label = 'State 1')
    plt.scatter(range(len(np.array(values['advertising']))), np.array(values['advertising']), marker = 'o', alpha = 0.5, color = 'r', s = 6, label = "State 2")
    plt.legend()   
    plt.xlabel("Iteration")
    plt.ylabel("V(s)")
    plt.show()


    
