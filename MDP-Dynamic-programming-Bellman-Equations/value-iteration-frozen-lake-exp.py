#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 19:27:39 2019

@author: nitin
"""
import gym
import numpy as np


class ValueIterationAgent:
    def __init__(self, env):
        self.env = env
        self.gamma = 1.0
        
    def value_iteration(self):
	    # initialize value table with zeros
	    value_table = np.zeros(self.env.observation_space.n)
	    
	    # set number of iterations and threshold
	    no_of_iterations = 100000
	    threshold = 1e-20
	    
	    for i in range(no_of_iterations):
	        
	        # On each iteration, copy the value table to the updated_value_table
	        updated_value_table = np.copy(value_table) 
	        
	        # Now we calculate Q Value for each actions in the state 
	        # and update the value of a state with maximum Q value
	        
	        for state in range(self.env.observation_space.n):
	            Q_value = []
	            for action in range(self.env.action_space.n):
	                next_states_rewards = []
	                for next_sr in self.env.P[state][action]: 
	                    trans_prob, next_state, reward_prob, _ = next_sr 
	                    next_states_rewards.append((trans_prob * (reward_prob + self.gamma * updated_value_table[next_state]))) 
	                
	                Q_value.append(np.sum(next_states_rewards))
	                
	            value_table[state] = max(Q_value) 
	            
	        # we will check whether we have reached the convergence i.e whether the difference 
	        # between our value table and updated value table is very small. But how do we know it is very
	        # small? We set some threshold and then we will see if the difference is less
	        # than our threshold, if it is less, we break the loop and return the value function as optimal
	        # value function
	        
	        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
	             print ('Value-iteration converged at iteration# %d.' %(i+1))
	             break    
	    return value_table
    
    def extract_policy(self, value_table):
        # initialize the policy with zeros
        policy = np.zeros(self.env.observation_space.n)    

        for state in range(self.env.observation_space.n):
            # initialize the Q table for a state
            Q_table = np.zeros(self.env.action_space.n)
            
            # compute Q value for all ations in the state
            for action in range(self.env.action_space.n):
                for next_sr in self.env.P[state][action]: 
                    trans_prob, next_state, reward_prob, _ = next_sr 
                    Q_table[action] += (trans_prob * (reward_prob + self.gamma * value_table[next_state]))
                    
            # select the action which has maximum Q value as an optimal action of the state
            policy[state] = np.argmax(Q_table)

        return policy

if __name__ == '__main__':
    env = gym.make('FrozenLake-v0')
    env.render()
    
    env = env.unwrapped
    agent = ValueIterationAgent(env)

    optimal_value_function = agent.value_iteration()
    optimal_policy = agent.extract_policy(optimal_value_function)

    print("Optimal policy: ", optimal_policy)



