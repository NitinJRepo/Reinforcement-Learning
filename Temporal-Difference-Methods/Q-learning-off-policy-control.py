#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 14:53:33 2019

@author: nitin
"""
# # Q-Learning Algorithm

import matplotlib.pyplot as plt
import numpy as np
import seaborn

from gridWorldEnvironment import GridWorld


class QAgent:
    def __init__(self, env):
        self.env = env

    def state_action_value(self):
        q = dict()
        for state, action, next_state, reward in self.env.transitions:
            q[(state, action)] = np.random.normal()
        for action in self.env.actions:
            q[0, action] = 0
        return q

    def generate_greedy_policy(self, Q):
        pi = dict()
        for state in self.env.states:
            actions = []
            q_values = []
            prob = []
            
            for a in self.env.actions:
                actions.append(a)
                q_values.append(Q[state,a])
                
            for i in range(len(q_values)):
                if i == np.argmax(q_values):
                    prob.append(1)
                else:
                    prob.append(0)       
                    
            pi[state] = (actions, prob)
        return pi

    def epsilon_greedy(self, e, q, state):
        actions = self.env.actions
        action_values = []
        prob = []
        
        for action in actions:
            action_values.append(q[(state, action)])
            
        for i in range(len(action_values)):
            if i == np.argmax(action_values):
                prob.append(1 - e + e/len(action_values))
            else:
                prob.append(e/len(action_values))
        return np.random.choice(actions, p = prob)

    def get_best_action(self, q, state):
        actions = self.env.actions
        action_values = []
        for action in actions:
            action_values.append(q[state, action])
        return actions[np.argmax(action_values)]

    def q_learning(self, epsilon, alpha, num_iter):
        Q = self.state_action_value()
        
        for _ in range(num_iter):
            current_state = np.random.choice(self.env.states)
            while current_state != 0:
                # select action 'a' in state 's' using epsilon-greedy policy
                current_action = self.epsilon_greedy(epsilon, Q, current_state)
                
                # perform action and move to next state and get the reward 
                next_state, reward = self.env.state_transition(current_state, current_action)
                
                # select the action which has maximum value  
                best_action = self.get_best_action(Q, next_state)
                
                # update q value in Q table
                Q[current_state, current_action] += alpha * (reward + self.env.gamma * Q[next_state, best_action] - Q[current_state, current_action])
                current_state = next_state
        return Q

    def show_policy(self, pi):
        temp = np.zeros(len(self.env.states) + 2)
        for s in self.env.states:
            a = pi_hat[s][0][np.argmax(pi_hat[s][1])]
            if a == "U":
                temp[s] = 0.25
            elif a == "D":
                temp[s] = 0.5
            elif a == "R":
                temp[s] = 0.75
            else:
                temp[s] = 1.0
                
        temp = temp.reshape(4,4)
        ax = seaborn.heatmap(temp, cmap = "prism", linecolor="#282828", cbar = False, linewidths = 0.1)
        plt.show()

    def get_source():
        


if __name__ == '__main__':
    # Creating gridworld environment
    gw = GridWorld(gamma = .9, theta = .5)
    agent = QAgent(gw)

    # Q-Learning: off-policy TD control
    Q = agent.q_learning(0.2, 1.0, 10000)

    # Visualizing policy
    
    ### RED = TERMINAL (0)
    ### GREEN = LEFT
    ### BLUE = UP
    ### PURPLE = RIGHT
    ### ORANGE = DOWN
    
    pi_hat = agent.generate_greedy_policy(Q)

    agent.show_policy(pi_hat)

