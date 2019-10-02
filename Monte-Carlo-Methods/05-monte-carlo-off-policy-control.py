#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 18:58:24 2019

@author: nitin
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn

from gridWorldEnvironment import GridWorld

class MCAgent:
    def __init__(self, env):
        self.env = env

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

    def generate_any_policy(self):
        pi = dict()
        
        for state in self.env.states:
            r = sorted(np.random.sample(3))
            actions = self.env.actions
            prob = [r[0], r[1] - r[0], r[2] - r[1], 1-r[2]]
            pi[state] = (actions, prob)
        return pi

    def state_action_value(self):
        q = dict()
        for state, action, next_state, reward in self.env.transitions:
            q[(state, action)] = 0
        return q

    def generate_episode(self, policy):
        episode = []
        done = False
        current_state = np.random.choice(self.env.states)
        action = np.random.choice(policy[current_state][0], p = policy[current_state][1])
        episode.append((current_state, action, -1))
        
        while not done:
            next_state, reward = gw.state_transition(current_state, action)
            action = np.random.choice(policy[current_state][0], p = policy[current_state][1])
            episode.append((next_state, action, reward))
            
            if next_state == 0:
                done = True
            current_state = next_state
            
        return episode

    def weight_cum_sum(self):
        c = dict()
        for state, action, next_state, reward in self.env.transitions:
            c[(state, action)] = 0
        return c

    def off_policy_mc_control(self, num_iter):
        Q = self.state_action_value()
        C = self.weight_cum_sum()
        pi = self.generate_greedy_policy(Q)
        
        for _ in range(num_iter):
            b = self.generate_any_policy()
            episode = self.generate_episode(b)
            G = 0
            W = 1
            for i in range(len(episode)-1, -1, -1):
                s, a, r = episode[i]
                if s != 0:
                    G = self.env.gamma * G + r
                    C[s,a] += W
                    Q[s,a] += (W / C[s,a]) * (G - Q[s,a])
                    pi = self.generate_greedy_policy(Q)
                    if a == pi[s][0][np.argmax(pi[s][1])]:
                        break
                    W *= 1 / b[s][1][b[s][0].index(a)]
    
        return Q, pi

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
        seaborn.heatmap(temp, cmap = "prism", linecolor="#282828", cbar = False, linewidths = 0.1)
        plt.show()

if __name__ == '__main__':
    # Creating gridworld environment
    gw = GridWorld(gamma = .9, theta = .5)
    agent = MCAgent(gw)

    # On policy MC
    # Obtained Estimates for Q & pi after 10 iterations
    Q_hat, pi_hat = agent.off_policy_mc_control(100000)

    # Visualizing policy
    
    ### RED = TERMINAL (0)
    ### GREEN = LEFT
    ### BLUE = UP
    ### PURPLE = RIGHT
    ### ORANGE = DOWN
    
    agent.show_policy(pi_hat)
