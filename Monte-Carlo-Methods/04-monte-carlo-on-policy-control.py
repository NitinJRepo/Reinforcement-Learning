#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 18:40:04 2019

@author: nitin
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn

from gridWorldEnvironment import GridWorld

class MCAgent:
    def __init__(self, env):
        self.env = env

    def generate_random_policy(self):
        pi = dict()
        
        for state in self.env.states:
            actions = []
            prob = []
            
            for action in self.env.actions:
                actions.append(action)
                prob.append(0.25)
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


    # first-visit MC
    def on_policy_mc(self, e, num_iter):
        Q = self.state_action_value()
        pi = self.generate_random_policy()
        returns = dict()
        for s, a in Q:
            returns[(s,a)] = []
            
        for i in range(num_iter):
            episode = self.generate_episode(pi)
            already_visited = set({0})
            
            for s, a, r in episode:
                if s not in already_visited:
                    already_visited.add(s)
                    idx = episode.index((s, a, r))
                    G = 0
                    j = 1
                    while j + idx < len(episode):
                        G = self.env.gamma * (G + episode[j + idx][-1])
                        j += 1
                    returns[(s,a)].append(G)
                    Q[(s,a)] = np.mean(returns[(s,a)])
                    
            for s, _, _ in episode:
                if s != 0:
                    actions = []
                    action_values = []
                    prob = []
    
                    for a in self.env.actions:
                        actions.append(a)
                        action_values.append(Q[s,a])         
                    for i in range(len(action_values)):
                        if i == np.argmax(action_values):
                            prob.append(1 - e + e/len(actions))
                        else:
                            prob.append(e/len(actions))        
                    pi[s] = (actions, prob)
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
    Q_hat, pi_hat = agent.on_policy_mc(0.2, 100000)

    # Visualizing policy
    
    ### RED = TERMINAL (0)
    ### GREEN = LEFT
    ### BLUE = UP
    ### PURPLE = RIGHT
    ### ORANGE = DOWN
    
    agent.show_policy(pi_hat)
