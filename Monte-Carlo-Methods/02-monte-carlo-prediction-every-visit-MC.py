#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 22:04:19 2019

@author: nitin
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn

from gridWorldEnvironment import GridWorld

class MCAgent:
    def __init__(self, env):
        self.env = env
        self.no_of_iterations = 10000

    def generate_random_episode(self):
        episode = []
        done = False
        current_state = np.random.choice(self.env.states)
        episode.append((current_state, -1))

        while not done:
            action = np.random.choice(self.env.actions)
            next_state, reward = self.env.state_transition(current_state, action)
            episode.append((next_state, reward))
            if next_state == 0:
                done = True
            current_state = next_state
        return episode

    def value_array(self):
        return np.zeros(len(self.env.states) + 2)  # Values in indices 0 and -1 are for terminal states ((0,0) and (3,3))

    def every_visit_mc(self):
        values = self.value_array()
        returns = dict()
        
        for state in self.env.states:
            returns[state] = list()
        
        for i in range(self.no_of_iterations):
            episode = self.generate_random_episode()

            for s, r in episode:
                if s != 0:    # exclude terminal state (0)
                    idx = episode.index((s, r))
                    G = 0
                    j = 1
                    while j + idx < len(episode):
                        G = self.env.gamma * (G + episode[j + idx][1])
                        j += 1
                    returns[s].append(G)
                    values[s] = np.mean(returns[s])
        return values, returns

    def show_values(self, values):
        values = values.reshape(4,4)
        seaborn.heatmap(values, cmap = "Blues_r", annot = True, linecolor="#282828", linewidths = 0.1)
        plt.show()

if __name__ == '__main__':
    # creating gridworld environment
    gw = GridWorld(gamma = .9, theta = .5)

    agent = MCAgent(gw)
    values, returns = agent.every_visit_mc()

    # Visualizing values in table
    # Lighter color in table means higher value for random policy
    agent.show_values(values)
