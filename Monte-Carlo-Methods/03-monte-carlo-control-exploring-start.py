import matplotlib.pyplot as plt
import numpy as np
import seaborn

from gridWorldEnvironment import GridWorld



class MCAgent:
    def __init__(self, env):
        self.env = env
        self.no_of_iteration = 10

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

    def generate_any_policy(self):
        pi = dict()
        
        for state in self.env.states:
            r = sorted(np.random.sample(3))
            actions = self.env.actions
            prob = [r[0], r[1] - r[0], r[2] - r[1], 1-r[2]]
            pi[state] = (actions, prob)
        return pi

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

    def state_action_value(self):
        q = dict()
        for state, action, next_state, reward in self.env.transitions:
            q[(state, action)] = - 1000        # arbitrary negative value to avoid infinite loop
        return q

    def generate_episode(self, s0, a0, policy):
        episode = []
        done = False
        current_state, action = s0, a0
        episode.append((current_state, action, -1))
        
        while not done:
            next_state, reward = self.env.state_transition(current_state, action)
            pr = policy[current_state][1]
            ## to make non-deterministic episode (mostly to avoid infinite episode due to greediness)
            pr[np.argmax(pr)] -= .2
            pr[np.random.choice(np.delete(np.arange(4), np.argmax(pr)))] += .1
            pr[np.random.choice(np.delete(np.arange(4), np.argmax(pr)))] += .1
            ##
            action = np.random.choice(policy[current_state][0], p = pr)
            episode.append((next_state, action, reward))
            
            if next_state == 0:   
                done = True
            current_state = next_state
        return episode[:-1]



    # Exploring start using first-visit MC
    def monte_carlo_es(self, num_iter):
        Q = self.state_action_value()
        pi = self.generate_any_policy()
        returns = dict()
        
        for s, a in Q.keys():
            returns[(s,a)] = []
        
        for _ in range(num_iter):
            s0 = np.random.choice(self.env.states)
            a0 = np.random.choice(self.env.actions)
            already_visited = set()
            
            episode = self.generate_episode(s0, a0, pi)
            
            for s, a, r in episode:
                if (s, a) not in already_visited:
                    already_visited.add((s, a))
                    idx = episode.index((s, a, r))
                    G = 0
                    for j in range(idx, len(episode)-1):
                        G = self.env.gamma * (G + episode[j][-1])
                    returns[(s,a)].append(G)
                    Q[(s,a)] = np.mean(returns[(s,a)])
                    
            pi = self.generate_greedy_policy(Q)
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

    # MC Exploring start
    # Obtained Estimates for Q & pi after 10 iterations
    Q_hat, pi_hat = agent.monte_carlo_es(10)

    # Visualizing policy
    
    ### RED = TERMINAL (0)
    ### GREEN = LEFT
    ### BLUE = UP
    ### PURPLE = RIGHT
    ### ORANGE = DOWN
    
    agent.show_policy(pi_hat)
