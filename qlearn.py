import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generateSequence(sequence_length = 100, trigger_num = 5, debug = False):
    """This method takes a single argument, length, and generates a sequence of ints at random according to
    a "rule". This rule is that a specific int, if generated, will always be followed by the int 9. The point
    of this method is to generate a sequence on which a q-learning implementation will train an agent to
    perform some action (betting) on some trigger number, which gives a reward if the trigger number is
    followed by the int 9.

    Parameters
    ----------
    sequence_length : int
    The length of the sequence to be generated.

    trigger_num : int
    The number which should always be followed by the number 9.

    debug : bool
    A flag which prints debugging statements if set to True.
    """
    
    sequence = []
    sequence.append(np.random.randint(0,9))
    while len(sequence) < sequence_length:
        if sequence[len(sequence) - 1] == trigger_num:
            sequence.append(9)
        else:
            sequence.append(np.random.randint(0,9))
    if debug:
        print("Trigger number is " + str(trigger_num) + ".")
        print("Generating a sequence of length " + str(sequence_length) + ".")
        print("Actual sequence length is " + str(len(sequence)) + ".")
    return sequence

class Grid:
    def __init__(self, x_dim = 10, y_dim = 10):
        # store those juicy parameters
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_dim = x_dim * y_dim
        self.action_dim = 4
        # (0,0) = top left = 0, (0,1) = 1 right from top left = 1, ... ,
        # (1, 0) = x_dim, (1,1) = x_dim + 1, ... ,
        # (x_dim, y_dim) = bottom right = x_dim*y_dim - 1
        self.state_space = np.zeros(x_dim * y_dim)
        # 0 = left, 1 = up, 2 = right, 3 = down
        self.action_space = [0,1,2,3]
        # initialize reward table
        self.setRewardTable(x_dim, y_dim)

    def setRewardTable(self, x_dim, y_dim):
        #TODO: this method (due to the modulus/quotient stuff) only works on nxn grids, need
        #to get this working for nxm as well
        self.reward_table = np.zeros(self.x_dim * self.y_dim)
        for i in range(self.x_dim * self.y_dim):
            x_pos = i % self.x_dim
            y_pos = i // self.y_dim
            manhattan_distance = (x_dim - 1 - x_pos) + (y_dim - 1 - y_pos)
            if manhattan_distance == 0:
                self.reward_table[i] = 100
            else:
                self.reward_table[i] = 1/manhattan_distance
            

    def getNextState(self, action, current_state):
        if action == 0:
            out_of_bounds = (current_state) % self.x_dim == 0
            if out_of_bounds:
                return False
            else:
                return current_state - 1
        elif action == 1:
            out_of_bounds = (current_state - self.x_dim) <= 0
            if out_of_bounds:
                return False
            else:
                return current_state - self.x_dim
        elif action == 2:
            out_of_bounds = (current_state + 1) % self.x_dim == 0
            if out_of_bounds:
                return False
            else:
                return current_state + 1
        elif action == 3:
            out_of_bounds = (current_state + self.x_dim) >= (self.x_dim * self.y_dim)
            if out_of_bounds:
                return False
            else:
                return current_state + self.x_dim

    def print(self):
        print("state_space:")
        for i in range(self.y_dim):
            print(self.state_space[(self.x_dim * i):(self.x_dim * (i + 1) - 1)])
        print("action_space:")
        print(self.action_space)
        print("reward_table")
        for i in range(self.y_dim):
            print(self.reward_table[(self.x_dim * i):(self.x_dim * (i + 1) - 1)])

class Simulation:
    def __init__(self, state_space = [0,1,2,3,4,5,6,7,8,9], action_space = [0,1], reward_num = 9):
        self.state_dim = len(state_space)
        self.action_dim = len(action_space)
        self.state_space = state_space
        self.action_space = action_space
        self.sequence = generateSequence()
        self.current_index = 0
        self.current_state = self.sequence[self.current_index]
        self.next_state = self.sequence[self.current_index + 1]
        self.reward_num = 9
        self.done = False

        #set up reward table
        self.reward_table = np.zeros((self.state_dim, self.action_dim))
        for i in range(self.state_dim):
            self.reward_table[i][1] = -1
        self.reward_table[self.reward_num][self.action_dim - 1] = 1

    def step(self, action, debug = False):
        try:
            self.next_state = self.sequence[self.current_index + 1]
        except:
            return self.current_state, 0, True
        reward = self.reward_table[self.next_state][action]
        if debug:
            print("current_state is " + str(self.current_state) + ".")
            print("next_state is " + str(self.next_state) + ".")
            print("current_action is " + str(action) + ".")
            print("reward is " + str(reward))
        self.current_index += 1
        self.current_state = self.sequence[self.current_index]
        return self.next_state, reward, self.done

    def print(self):
        print("\n")
        print("Simulation is located at " + str(self))
        print("state_space is " + str(self.state_space) + ".")
        print("action_space is " + str(self.action_space) + ".")
        print("sequence is " + str(self.sequence) + ".")
        print("current_index is " + str(self.current_index) + ".")
        print("reward_table is " + str(self.reward_table) + ".")

class SimulationPrime:
    def __init__(self, env):
        self.env = env
        self.state_space = env.state_space
        self.action_space = env.action_space
        self.state_dim = len(env.state_space)
        self.action_dim = len(env.action_space)
        #TODO find some way to input custom starting states
        self.current_state = 0
        self.done = False
        self.reward_table = env.reward_table

    def step(self, action, debug = False):
        next_state = self.env.getNextState(action, self.current_state)
        if next_state == False:
            self.done = True
            next_state = 0
            reward = -1
        elif next_state == self.state_dim - 1:
            self.done = True
            reward = self.reward_table[next_state]
        else:
            reward = self.reward_table[next_state]
        if debug:
            print("current_state is " + str(self.current_state) + ".")
            print("next_state is " + str(self.next_state) + ".")
            print("current_action is " + str(action) + ".")
            print("reward is " + str(reward))
        self.current_state = next_state
        return next_state, reward, self.done

    def print(self):
        print("\n")
        print("Simulation is located at " + str(self))
        print("state_space is " + str(self.state_space) + ".")
        print("action_space is " + str(self.action_space) + ".")
        print("sequence is " + str(self.sequence) + ".")
        print("current_index is " + str(self.current_index) + ".")
        print("reward_table is " + str(self.reward_table) + ".")        

class Agent:
    def __init__(self, state_dim = 10, action_dim = 2, learning_rate = .1, gamma = .8, p_exploration = .3):
        #initialize q table
        self.q_table = np.zeros((state_dim, action_dim))
        #set learning rate and gamma
        self.learning_rate = learning_rate
        self.gamma = gamma
        #set p_exploration
        self.p_exploration = 1

    def take_action(self, current_state):
        if np.random.uniform(0,1) < self.p_exploration:
            action = np.random.choice([0,1,2,3])
        else:
            action = np.argmax(self.q_table[current_state,:])
        return action

    def update(self, current_state, action, next_state, reward):
##        print("Current state is " + str(current_state))
##        print("Action is " + str(action))
##        print("Next state is " + str(next_state))
##        print("Reward is " + str(reward))
##        print("Before Update:")
##        print(self.q_table)
        self.q_table[current_state, action] = (1 - self.learning_rate) * self.q_table[current_state, action] \
                                              + self.learning_rate * (reward + self.gamma * max(self.q_table[next_state,:]))
##        print("After Update:")
##        print(self.q_table)
        
    def print(self):
        print("\n")
        print("Agent is located at " + str(self))
        print("q_table is " + str(self.q_table) + ".")

def runSequenceSim():
    a = Agent(10,2)
    sim = Simulation()
    num_episodes = 100
    for i in range(num_episodes):
        sim = Simulation()
        done = False
        while not done:
            current_state = sim.current_state
            action = a.take_action(current_state)
            next_state, reward, done = sim.step(action)
            a.update(current_state, action, next_state, reward)
##        print("After " + str(i) + " episodes our agent has the following q_table:")
##        a.print()
##        print("\n")
    return a

def runGridSim(grid_dimensions = [5,5], num_episodes = 100000):
    g = Grid(grid_dimensions[0],grid_dimensions[1])
    a = Agent(state_dim = g.state_dim, action_dim = g.action_dim)
    for i in range(num_episodes):
        if i % 100 == 0:
            print(i)
        sim = SimulationPrime(g)
        while not sim.done:
            current_state = sim.current_state
            action = a.take_action(current_state)
            next_state, reward, done = sim.step(action)
            a.update(current_state, action, next_state, reward)
    a.print()
    return a

trained_grid_agent = runGridSim()
tex_string = """\\begin{bmatrix}
	1 & z_1 \\\\
	1 & z_2 \\\\
	\\end{bmatrix}"""
import matplotlib.pyplot as plt
plt.plot()
##plt.text(0.5, 0.5,'$%s$'%tex_string)
plt.matshow(trained_grid_agent.q_table)
plt.show()
