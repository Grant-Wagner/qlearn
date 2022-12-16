import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generateSequence(sequence_length = 100, trigger_num = 5, debug = False):
    """This method takes a single argument, length, and generates a sequence of ints at random according to a "rule". This rule is that a specific int, if generated, will always be followed by the int 9. The point of this method is to generate a sequence on which a q-learning implementation will train an agent to perform some action (betting) on some trigger number, which gives a reward if the trigger number is followed by the int 9.

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
    """A simple grid with a built in reward table. Reward for the bottom right tile is always 1, rewards along other tiles depend on the value of the manhattan_distance parameter.

    Attributes
    ----------
    x_dim : int
        width of the grid
    y_dim : int
        height of the grid
    state_dim : array
        actual grid
    action_space : array
        possible actions that can be taken while on the grid (move left, up, right, down)
    manhattan_distance : bool
        flag for whether or not rewards on the grid should be proportional to the inverse manhattan distance from the "goal tile"
    reward_table : array
        rewards to give to the agent on each tile
    
    Methods
    -------
    setRewardTable(x_dim, y_dim)
        sets the reward_table attribute, called from init
    getNextState(action, current_state)
        returns the next state that the agent will be in after taking an action
    print()
        prints attributes of the grid
    """

    def __init__(self, x_dim = 10, y_dim = 10, manhattan_distance = True):
        # store those juicy parameters
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.state_dim = x_dim * y_dim
        self.action_dim = 4
        self.manhattan_distance = manhattan_distance
        # (0,0) = top left = 0, (0,1) = 1 right from top left = 1, ... ,
        # (1, 0) = x_dim, (1,1) = x_dim + 1, ... ,
        # (x_dim, y_dim) = bottom right = x_dim*y_dim - 1
        self.state_space = np.zeros(x_dim * y_dim)
        # 0 = left, 1 = up, 2 = right, 3 = down
        self.action_space = [0,1,2,3]
        # initialize reward table
        self.setRewardTable(x_dim, y_dim)

    def setRewardTable(self, x_dim, y_dim):
        """Helper method to set the reward_table attribute from the init method.
        """

        self.reward_table = np.zeros(self.x_dim * self.y_dim)
        if self.manhattan_distance:
            for i in range(self.x_dim * self.y_dim):
                x_pos = i % self.x_dim
                y_pos = i // self.x_dim
                manhattan_distance = (x_dim - 1 - x_pos) + (y_dim - 1 - y_pos)
                if manhattan_distance == 0:
                    self.reward_table[i] = 2
                else:
                    self.reward_table[i] = (1 / manhattan_distance)
        else:
            self.reward_table[x_dim * y_dim - 1] = 1
        print(self.reward_table)
            
    def getNextState(self, action, current_state):
        """Returns the next state that an agent should be in based on their current location (current_state) and the action they take (left, up, right, down). Returns False if the agent has wandered out of the boundaries of the grid.

        Attributes
        ----------
        action : int
            value of the direction the agent is walking in
        current_state : int
            current location of the agent, defined by an index of the state_space array. 
        """

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
        """Pretty self explanatory
        """

        print("state_space:")
        for i in range(self.y_dim):
            print(self.state_space[(self.x_dim * i):(self.x_dim * (i + 1) - 1)])
        print("action_space:")
        print(self.action_space)
        print("reward_table:")
        for i in range(self.y_dim):
            print(self.reward_table[(self.x_dim * i):(self.x_dim * (i + 1) - 1)])

class Simulation:
    """Simple simulation class
    
    Was initially an ill-conceived attempt to reinforcement train on sequences which upon a moment of reflection should have revealed itself as a farce - agent actions have no impact on the next value of a random-ish sequence and so gamm has to be set to 0 for a q-agent for this to work, might as well just do statistics.
    
    Honestly not going to go into attrs and methods; a sequence gets passed in, along with a simple action space (bet or no bet) and agent queries into the sequence via the next method with its actions to receive rewards. Also has a simple print.
    
    Superceded by the SimulationPrime class which while flawed actually is a reasonable implementation of the q-learning paradigm since it gets passed a simulation environment where actions have consequences.
    """

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
        """Method for stepping through the sequence and returning rewards."""

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
    """Simple simulation class.

    A step up from the Simulation class, this one gets a sim environment passed in and calls that environment for information on next states after an agent takes an action. There are still a few kinks though:
    
        - reward_table gets stored both here and in the env, which it really should just be here. Possible fixes are to allow users to set a reward structure somehow, and give default behavior for some environments? Or commit to the reward table being stored in the environment and just have rewards returned from there. At that point does this class even really need to exist? 

        - step function is coupled to the Grid environment class which is stupid. Probably the environment and sim classes should all be one class, thinking that more and more as I type.
    
    #TODO find some way to input custom starting states
    #TODO decouple the step function from the very specific grid environment I've jerry-rigged together
    """
    def __init__(self, env):
        self.env = env
        self.state_space = env.state_space
        self.action_space = env.action_space
        self.state_dim = len(env.state_space)
        self.action_dim = len(env.action_space)
        self.current_state = 0
        self.done = False
        self.reward_table = env.reward_table

    def step(self, action, debug = False):
        """Takes an agent generated action and returns the next state, associated reward and a flag for whether the episode is over.

        Horrendously coupled with the Grid environment class, leading to some gross custom if-else logic. Fix this.
        """
        next_state = self.env.getNextState(action, self.current_state)
        if next_state == False:
            self.done = True
            next_state = ((self.env.x_dim * self.env.y_dim) - 1)
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
    """Simple agent class for q learning.
    
    Makes a little agent that can step through its environment and learn a q-table. Relies on the Simulation (or SimulationPrime) class when actually moving through the sim.
    
    Attributes
    ----------
    state_dim : int
        size of the state space
    action_dim : int
        size of the action space. actions are taken by randomly choosing an int from the set [0,...,action_dim]
    learning_rate : float
        rate at which the agent should learn, see any text on q learning for how this weights the adjustment to our q values
    gamma : float
        future reward discount, same note as learning_rate
    p_exploration : float
        rate at which the agent decides to explore (pick a random action) vs. exploit (pick the action which its q_table says is the best) when stepping through a simulation environment
    q_table : arr
        array of q-values, which correspond to expected rewards from taking a particular action while in a particular state
        
    Methods
    -------
    take_action(current_state)
        returns an action (int) based on the current state the agent is in
    update(current_state, action, next_state)
        updates the q_table of the agent
    """
    
    def __init__(self, state_dim = 10, action_dim = 2, learning_rate = .1, gamma = .8, p_exploration = .3):
        #initialize q table
        self.q_table = np.zeros((state_dim, action_dim))
        #set learning rate and gamma
        self.learning_rate = learning_rate
        self.gamma = gamma
        #set p_exploration
        self.p_exploration = 1

    def take_action(self, current_state):
        """Method for generating an action based on the current state
        """

        if np.random.uniform(0,1) < self.p_exploration:
            action = np.random.choice([0,1,2,3])
        else:
            action = np.argmax(self.q_table[current_state,:])
        return action

    def update(self, current_state, action, next_state, reward):
        """Method for updating the q_table of the agent based on feedback from the Simulation or SimulationPrime class. Ignore ugly prints
        """

        # print("Current state is " + str(current_state))
        # print("Action is " + str(action))
        # print("Next state is " + str(next_state))
        # print("Reward is " + str(reward))
        # print("Before Update:")
        # print(self.q_table)
        self.q_table[current_state, action] = (1 - self.learning_rate) * self.q_table[current_state, action] \
                                               + self.learning_rate * (reward + self.gamma * max(self.q_table[next_state,:]))
        # print("After Update:")
        # print(self.q_table)
        
    def print(self):
        print("\n")
        print("Agent is located at " + str(self))
        print("q_table is:\n" + str(self.q_table) + ".")

def runSequenceSim():
    """A really basic loop that runs the sequence rl attempt. 
    
    We do get actual results out of this (as in, the q-table of the trained agent reflects the pattern we generated in the sequence if gamma is set to 0) but as detailed in the docstring of the Simulation class, this is a wrongheaded approach to the problem.
    """

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
        print("After " + str(i) + " episodes our agent has the following q_table:")
        a.print()
        print("\n")
    return a

def runGridSim(grid_dimensions = [10,10], num_episodes = 100000):
    """Runs a simulation of an agent stepping through a grid.
    
    The agent starts each episode at the left top corner and "wants" to get to the bottom right where it can enjoy a tasty treat of +1 reward. The p_exploration of the agent is reduced linearly from 1 to 0 as the loop runs so the agent relies more and more on its q_table in later episodes.
    
    Parameters
    ----------
    grid_dimensions : array
        dimensions of the grid that we want our agent to play on
    num_episodes : int
        number of episodes to run the simulation
    """

    g = Grid(grid_dimensions[0],grid_dimensions[1], manhattan_distance = False)
    a = Agent(state_dim = g.state_dim, action_dim = g.action_dim, p_exploration = 1,
     learning_rate = .5, gamma = .8)
    for i in range(num_episodes):
        if i % 1000 == 0:
            print(i)
        sim = SimulationPrime(g)
        while not sim.done:
            current_state = sim.current_state
            action = a.take_action(current_state)
            next_state, reward, done = sim.step(action)
            a.update(current_state, action, next_state, reward)
            a.p_exploration = 1- i/num_episodes 
    a.print()
    return a

def showActionMatrices(agent):
    """Generates a matplotlib image of the rewards that a grid agent is expecting on each tile, visualized as a heat map.

    #TODO fix logic for the matrices so non-square matrices are allowed. or don't, that would require some rewriting of the agent class that would couple it with grids. ugh
    """

    mat = agent.q_table
    x_dim = int(np.sqrt(len(mat)))
    left_mat = [[mat[i + x_dim*n][0] for i in range(x_dim)] for n in range(x_dim)]
    up_mat = [[mat[i + x_dim*n][1] for i in range(x_dim)] for n in range(x_dim)]
    right_mat = [[mat[i + x_dim*n][2] for i in range(x_dim)] for n in range(x_dim)]
    down_mat = [[mat[i + x_dim*n][3] for i in range(x_dim)] for n in range(x_dim)]

    fig, axes = plt.subplots(2,2)
    axes[0, 0].set_title("Left Incentives")
    pcm = axes[0, 0].pcolormesh(left_mat, cmap='RdBu_r')
    axes[0,0].invert_yaxis()
    fig.colorbar(pcm, ax=axes[0, 0])
    axes[0,0].set_xticks([])
    axes[0,0].set_yticks([])

    axes[0, 1].set_title("Up Incentives")
    pcm = axes[0, 1].pcolormesh(up_mat, cmap='RdBu_r')
    axes[0,1].invert_yaxis()
    fig.colorbar(pcm, ax=axes[0, 1])
    axes[0,1].set_xticks([])
    axes[0,1].set_yticks([])

    axes[1, 0].set_title("Right Incentives")
    pcm = axes[1, 0].pcolormesh(right_mat, cmap='RdBu_r')
    axes[1,0].invert_yaxis()
    fig.colorbar(pcm, ax=axes[1, 0])
    axes[1,0].set_xticks([])
    axes[1,0].set_yticks([])

    axes[1, 1].set_title("Down Incentives")
    pcm = axes[1, 1].pcolormesh(down_mat, cmap='RdBu_r')
    axes[1,1].invert_yaxis()
    fig.colorbar(pcm, ax=axes[1, 1])
    axes[1,1].set_xticks([])
    axes[1,1].set_yticks([])

    plt.show()

def outputQTables(num_agents, grid_dimensions, episode_step_size):
    """A method that runs several simulation loops of varying size and outputs the trained agent q tables to the figs folder.

    Parameters
    ----------
    num_agents : int
        number of agents to train
    grid_dimensions : array
        size of the grid on which to train the agents
    episode_step_size : int
        the step size by which we increase the number of episodes that we train each agent on
    """

    for i in range(num_agents):
        trained_grid_agent = runGridSim(grid_dimensions = grid_dimensions, num_episodes = i * episode_step_size)
        plt.plot()
        plt.matshow(trained_grid_agent.q_table)
        plt.savefig("figs/" + str(i * 10000) + "runs.png")

def main():
    trained_grid_agent = runGridSim(grid_dimensions = [7,7], num_episodes = 100000)
    showActionMatrices(trained_grid_agent)

if __name__ == "__main__":
    main()
