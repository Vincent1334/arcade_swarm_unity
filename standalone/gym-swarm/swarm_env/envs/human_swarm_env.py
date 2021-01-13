import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
from gym.envs.toy_text import discrete
import json


def target_coords(operator, GRID_X, GRID_Y):
    temp_max = -10
    max_i = 0
    max_j = 0
    for i in range(GRID_Y):
        for j in range(GRID_X):                    
            temp = (1 - operator.confidence_map[i][j]) * operator.internal_map[i][j]
            if temp > temp_max:
                temp_max = temp
                max_i = i
                max_j = j
    return max_i, max_j

# Our customized environment for human-swarm simulation.
class HumanSwarmEnv(discrete.DiscreteEnv):
    metadata = {"render.modes": ["human"]}

    # Initializing the values inside environment.
    def __init__(self, maze_size=None):

        self.maze_size = maze_size
        
        current_screen = None
        nA = 4
        nS = self.maze_size * self.maze_size

        isd = np.zeros((self.maze_size, self.maze_size)).astype('float64')
        isd[int(self.maze_size/2), int(self.maze_size/2)] = 1
        isd.ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        super(HumanSwarmEnv, self).__init__(nS, nA, P, isd)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.current_screen = None

    # Setting up the step function for tsdfsdfhis environment.
    def step(self, action, disaster, drone, operator, GRID_X, GRID_Y):#, min_distance):

        print("***** calling step ***************")
        goal_x, goal_y = target_coords(operator, GRID_X, GRID_Y)
        goal_x = 39
        goal_y = 39
        new_state = goal_x*GRID_X + goal_y	
        i = int (action / GRID_X)
        j = action % GRID_X
        operator.confidence_map[i][j] = -10
        
        # Calculating state distance to the disaster
        distance_x = (drone.center_x - disaster.center_x)
        distance_y = (drone.center_y - disaster.center_y)
        distance = int(np.sqrt(pow(distance_x, 2) + pow(distance_y, 2)))

        #reward = 1 - distance
        #reward = operator.confidence_map[goal_x][goal_y]
        reward = 0
        for i in range(GRID_Y):
            for j in range(GRID_X):                    
                r = np.sqrt((j-goal_x)**2 + (i-goal_y)**2)                    
                if (r < 5):
                    reward+=operator.confidence_map[i][j]        
        '''
        if (min_distance < distance):
            reward = 1 - min_distance
        else:
            reward = 1 - distance
            min_distance = distance
        '''
        done = False

        print("***** Distance *****")
        print(distance)
        print("*** Action value ***")
        print(action)
        print("*** State value ***")
        print(new_state)
        print("*** Reward value ***")
        print(reward)

        info = {}

        return new_state, reward, done, info, operator.confidence_map
