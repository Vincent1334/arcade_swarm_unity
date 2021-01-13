import time
import os
import arcade
import argparse
import gym
from gym import spaces
import swarm_env
import numpy as np
import random
import sys
sys.path.insert(0, '..')
from objects import SwarmSimulator

# Running experiment 22 in standalone file.

def experiment_runner(SWARM_SIZE = 15, ARENA_WIDTH = 600, ARENA_HEIGHT = 600, name_of_experiment = time.time(), INPUT_TIME = 300, GRID_X = 40, GRID_Y = 40,
               disaster_size = 1, disaster_location = 'random', operator_size = 1, operator_location = 'random', reliability = (100, 101), unreliability_percentage = 0, 
               moving_disaster = False, communication_noise = 0, alpha = 10, normal_command = None, command_period = 0, constant_repulsion = False, 
               operator_vision_radius = 150, communication_range = 8, vision_range = 2, velocity_weight_coef = 0.01, boundary_repulsion = 1, aging_factor = 0.9999,
               gp = False, gp_step = 50, maze = None, through_walls = True, rl_sim = None):

    ########### q-learning parameter setup #############

    max_steps_per_episode = 10 # Steps allowed in a single episode.

    learning_rate = 0.1 # alpha in bellman.
    discount_rate = 0.99 # gamma in bellman for discount.

    # Epsilon greedy policy vars.
    exploration_rate = 1 # To set exploration (1 means 100% exploration)
    max_exploration_rate = 1 # How large can exploration be.
    min_exploration_rate = 0.01 # How small can exploration be.
    exploration_decay_rate = 0.001 # decay rate for exploration.
    rewards_all_episodes = [] # Saving all scores in rewards.

    gym_swarm_env = gym.make('humanswarm-v0', maze_size=GRID_X) # Creating the environment for swarm learning.
    gym_swarm_env.action_space = np.zeros((GRID_X, GRID_Y))
    q_table = np.zeros((gym_swarm_env.observation_space.n , gym_swarm_env.action_space.size)) # Creating q-table for measuring score.
    action = np.zeros((gym_swarm_env.action_space.size))

    print('\n')
    print("===== Reinforcement Parameters =====")
    print("# Discount rate: " + str(discount_rate))
    print("# Learning rate: " + str(learning_rate))
    print("# Max steps per iteration: " + str(max_steps_per_episode))
    print("# Max exploration rate: " + str(max_exploration_rate))
    print("# Min exploration rate: " + str(min_exploration_rate))
    print("# Exploration decay rate: " + str(exploration_decay_rate))
    print("# Algorithm: " + str(rl_sim))
    print("# State space size: " + str(gym_swarm_env.observation_space.n))
    print("# Action space size: " + str(gym_swarm_env.action_space.size))
    print("# Q-table size: " + str(q_table.shape))
    print("====================================")
    print('\n')

    # Implemeting Q-learning algorithm.
    done = False
    state = gym_swarm_env.reset()
    s_list = []
    for step in range(max_steps_per_episode):
        print('\n' + "============ start of step " + str(step) + " =============")
        """
        In this loop we will set up exploration-exploitation trade-off,
        Taking new action,
        Updating Q-table,
        Setting new state,
        Adding new reward.
        """
        # Simulation functions
        sim = SwarmSimulator(ARENA_WIDTH, ARENA_HEIGHT, name_of_experiment, SWARM_SIZE, INPUT_TIME, GRID_X, GRID_Y, rl_sim)
        sim.setup(disaster_size, disaster_location, operator_size, operator_location, reliability[0], reliability[1], unreliability_percentage, moving_disaster, communication_noise, 
                alpha, normal_command, command_period, constant_repulsion, operator_vision_radius,
                communication_range, vision_range, velocity_weight_coef, boundary_repulsion, aging_factor, gp, gp_step, maze, through_walls)

        if (not os.path.isdir('../outputs/' + name_of_experiment)):
            os.mkdir('../outputs/' + name_of_experiment)
        if (not os.path.isdir('../outputs/' + name_of_experiment + '/step_' + str(step))):
            os.mkdir('../outputs/' + name_of_experiment + '/step_' + str(step))
        if (not os.path.isdir('../outputs/' + name_of_experiment + '/step_' + str(step) + '/data')):
            os.mkdir('../outputs/' + name_of_experiment + '/step_' + str(step) + '/data')
        if (not os.path.isdir('../outputs/' + name_of_experiment + '/step_' + str(step) + '/data' + '/results')):
            os.mkdir('../outputs/' + name_of_experiment + '/step_' + str(step) + '/data' + '/results')

        sim.directory = str('../outputs/' + name_of_experiment + '/data/results/'+ str(time.time()))
        
        while os.path.isdir(sim.directory):
            sim.directory = str('../outputs/' + name_of_experiment + '/step_'+ str(step) + '/data/results/' + str(time.time()))

        sim.directory = str('../outputs/' + name_of_experiment + '/step_'+ str(step) + '/data/results/'+ str(time.time()))
        
        while os.path.isdir(sim.directory):
            sim.directory = str('../outputs/' + name_of_experiment + '/step_'+ str(step) + '/data/results/' + str(time.time()))

        directory = sim.directory
            
        os.mkdir(directory)
        sim.log_setup(directory)
        # Adding new RL parameters to log #
        with open(directory + "/log_setup.txt", "a") as file:
            file.write('\n')
            file.write('REINFORCEMENT LEARNING INFO:' + '\n')
            file.write('  -- DISCOUNT RATE: ' + str(discount_rate) + '\n')
            file.write('  -- LEARNING RATE: ' + str(learning_rate) + '\n')
            file.write('  -- MAX STEPS PER ITERATION: ' + str(max_steps_per_episode) + '\n')
            file.write('  -- MAX EXPLORATION RATE: ' + str(max_exploration_rate) + '\n')
            file.write('  -- MIN EXPLORATION RATE: ' + str(min_exploration_rate) + '\n')
            file.write('  -- EXPLORATION DECAY RATE: ' + str(exploration_decay_rate) + '\n')
            file.write('  -- ALGORITHM: ' + str(rl_sim) + '\n')        
            file.write('  -- STATE SPACE SIZE: ' + str(gym_swarm_env.observation_space.n) + '\n')
            file.write('  -- ACTION SPACE SIZE: ' + str(gym_swarm_env.action_space.size) + '\n')
            file.write('  -- Q-TABLE SIZE: ' + str(q_table.shape) + '\n')
        arcade.run()

        ########################

        ##### Exploration and explotation block. ####

        exploration_rate_threshold = random.uniform(0, 1) # Setting a random number that will be compared to exploration_rate.
        if exploration_rate_threshold > exploration_rate:
            i, j = np.unravel_index(np.argmax(q_table[state, :]), q_table.shape)
            print ("i ", i, " , j ", j)
            #action = (i, j) # Choosing the action that had the highest q-value in q-table.
            action = i*GRID_X + j  # Choosing the action that had the highest q-value in q-table.
            #print (action)
            #exit(0)
        else:
            i = random.randint(0, GRID_X - 1)
            j = random.randint(0, GRID_Y - 1)
            action = i*GRID_X + j # Sample an action randomly to explore.
        ##### Exploration and explotation block. ####

        ##### Taking appropriate action after choosing the action. ####
        new_state, reward, done, info, operator_cm = gym_swarm_env.step(action, sim.operator_list[0], GRID_X, GRID_Y) # Returns a tuple contaning the new state, the reward for this action, the end status of action, some additional info.

        sim.operator_list[0].confidence_map = operator_cm

        # Updating q-table values                
        q_table[state, action]=q_table[state, action] * (1 - learning_rate) + \
                                learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        print('*** State-Action pair in Q-table ***')
        print('Q[' + str(state) + ',  ' + str(action) + '] = '+ str(q_table[state, action]))

        state = new_state
        if done == True:
            break
        
        ##### Taking appropriate action after choosing the action. ####
        print("============= End of step " + str(step) + " =============")
        
        """
        # logging q-table
        if self.directory == None:      
            self.q_table.tofile(self.directory + "/q_table" + "_step" + str(step) + "_timer" + str(self.timer) + ".txt", sep=" ", format="%s")
        else:
            self.q_table.tofile(self.directory + "/q_table" + "_step" + str(step) + "_timer" + str(self.timer) + ".txt", sep=" ", format="%s")
        """
        
        # Decay exploration rate using a formula.
        exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * step)
    ######## END of q-learning parameter setup #########


if __name__ == '__main__':
    experiment_runner(operator_vision_radius=40, INPUT_TIME=0, command_period=200, alpha=10, moving_disaster=False, disaster_location=[(500, 500)], operator_location=[(450, 300)], name_of_experiment='RL model experiment_r40_t200')