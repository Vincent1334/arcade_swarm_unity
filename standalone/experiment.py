import os
import time
import random
import argparse
from datetime import datetime
import arcade
import gym
from gym import spaces
import torch
import torch.optim as optim
import numpy as np
import deep_q_modules
from deep_q_modules import DQN, Experience, ReplayMemory, EpsilonGreedyStrategy, Agent, QValues, extract_tensors
import objects


# Simply collects the belief error and the confidence of the swarm at each 5 steps
# Could be used with different swarm sizes, reliability ranges and percentages, and communication noise
def experiment(SWARM_SIZE = 15, ARENA_WIDTH = 600, ARENA_HEIGHT = 600, name_of_experiment = time.time(), INPUT_TIME = 300, GRID_X = 40, GRID_Y = 40,
               disaster_size = 1, disaster_location = 'random', operator_size = 1, operator_location = 'random', reliability = (100, 101), unreliability_percentage = 0, 
               moving_disaster = False, communication_noise = 0, alpha = 10, normal_command = None, command_period = 0, constant_repulsion = False, 
               operator_vision_radius = 150, communication_range = 8, vision_range = 2, velocity_weight_coef = 0.01, boundary_repulsion = 1, aging_factor = 0.9999,
               gp = False, gp_step = 50, maze = None, through_walls = True, rl_sim = None, learning_iteration = 40, sim_timer = 1000):

    ###### In case of a reinforcement learning simulation experiment. ######

    ########### q-learning parameter setup #############
    if rl_sim == 'q-learning':
        
        max_steps_per_episode = learning_iteration # Steps allowed in a single episode.


        learning_rate = 0.1 # alpha in bellman.
        discount_rate = 0.99 # gamma in bellman for discount.

        # Epsilon greedy policy vars.
        exploration_rate = 1 # To set exploration (1 means 100% exploration)
        max_exploration_rate = 1 # How large can exploration be.
        min_exploration_rate = 0.01 # How small can exploration be.
        exploration_decay_rate = 0.001 # decay rate for exploration.

        min_distance = 2*600
        
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
        print("# Min distance: " + str(min_distance))
        print("# Exploration decay rate: " + str(exploration_decay_rate))
        print("# Algorithm: " + str(rl_sim))
        print("# State space size: " + str(gym_swarm_env.observation_space.n))
        print("# Action space size: " + str(gym_swarm_env.action_space.size))
        print("# Q-table size: " + str(q_table.shape))
        print("====================================")
        print('\n')

        # Implemeting Q-learning algorithm.
        state = gym_swarm_env.reset()
        cm_list = []
        reward = -10000
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
            rl_name_of_experiment = name_of_experiment + "_learningCount: " + str(step)
            sim = objects.SwarmSimulator(ARENA_WIDTH, ARENA_HEIGHT, rl_name_of_experiment, SWARM_SIZE, INPUT_TIME, GRID_X, GRID_Y, rl_sim, sim_timer)
            sim.setup(disaster_size, disaster_location, operator_size, operator_location, reliability[0], reliability[1], unreliability_percentage, moving_disaster, communication_noise, 
                    alpha, normal_command, command_period, constant_repulsion, operator_vision_radius,
                    communication_range, vision_range, velocity_weight_coef, boundary_repulsion, aging_factor, gp, gp_step, maze, through_walls)

            if (not os.path.isdir('outputs/' + name_of_experiment)):
                os.mkdir('outputs/' + name_of_experiment)
            if (not os.path.isdir('outputs/' + name_of_experiment + '/step_' + str(step))):
                os.mkdir('outputs/' + name_of_experiment + '/step_' + str(step))
            if (not os.path.isdir('outputs/' + name_of_experiment + '/step_' + str(step) + '/data')):
                os.mkdir('outputs/' + name_of_experiment + '/step_' + str(step) + '/data')
            if (not os.path.isdir('outputs/' + name_of_experiment + '/step_' + str(step) + '/data' + '/results')):
                os.mkdir('outputs/' + name_of_experiment + '/step_' + str(step) + '/data' + '/results')

            sim.directory = str('outputs/' + name_of_experiment + '/data/results/'+ str(time.time()))
            
            while os.path.isdir(sim.directory):
                sim.directory = str('outputs/' + name_of_experiment + '/step_'+ str(step) + '/data/results/' + str(time.time()))

            sim.directory = str('outputs/' + name_of_experiment + '/step_'+ str(step) + '/data/results/'+ str(time.time()))
            
            while os.path.isdir(sim.directory):
                sim.directory = str('outputs/' + name_of_experiment + '/step_'+ str(step) + '/data/results/' + str(time.time()))

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
                file.write('  -- MIN DISTANCE: ' + str(min_distance) + '\n')
                file.write('  -- EXPLORATION DECAY RATE: ' + str(exploration_decay_rate) + '\n')
                file.write('  -- ALGORITHM: ' + str(rl_sim) + '\n')        
                file.write('  -- STATE SPACE SIZE: ' + str(gym_swarm_env.observation_space.n) + '\n')
                file.write('  -- ACTION SPACE SIZE: ' + str(gym_swarm_env.action_space.size) + '\n')
                file.write('  -- Q-TABLE SIZE: ' + str(q_table.shape) + '\n')

            ########################

            ##### Exploration and explotation block. ####

            exploration_rate_threshold = random.uniform(0, 1) # Setting a random number that will be compared to exploration_rate.
            if exploration_rate_threshold > exploration_rate:
                i, j = np.unravel_index(np.argmax(q_table[state, :]), q_table.shape)
                #print ("i ", i, " , j ", j)
                #action = (i, j) # Choosing the action that had the highest q-value in q-table.
                action = i*GRID_X + j  # Choosing the action that had the highest q-value in q-table.
                #print (action)
                #exit(0)
            else:
                i = random.randint(0, GRID_X - 1)
                j = random.randint(0, GRID_Y - 1)
                action = i*GRID_X + j # Sample an action randomly to explore.

            ##### Exploration and explotation block. ####
            
            #if step > 0:
            #    sim.operator_list[0].confidence_map = cm_list[-1]
            
            sim.operator_list[0].confidence_map= np.array([[0.0 for i in range(GRID_X)] for j in range(GRID_Y)])
            
            #new_state, new_reward, done, info, operator_cm, min_distance = gym_swarm_env.step(action, sim.disaster_list[0], sim.drone_list[0], sim.operator_list[0], GRID_X, GRID_Y, min_distance) # Returns a tuple contaning the new state, the reward for this action, the end status of action, some additional info.
            new_state, new_reward, done, info, operator_cm = gym_swarm_env.step(action, sim.disaster_list[0], sim.drone_list[0], sim.operator_list[0], GRID_X, GRID_Y) # Returns a tuple contaning the new state, the reward for this action, the end status of action, some additional info.
            
            for i in range(GRID_Y):
                for j in range(GRID_X): 
                    if (operator_cm[i][j] < 0): 
                        print ("i,j: ", i , " , ", j)
                        
            arcade.run()
            
            ##### Taking appropriate action after choosing the action. ####
            #new_state, new_reward, done, info, operator_cm, min_distance = gym_swarm_env.step(action, sim.disaster_list[0], sim.drone_list[0], sim.operator_list[0], GRID_X, GRID_Y, min_distance) # Returns a tuple contaning the new state, the reward for this action, the end status of action, some additional info.
            new_state, new_reward, done, info, operator_cm = gym_swarm_env.step(action, sim.disaster_list[0], sim.drone_list[0], sim.operator_list[0], GRID_X, GRID_Y) # Returns a tuple contaning the new state, the reward for this action, the end status of action, some additional info.
            

            if step != 0:
                if new_reward > reward:
                    reward = new_reward
                    cm_list.append(operator_cm)
            else:
                cm_list.append(operator_cm)


            # Updating q-table values                
            q_table[state, action]=q_table[state, action] * (1 - learning_rate) + \
                                    learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

            print('*** State-Action pair in Q-table ***')
            print('Q[' + str(state) + ',  ' + str(action) + '] = '+ str(q_table[state, action]))

            state = new_state
            
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

            # Saving images of plots
            os.makedirs(directory + '/map_images')
            sim.save_image_plot_heatmaps(sim.operator_confidence_maps, 'operator confidence', directory)
            sim.save_image_plot_heatmaps(sim.operator_belief_maps, 'operator belief', directory)

            sim.save_image_plot_boxplots(sim.operator_confidence, 'operator_confidence_time', directory)
            sim.save_image_plot_boxplots(sim.operator_internal_error, 'operator_belief_error', directory)
        #for str(q_table[state, action]))
    ######## END of q-learning parameter setup #########

    ########## deep_q_learning parameter setup #########
    elif rl_sim == 'deep_q_learning':

        max_steps_per_episode = learning_iteration # Steps allowed in a single episode.

        min_distance = 2*600

        batch_size = 256
        gamma = 0.999
        eps_start = 1
        eps_end = 0.01
        eps_decay = 0.001
        target_update = 10
        memory_size = 100000
        lr = 0.001

        # Q-table & Gym environment definition
        gym_swarm_env = gym.make('humanswarm-v0', maze_size=GRID_X) # Creating the environment for swarm learning.
        gym_swarm_env.action_space = np.zeros((GRID_X, GRID_Y))
        q_table = np.zeros((gym_swarm_env.observation_space.n , gym_swarm_env.action_space.size)) # Creating q-table for measuring score.
        action = np.zeros((gym_swarm_env.action_space.size))

        # Memory & agent definition
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
        agent = Agent(strategy, gym_swarm_env.action_space.size , device)
        memory = ReplayMemory(memory_size)
        
        # Neural Network layers
        policy_net = DQN(gym_swarm_env.observation_space.n , gym_swarm_env.action_space.size).to(device)
        target_net = DQN(gym_swarm_env.observation_space.n , gym_swarm_env.action_space.size).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
        optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

        episode_durations = []

        print('\n')
        print("===== Reinforcement Parameters =====")
        print("# Learning Rate: " + str(lr))
        print("# Batch Size: " + str(batch_size))
        print("# Bellman Discount(Gamma): " + str(gamma))
        print("# Max steps per iteration: " + str(max_steps_per_episode))
        print("# Episode Start(exploration rate): " + str(eps_start))
        print("# Episode End(exploration rate): " + str(eps_end))
        print("# Episode Decay: " + str(min_distance))
        print("# Target Update: " + str(target_update))
        print("# Memory Size: " + str())
        print("# Algorithm: " + str(rl_sim))
        print("# State space size: " + str(gym_swarm_env.observation_space.n))
        print("# Action space size: " + str(gym_swarm_env.action_space.size))
        print("# Q-table size: " + str(q_table.shape))
        print("====================================")
        print('\n')

        # Implemeting Q-learning algorithm.
        gym_swarm_env.reset()
        state = deep_q_modules.get_state(gym_swarm_env)
        cm_list = []
        episode_durations = []
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
            rl_name_of_experiment = name_of_experiment + "_learningCount: " + str(step)
            sim = objects.SwarmSimulator(ARENA_WIDTH, ARENA_HEIGHT, rl_name_of_experiment, SWARM_SIZE, INPUT_TIME, GRID_X, GRID_Y, rl_sim, sim_timer)
            sim.setup(disaster_size, disaster_location, operator_size, operator_location, reliability[0], reliability[1], unreliability_percentage, moving_disaster, communication_noise, 
                    alpha, normal_command, command_period, constant_repulsion, operator_vision_radius,
                    communication_range, vision_range, velocity_weight_coef, boundary_repulsion, aging_factor, gp, gp_step, maze, through_walls)

            if (not os.path.isdir('outputs/' + name_of_experiment)):
                os.mkdir('outputs/' + name_of_experiment)
            if (not os.path.isdir('outputs/' + name_of_experiment + '/step_' + str(step))):
                os.mkdir('outputs/' + name_of_experiment + '/step_' + str(step))
            if (not os.path.isdir('outputs/' + name_of_experiment + '/step_' + str(step) + '/data')):
                os.mkdir('outputs/' + name_of_experiment + '/step_' + str(step) + '/data')
            if (not os.path.isdir('outputs/' + name_of_experiment + '/step_' + str(step) + '/data' + '/results')):
                os.mkdir('outputs/' + name_of_experiment + '/step_' + str(step) + '/data' + '/results')

            sim.directory = str('outputs/' + name_of_experiment + '/data/results/'+ str(time.time()))
            
            while os.path.isdir(sim.directory):
                sim.directory = str('outputs/' + name_of_experiment + '/step_'+ str(step) + '/data/results/' + str(time.time()))

            sim.directory = str('outputs/' + name_of_experiment + '/step_'+ str(step) + '/data/results/'+ str(time.time()))
            
            while os.path.isdir(sim.directory):
                sim.directory = str('outputs/' + name_of_experiment + '/step_'+ str(step) + '/data/results/' + str(time.time()))

            directory = sim.directory
                
            os.mkdir(directory)
            sim.log_setup(directory)

            # Adding new RL parameters to log #
            with open(directory + "/log_setup.txt", "a") as file:
                file.write('\n')
                file.write('REINFORCEMENT LEARNING INFO:' + '\n')
                file.write('  -- LEARNING RATE: ' + str(lr) + '\n')
                file.write('  -- BATCH SIZE: ' + str(batch_size) + '\n')
                file.write('  -- BELLMAN DISCOUNT: ' + str(gamma) + '\n')
                file.write('  -- MAX STEPS PER ITERATION: ' + str(max_steps_per_episode) + '\n')
                file.write('  -- EPISODE START: ' + str(eps_start) + '\n')
                file.write('  -- EPISODE END: ' + str(eps_end) + '\n')
                file.write('  -- MIN DISTANCE: ' + str(min_distance) + '\n')
                file.write('  -- EPISODE DECAY: ' + str(eps_decay) + '\n')
                file.write('  -- TARGET UPDATE: ' + str(target_update) + '\n')
                file.write('  -- MEMORY SIZE: ' + str(memory_size) + '\n')
                file.write('  -- ALGORITHM: ' + str(rl_sim) + '\n')        
                file.write('  -- STATE SPACE SIZE: ' + str(gym_swarm_env.observation_space.n) + '\n')
                file.write('  -- ACTION SPACE SIZE: ' + str(gym_swarm_env.action_space.size) + '\n')
                file.write('  -- Q-TABLE SIZE: ' + str(q_table.shape) + '\n')

            ########################
            ###################################################################################################################
            ####################### Running simulation before pushing experiment to replay memory #############################
            ###################################################################################################################
            arcade.run()

            action = deep_q_modules.Agent.select_action(state, policy_net)
            reward = deep_q_modules.take_action(action)
            next_state = deep_q_modules.get_state(gym_swarm_env)
            memory.push(Experience(state, action, next_state, reward))
            state = next_state

            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)

                # Getting the experiment tensors (states, actions, rewards, next_states)
                states, actions, rewards, next_states = deep_q_modules.extract_tensors(experiences)

                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards

                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if gym_swarm_env.done:
                    episode_durations.append(timestep)
                    plot(episode_durations, 100)
                    break

                if episode % target_update == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            # Saving images of plots
            os.makedirs(directory + '/map_images')
            sim.save_image_plot_heatmaps(sim.operator_confidence_maps, 'operator confidence', directory)
            sim.save_image_plot_heatmaps(sim.operator_belief_maps, 'operator belief', directory)

            sim.save_image_plot_boxplots(sim.operator_confidence, 'operator_confidence_time', directory)
            sim.save_image_plot_boxplots(sim.operator_internal_error, 'operator_belief_error', directory)
        #for str(q_table[state, action]))
    ######## END of deep_q_learning parameter setup #########
    else:
        sim = objects.SwarmSimulator(ARENA_WIDTH, ARENA_HEIGHT, name_of_experiment, SWARM_SIZE, INPUT_TIME, GRID_X, GRID_Y, rl_sim, sim_timer)
        
        sim.setup(disaster_size, disaster_location, operator_size, operator_location, reliability[0], reliability[1], unreliability_percentage, moving_disaster, communication_noise, 
                alpha, normal_command, command_period, constant_repulsion, operator_vision_radius,
                communication_range, vision_range, velocity_weight_coef, boundary_repulsion, aging_factor, gp, gp_step, maze, through_walls)  

        if not os.path.isdir('outputs'):
            os.mkdir('outputs')
        if (not os.path.isdir('outputs/' + name_of_experiment)):
            os.mkdir('outputs/' + name_of_experiment)
        if (not os.path.isdir('outputs/' + name_of_experiment + '/data')):
            os.mkdir('outputs/' + name_of_experiment + '/data')
        if (not os.path.isdir('outputs/' + name_of_experiment + '/data' + '/results')):
            os.mkdir('outputs/' + name_of_experiment + '/data' + '/results')

        sim.directory = str('outputs/' + name_of_experiment + '/data/results/'+ str(time.time()))
        
        while os.path.isdir(sim.directory):
            sim.directory = str('outputs/' + name_of_experiment + '/data/results/' + str(time.time()))

        directory = sim.directory
            
        os.mkdir(directory)
        sim.log_setup(directory)      
        arcade.run()                 
        
        #sim.plot_heatmaps(sim.random_drone_confidence_maps, 'Random drone confidence')
        #sim.plot_heatmaps(sim.random_drone_belief_maps, 'Random drone belief')
        
        #sim.plot_boxplots(sim.swarm_confidence, 'Swarm confidence over time')
        #sim.plot_boxplots(sim.swarm_internal_error, 'Swarm belief map error over time')
        
        #sim.plot_heatmaps(sim.operator_confidence_maps, 'Operator confidence')
        #sim.plot_heatmaps(sim.operator_belief_maps, 'Operator belief')
        
        #sim.plot_boxplots(sim.operator_confidence, 'Operator confidence over time')
        #sim.plot_boxplots(sim.operator_internal_error, 'Operator belief map error over time')
        
        sim.save_positions(sim, directory)
        sim.save_boxplots(sim.swarm_confidence, 'confidence_time', directory)
        sim.save_boxplots(sim.swarm_internal_error, 'belief_error', directory)

        sim.save_boxplots(sim.operator_confidence, 'operator_confidence_time', directory)
        sim.save_boxplots(sim.operator_internal_error, 'operator_belief_error', directory)
        sim.save_boxplots(sim.operator_internal_error, 'operator_belief_error', directory)
        
        # Saving images of plots
        os.makedirs(directory + '/map_images')
        sim.save_image_plot_heatmaps(sim.operator_confidence_maps, 'operator confidence', directory)
        sim.save_image_plot_heatmaps(sim.operator_belief_maps, 'operator belief', directory)

        sim.save_image_plot_boxplots(sim.operator_confidence, 'operator_confidence_time', directory)
        sim.save_image_plot_boxplots(sim.operator_internal_error, 'operator_belief_error', directory)
        print('END')

def merge(list1, list2):       
    merged_list = [] 
    for i in range(max((len(list1), len(list2)))):   
        while True: 
            try: 
                tup = (list1[i], list2[i]) 
            except IndexError: 
                if len(list1) > len(list2): 
                    list2.append('') 
                    tup = (list1[i], list2[i]) 
                elif len(list1) < len(list2): 
                    list1.append('') 
                    tup = (list1[i], list2[i]) 
                continue  
            merged_list.append(tup)
            break
    return merged_list 

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-size', type = int, default = 15) #swarm_size
    parser.add_argument('-name', type = str, default = str(time.time())) #experiment_name
    parser.add_argument('-d_size', type=int, default = 1)
    parser.add_argument('-d_xs', nargs='+', type=int, default = [500])
    parser.add_argument('-d_ys', nargs='+', type=int, default = [500])
    parser.add_argument('-d_move', type = bool, default = False)#moving disaster
    parser.add_argument('-op_size', type=int, default = 1)
    parser.add_argument('-op_xs', nargs='+', type=int, default = [450])
    parser.add_argument('-op_ys', nargs='+', type=int, default = [300])
    parser.add_argument('-noise', type = int, default = 0) #communication_noise
    parser.add_argument('-r_min', type = int, default = 100) #min_reliability
    parser.add_argument('-r_max', type = int, default = 100) #max_reliability
    parser.add_argument('-r_perc', type = int, default = 0) #unreliability_percentage
    parser.add_argument('-cmd', type = str, default = None) #normal_command
    parser.add_argument('-cmd_t', type = int, default = 0) #command_period
    parser.add_argument('-const_repel', type = bool, default = False) #constant_repulsion
    parser.add_argument('-alpha', type = float, default = 10) #command strength
    parser.add_argument('-comm_range', type = int, default = 4) #communication_range
    parser.add_argument('-vis_range', type = int, default = 2) #vision_range
    parser.add_argument('-w', type = float, default = 0.01) #velocity_weight_coef
    parser.add_argument('-bound', type = float, default = 1) #boundary_repulsion
    parser.add_argument('-aging', type = float, default = 0.9999) #boundary_repulsion
    parser.add_argument('-hum_r', type = int, default = 100)#operator_vision_radius    
    parser.add_argument('-height', type = int, default = 600) #arena_height
    parser.add_argument('-width', type = int, default = 600) #arena_width
    parser.add_argument('-grid_x', type = int, default = 40) #grid_x
    parser.add_argument('-grid_y', type = int, default = 40) #grid_y
    parser.add_argument('-input_time', type = int, default = 300) #input_time
    parser.add_argument('-gp', type = bool, default = False) #gaussian processes
    parser.add_argument('-gp_step', type = int, default = 50) #gaussian processes step
    parser.add_argument('-maze', type = str, default = None) #maze
    parser.add_argument('-walls', type = bool, default = False) #communication through walls
    parser.add_argument('-rl_sim', type = str, default = None) #define reinforcement learning algorithm
    parser.add_argument('-learning_iteration', type = int, default = None) #defines learning iteration
    parser.add_argument('-sim_timer', type = int, default = 1000) #assign's agent's active time on each iteration
    args = parser.parse_args()
    
    disasters_locations = merge(args.d_xs, args.d_ys)
    operators_locations = merge(args.op_xs, args.op_ys)
    
    if args.d_size > len(args.d_xs):
        disasters_locations += [('random', 'random')]*(args.d_size - len(args.d_xs))
        
    if args.op_size > len(args.op_xs):
        operators_locations += [('random', 'random')]*(args.op_size - len(args.op_xs))

    experiment(args.size, args.width, args.height, args.name, args.input_time, args.grid_x, args.grid_y, len(disasters_locations), disasters_locations, 
                   len(operators_locations), operators_locations, (args.r_min, args.r_max), args.r_perc, args.noise, args.d_move, args.alpha, args.cmd, 
                   args.cmd_t, args.const_repel, args.hum_r, args.comm_range, args.vis_range, args.w, args.bound, args.aging, args.gp, args.gp_step,
                   args.maze, args.walls, args.rl_sim, args.learning_iteration, args.sim_timer)
