import time
import os
import arcade
import simulation
import argparse

# Simply collects the belief error and the confidence of the swarm at each 5 steps
# Could be used with different swarm sizes, reliability ranges and percentages, and communication noise
def init(SWARM_SIZE = 15, ARENA_WIDTH = 600, ARENA_HEIGHT = 600, name_of_experiment = time.time(), run_time = 1000, INPUT_TIME = 300, GRID_X = 40, GRID_Y = 40,
               disaster_size = 1, disaster_location = 'random', operator_size = 1, operator_location = 'random', reliability = (100, 101), unreliability_percentage = 0, 
               moving_disaster = False, communication_noise = 0, alpha = 10, normal_command = None, command_period = 0, constant_repulsion = False, 
               operator_vision_radius = 150, communication_range = 8, vision_range = 2, velocity_weight_coef = 0.01, boundary_repulsion = 1, aging_factor = 0.9999,
               gp = False, gp_step = 50, maze = None, through_walls = True,
               communication_noise_strength = 0, communication_noise_prob = 0, positioning_noise_strength = 0, positioning_noise_prob = 0, sensing_noise_strength = 0, sensing_noise_prob = 0):
    
    sim = simulation.SwarmSimulator(ARENA_WIDTH, ARENA_HEIGHT, name_of_experiment, SWARM_SIZE, run_time, INPUT_TIME, GRID_X, GRID_Y)
    
    sim.setup(disaster_size, disaster_location, operator_size, operator_location, reliability[0], reliability[1], unreliability_percentage, moving_disaster, communication_noise, 
              alpha, normal_command, command_period, constant_repulsion, operator_vision_radius,
              communication_range, vision_range, velocity_weight_coef, boundary_repulsion, aging_factor, gp, gp_step, maze, through_walls,
              communication_noise_strength, communication_noise_prob, positioning_noise_strength, positioning_noise_prob,sensing_noise_strength, sensing_noise_prob)  

    #if (not os.path.isdir(name_of_experiment)):
    #    os.mkdir(name_of_experiment)
    #if (not os.path.isdir(name_of_experiment + '/data')):
    #    os.mkdir(name_of_experiment + '/data')
         
    sim.directory = str(name_of_experiment) + '/'+ str(time.time())

    directory = sim.directory
        
    os.makedirs(directory)
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
    parser.add_argument('-run_time', type = int, default = 1000) #communication through walls


    # New experiments
    parser.add_argument('-communication_noise_strength', type = float, default = 0) 
    parser.add_argument('-communication_noise_prob', type = float, default = 0) 

    parser.add_argument('-positioning_noise_strength', type = float, default = 0) 
    parser.add_argument('-positioning_noise_prob', type = float, default = 0) 
    
    parser.add_argument('-sensing_noise_strength', type = float, default = 0) 
    parser.add_argument('-sensing_noise_prob', type = float, default = 0) 
    
    args = parser.parse_args()
    
    disasters_locations = merge(args.d_xs, args.d_ys)
    operators_locations = merge(args.op_xs, args.op_ys)
    
    if args.d_size > len(args.d_xs):
        disasters_locations += [('random', 'random')]*(args.d_size - len(args.d_xs))
        
    if args.op_size > len(args.op_xs):
        operators_locations += [('random', 'random')]*(args.op_size - len(args.op_xs))

    init(args.size, args.width, args.height, args.name, args.run_time, args.input_time, args.grid_x, args.grid_y, len(disasters_locations), disasters_locations, 
                   len(operators_locations), operators_locations, (args.r_min, args.r_max), args.r_perc, args.noise, args.d_move, args.alpha, args.cmd, 
                   args.cmd_t, args.const_repel, args.hum_r, args.comm_range, args.vis_range, args.w, args.bound, args.aging, args.gp, args.gp_step,
                   args.maze, args.walls,
                   args.communication_noise_strength, args.communication_noise_prob,
                   args.positioning_noise_strength, args.positioning_noise_prob,
                   args.sensing_noise_strength, args.sensing_noise_prob)
