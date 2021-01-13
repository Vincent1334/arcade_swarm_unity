import sys
import subprocess
import re
from datetime import datetime

''' COMMANDS:
-size            type = int     default = 15           swarm_size
-name            type = str     default = time.time()  experiment_name
-d_size          type = int     default = 1            number of disasters
-d_xs            type = int     default = [500]        disaster_x's
-d_ys            type = int     default = [500]        disaster_y's
-d_move          type = bool    default = False        moving_disaster
-op_size         type = int     default = 1            operator_size
-op_x            type = int     default = [450]        operator_x's
-op_y            type = int     default = [300]        operator_y's
-noise           type = int     default = 0            communication_noise
-r_min           type = int     default = 100          min_reliability
-r_max           type = int     default = 100          max_reliability
-r_perc          type = int     default = 0            unreliability_percentage
-cmd             type = str     default = None         normal_command
-cmd_t           type = int     default = 0            command_period
-const_repel     type = bool    default = False        constant_repulsion
-alpha           type = int     default = 10           command strength
-comm_range      type = int     default = 8            communication_range
-vis_range       type = int     default = 2            vision_range
-w               type = int     default = 0.01         velocity_weight_coef
-bound           type = int     default = 1            boundary_repulsion
-aging           type = int     default = 0.9999       boundary_repulsion
-hum_r           type = int     default = 150          operator_vision_radius
-height          type = int     default = 600          arena_height
-width           type = int     default = 600          arena_width
-grid_x          type = int     default = 40           grid_x
-grid_y          type = int     default = 40           grid_y
-input_time      type = int     default = 300          input_time
-gp              type = bool    default = False        gaussian processes
-gp_step         type = int     default = 50           gaussian processes_step
-maze            type = str     default = True         maze type
-rl_sim          type = str     default = None         reinforcement learning algorithm
-learning_iteration type = int  default = 10           defines learning iteration
-sim_timer       type = int     default = 1000         assigns agent's active time on each iteration
'''

def different_sizes(swarm_sizes = [1, 5, 10, 15, 20, 25, 50]):    
    cmds = []   
    for size in swarm_sizes:             
        cmds.append('-name "first experiment_{0}" -d_x 500 -d_y 500'.format(size) + ' -size ' + size)
        cmds.append('-name "first experiment_{0}" -d_x 100 -d_y 500'.format(size) + ' -size ' + size)
        cmds.append('-name "first experiment_{0}" -d_x 300 -d_y 100'.format(size) + ' -size ' + size)
    return cmds    

def ordinary_experiment():    
    cmds = []                
    cmds.append('-name "second experiment" -d_x 500 -d_y 500')
    cmds.append('-name "second experiment" -d_x 100 -d_y 500')
    cmds.append('-name "second experiment" -d_x 300 -d_y 100')
    return cmds
    
def reliability_experiment(min_reliability = 95, max_reliability = 100, unreliability_percentage = [0, 20, 50]):    
    cmds = []    
    addition = ' -r_min '+ str(min_reliability) + ' -r_max '+ str(max_reliability)     
    for level in unreliability_percentage:     
        cmds.append('-name "third experiment_{0}" -d_x 500 -d_y 500'.format(level) + addition + ' -r_perc ' + str(level) )
        cmds.append('-name "third experiment_{0}" -d_x 100 -d_y 500'.format(level) + addition + ' -r_perc ' + str(level) )
        cmds.append('-name "third experiment_{0}" -d_x 300 -d_y 100'.format(level) + addition + ' -r_perc ' + str(level) )
    return cmds
    
def noise_experiment(noise_levels = [0, 10, 15]):    
    cmds = []        
    for level in noise_levels:        
        cmds.append('-name "fourth experiment_{0}" -d_x 500 -d_y 500 -noise '.format(level) + str(level))
        cmds.append('-name "fourth experiment_{0}" -d_x 100 -d_y 500 -noise '.format(level) + str(level))
        cmds.append('-name "fourth experiment_{0}" -d_x 300 -d_y 100 -noise '.format(level) + str(level))
    return cmds
 
def ordinary_repulsion(ranges = [50, 100, 150], t = 25):    
    cmds = []        
    addition = ' -cmd "center repel" -cmd_t ' + str(t) 
    for r in ranges:        
        cmds.append('-name "ordinary repulsion experiment_r{0}_t{1}" -d_x 500 -d_y 500'.format(r, t) + addition + ' -hum_r ' + str(r))
        cmds.append('-name "ordinary repulsion experiment_r{0}_t{1}" -d_x 100 -d_y 500'.format(r, t) + addition + ' -hum_r ' + str(r))
        cmds.append('-name "ordinary repulsion experiment_r{0}_t{1}" -d_x 300 -d_y 100'.format(r, t) + addition + ' -hum_r ' + str(r))
    return cmds
        
def ordinary_attraction(ranges = [50, 100, 150], t = 25):    
    cmds = []        
    addition = ' -cmd "center attract" -cmd_t ' + str(t) 
    for r in ranges:        
        cmds.append('-name "ordinary attraction experiment_r{0}_t{1}" -d_x 500 -d_y 500'.format(r, t) + addition + ' -hum_r ' + str(r))
        cmds.append('-name "ordinary attraction experiment_r{0}_t{1}" -d_x 100 -d_y 500'.format(r, t) + addition + ' -hum_r ' + str(r))
        cmds.append('-name "ordinary attraction experiment_r{0}_t{1}" -d_x 300 -d_y 100'.format(r, t) + addition + ' -hum_r ' + str(r))
    return cmds

def learning_attraction(swarm_size = 15, ranges = [300], input_t = 0, t = 1000):    
    cmds = []        
    addition = ' -cmd "important points" -input_time '+str(input_t) + ' -cmd_t ' + str(t) 
    for r in ranges:        
        cmds.append('-name "ordinary attraction experiment_r{0}_t{1}" -d_x 500 -d_y 500'.format(r, t) + addition + ' -hum_r ' + str(r) + ' -size ' +str(swarm_size))
        #cmds.append('-name "ordinary attraction experiment_r{0}_t{1}" -d_x 100 -d_y 500'.format(r, t) + addition + ' -hum_r ' + str(r))
        #cmds.append('-name "ordinary attraction experiment_r{0}_t{1}" -d_x 300 -d_y 100'.format(r, t) + addition + ' -hum_r ' + str(r))
    return cmds


def learning_attraction_2(swarm_size = 15, ranges = [40], input_t = 0, t = 1000, alpha=10, moving_d=True):    
    cmds = []        
    addition = ' -cmd "important points" -input_time '+str(input_t) + ' -cmd_t ' + str(t) + ' -alpha ' + str(alpha) + ' -d_move ' + str(moving_d)
    for r in ranges:        
        cmds.append('-name "Learning attraction 2 experiment_r{0}_t{1}" -d_x 500 -d_y 500'.format(r, t) + addition + ' -hum_r ' + str(r) + ' -size ' +str(swarm_size))
    return cmds

def simple_rl_model(swarm_size = 1, ranges = [40], input_t = 0, t = 100, alpha=10, moving_d=False, comm_range=300, rl_sim='q-learning', learning_iteration='10', sim_timer=200):
    dirtime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')    
    cmds = []        
    addition =  ' -cmd "q-learning" -input_time '+ str(input_t) + ' -cmd_t ' + str(t) + ' -alpha ' + str(alpha) \
                + ' -comm_range ' + str(comm_range) + ' -rl_sim ' + str(rl_sim) + ' -learning_iteration ' + str(learning_iteration) + ' -sim_timer ' + str(sim_timer)
    for r in ranges:        
        cmds.append(' -name ' + str(rl_sim) + "_" + str(dirtime) + f'_learn{learning_iteration}_steps{sim_timer} ' + addition + ' -hum_r ' + str(r) + ' -size ' +str(swarm_size))
    return cmds

def deep_q_learning(swarm_size = 1, ranges = [40], input_t = 0, t = 100, alpha=10, moving_d=False, comm_range=300, rl_sim='deep_q_learning', learning_iteration='10', sim_timer=200):
    dirtime = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')    
    cmds = []        
    addition =  ' -cmd "q-learning" -input_time '+ str(input_t) + ' -cmd_t ' + str(t) + ' -alpha ' + str(alpha) \
                + ' -comm_range ' + str(comm_range) + ' -rl_sim ' + str(rl_sim) + ' -learning_iteration ' + str(learning_iteration) + ' -sim_timer ' + str(sim_timer)
    for r in ranges:        
        cmds.append(' -name ' + str(rl_sim) + "_" + str(dirtime) + f'_learn{learning_iteration}_steps{sim_timer} ' + addition + ' -hum_r ' + str(r) + ' -size ' +str(swarm_size))
    return cmds

def constant_center_repel(ranges = [50, 100, 150]):    
    cmds = []         
    for r in ranges:        
        cmds.append('-name "constant repulsion experiment_r{0}" -d_x 500 -d_y 500 -const_repel "True"'.format(r) + ' -hum_r ' + str(r))
        cmds.append('-name "constant repulsion experiment_r{0}" -d_x 100 -d_y 500 -const_repel "True"'.format(r) + ' -hum_r ' + str(r))
        cmds.append('-name "constant repulsion experiment_r{0}" -d_x 300 -d_y 100 -const_repel "True"'.format(r) + ' -hum_r ' + str(r))
    return cmds

def moving_disaster():    
    cmds = []             
    cmds.append('-name "moving_disaster" -d_x 500 -d_y 500 -d_move "True"')
    cmds.append('-name "moving_disaster" -d_x 100 -d_y 500 -d_move "True"')
    cmds.append('-name "moving_disaster" -d_x 500 -d_y 100 -d_move "True"')
    return cmds

def obstacle():
    return ['-name "obstacle_gp" -d_x 200 -d_y 400 -gp "True" -size 2 -grid_x 60 -grid_y 60']

def maze(maze_type, additions = '-op_xs 500 -op_ys 300 -d_xs 100 -d_ys 300', through_walls = False):
    return ['-name "maze4_{0}_{1}" -maze "{0}" -walls {1} '.format(maze_type, through_walls) + additions]

def new_maze(maze_type):
    return maze(maze_type, '-op_xs 500 -op_ys 300 -d_xs 100 -d_ys 300 ')


def maze_experiments(maze_type = None, operator_range = [150, 100, 50], drone_range = [8, 4], gp = False, addition = '-op_xs 500 -op_ys 300 -d_xs 100 -d_ys 300'):

    cmds = []         
    for op_r in operator_range:
        for r in drone_range:
            if gp == True:
                if maze_type == None:
                    cmds.append('-name "gp_no_maze_op_{0}_dr_{1}" -gp "True" -hum_r {0} -comm_range {1} '.format(op_r,r) + addition)
                else:                
                    cmds.append('-name "gp_maze_{0}_op_{1}_dr_{2}" -maze "{0}" -gp "True" -aging 0.995 -w 0.005 -hum_r {1} -comm_range {2} '.format(maze_type,op_r,r) + addition)
            else:
                if maze_type == None:
                    cmds.append('-name "gp_no_maze_op_{0}_dr_{1}" -hum_r {0} -comm_range {1} '.format(op_r,r) + addition)
                else:  
                    cmds.append('-name "gp_maze_{0}_op_{1}_dr_{2}" -maze "{0}" -aging 0.995 -w 0.005 -hum_r {1} -comm_range {2} '.format(maze_type,op_r,r) + addition)

    return cmds

    
def trim_cmd(cmd):    
    PATTERN = re.compile(r'''((?:[^ "]|"[^"]*")+)''')
    cmd = PATTERN.split(cmd)[1::2]
    
    for i in range(len(cmd)):
        cmd[i] = re.sub('"','', cmd[i])        
    return cmd

if __name__ == "__main__":    
    exp = int(sys.argv[1])
    
    #N = 15
    N = 1
    procs = []
    for _ in range(N):
        if exp == 1:
            cmds = different_sizes()
        elif exp == 2:
            #cmds = ordinary_experiment()
            cmds = ['-name "no_maze" -op_xs 500 -op_ys 300 -d_xs 100 -d_ys 300']
        elif exp == 3:
            cmds = reliability_experiment()
        elif exp == 4:
            cmds = noise_experiment()
        elif exp == 5:
            cmds = ordinary_attraction(t=0)
        elif exp == 6:
            cmds = moving_disaster()
        elif exp == 7:
            cmds = constant_center_repel(ranges = [100])
        elif exp == 8:
            cmds = obstacle()
        elif exp == 9:
            cmds = maze('simple_new', additions = '-op_xs 500 -op_ys 300 -d_xs 100 -d_ys 300 -cmd_t 5 -cmd "disaster attract"')
            print(cmds)
        elif exp == 10:
            cmds = new_maze('hard_new')
        elif exp == 11:
            cmds = maze_experiments(maze_type = 'simple_new')
        elif exp == 12:
            cmds = maze_experiments(maze_type = 'hard_new')
        elif exp == 13:
            cmds = maze_experiments(maze_type = 'extreme_new')
        elif exp == 14:
            cmds = maze_experiments(maze_type = 'simple_new', gp = True)
        elif exp == 15:
            cmds = maze_experiments(maze_type = 'hard_new', gp = True)
        elif exp == 16:
            cmds = maze_experiments(maze_type = 'extreme_new', gp = True)
        elif exp == 17:
            cmds = maze_experiments()
        elif exp == 18:
            cmds = maze_experiments(gp = True)            
        elif exp == 19:
            cmds = ['-name "distances"']
        elif exp==20:
            cmds = learning_attraction()
        elif exp==21:
            cmds = learning_attraction_2()
        elif exp==22:
            cmds = simple_rl_model()
        elif exp==23:
            cmds = deep_q_learning()
        else:
            cmds = ['-d_size 4 -size 3 -grid_x 100 -grid_y 100 -vis_range 1']
        
        for cmd in cmds:            
            proc = subprocess.Popen([sys.executable, 'experiment.py'] + trim_cmd(cmd), bufsize=0)
            procs.append(proc)       
    
    for proc in procs:
        proc.wait()        
