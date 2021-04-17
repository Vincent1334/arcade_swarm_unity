import sys
import subprocess
import re

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
-run_time        type = int     default = 1000         simulation steps
'''
  
def compare_swarm_sizes(swarm_sizes = [1, 5, 10, 15, 20, 25, 50]):    
    cmds = []   
    for size in swarm_sizes:             
        cmds.append('-name "first experiment_{0}" -d_x 500 -d_y 500'.format(str(size)) + ' -size ' + str(size))
        cmds.append('-name "first experiment_{0}" -d_x 100 -d_y 500'.format(str(size)) + ' -size ' + str(size))
        cmds.append('-name "first experiment_{0}" -d_x 300 -d_y 100'.format(str(size)) + ' -size ' + str(size))
    return cmds    

def simple_experiment():    
    cmds = []                
    cmds.append('-name "second experiment" -d_x 500 -d_y 500 -size 150')
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

def constant_center_repel(ranges = [50, 100, 150]):    
    cmds = []         
    for r in ranges:        
        cmds.append('-name "constant repulsion experiment_r{0}" -d_x 500 -d_y 500 -const_repel "True"'.format(r) + ' -hum_r ' + str(r))
        cmds.append('-name "constant repulsion experiment_r{0}" -d_x 100 -d_y 500 -const_repel "True"'.format(r) + ' -hum_r ' + str(r))
        cmds.append('-name "constant repulsion experiment_r{0}" -d_x 300 -d_y 100 -const_repel "True"'.format(r) + ' -hum_r ' + str(r))
    return cmds

def moving_disaster():    
    cmds = []             
    #cmds.append('-name "moving_disaster" -d_x 500 -d_y 500 -d_move "True"')
    #cmds.append('-name "moving_disaster" -d_x 100 -d_y 500 -d_move "True"')
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

def communication_noise(communication_noise_prob = [0.1, 0.25, 0.5, 0.75, 1]):
    cmds = []
    for p in communication_noise_prob:
        #cmds.append('-name "communication_noise experiment_{0}" -d_x 500 -d_y 500 -communication_noise_prob {0}'.format(p))
        #cmds.append('-name "communication_noise experiment_{0}" -d_x 100 -d_y 500 -communication_noise_prob {0}'.format(p))
        cmds.append('-name "communication_noise experiment_{0}" -comm_range 30 -run_time 2000 -size 30 -grid_x 90 -grid_y 90 -d_x 600 -d_y 800 -communication_noise_prob {0} -height 900 -width 900'.format(p))
    return cmds

def positioning_noise(positioning_noise_strength = [1, 2, 3], positioning_noise_prob = [0.1, 0.25, 0.5, 0.75, 1]):
    cmds = []
    for s in positioning_noise_strength:
        for p in positioning_noise_prob:
            cmds.append('-name "positioning_noise experiment_{0}_{1}" -d_x 500 -d_y 500 -positioning_noise_strength {0} -positioning_noise_prob {1}'.format(s,p))
            #cmds.append('-name "positioning_noise experiment_{0}_{1}" -d_x 100 -d_y 500 -positioning_noise_strength {0} -positioning_noise_prob {1}'.format(s,p))
            #cmds.append('-name "positioning_noise experiment_{0}_{1}" -d_x 300 -d_y 100 -positioning_noise_strength {0} -positioning_noise_prob {1}'.format(s,p))
    return cmds

def sensing_noise(sensing_noise_strength = [0.1, 0.25, 0.5, 0.75, 1], sensing_noise_prob = [0.1, 0.25, 0.5, 0.75, 1]):
    cmds = []
    for s in sensing_noise_strength:
        for p in sensing_noise_prob:
            cmds.append('-name "sensing_noise experiment_{0}_{1}" -d_x 500 -d_y 500 -sensing_noise_strength {0} -sensing_noise_prob {1}'.format(s,p))
            #cmds.append('-name "sensing_noise experiment_{0}_{1}" -d_x 100 -d_y 500 -sensing_noise_strength {0} -sensing_noise_prob {1}'.format(s,p))
            #cmds.append('-name "sensing_noise experiment_{0}_{1}" -d_x 300 -d_y 100 -sensing_noise_strength {0} -sensing_noise_prob {1}'.format(s,p))
    return cmds

def bottleneck_tests(swarm_size = 100, r = 40, alpha = 0.99, t = 300, comm_range = 2):
    cmds = []  
    for exp_num in range(1):
        cmds.append('-name bottleneck_test_experiment_S{}_E{}'.format(swarm_size, exp_num) + ' -size ' +str(swarm_size) + ' -hum_r ' + str(r) + ' -cmd_t ' +  str(t)
                    + ' -alpha ' + str(alpha) + ' -comm_range ' + str(comm_range))
    return cmds

def online_experiment(swarm_size = 15, r = 40, vs_range=2, alpha = 0.99, t = 300, comm_range = 15, online_exp = "normal_network", ex_time = 500):
    cmds = []  
    for exp_num in range(1):
        cmds.append('-name Online_experiment_S{}'.format(swarm_size) + ' -exp_type ' + str(online_exp) + ' -run_time ' + str(ex_time) \
                    + ' -alpha ' + str(alpha) + ' -comm_range ' + str(comm_range) + ' -size ' + str(swarm_size))
    return cmds
    
def accuracy_diag(swarm_size = 1, r = 40, alpha = 0.99, t = 300, comm_range = 15,  ex_time = 500):
    cmds = []  
    for exp_num in range(1):
        cmds.append('-name accuracy_test_S{}_E{}_CR{}'.format(swarm_size, exp_num, comm_range) + ' -size ' +str(swarm_size) + ' -hum_r ' + str(r) + ' -cmd_t ' +  str(t)
                    + ' -alpha ' + str(alpha) + ' -comm_range ' + str(comm_range)+ ' -run_time ' + str(ex_time))
    return cmds

def user_study1(swarm_size = 15, r = 40, vs_range=2, alpha = 0.99, t = 300, comm_range = 15, exp_type = "user_study", ex_time = 200):
    cmds = []  
    for exp_num in range(1):
        cmds.append('-name User_Study_1_S{}'.format(swarm_size) + ' -exp_type ' + str(exp_type) + ' -run_time ' + str(ex_time) \
                    + ' -alpha ' + str(alpha) + ' -comm_range ' + str(comm_range) + ' -size ' + str(swarm_size) + ' -d_move ' + str(True))
    return cmds

def trim_cmd(cmd):    
    PATTERN = re.compile(r'''((?:[^ "]|"[^"]*")+)''')
    cmd = PATTERN.split(cmd)[1::2]
    
    for i in range(len(cmd)):
        cmd[i] = re.sub('"','', cmd[i])        
    return cmd

if __name__ == "__main__":    
    exp = int(sys.argv[1])
    N = 1
    
    procs = []
    for _ in range(N):
        if exp == 1:
            cmds = compare_swarm_sizes()
        elif exp == 2:
            cmds = simple_experiment()
            #cmds = ['-name "no_maze" -op_xs 500 -op_ys 300 -d_xs 100 -d_ys 300']
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
            cmds = maze('simple_new', additions = '-op_xs 500 -op_ys 300 -d_xs 100 -d_ys 300 -cmd_t 5 -cmd "disaster_attract"')
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
        elif exp == 20:
            cmds = ['-d_size 1 -size 15 -grid_x 100 -grid_y 100 ']
        elif exp == 21:
            cmds = communication_noise(communication_noise_prob =[0])#= [0.9995])#threshold = 0.997-0.999
        elif exp == 22:
            cmds = positioning_noise(positioning_noise_strength = [2], positioning_noise_prob = [.1, .5])
        elif exp == 23:
            cmds = sensing_noise(sensing_noise_strength = [0.5, 0.75, 1], sensing_noise_prob = [0.8, 0.9, 0.99])
        elif exp == 24:
            cmds = bottleneck_tests()
        elif exp == 25:
            cmds = online_experiment()
        elif exp == 26:
            cmds = accuracy_diag()
        elif exp == 27:
            cmds = user_study1()

            
        for cmd in cmds:            
            proc = subprocess.Popen([sys.executable, 'init.py'] + trim_cmd(cmd))
            procs.append(proc)       
    
    for proc in procs:
        proc.wait()        
