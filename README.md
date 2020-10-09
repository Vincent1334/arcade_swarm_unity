# arcade_swarm
### Installation 

We recommend installing the project's dependencies on a separate virtual environment for this project. You can use the instructions [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) to make your own virtual environment.

After activating your virtual environment you can install the required dependencies by following instruction bellow:

- [x] First clone the project to your desired directory.
- [x] You can install all the project's dependencies by changing the directory to the repository and run the command below:
    ```python
    pip install -r requirements.txt 
    ```
To run the deep learning related experiments (experiment 22 & 23) You have to install the customized gym environment:

- [x] Simply go to gym-swarm directory inside the cloned repository
- [x] Run the following command to install the gym environment:
    ```python
    pip install e . 
    ```
_All required dependencies are listed in [requirements.txt](requirements.txt) file._

_We recommend running the project on python3.8 but if you decided to run the project on other python3.X versions you might install different versions of packages in the [requirements.txt](requirements.txt) file._

### Running

You can run different experiment by running [run.py](run.py) file inside your terminal.

```python
python3 run.py [experiment number (1-23)] 
```

for example if you want to run experiment 20 you simply run the following command:

```python
python3 run.py 20 
```

### [objects.py](objects.py)

[objects.py](objects.py) simply stores the different classes (Obstacle, Disaster, Drone, Human, SwarmSimulation). These define all the functionality used in the simulation.  

It allows for creating and running an extremely configurable simulation based on the algorithms explained in the paper.

The exact configuration of each simulation is stored in a log file in the same folder so that all parameters used are recorded.

An example log file is:
```
GENERAL INFO:
  -- ARENA HEIGHT: 600
  -- ARENA WIDTH: 600
  -- GRID_X: 40
  -- GRID_Y: 40
  -- GRANULARITY: 10
  -- COMMAND STRENGTH: 10
  -- ORDINARY COMMAND: None
  -- ORDINARY COMMAND INTERVAL: 0
  -- CONSTANT REPULSION CENTER: False
  -- VELOCITY WEIGHT COEFFICIENT: 0.005
  -- BOUNDARY REPULSION: 1
  -- AGING COEFFICIENT: 0.995
  -- INPUT TIME: 300
  -- GAUSSIAN PROCESSES: True at each 50 steps 

OPERATORS INFO:
  -- OPERATORS SIZE: 1
  -- operator 0: x position: 450 & y position: 300

DISASTERS INFO:
  -- DISASTERS SIZE: 1
  -- disaster 0: x position: 500 & y position: 500 & is moving: False

DRONE INFO:
  -- SWARM SIZE: 15
  -- DRONE RELIABILITY RANGE: (100, 99)
  -- DRONE UNRELIABILITY PERCENTAGE: 0
  -- DRONE COMMUNICATION NOISE RANGE: (0.0, 0.0)
  -- DRONE COMMUNICATION RANGE: 8
  -- DRONE BELIEF VISION RANGE: 4
  -- DRONE CONFIDENCE VISION RANGE: 2

  -- drone 0: reliability: 1 & communication noise: 0.0
  -- drone 1: reliability: 1 & communication noise: 0.0
  -- drone 2: reliability: 1 & communication noise: 0.0
  -- drone 3: reliability: 1 & communication noise: 0.0
  -- drone 4: reliability: 1 & communication noise: 0.0
  -- drone 5: reliability: 1 & communication noise: 0.0
  -- drone 6: reliability: 1 & communication noise: 0.0
  -- drone 7: reliability: 1 & communication noise: 0.0
  -- drone 8: reliability: 1 & communication noise: 0.0
  -- drone 9: reliability: 1 & communication noise: 0.0
  -- drone 10: reliability: 1 & communication noise: 0.0
  -- drone 11: reliability: 1 & communication noise: 0.0
  -- drone 12: reliability: 1 & communication noise: 0.0
  -- drone 13: reliability: 1 & communication noise: 0.0
  -- drone 14: reliability: 1 & communication noise: 0.0

MAZE: hard 
OBSTACLE INFO:
  -- TYPE: vertical, POSITION: (500, 460), VELOCITY: (0, 0)
  -- TYPE: horizontal, POSITION: (450, 350), VELOCITY: (0, 0)
  -- TYPE: vertical, POSITION: (570, 460), VELOCITY: (0, 0)
  -- TYPE: horizontal, POSITION: (450, 570), VELOCITY: (0, 0)
  -- TYPE: horizontal, POSITION: (300, 570), VELOCITY: (0, 0)
  -- TYPE: horizontal, POSITION: (200, 570), VELOCITY: (0, 0)
  -- TYPE: vertical, POSITION: (60, 460), VELOCITY: (0, 0)
  -- TYPE: horizontal, POSITION: (300, 350), VELOCITY: (0, 0)
  -- TYPE: vertical, POSITION: (60, 260), VELOCITY: (0, 0)
  -- TYPE: vertical, POSITION: (330, 240), VELOCITY: (0, 0)
```

### [experiment.py](experiment.py)
[experiment.py](experiment.py) allows for running a configurable simulation by parsing predefined arguments which are passed by the user. 

##### List of predefined arguments:
| Command       | Type   | Default     | Comments                                                                                                   |
| ------------- |:------:| ---------:  |-----------------------------------------------------------------------------------------------------------:|
-name           |  str   | time.time() | name of experiment, the results of the simulation are then saved in a folder with the same name            | 
-size           |  int   | 15          | number of drones                                                                                           | 
-d_size         |  int   | 1           | number of disasters                                                                                        |           
-d_xs           |  int   | [500]       | the x coordinates of the disasters, if some of them are not specified, then then are assigned at random    |      
-d_ys           |  int   | [500]       | the y coordinates of the disasters, if some of them are not specified, then then are assigned at random    |       
-d_move         |  bool  | False       | will the disasters move in a predefined manner                                                               |             
-op_size        |  int   | 1           | number of operator                                                                                         |          
-op_x           |  int   | [450]       | the x coordinates of the operators, if some of them are not specified, then then are assigned at random    |                
-op_y           |  int   | [300]       | the y coordinates of the operators, if some of them are not specified, then then are assigned at random    |            
-noise          |  int   | 0           | communication noise                                                                                        |                
-r_min          |  int   | 100         | min_reliability                                                                                            |           
-r_max          |  int   | 100         | max_reliability                                                                                            |           
-r_perc         |  int   | 0           | unreliability_percentage                                                                                   |               
-cmd            |  str   | None        | operator command - 'boundary', 'corner', 'center attract', 'center repel', 'small center repel'            |             
-cmd_t          |  int   | 0           | period of transmission of the operator command                                                             |                 
-const_repel    |  bool  | False       | constant repulsion                                                                                         |                  
-alpha          |  int   | 10          | operator command strength                                                                                  |              
-comm_range     |  int   | 8           | communication range                                                                                        |                      
-vis_range      |  int   | 2           | vision range                                                                                               |                  
-w              |  int   | 0.01        | velocity weight coefficient                                                                                |                   
-bound          |  int   | 1           | boundary repulsion                                                                                         |                           
-aging          |  int   | 0.9999      | aging factor                                                                                               |                   
-hum_r          |  int   | 150         | operator communication radius                                                                              |                     
-height         |  int   | 600         | arena height                                                                                               |                 
-width          |  int   | 600         | arena width                                                                                                |                   
-grid_x         |  int   | 40          | grid_x                                                                                                     |                     
-grid_y         |  int   | 40          | grid_y                                                                                                     |                   
-input_time     |  int   | 300         | input_time                                                                                                 |                 
-gp             |  bool  | False       | gaussian processes                                                                                         |                       
-gp_step        |  int   | 50          | step of gaussian processes predictions                                                                     |                           
-maze           |  str   | None        | maze type - 'simple', 'big', 'hard', 'extreme'                                                             |                    
-rl_sim         |  str   | None        | reinforcement leanring algorithm                                                                           |                    
-learning_iteration |  int   | 10      | defines learning iteration 
|
-sim_timer      |  int   | 1000        | assign agents active time on each iteration                                                                           

Note that if a parameter is not specified, it will have its default value.

An example experiment may be run with 
```python
python3 experiment.py
```
This will run a simulation with its default parameters. 

You can configure those parameters by using the various predefined commands listed above when running the code. 

An example simulations may be 
```python
python3 experiment.py -d_x 500 -d_y 500 -noise 10
```
which would initialise the disaster at (500, 500) and all agents will have a communication noise of 0.001, 

```python
python3 experiment.py -size 5 -d_size 3 
```
which would initialise only 5 drones and the disaster at (500, 500) and other 2 disasters at random locations or

```python
python3 experiment.py -maze "hard" -gp "True" -w 0.005 -aging 0.995
```
which would initialise a simulation with a 'hard' maze and a swarm which uses Gaussian Processes (plus the right parameters so that drones avoid collisions with obstacles).



### [run.py](run.py)
[run.py](run.py) uses [experiment.py](experiment.py) to implement the experiments used throughout the paper. It contains 9 predefined experiments which include:

different swarm sizes
different reliability percentages
different noise levels
center repel/attract command with different communication ranges of the operators
moving disaster
constant repulsion
mazes

These can also be greatly configured by passing different arguments to the functions in the run.py file.

### [deep_q_modules.py](deep_q_modules.py)
All the modules needed for running experiment 23 (Deep-Q-Learning) are in [deep_q_modules.py](deep_q_modules.py) file.

These classes are being used inside the experiment function on line 207 to 367 of [experiment.py](experiment.py) file.

The neural net initializations are being done from line 207 to 241 of [experiment.py](experiment.py) file.

The main configurations for DQN inside step loop are done from line 331 to 359 of [experiment.py](experiment.py) file.
