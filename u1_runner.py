import simulation
import arcade
import datetime
import time
import os
from threading import Thread


def main():  
        print("Please wait for experiment 1 to load...")
        
        EXP_D_T = datetime.datetime.now().strftime("%d-%m-%Y_%H_%M_%S")
        name_of_experiment = "userstudy1"
          
        sim = simulation.SwarmSimulator(600, 600, name_of_experiment,  30, 500, 300, 40, 40, "user_study")
        sim.setup(1, [(500, 500)], 1, [(450, 300)], 100, 100, 0, True, 0, 0.99, None, 0, False, 
                  100, 5, 1, 0.01, 1, 0.995, False, 50, None, False, 0, 0, 0, 0, 0, 0)

        # if not os.path.isdir('outputs'):
        #     os.mkdir('outputs')
        # if (not os.path.isdir('outputs/' + name_of_experiment)):
        #     os.mkdir('outputs/' + name_of_experiment)
        # if (not os.path.isdir('outputs/' + name_of_experiment + "/" + str(EXP_D_T))):
        #     os.mkdir('outputs/' + name_of_experiment + "/" + str(EXP_D_T))
        # if (not os.path.isdir('outputs/' + name_of_experiment + "/" + str(EXP_D_T) + '/performance_test')):
        #     os.mkdir('outputs/' + name_of_experiment + "/" + str(EXP_D_T) + '/performance_test')
            
        sim.directory = str('outputs/' + name_of_experiment + "/" + str(EXP_D_T))

        directory = sim.directory
            
        # sim.log_setup(directory)   
        # if exp_type == "user_study_2":
        #     sim.set_visible(False)

        arcade.run()
    
if __name__ == "__main__":
    main()