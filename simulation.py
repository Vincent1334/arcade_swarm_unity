import arcade
from threading import *
import numpy as np
import random
import scipy, scipy.ndimage
import math
import time
from matplotlib import pyplot as plt
import collections
import os
import timeit
import json
import pyglet
import requests
from fps_test_modules import FPSCounter
import time as timeclock
import datetime
import argparse

np.seterr(divide='ignore', invalid='ignore')

EXP_D_T = datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")

'''
Class Object (abstract class)
    - A simple class which implements the simplest behaviour of each object in the simulations
    - Stores position, velocity and implements a basic movement
'''
class Object(arcade.Sprite):
    def __init__(self, x, y, change_x, change_y, scl, img, sim):
       super().__init__(img, scl)      
       
       self.center_x = x
       self.center_y = y

       self.change_x = change_x ## speed of movement in x direction per frame
       self.change_y = change_y ## speed of movement in y directieno per frame
       
       self.simulation = sim
       
    # Basic movement and keeps object in arena
    def update(self):   
        
        if self.center_x + self.change_x >= self.simulation.ARENA_WIDTH or self.center_x + self.change_x <= 0:
            self.change_x = 0
        if self.center_y + self.change_y >= self.simulation.ARENA_HEIGHT or self.center_y + self.change_y <= 0:
            self.change_y = 0
            
        self.center_x += self.change_x
        self.center_y += self.change_y

class Obstacle(Object):
    def __init__(self, x, y, change_x, change_y, scl, sim, obstacle_type):
        self.type = obstacle_type
        
        if obstacle_type == 'disk':        
            super().__init__(x, y, change_x, change_y, scl, "images/obstacle.png", sim)
        elif obstacle_type == 'horizontal':        
            super().__init__(x, y, change_x, change_y, scl, "images/obstacle2.png", sim)
        elif obstacle_type == 'vertical':        
            super().__init__(x, y, change_x, change_y, scl, "images/obstacle5_thin.png", sim)
        elif obstacle_type == 'vertical_thin':
            super().__init__(x, y, change_x, change_y, scl, "images/obstacle1_thin.png", sim)
            
    def update(self):
        super().update()
'''
Class Disaster
    - A simple class which implements the behaviour of a disaster
    - Extends the behaviour of the class Object
'''

class Disaster(Object):
    def __init__(self, x, y, change_x, change_y, scl, sim, moving = False, img = "images/disaster.png"):
        super().__init__(x, y, change_x, change_y, scl, img, sim)        
        self.moving = moving
    
    def update(self):       

        '''
        if self.moving == True: 
            if self.simulation.timer >= self.simulation.INPUT_TIME/2 and self.simulation.timer < self.simulation.INPUT_TIME/2 + 50:               
                self.change_y = -5  
            else:            
                self.change_y = 0
        '''
        
        # Simple movement
        if self.moving == True: 
            if self.simulation.timer >= self.simulation.INPUT_TIME/2 and self.simulation.timer < self.simulation.INPUT_TIME/2 + 100:
                if self.change_y == 0:
                    if self.center_y <= self.simulation.ARENA_HEIGHT/2:
                        self.change_y = 3
                    else:
                        self.change_y = -3
                self.change_x = 0                
            elif self.simulation.timer >= self.simulation.INPUT_TIME/2 + 100 and self.simulation.timer < self.simulation.INPUT_TIME/2 + 200:
                if self.change_x == 0:
                    if self.center_x <= self.simulation.ARENA_WIDTH/2:
                        self.change_x = 3
                    else:
                        self.change_x = -3   
                self.change_y = 0
            else:            
                self.change_x = 0
                self.change_y = 0
        
        
        super().update()

'''
Class Agent (abstract class)
    - Implements the functionality of all agents
'''
class Agent(Object):
    def __init__(self, x, y, change_x, change_y, scl, img, sim, reliability = 1, communication_noise = 0):
        super().__init__(x, y, change_x, change_y, scl, img, sim)
        
        self.confidence_map= np.array([[0.0 for i in range(self.simulation.GRID_X)] for j in range(self.simulation.GRID_Y)])
        self.internal_map= np.array([[0.0 for i in range(self.simulation.GRID_X)] for j in range(self.simulation.GRID_Y)])       
        
        self.reliability = reliability
        self.communication_noise = communication_noise

        self.grid_pos_x = math.trunc((self.center_x * (self.simulation.GRID_X - 1) / self.simulation.ARENA_WIDTH))
        self.grid_pos_y = math.trunc(((1 - self.center_y / self.simulation.ARENA_HEIGHT ) * (self.simulation.GRID_Y - 1)))
        
        self.have_communicated = False
        self.message_count_succ = 0
        self.message_count_fail = 0
        
        self.message = ""
        
    def update(self):
        #sensing_noise = np.random.uniform(0, self.simulation.sensing_noise_strength, (self.simulation.GRID_Y, self.simulation.GRID_X))

        #s = random.uniform(0, self.simulation.sensing_noise_strength)
        
        #if random.random() < self.simulation.sensing_noise_prob:
        #    self.internal_map *= (1-s)
        
        super().update()
    
    def communicate(self, agent, how):    
        '''
        if self.simulation.through_walls == False:

            #print('ping')
            from bresenham import bresenham
            
            visible = True
            
            line_segment = list(bresenham(self.grid_pos_y, self.grid_pos_x, agent.grid_pos_y, agent.grid_pos_x))

            for k,j in line_segment:
                obstacle = self.is_obstacle_at_position(k, j)

                if obstacle == True:
                    visible = False
                    break
                            
            if visible == True:
                self.exchange_data(agent, how)

        else:           
        '''
        # if random.random() > self.simulation.communication_noise_prob:
        #     #self.communication_noise = random.uniform(0, self.simulation.communication_noise_strength)
        #     if self.have_communicated == False:
        #         self.exchange_data(agent, how)
        #         self.message_count_succ += 1
        #         self.have_communicated = True
        # else: 
        #     self.message_count_fail += 1
    
        self.exchange_data(agent, how)
        self.message_count_succ += 1
        self.have_communicated = True

    def exchange_data_old(self, agent, how):
        '''if (random.randrange(0, 100) < 50):
            coeff = +1 
        else:
            coeff = -1
        '''
        coeff = 0
        if how == 'max':
            for j in range(self.simulation.GRID_X):
                for i in range(self.simulation.GRID_Y):
                    if (agent.confidence_map[i][j] > self.confidence_map[i][j]):                            
                        self.internal_map[i][j] =  agent.reliability * agent.internal_map[i][j] + coeff * self.communication_noise                        
                        self.confidence_map[i][j] = agent.confidence_map[i][j] + coeff * self.communication_noise                        
                    else:
                        agent.internal_map[i][j] =  self.reliability * self.internal_map[i][j] + coeff * self.communication_noise                        
                        agent.confidence_map[i][j] = self.confidence_map[i][j] + coeff * self.communication_noise
        elif how == 'average':            
            #self.message_count_succ += 1
            agent.internal_map = self.internal_map = (self.reliability * self.internal_map + agent.reliability * agent.internal_map)/2  + coeff * self.communication_noise                                        
            agent.confidence_map = self.confidence_map = (self.confidence_map + agent.confidence_map)/2  + coeff * self.communication_noise

    def exchange_data(self, agent, how):
        '''if (random.randrange(0, 100) < 50):
            coeff = +1 
        else:
            coeff = -1
        '''
        coeff = 0
        ################################# 

        b_a_mask = np.bitwise_and(agent.confidence_map > 0.8, agent.confidence_map > self.confidence_map)
        np.putmask(self.internal_map, b_a_mask, agent.reliability * agent.internal_map + coeff * self.communication_noise)
        np.putmask(self.confidence_map, b_a_mask, agent.confidence_map + coeff * self.communication_noise)
        
        b_s_mask = np.bitwise_and(self.confidence_map > 0.8, agent.confidence_map < self.confidence_map)
        np.putmask(agent.internal_map, b_s_mask, self.reliability * self.internal_map + coeff * self.communication_noise)
        np.putmask(agent.confidence_map, b_s_mask, self.confidence_map + coeff * self.communication_noise)
        
        l_a_mask = np.bitwise_and(agent.confidence_map < 0, agent.confidence_map < self.confidence_map)
        np.putmask(self.internal_map, l_a_mask, agent.reliability * agent.internal_map + coeff * self.communication_noise)
        np.putmask(self.confidence_map, l_a_mask, agent.confidence_map + coeff * self.communication_noise)
        
        l_s_mask = np.bitwise_and(self.confidence_map < 0, agent.confidence_map > self.confidence_map)
        np.putmask(agent.internal_map, l_s_mask, self.reliability * self.internal_map + coeff * self.communication_noise)
        np.putmask(agent.confidence_map, l_s_mask, self.confidence_map + coeff * self.communication_noise)
        
        ll_a_mask = np.bitwise_or(self.confidence_map < 0.8, self.confidence_map > 0.8)
        np.putmask(agent.internal_map, ll_a_mask, (self.reliability * self.internal_map + agent.reliability * agent.internal_map)/2  + coeff * self.communication_noise)
        np.putmask(agent.confidence_map, ll_a_mask, (self.confidence_map + agent.confidence_map)/2  + coeff * self.communication_noise)
        
        ll_s_mask = np.bitwise_or(agent.internal_map < 0.8, agent.internal_map > 0)
        np.putmask(self.internal_map, ll_s_mask, (self.reliability * self.internal_map + agent.reliability * agent.internal_map)/2  + coeff * self.communication_noise)
        np.putmask(self.confidence_map, ll_s_mask, (self.confidence_map + agent.confidence_map)/2  + coeff * self.communication_noise)
        
        # for j in range(self.simulation.GRID_X):
        #     for i in range(self.simulation.GRID_Y):
        #         agent_confidence = agent.confidence_map[i][j]
        #         agent_belief = agent.internal_map[i][j]
				
        #         self_confidence = self.confidence_map[i][j]
        #         self_belief = self.internal_map[i][j]
        #         #print ('confidence is ' + str(agent_confidence))
        #         if (agent_confidence > 0.8) or (self_confidence > 0.8):
        #             if (agent_confidence > self_confidence):                            
        #                 self_belief =  agent.reliability * agent_belief + coeff * self.communication_noise                        
        #                 self_confidence = agent_confidence + coeff * self.communication_noise                        
        #             else:
        #                 agent_belief =  self.reliability * self_belief + coeff * self.communication_noise                        
        #                 agent_confidence = self_confidence + coeff * self.communication_noise
        #         elif (agent_confidence < 0) or (self_confidence < 0):
        #             if (agent_confidence < self_confidence):                            
        #                 self_belief =  agent.reliability * agent_belief + coeff * self.communication_noise                        
        #                 self_confidence = agent_confidence + coeff * self.communication_noise                        
        #             else:
        #                 agent_belief =  self.reliability * self_belief + coeff * self.communication_noise                        
        #                 agent_confidence = self_confidence + coeff * self.communication_noise
        #         else:
        #             agent_belief = self_belief = (self.reliability * self_belief + agent.reliability * agent_belief)/2  + coeff * self.communication_noise                                        
        #             agent_confidence = self_confidence = (self_confidence + agent_confidence)/2  + coeff * self.communication_noise
                
        #         agent.confidence_map[i][j] = agent_confidence
        #         agent.internal_map[i][j] = agent_belief
        #         self.confidence_map[i][j]= self_confidence 
        #         self.internal_map[i][j] = self_belief
        '''
        #agent.internal_map = self.internal_map = (self.reliability * self.internal_map + agent.reliability * agent.internal_map)/2  + coeff * self.communication_noise                                        
        #agent.confidence_map = self.confidence_map = (self.confidence_map + agent.confidence_map)/2  + coeff * self.communication_noise

        agent.internal_map = (self.reliability * self.internal_map + agent.reliability * agent.internal_map)/2  + coeff * self.communication_noise
        agent.internal_map [agent.internal_map > 1] = 1
        agent.internal_map [agent.internal_map < -10] = -10
        
        self.internal_map = (self.reliability * self.internal_map + agent.reliability * agent.internal_map)/2  + coeff * self.communication_noise
        self.internal_map [self.internal_map  > 1] = 1
        self.internal_map [self.internal_map < -10] = -10
        
        agent.confidence_map = (self.reliability * self.confidence_map + agent.reliability * agent.confidence_map)/2  + coeff * self.communication_noise                                        
        agent.confidence_map [agent.confidence_map  > 1] = 1
        agent.confidence_map [agent.confidence_map < -10] = -10
        
        self.confidence_map = (self.reliability * self.confidence_map + agent.reliability * agent.confidence_map)/2  + coeff * self.communication_noise                                        
        self.confidence_map [self.confidence_map > 1] = 1
        self.confidence_map [self.confidence_map < -10] = -10
        '''
        #################################
    def is_obstacle_at_position(self, k, j):

        obstacle = False
        for obstacle in self.simulation.obstacle_list:
            obstacle_grid_center_pos_x = math.trunc((obstacle.center_x * (self.simulation.GRID_X -1)/self.simulation.ARENA_WIDTH) )
            obstacle_grid_center_pos_y = math.trunc((self.simulation.GRID_Y -1)*(1 - obstacle.center_y / self.simulation.ARENA_HEIGHT) )
							 
            obstacle_witdh = obstacle.width * (self.simulation.GRID_X -1)/self.simulation.ARENA_WIDTH
            obstacle_height = obstacle.height * (self.simulation.GRID_Y -1)/self.simulation.ARENA_HEIGHT

            if (j >= obstacle_grid_center_pos_x - obstacle_witdh/2 and j <= obstacle_grid_center_pos_x + obstacle_witdh/2) and (k >= obstacle_grid_center_pos_y - obstacle_height/2 and k <= obstacle_grid_center_pos_y + obstacle_height/2):                                 
                obstacle = True
                break
        return obstacle
        
'''
Class Human
    - For the moment just extends the class Agent
'''
class Human(Agent):    
    def __init__(self, x, y, scl, change_x = 0, change_y = 0, sim = None, img = "images/human.png"):
        super().__init__(x, y, change_x, change_y, scl, img, sim)

    def communicate(self, agent, how = 'max'):
        super().communicate(agent, how)
    
    def update(self, gp_operator = False):


        self.confidence_map *= self.simulation.LOSING_CONFIDENCE_RATE
        self.internal_map *= self.simulation.LOSING_CONFIDENCE_RATE
        
        '''
        t0 = self.simulation.INPUT_TIME - 1
        t1 = self.simulation.INPUT_TIME + 50
        if self.simulation.timer == t0 or self.simulation.timer == t1 or self.simulation.timer == t1 + 50 or self.simulation.timer == t1 + 100 or self.simulation.timer == t1 + 150 or self.simulation.timer == t1 + 200 or self.simulation.timer == t1 + 250 or self.simulation.timer == t1 + 300 or self.simulation.timer == t1 + 350 or self.simulation.timer == t1 + 400 or self.simulation.timer == t1 + 450 or self.simulation.timer == t1 + 500 or self.simulation.timer == t1 + 550 or self.simulation.timer == t1 + 600 or self.simulation.timer == t1 + 650:
            self.simulation.save_one_heatmap(self.internal_map, 'belief_' + str(self.simulation.timer), self.simulation.directory)
        '''
        #if self.simulation.timer == 10 or self.simulation.timer%100 == 0:
            #self.simulation.save_one_heatmap(self.internal_map, 'belief_' + str(self.simulation.timer), self.simulation.directory)
        
        # if self.simulation.timer % 100 == 0:
        #     self.simulation.save_one_heatmap(self.confidence_map, 'confidence_' + str(self.simulation.timer), self.simulation.directory)
        
        if gp_operator == True:
            if self.simulation.timer % 100 == 0:
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF

                threshold = 0.1
                #dist = 20
                #evolve_top = 20

                #Belief
                #if evolve_top != None:           
                 #   X_high = np.array(np.dstack(np.unravel_index(np.argsort(self.internal_map.ravel()), self.internal_map.shape))[:,-evolve_top:])[0]
                 #   X_low = np.array(np.dstack(np.unravel_index(np.argsort(self.internal_map.ravel()), self.internal_map.shape))[:,:evolve_top])[0]
                    
                  #  X = np.concatenate((X_high, X_low))
                #else:
                X = [[i,j] for i in range(self.simulation.GRID_Y) for j in range(self.simulation.GRID_X) if self.internal_map[i,j] > threshold]#and np.sqrt((self.grid_pos_x-j)**2 + (self.grid_pos_y-i)**2) >= dist]
                
                if len(X) == 0:
                    return None

                #print(len(X))
                
                Y = [self.internal_map[i,j] for i,j in X]
            
                gpr = GaussianProcessRegressor(1.0*RBF(1.0)).fit(X, Y)

                X_pred = [[i,j] for i in range(self.simulation.GRID_Y) for j in range(self.simulation.GRID_X)]# if np.sqrt((self.grid_pos_x-j)**2 + (self.grid_pos_y-i)**2) >= dist]

                self.internal_map = np.reshape(gpr.predict(X_pred), (-1, self.simulation.GRID_X))
                #self.internal_map[self.internal_map > 1] = 1
                #self.internal_map[self.internal_map < 0] = 0

                '''
                #Confidence
                if evolve_top != None:           
                    X_high = list(np.dstack(np.unravel_index(np.argsort(self.confidence_map.ravel()), self.confidence_map.shape))[:,-evolve_top:])[0]
                    X_low = list(np.dstack(np.unravel_index(np.argsort(self.confidence_map.ravel()), self.confidence_map.shape))[:,:evolve_top])[0]

                    X = np.concatenate((X_high, X_low))
                else:
                    X = [[i,j] for i in range(self.simulation.GRID_Y) for j in range(self.simulation.GRID_X) if self.confidence_map[i,j] > threshold]#and np.sqrt((self.grid_pos_x-j)**2 + (self.grid_pos_y-i)**2) >= dist]
                
                if len(X) == 0:
                    return None

                Y = [self.confidence_map[i,j] for i,j in X]
            
                gpr = GaussianProcessRegressor(100.0*RBF(1)).fit(X, Y)

                X_pred = [[i,j] for i in range(self.simulation.GRID_Y) for j in range(self.simulation.GRID_X)]# if np.sqrt((self.grid_pos_x-j)**2 + (self.grid_pos_y-i)**2) >= dist]

                self.confidence_map = np.reshape(gpr.predict(X_pred), (-1, self.simulation.GRID_X))
                #self.confidence_map[self.confidence_map > 1] = 1
                #self.confidence_map[self.confidence_map < 0] = 0
                '''
        
        super().update()
    
'''
Class Drone
    - Extends the class Agent and a controller of the drones
'''
class Drone(Agent):  
    def __init__(self, x, y, speed, radius, name, sim, img = 'images/drone.png', reliability = 1, communication_noise = 0):
        
         super().__init__(x, y, 0, 0, radius*2, img, sim, reliability, communication_noise)         
         
         self.name = name
         
         self.random_walks = 0
         self.local_forces = []
         self.global_forces = []
	 
         self.sobel_x = self.custom_sobel((self.simulation.VISION_BOUNDARY*3 + 1, self.simulation.VISION_BOUNDARY*3 + 1), axis = 0)*(-1)         
         self.sobel_y = self.custom_sobel((self.simulation.VISION_BOUNDARY*3 + 1, self.simulation.VISION_BOUNDARY*3 + 1), axis = 1)
         
         self.global_sobel_x = self.custom_sobel((self.simulation.GRID_X*2 + 1, self.simulation.GRID_Y*2 + 1), axis = 0)*(-1)
         self.global_sobel_y = self.custom_sobel((self.simulation.GRID_X*2 + 1, self.simulation.GRID_Y*2 + 1), axis = 1)

	 
	 #self.obstacles_current_range = np.array([[0.0 for i in range(self.simulation.GRID_X)] for j in range(self.simulation.GRID_Y)])                
    
    def communicate(self, agent, how = 'max'):            
        super().communicate(agent, how)

    def custom_sobel(self, shape, axis):
        """
        shape must be odd: eg. (5,5)
        axis is the direction, with 0 to positive x and 1 to positive y
        """
        if shape[0] % 2 == 0:
            shape = (shape[0]+1,shape[1]+1)
        k = np.zeros(shape)
        p = [(j,i) for j in range(shape[0]) 
               for i in range(shape[1]) 
               if not (i == (shape[1] -1)/2. and j == (shape[0] -1)/2.)]
    
        for j, i in p:
            j_ = int(j - (shape[0] -1)/2.)
            i_ = int(i - (shape[1] -1)/2.)
            k[j,i] = (i_ if axis==0 else j_)/float(i_*i_ + j_*j_)
        return k

    def get_gradient_velocity(self):
        dx1 = self.convolve(self.sobel_x)
        dy1 = self.convolve(self.sobel_y)        
        dx2 = self.convolve(self.global_sobel_x)
        dy2 = self.convolve(self.global_sobel_y)

        self.local_forces.append(10 * np.sqrt(dx1*dx1 + dy1*dy1))
        self.global_forces.append(np.sqrt(dx2*dx2 + dy2*dy2))
        
        dx = dx1*(1 - self.simulation.VELOCITY_WEIGHT) + dx2*self.simulation.VELOCITY_WEIGHT
        dy = dy1*(1 - self.simulation.VELOCITY_WEIGHT) + dy2*self.simulation.VELOCITY_WEIGHT

        norm = np.sqrt(dx*dx + dy*dy)    
        threshold = 0.1   
        if norm <= threshold:# and self.simulation.timer < 3:
            self.random_walks += 1
            return dx + self.simulation.GRANULARITY*random.randrange(-100,100)/100.0, dy + self.simulation.GRANULARITY*random.randrange(-100,100)/100.0
        #if norm <= 1:
       #     return (self.simulation.GRANULARITY*dx/2), (self.simulation.GRANULARITY*dy/2)
        
        return (self.simulation.GRANULARITY*dx/2)/norm, (self.simulation.GRANULARITY*dy/2)/norm
 
    def convolve(self, template):     
        r = math.floor(len(template)/2) 
        
        if r > 20:            
            j = self.simulation.GRID_Y - self.grid_pos_y
            jj = self.simulation.GRID_Y + j
            i = self.simulation.GRID_X - self.grid_pos_x
            ii = self.simulation.GRID_X + i
            # k = self.grid_pos_y - j
            # l = self.grid_pos_x - i
            '''

            res = np.sum(np.multiply(self.confidence_map, template[j:jj,i:ii]))
            if k >= 0 and l >= 0:
                return res + self.simulation.BOUNDARY_REPULSION * (np.sum(template[jj:jj+k,i:ii+l]) + np.sum(template[j:jj,ii:ii+l]))
            elif k >= 0 and l < 0:
                return res + self.simulation.BOUNDARY_REPULSION * (np.sum(template[jj:jj+k,i+l:ii]) + np.sum(template[j:jj,i+l:i]))
            elif k < 0 and l >= 0:
                return res + self.simulation.BOUNDARY_REPULSION * (np.sum(template[j+k:j,i:ii+l]) + np.sum(template[j:jj,ii:ii+l]))
            else:
                return res + self.simulation.BOUNDARY_REPULSION * (np.sum(template[j+k:j,i+l:ii]) + np.sum(template[j:jj,i+l:i]))
            '''
            #############################################################################################
            ####################################### CPU Intensive #######################################
            #############################################################################################
            return np.sum(np.multiply(self.confidence_map, template[j:jj,i:ii])) + self.simulation.BOUNDARY_REPULSION * ( # Time: 8.1% ######## Hits: 600000 ######## Per Hit: 1.3
                    np.sum(template[:j,:]) + np.sum(template[jj:,:]) + np.sum(template[j:jj,:i]) + np.sum(template[j:jj,ii:])) # Time: 14.7% ######## Hits: 300000 ######## Per Hit: 42.4
            #############################################################################################
            ####################################### /CPU Intensive ######################################
            #############################################################################################

            # Other ways of calculating the velocity, may end up being faster
            #a = np.pad(self.confidence_map, ((self.simulation.GRID_Y - self.grid_pos_y, self.grid_pos_y + 1), (self.simulation.GRID_X - self.grid_pos_x, self.grid_pos_x + 1)), 'constant', constant_values=(self.simulation.BOUNDARY_REPULSION))            
            
            #return a.ravel().dot(template.ravel())
            #return np.einsum('ij,ij',np.pad(self.confidence_map, ((self.simulation.GRID_Y - self.grid_pos_y, self.grid_pos_y + 1), (self.simulation.GRID_X - self.grid_pos_x, self.grid_pos_x + 1)), 'constant', constant_values=(self.simulation.BOUNDARY_REPULSION)),template)
            #return np.sum(np.multiply(np.pad(self.confidence_map, ((self.simulation.GRID_Y - self.grid_pos_y, self.grid_pos_y + 1), (self.simulation.GRID_X - self.grid_pos_x, self.grid_pos_x + 1)), 'constant', constant_values=(self.simulation.BOUNDARY_REPULSION)), template))
        else:
            result = 0 
            #############################################################################################
            ####################################### CPU Intensive #######################################
            #############################################################################################
            for i in range(self.grid_pos_y - r, self.grid_pos_y + r + 1): # Time: 1.5% ######## Hits: 2400000 ######## Per Hit: 0.5
                for j in range(self.grid_pos_x - r, self.grid_pos_x + r + 1): # Time: 9.8% ######## Hits: 16800000 ######## Per Hit: 0.5
                    if(i >= 0 and i < self.simulation.GRID_Y and j >= 0 and j < self.simulation.GRID_X): # Time: 11.9% ######## Hits: 14700000 ######## Per Hit: 0.7
                        # pixel = self.confidence_map[i][j] * (1 - self.internal_map[i][j]) # Time: 28.6% ######## Hits: 14344566 ######## Per Hit: 1.7
                        pixel = self.confidence_map[i][j]
                    else:
                        pixel = 1*self.simulation.BOUNDARY_REPULSION                    
                    result += pixel * template[i - self.grid_pos_y + r][j - self.grid_pos_x + r] # Time: 22.7% ######## Hits: 14700000 ######## Per Hit: 1.3
            return result
            #############################################################################################
            ####################################### /CPU Intensive ######################################
            #############################################################################################
            
    def predict_belief_map4(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF
        
        r = np.linspace(0, 1, self.simulation.GRID_Y) 
        c = np.linspace(0, 1, self.simulation.GRID_X)

        rr, cc = np.meshgrid(r, c)
        vals = self.internal_map >= 0

        if (vals == True).any() == False:
            return
        
        gp = GaussianProcessRegressor(RBF(1))
        gp.fit(X=np.column_stack([rr[vals],cc[vals]]), y=self.internal_map[vals])
        
        rr_cc_as_cols = np.column_stack([rr.flatten(), cc.flatten()])

        self.internal_map1 = gp.predict(rr_cc_as_cols).reshape(self.internal_map.shape)

    def predict_belief_map3(self):
        import scipy.interpolate as interpolate
        
        r = np.linspace(0, 1, self.simulation.GRID_Y) 
        c = np.linspace(0, 1, self.simulation.GRID_X)

        rr, cc = np.meshgrid(r, c)

        vals = self.internal_map >= 0

        if (vals == True).any() == False:
            return
        
        f = interpolate.Rbf(rr[vals], cc[vals], self.internal_map[vals], function='linear')
        interpolated = f(rr, cc)

        self.internal_map1 = interpolated
        
        
    def predict_belief_map2(self, evolve_top = None):
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF

        # threshold = 0
        #dist = 20

        if evolve_top != None:           
            X = list(np.dstack(np.unravel_index(np.argsort(self.internal_map.ravel()), self.internal_map.shape))[:,-evolve_top:])[0]
        else:
            X = [[i,j] for i in range(self.simulation.GRID_Y) for j in range(self.simulation.GRID_X) if self.internal_map[i,j] >= 0]#and np.sqrt((self.grid_pos_x-j)**2 + (self.grid_pos_y-i)**2) >= dist]
            
        if len(X) == 0:
            return None

        Y = [self.internal_map[i,j] for i,j in X]
        
        gpr = GaussianProcessRegressor(1.0*RBF(1.0)).fit(X, Y)

        X_pred = [[i,j] for i in range(self.simulation.GRID_Y) for j in range(self.simulation.GRID_X)]# if np.sqrt((self.grid_pos_x-j)**2 + (self.grid_pos_y-i)**2) >= dist]

        #for i,j in X_pred:
         #   self.internal_map[i,j] = gpr.predict(np.array([i,j]).reshape(-1,2))
        #self.internal_map = (self.internal_map + np.reshape(gpr.predict(X_pred), (-1, self.simulation.GRID_X)))/2
        self.internal_map1 = np.reshape(gpr.predict(X_pred), (-1, self.simulation.GRID_X))
     
    def predict_belief_map(self):
        
        X_org = []
        Y = []
        
        threshold = 0.5
        # radius = self.simulation.BOUDARY_DIAMETER
        
        # Examine neigbourhood only
        
        for i in range(self.internal_map.shape[0]):
          for j in range(self.internal_map.shape[1]):
            if self.internal_map[i][j] >= threshold:
                #dist = np.sqrt((self.grid_pos_x - j)**2 + (self.grid_pos_y - i)**2) 
                #if dist <= radius:
                  X_org.append((i,j))
                  Y.append(self.internal_map[i][j])
        
        X_org = np.array(X_org)
        Y = np.array(Y)        
        #print('X_org: ', X_org.shape)
        
        X_dim1 = X_org.shape[0]
        
        if X_dim1 == 0:
            return None
        
        
        sigma_n = 0.1
        
        sigma_f = 1.0
        
        l1 = 4.0
        l2 = 4.0
        
        #print(X_org.shape)
        X = X_org[:,:,np.newaxis]
        #print('X: ', X.shape)
        
        
        # D = np.zeros((X_dim1,X_dim1))
        
        K = np.zeros((X_dim1,X_dim1))
        
        X1 = X[:,0,:]
        #print('X1: ', X1.shape)
        #D1 = (X1 - X1.T)**2 / (2.0 * l1**2 )
        D1 = (X1 - X1.T) / (2.0 * l1**2 )
        
        
        X2 = X[:,1,:]
        #print('X2: ', X2.shape)
        
        #D2 = (X2 - X2.T)**2 / (2.0 * l2**2 )
        D2 = (X2 - X2.T) / (2.0 * l2**2 )
        

        K = np.exp(-(D1+D2))*(sigma_f**2)
        #print('K: ', K.shape)
        #print(K.shape)
        
        np.fill_diagonal(K, K.diagonal() +sigma_n**2 )
        
        
        
        
        '''
        
        from pylab import figure, cm
        
        X1_new, X2_new = np.meshgrid(np.arange(0, self.simulation.GRID_Y, 1), np.arange(0, self.simulation.GRID_X - 1, 1))
        
        X1_new_dim = X1_new.shape
        
        K_new = np.zeros((X_dim1, X1_new_dim[0], X1_new_dim[1]))
        
        Y_predict = np.zeros(X1_new_dim)
        
        Y_predict_var = np.zeros(X1_new_dim)
        
        for i in range(X_dim1):
        
            D1 = X1_new - X1[i]
        
            #D1 = ( D1**2 ) / l1**2
            D1 = ( D1 ) / l1**2
        
            D2 = X2_new - X2[i]
        
            #D2 = ( D2**2 ) / l2**2
            D2 = ( D2 ) / l2**2
            
            K_new[i,:,:] = sigma_f**2 * np.exp( - 0.5 * (D1 + D2) )
        '''
        
        from numpy.linalg import inv
        K_inv = inv(K)
        
        
        #Predict neigbourhood only
        
        for i in range(self.simulation.GRID_Y):
            for j in range(self.simulation.GRID_X):
                #dist = np.sqrt((self.grid_pos_x - j)**2 + (self.grid_pos_y - i)**2) 
               # if dist <= radius:
                
                    #m1 = np.dot(K_new[:,i,j],K_inv)     
                    self.internal_map[i,j] = np.dot(K_inv,Y)
                    #Y_predict_var[i,j] = K[0,0] - K_new[:,i,j].dot(K_inv.dot(np.transpose(K_new[:,i,j])))


        #self.internal_map = Y_predict
        
        return 1
        
    def update(self):      
        
        #Momentum
        '''
        x,y = self.get_gradient_velocity()        
        t = 0.1        
        self.change_x = t*self.change_x + (1-t)*x
        self.change_y = t*self.change_y + (1-t)*y
        '''
        
        if self.simulation.GP == True:
            if self.simulation.timer % self.simulation.gp_step == 0:
                self.predict_belief_map2()

        self.change_x, self.change_y = self.get_gradient_velocity()     

        '''
        positioning_noise_x = random.uniform(-self.simulation.positioning_noise_strength, self.simulation.positioning_noise_strength)
        positioning_noise_y = random.uniform(-self.simulation.positioning_noise_strength, self.simulation.positioning_noise_strength)

        if random.random() < self.simulation.positioning_noise_prob:            
            self.change_x = self.change_x + positioning_noise_x * self.simulation.ARENA_WIDTH/self.simulation.GRID_X
            self.change_y = self.change_y + positioning_noise_y * self.simulation.ARENA_HEIGHT/self.simulation.GRID_Y      
        '''   
        super().update()
        
        self.grid_pos_x = int(np.trunc((self.center_x * (self.simulation.GRID_X - 1) / self.simulation.ARENA_WIDTH)))
        self.grid_pos_y = int(np.trunc(((1-self.center_y / self.simulation.ARENA_HEIGHT) * (self.simulation.GRID_Y - 1))))
        
    def update_confidence_and_belief(self):		 
        self.confidence_map *= self.simulation.LOSING_CONFIDENCE_RATE
        #self.internal_map *= self.simulation.LOSING_CONFIDENCE_RATE
        
        for j in range(self.grid_pos_x - int(self.simulation.BOUDARY_DIAMETER/2), self.grid_pos_x + int(self.simulation.BOUDARY_DIAMETER/2) + 1):    
            for k in range(self.grid_pos_y - int(self.simulation.BOUDARY_DIAMETER/2), self.grid_pos_y + int(self.simulation.BOUDARY_DIAMETER/2) + 1):
                  
                if j < 0 or j >= self.simulation.GRID_X or k < 0 or k >= self.simulation.GRID_Y: 
                    continue
                
                dx = (self.grid_pos_x - j)
                dy = (k - self.grid_pos_y)
                distance = math.sqrt(dx*dx + dy*dy)
                  
                if (distance <= self.simulation.VISION_BOUNDARY):
                    obstacle = self.is_obstacle_at_position(k,j)
                    if obstacle == True:
                        self.confidence_map[k][j] = self.reliability * self.simulation.BOUNDARY_REPULSION * 5
                        self.internal_map[k][j] = 0
                    else:
                        #self.confidence_map[k][j] = self.reliability
                        
                        s = random.uniform(0, self.simulation.sensing_noise_strength)
        
                        if random.random() < self.simulation.sensing_noise_prob:
                            self.internal_map[k][j] = self.reliability * self.simulation.global_map[k][j] * int(1-s)
                        else:
                            self.internal_map[k][j] = self.reliability * self.simulation.global_map[k][j]
                        
                        positioning_noise_x = int(random.uniform(-self.simulation.positioning_noise_strength, self.simulation.positioning_noise_strength))
                        positioning_noise_y = int(random.uniform(-self.simulation.positioning_noise_strength, self.simulation.positioning_noise_strength))                        
                        k_noise = positioning_noise_x + k #*= self.simulation.ARENA_WIDTH/self.simulation.GRID_X
                        if (k_noise < 0):
                            k_noise = 0
                        elif (k_noise >= self.simulation.GRID_X):
                            k_noise = self.simulation.GRID_X - 1
                        
                        j_noise = positioning_noise_y  + j #*= self.simulation.ARENA_HEIGHT/self.simulation.GRID_Y
                        if (j_noise < 0):
                            j_noise = 0
                        elif (j_noise >= self.simulation.GRID_Y):
                            j_noise = self.simulation.GRID_Y - 1                            
                        
                        if random.random() < self.simulation.positioning_noise_prob:            
                            self.internal_map[k_noise][j_noise] = self.simulation.global_map[k][j]
                            self.confidence_map[k_noise][j_noise] = self.reliability
                        else:
                            self.internal_map[k][j] = self.simulation.global_map[k][j]
                            self.confidence_map[k][j] = self.reliability                            
                                                                    
'''
    Drone swarm Simulator
'''

def listener(sim):
    while True:
        r = input()

        if r == 'close':
            arcade.close_window()
            break
        else:
            r = r.split(',')
            if r[0] == 'attract':
                sim.c_count += 1
                sim.network_command("attract", int(r[1]), int(r[2]))
            elif r[0] == 'deflect':
                sim.c_count += 1
                sim.network_command("deflect", int(r[1]), int(r[2]))
            else:
                print("Wrong Command!")

def find_nbs(matrix, indices):
    
    matrix = np.array(matrix)
    indices = tuple(np.transpose(np.atleast_2d(indices)))
    arr_shape = np.shape(matrix)
    
    dist = np.ones(arr_shape)
    dist[indices] = 0
    dist = scipy.ndimage.distance_transform_cdt(dist, metric='chessboard')
    
    nb_indexes = np.transpose(np.nonzero(dist == 1))
    
    return nb_indexes
    
class SwarmSimulator(arcade.Window):
    
    def __init__(self, ARENA_WIDTH, ARENA_HEIGHT, ARENA_TITLE, SWARM_SIZE, RUN_TIME, INPUT_TIME, GRID_X, GRID_Y, exp_type):
        
        self.ARENA_TITLE = ARENA_TITLE
        self.ARENA_WIDTH = ARENA_WIDTH
        self.ARENA_HEIGHT = ARENA_HEIGHT   
        self.ARENA_TITLE = ARENA_TITLE        
        self.SWARM_SIZE = SWARM_SIZE        
        
        self.INPUT_TIME = INPUT_TIME
        self.GRANULARITY = 10
        self.GRID_X = GRID_X #int(ARENA_WIDTH / GRANULARITY)
        self.GRID_Y = GRID_Y #nt(ARENA_HEIGHT / GRANULARITY)
        
        self.run_time = RUN_TIME
        self.global_map = None
        
        self.swarm_confidence = []
        self.swarm_internal_error = []        
        self.drones_positions = np.array([[0.0 for i in range(self.GRID_X)] for j in range(self.GRID_Y)])  

        self.operator_confidence = []
        self.operator_internal_error = [] 
                
        self.timer = 0
        self.begining = time.time()   
        
        self.exp_type = exp_type
        self.sim_net_id = ''
            
        # FPS TEST parameters
        self.processing_time = 0
        self.draw_time = 0
        self.program_start_time = timeit.default_timer()
        self.fps_list = []
        self.processing_time_list = []
        self.drawing_time_list = []
        self.last_fps_reading = 0
        self.fps = FPSCounter()
        
        # Open file to save timings
        self.results_file = None
        
        # User-study
        if exp_type == "user_study" or "user_study_2":
            self.belief_fig = None
            self.ax = None
            self.ax2 = None
            self.im = None
            self.im2 = None
            
            # drones
            self.picked_drone = None
            
            # game
            self.u_name = None
            self.c_count = 0
            self.click_map = []
            self.u_timer = 0
            self.w_time = 0
            
            # U2 Warning
            self.u2_warning = None
            
        super().__init__(ARENA_WIDTH, ARENA_HEIGHT, ARENA_TITLE)
        #super().set_location(50,50)
        arcade.set_background_color(arcade.color.WHITE)
        
    def setup(self, disaster_size = 1, disaster_location = 'random', operator_size = 1, operator_location = 'random',
              min_reliability = 100, max_reliability = 101, unreliability_percentage = 0, moving_disaster = False, communication_noise = 0, 
              alpha = 10, normal_command = None, command_period = 0, constant_repulsion = False, operator_vision_radius = 150, 
              communication_range = 8, vision_range = 2, velocity_weight_coef = 0.01, boundary_repulsion = 1, aging_factor = 0.9999, GP = False, gp_step = 50,
              maze = None, through_walls = True,
              communication_noise_strength = 0, communication_noise_prob = 0, positioning_noise_strength = 0, positioning_noise_prob = 0, sensing_noise_strength = 0,
              sensing_noise_prob = 0, sim_net_id = ''):        

        #Communication noise
        self.communication_noise_strength = communication_noise_strength
        self.communication_noise_prob = communication_noise_prob
        # print(self.communication_noise_strength, self.communication_noise_prob)

        #Positioning noise
        self.positioning_noise_strength = positioning_noise_strength
        self.positioning_noise_prob = positioning_noise_prob

        #Sensing noise
        self.sensing_noise_strength = sensing_noise_strength
        self.sensing_noise_prob = sensing_noise_prob
        
        self.BOUDARY_DIAMETER = communication_range
        self.VISION_BOUNDARY = vision_range
        self.VELOCITY_WEIGHT = velocity_weight_coef
        
        self.BOUNDARY_REPULSION = boundary_repulsion
        
        self.LOSING_CONFIDENCE_RATE = aging_factor
        
        self.drone_list = arcade.SpriteList()  
        self.operator_list = arcade.SpriteList()
        self.obstacle_list = arcade.SpriteList() 
        
        self.influenced_drone = None
        
        self.median_swarm_confidence = 0
        self.median_internal_error = 0
                
        self.min_reliability = min_reliability
        self.max_reliability = max_reliability
        self.unreliability_percentage = unreliability_percentage
        self.communication_noise = communication_noise        
        
        self.alpha = alpha
        self.normal_command = normal_command
        self.command_period = command_period
        self.constant_repulsion = constant_repulsion
        self.operator_vision_radius = operator_vision_radius
        
        self.moving_disaster = moving_disaster
        self.disaster_location = disaster_location
        self.operator_location = operator_location

        self.GP = GP
        self.gp_step = gp_step
        self.maze = maze
        self.through_walls = through_walls
        
        self.directory = ''

        self.collision_counter = 0

        self.drone_distances = collections.Counter({k:0 for k in range(851)})
        
        self.sim_net_id = sim_net_id
        
        # initializing disaster objects
        self.disaster_size = disaster_size

        self.disaster_list = arcade.SpriteList()        
        for i in range (self.disaster_size):            
            d_x, d_y = self.disaster_location[i]            
            if d_x == 'random' or d_x == '':
                d_x = random.randrange(100, self.ARENA_WIDTH - 100)
            if d_y == 'random' or d_y == '':
                d_y = random.randrange(100, self.ARENA_WIDTH - 100)
                
            x, y = d_x, d_y
            spd_x = 0 #random.randrange(-100, 100)/100
            spd_y = 0 #random.randrange(-100, 100)/100    
            scl = 0.1 #random.randrange(25, 100)/100
            
            adisaster = Disaster(x,y,spd_x,spd_y,scl,sim = self,moving=self.moving_disaster)
            self.disaster_list.append(adisaster)   
            
            if self.exp_type == "user_study":
                self.disaster_list[-1].alpha = 0
                        
        
        # initialize drones
        self.drone_info = []

        if self.exp_type == "user_study" or "user_study_2":
            area = self.ARENA_HEIGHT * self.ARENA_HEIGHT
            area_p = area / self.SWARM_SIZE
            area_l = math.sqrt(area_p)
            di_x = area_l / 2
            di_y = area_l / 2
                
        for i in range(self.SWARM_SIZE): 
            if (random.randrange(0,100) < self.unreliability_percentage):
                reliability = random.randrange(self.min_reliability, self.max_reliability + 1)/100.0
            else:
                reliability = 1
                
            noise = random.randrange(0, self.communication_noise + 1)/10000.0
            
            self.drone_info.append('drone ' + str(i) + ": reliability: " + str(reliability) + " & communication noise: " + str(noise))            
            
            if self.exp_type == "user_study" or self.exp_type == "user_study_2":
                if i == 0:
                    pass
                else:
                    di_x = di_x + area_l
                    
                if di_x > self.ARENA_WIDTH:
                    di_x = area_l / 2
                    if di_y + area_l > self.ARENA_WIDTH:
                        gt = False
                        ll = area_l
                        while gt == False:
                            ll = ll / 2
                            if di_y + ll < self.ARENA_WIDTH:
                                gt = True
                        di_y = di_y + ll
                    else:
                        di_y = di_y + area_l
            else:
                di_x = random.randrange(310, self.ARENA_WIDTH)
                di_y = random.randrange(0, self.ARENA_HEIGHT)
                
            self.drone_list.append(
                    Drone(x = di_x, y = di_y, speed = 1, radius = 0.02, name = "drone "+str(i), sim = self, 
                          reliability = reliability, communication_noise = noise))     
        
        #for info in self.drone_info:  
        #    print(info)
        
        # initialize human operatorss
        self.operators_size = operator_size
        for i in range(self.operators_size): 
            op_x, op_y = self.operator_location[i]            
            if op_x == 'random' or op_x == '':
                op_x = random.randrange(100, self.ARENA_WIDTH - 100)
            if op_y == 'random' or op_y == '':
                op_y = random.randrange(100, self.ARENA_WIDTH - 100)
                        
            x, y = op_x, op_y            
            aoperator = Human(x, y, scl = 0.05, sim = self)                
            self.operator_list.append(aoperator) 

        # initialize obstacles

        if self.maze != None:            
            if self.maze == 'simple':
                self.obstacle_list.append(Obstacle(x=500,y=460,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))

            elif self.maze == 'big':
                self.obstacle_list.append(Obstacle(x=450,y=350,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))
                self.obstacle_list.append(Obstacle(x=570,y=460,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))
                self.obstacle_list.append(Obstacle(x=450,y=570,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "horizontal"))
                self.obstacle_list.append(Obstacle(x=300,y=570,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "horizontal"))
                self.obstacle_list.append(Obstacle(x=200,y=570,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "horizontal"))
                self.obstacle_list.append(Obstacle(x=60,y=460,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))
                self.obstacle_list.append(Obstacle(x=300,y=350,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "horizontal"))
                self.obstacle_list.append(Obstacle(x=60,y=260,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))
                self.obstacle_list.append(Obstacle(x=330,y=240,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))

                
            elif self.maze == 'hard':
                self.obstacle_list.append(Obstacle(x=500,y=460,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))       
                self.obstacle_list.append(Obstacle(x=450,y=350,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "horizontal"))
                self.obstacle_list.append(Obstacle(x=570,y=460,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))
                self.obstacle_list.append(Obstacle(x=450,y=570,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "horizontal"))
                self.obstacle_list.append(Obstacle(x=300,y=570,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "horizontal"))
                self.obstacle_list.append(Obstacle(x=200,y=570,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "horizontal"))
                self.obstacle_list.append(Obstacle(x=60,y=460,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))
                self.obstacle_list.append(Obstacle(x=300,y=350,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "horizontal"))
                self.obstacle_list.append(Obstacle(x=60,y=260,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))
                self.obstacle_list.append(Obstacle(x=330,y=240,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))

            elif self.maze == 'extreme':
                self.obstacle_list.append(Obstacle(x=500,y=460,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))          
                self.obstacle_list.append(Obstacle(x=250,y=450,change_x = 0, change_y = 0,scl = 0.2,sim = self, obstacle_type = "disk"))
                self.obstacle_list.append(Obstacle(x=450,y=350,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "horizontal"))
                self.obstacle_list.append(Obstacle(x=570,y=460,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))
                self.obstacle_list.append(Obstacle(x=450,y=570,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "horizontal"))
                self.obstacle_list.append(Obstacle(x=300,y=570,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "horizontal"))
                self.obstacle_list.append(Obstacle(x=200,y=570,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "horizontal"))
                self.obstacle_list.append(Obstacle(x=60,y=460,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))
                self.obstacle_list.append(Obstacle(x=300,y=350,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "horizontal"))
                self.obstacle_list.append(Obstacle(x=60,y=260,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))
                self.obstacle_list.append(Obstacle(x=330,y=240,change_x = 0, change_y = 0,scl = 0.5,sim = self, obstacle_type = "vertical"))

            elif self.maze == 'simple_new':
                self.obstacle_list.append(Obstacle(x=300,y=300,change_x = 0, change_y = 0,scl = 0.6,sim = self, obstacle_type = "vertical")) 

            elif self.maze == 'hard_new':
                self.obstacle_list.append(Obstacle(x=400,y=400,change_x = 0, change_y = 0,scl = 1,sim = self, obstacle_type = "vertical_thin"))
                self.obstacle_list.append(Obstacle(x=250,y=200,change_x = 0, change_y = 0,scl = 1,sim = self, obstacle_type = "vertical_thin")) 

            elif self.maze == 'extreme_new':
                self.obstacle_list.append(Obstacle(x=450,y=400,change_x = 0, change_y = 0,scl = 1,sim = self, obstacle_type = "vertical_thin"))
                self.obstacle_list.append(Obstacle(x=330,y=200,change_x = 0, change_y = 0,scl = 1,sim = self, obstacle_type = "vertical_thin"))
                self.obstacle_list.append(Obstacle(x=210,y=400,change_x = 0, change_y = 0,scl = 1,sim = self, obstacle_type = "vertical_thin"))
            
        self.random_drone = self.drone_list[random.randint(0, self.SWARM_SIZE - 1)]    
        self.random_drone_belief_maps = []
        self.random_drone_confidence_maps = []

        self.operator_belief_maps = []
        self.operator_confidence_maps = []
        
        if self.exp_type == "user_study":
            # Belief Plot
            self.belief_fig = plt.figure("Operator's Belief Map")
            self.ax = self.belief_fig.add_subplot(111)            
            self.ax.set_xticks([])
            self.ax.set_yticks([])        
            self.im = self.ax.imshow(np.random.rand(40, 40), cmap='coolwarm', interpolation='nearest')
            self.belief_fig.show()
    
        if self.exp_type == "user_study_2":
            self.u2_warning = "Click on confidence or belief map."
            # Belief Plot
            self.belief_fig, (self.ax, self.ax2) = plt.subplots(nrows=1, ncols=2,  figsize=(8, 8))

            self.belief_fig.suptitle("Status: Pause\n\n"
                "Enter your name in terminal to Start! \n\n"
                    "{}s elapsed\n\n".format(0, self.run_time), fontsize=16)
            
            self.ax.set_title("Operator's Belief Map")
            self.ax.set_xticks([])
            self.ax.set_yticks([])  
            
            self.ax2.set_title("Operator's Confidence Map")
            self.ax2.set_xticks([])
            self.ax2.set_yticks([])

            self.belief_fig.canvas.mpl_connect('button_press_event', self.on_map_click)
            self.im = self.ax.imshow(np.random.rand(40, 40), cmap='coolwarm', interpolation='nearest')
            self.im2 = self.ax2.imshow(np.random.rand(40, 40), cmap='coolwarm', interpolation='nearest')

            self.belief_fig.show()
            


    def log_setup(self, directory = None):
        if directory == None:      
            log = open("log_setup.txt", "a")
        else:
            log = open(directory + "/log_setup.txt", "a")        
        self.directory = directory
        
        log.write('GENERAL INFO:' + '\n')
        log.write('  -- ARENA HEIGHT: ' + str(self.ARENA_WIDTH) + '\n')        
        log.write('  -- ARENA WIDTH: ' + str(self.ARENA_HEIGHT) + '\n')
        log.write('  -- GRID_X: ' + str(self.GRID_X) + '\n')
        log.write('  -- GRID_Y: ' + str(self.GRID_Y) + '\n')
        log.write('  -- GRANULARITY: ' + str(self.GRANULARITY) + '\n')
        
        log.write('  -- COMMAND STRENGTH: ' + str(self.alpha) + '\n')
        log.write('  -- ORDINARY COMMAND: ' + str(self.normal_command) + '\n')
        log.write('  -- ORDINARY COMMAND INTERVAL: ' + str(self.command_period) + '\n')
        log.write('  -- CONSTANT REPULSION CENTER: ' + str(self.constant_repulsion) + '\n')        
        log.write('  -- VELOCITY WEIGHT COEFFICIENT: ' + str(self.VELOCITY_WEIGHT) + '\n')
        log.write('  -- BOUNDARY REPULSION: ' + str(self.BOUNDARY_REPULSION) + '\n')
        log.write('  -- AGING COEFFICIENT: ' + str(self.LOSING_CONFIDENCE_RATE) + '\n')     
        
        log.write('  -- INPUT TIME: ' + str(self.INPUT_TIME) + '\n')
        log.write('  -- GAUSSIAN PROCESSES: ' + str(self.GP)  + ' at each ' + str(self.gp_step) + ' steps ' + '\n')
        
        log.write('\n')
        log.write('OPERATORS INFO:' + '\n')
        log.write('  -- OPERATORS COOMUNICATION RANGE: ' + str(self.operator_vision_radius) + '\n')
        log.write('  -- OPERATORS SIZE: ' + str(self.operators_size) + '\n')
        for i in range(len(self.operator_list)):
            log.write('  -- operator ' + str(i) + ": x position: " + str(self.operator_list[i].center_x) + " & y position: " + str(self.operator_list[i].center_y) + '\n')
        
        log.write('\n')
        log.write('DISASTERS INFO:' + '\n')
        log.write('  -- DISASTERS SIZE: ' + str(self.disaster_size) + '\n')
        for i in range(len(self.disaster_list)):
            log.write('  -- disaster ' + str(i) + ": x position: " + str(self.disaster_list[i].center_x) + " & y position: " + str(self.disaster_list[i].center_y) + " & is moving: " + str(self.disaster_list[i].moving) + '\n')
        
        log.write('\n')
        log.write('DRONE INFO:' + '\n')
        log.write('  -- SWARM SIZE: ' + str(self.SWARM_SIZE) + '\n')
        log.write('  -- DRONE RELIABILITY RANGE: (' + str(self.min_reliability) + ', ' + str(self.max_reliability - 1) + ')' + '\n')
        log.write('  -- DRONE UNRELIABILITY PERCENTAGE: ' + str(self.unreliability_percentage) + '\n')
        log.write('  -- DRONE COMMUNICATION NOISE RANGE: (0.0, ' + str(self.communication_noise/10000.0) + ')' + '\n')
        log.write('  -- DRONE COMMUNICATION RANGE: ' + str(self.BOUDARY_DIAMETER) + '\n')
        log.write('  -- DRONE BELIEF VISION RANGE: ' + str(int(self.BOUDARY_DIAMETER/2)) + '\n')
        log.write('  -- DRONE CONFIDENCE VISION RANGE: ' + str(self.VISION_BOUNDARY) + '\n')
        log.write('\n')        
        for info in self.drone_info:
            log.write('  -- ' + info + '\n')

        if self.maze != None:
            log.write('\n')
            log.write('MAZE: ' + self.maze + ' \n')
            log.write('OBSTACLE INFO:' + '\n')
            
            for obstacle in self.obstacle_list:
                log.write('  -- ' + 'TYPE: ' + obstacle.type + ', POSITION: (' + str(obstacle.center_x) + ', ' + str(obstacle.center_y) + '), ' +
                          'VELOCITY: (' + str(obstacle.change_x) + ', ' + str(obstacle.change_y) + ')'+ '\n')
        
        log.close()

        # Open file to save timings
        self.results_file = open(self.directory + '/performance_test/' + 'stress_test_results.csv', "w")

    def get_current_drones_positions(self):        
        positions = np.array([[0.0 for i in range(self.GRID_X)] for j in range(self.GRID_Y)])
        
        for drone in self.drone_list:
            positions[drone.grid_pos_y, drone.grid_pos_x] += 1
        
        return positions       
        
    def get_swarm_confidence(self):        
        confidence = []
        
        for drone in self.drone_list:
            confidence.append(drone.confidence_map.sum())
            
        return confidence

    def get_operator_confidence(self):        
        confidence = []
        
        for operator in self.operator_list:
            confidence.append(operator.confidence_map.sum())
            
        return confidence
    
    def get_median_confidence(self):
        return np.median(self.get_swarm_confidence())     
    
    def get_swarm_internal_error(self, belief_map = 'belief_map'):        
        errors = []
        for drone in self.drone_list:
            if belief_map == 'belief_map':
                errors.append(np.sum(np.abs(self.global_map - drone.internal_map)))        
            elif belief_map == 'gp_predict':
                errors.append(np.sum(np.abs(self.global_map - drone.internal_map1)))  
        return errors

    def get_operator_internal_error(self):        
        errors = []
        for operator in self.operator_list:            
            errors.append(np.sum(np.abs(self.global_map - operator.internal_map)))       
        return errors
            
    def get_median_belief_error(self, belief_map = 'belief_map'):
        return np.median(self.get_swarm_internal_error(belief_map))       

    def update_map(self):       
        self.global_map=[[0 for i in range(self.GRID_X)] for j in range(self.GRID_Y)]   
        
        for j in range(self.GRID_X):
            for i in range(self.GRID_Y):
                 sum_value = 0
                 
                 for adisaster in self.disaster_list:
                     disaster_grid_center_pos_x = math.trunc((adisaster.center_x * (self.GRID_X -1)/self.ARENA_WIDTH) )
                     disaster_grid_center_pos_y = math.trunc((adisaster.center_y * (self.GRID_Y -1)/self.ARENA_HEIGHT) )
                     dist_x = j -disaster_grid_center_pos_x
                     dist_y = i-disaster_grid_center_pos_y
                     disaster_witdh = adisaster.width * (self.GRID_X -1)/self.ARENA_WIDTH

                     if ((dist_x*dist_x) + (dist_y*dist_y) < disaster_witdh/2 * disaster_witdh/2):
                         sum_value=1
                 
                 for aoperator in self.operator_list:
                     operator_grid_center_pos_x = math.trunc((aoperator.center_x * (self.GRID_X -1)/self.ARENA_WIDTH) )
                     operator_grid_center_pos_y = math.trunc((aoperator.center_y * (self.GRID_Y -1)/self.ARENA_HEIGHT) )
                     dist_x = j - operator_grid_center_pos_x
                     dist_y = i - operator_grid_center_pos_y
                     operator_width = aoperator.width * (self.GRID_X -1)/self.ARENA_WIDTH + 1
                     operator_height = aoperator.height * (self.GRID_Y -1)/self.ARENA_HEIGHT + 1
                     
                     # As players are not in circular shape, this needs to be changed later 
                     if ((dist_x*dist_x) + (dist_y*dist_y) < operator_width/2 * operator_height/2):
                           sum_value=1
                 '''
                 for obstacle in self.obstacle_list:
                     obstacle_grid_center_pos_x = math.trunc((obstacle.center_x * (self.GRID_X -1)/self.ARENA_WIDTH) )
                     obstacle_grid_center_pos_y = math.trunc((obstacle.center_y * (self.GRID_Y -1)/self.ARENA_HEIGHT) )
                     
                     dist_x = j - obstacle_grid_center_pos_x
                     dist_y = i - obstacle_grid_center_pos_y
                     
                     obstacle_witdh = obstacle.width * (self.GRID_X -1)/self.ARENA_WIDTH

                     if ((dist_x*dist_x) + (dist_y*dist_y) < obstacle_witdh/2 * obstacle_witdh/2):
                         sum_value=0
                 '''
                 
                 if (sum_value==1):
                     self.global_map[self.GRID_Y - i - 1][j] = 1
                 else:
                     self.global_map[self.GRID_Y - i - 1][j] = 0

    def get_drone_distances(self):
         distances = []
         
         for i in range(len(self.drone_list) - 1):             
             drone_1 = self.drone_list[i]
            
             for j in range(i + 1, len(self.drone_list)):
                 drone_2 = self.drone_list[j]
                 
                 dx = (drone_1.center_x - drone_2.center_x)
                 dy = (drone_1.center_y - drone_2.center_y)
                 distances.append(int(math.sqrt(dx*dx + dy*dy)))

         return collections.Counter(distances)
            
    def on_update(self, delta_time):
        if self.timer == 2:
            self.w_time = delta_time
        self.u_timer += delta_time
              
    def update(self, interval):
        if self.timer == 1:
            if self.exp_type == "normal_network":
                Thread(target=listener, args=[self]).start()
                
            # elif self.exp_type == "user_study":
            #     self.u_name = input("Please enter your name: ")
                
            # elif self.exp_type == "user_study_2":
            #     self.u_name = input("Please enter your name: ")
                
        if self.timer >= self.run_time:
            #  if self.exp_type == "user_study" or "user_study_2":
            #     directory = self.directory
            #     if directory == None:      
            #         log = open("log_setup.txt", "a")
            #     else:
            #         log = open(directory + "/log_setup.txt", "a") 

            #     log.write('\nUser study info:' + '\n')
            #     if self.exp_type == "user_study":
            #         log.write('  -- Experiment: ' + str("User study 1") + '\n') 
            #     if self.exp_type == "user_study_2":
            #         log.write('  -- Experiment: ' + str("User study 2") + '\n')   
            #     log.write('  -- Player: ' + str(self.u_name) + '\n') 
            #     log.write('  -- Data: ' + str(datetime.datetime.now().strftime("%d-%m-%Y_%H:%M:%S")) + '\n') 
            #     log.write('  -- Click count: ' + str(self.c_count) + '\n') 

            #     log.close()
                
            #     self.save_click_map()
             arcade.close_window()
             
             local_f = []
             global_f = []
             for drone in self.drone_list:
                 local_f += drone.local_forces
                 global_f += drone.global_forces

            #  print('LOCAL: ', np.mean(local_f))
            #  print('GLOBAL: ', np.mean(global_f))
             
             import pandas as pd
             distances = pd.DataFrame([(v) for k, v in self.drone_distances.items()])
             distances.to_csv(self.directory + '/distances.csv', sep=',')

             '''
             import csv
             with open(self.directory + '\\distances.csv','w') as f:
                 w = csv.writer(f)
                 w.writerow(self.drone_distances.keys())
                 w.writerow(self.drone_distances.values())
             '''

        if self.exp_type == "normal_network":
            # Sending maps to web-api for game interface
                self.send_data(self.operator_list[0])
        
        # Start update timer
        start_time = timeit.default_timer()

        self.timer += 1

        if self.exp_type == "user_study":                      
            self.im.set_array(self.operator_list[0].internal_map)
            self.belief_fig.canvas.draw()

        if self.exp_type == "user_study_2":
            if self.timer > 1:
                t_now_s = int(self.u_timer) % 60
                t_now_m = int(self.u_timer) // 60
                self.belief_fig.suptitle("Status: Running\n\n"
                        "{}m:{}s elapsed\n\n"
                        "Last coordinates: {}".format(t_now_m, t_now_s, self.u2_warning), fontsize=16)

            self.im.set_array(self.operator_list[0].internal_map)
            self.im2.set_array(self.operator_list[0].confidence_map)
            self.belief_fig.canvas.flush_events()
            self.belief_fig.canvas.draw()

        # # To refresh the communications in drones
        # for drone in self.drone_list:
        #     drone.have_communicated = False
        
        #if self.timer % 100 == 0:             
        #     print(self.timer)
             
        # if self.timer == 100:
        #     print('***', time.time() - self.begining, '***')
        
        self.disaster_list.update_animation()        
        self.disaster_list.update()

        if self.maze != None:
            self.obstacle_list.update_animation()
            self.obstacle_list.update()

        """
        if self.timer >= 50:
            self.drone_distances.update(self.get_drone_distances().elements())
        """
        
        self.drone_list.update()
        
        #update disaster info in global_map  
        if self.global_map is None or self.moving_disaster is True:
            self.update_map()  

        current_positions = self.get_current_drones_positions()
        '''
        if self.timer == 2 or self.timer % 100 == 0:
            self.display_selected_drone_info(self.drone_list[0])
            self.save_one_heatmap(current_positions, 'swarm_positions_' + str(self.timer), self.directory)
        '''
        self.drones_positions += current_positions

        for drone in self.drone_list :
             drone.update_confidence_and_belief()  
        
        if self.constant_repulsion == True:                        
            for operator in self.operator_list:                
                for drone in self.drone_list:                        
                    dx = (drone.center_x - operator.center_x)
                    dy = (drone.center_y - operator.center_y)
                    if(math.sqrt(dx * dx + dy * dy) <= self.operator_vision_radius):
                        self.send_gradual_indirect_command("small center repel", drone, self.alpha)
        
        # exchange messages in certain distance between drones
        for i in range(len(self.drone_list) - 1):             
             drone_1 = self.drone_list[i]
            
             for j in range(i + 1, len(self.drone_list)):
                 drone_2 = self.drone_list[j]
                 
                 dx = (drone_1.grid_pos_x - drone_2.grid_pos_x)
                 dy = (drone_1.grid_pos_y - drone_2.grid_pos_y)
                 distance = math.sqrt(dx*dx + dy*dy)
                 if (distance <= self.BOUDARY_DIAMETER):
                     if distance <= 1:
                         self.collision_counter += 1
                         
                     drone_1.communicate(drone_2)
        
        # exchange messages in certain distance between drones and operator
        for operator in self.operator_list:                 
             for drone in self.drone_list:
                 
                 dx = (drone.center_x - operator.center_x)
                 dy = (drone.center_y - operator.center_y)
                 distance = math.sqrt(dx*dx + dy*dy)

                 if (distance <= self.operator_vision_radius):
                     operator.communicate(drone)
        
        self.operator_list.update()
        
        if self.timer % 5 == 0:                 
            self.swarm_confidence.append(self.get_swarm_confidence())
            
            #if self.GP == True:
            #    self.swarm_internal_error.append(self.get_swarm_internal_error(belief_map = 'gp_predict'))
            #else:
            self.swarm_internal_error.append(self.get_swarm_internal_error())
            
            self.operator_confidence.append(self.get_operator_confidence())
            #if self.GP == True:
            #    self.operator_internal_error.append(self.get_operator_internal_error(belief_map = 'gp_predict'))
           # else:
            self.operator_internal_error.append(self.get_operator_internal_error())
          
        
        if self.timer % 100 == 0:
            #if self.GP == True:
                #self.random_drone_belief_maps.append(self.random_drone.internal_map1.copy())
            #else:
            self.random_drone_belief_maps.append(self.random_drone.internal_map.copy())
            self.random_drone_confidence_maps.append(self.random_drone.confidence_map.copy())

            self.operator_belief_maps.append(self.operator_list[0].internal_map.copy())
            self.operator_confidence_maps.append(self.operator_list[0].confidence_map.copy())

            #np.savetxt(self.directory + '\\t_{0}_random_droneconfidence_map.csv'.format(self.timer), self.random_drone.confidence_map, delimiter=",")
        #print(self.timer)
        #np.savetxt(self.directory + '\\belief\\t_{0}_random_drone_internal_map.csv'.format(self.timer), self.random_drone.internal_map, delimiter=",")
        #np.savetxt(self.directory + '\\confidence\\t_{0}_random_droneconfidence_map.csv'.format(self.timer), self.random_drone.confidence_map, delimiter=",")
            
        if (self.normal_command != None):             
             period = self.command_period
             '''
             if (self.timer == self.INPUT_TIME - 1):
                 print ("************ readdy for indirect ***************")	
                 #self.display_selected_drone_info(self.random_drone)
             '''    
             if (self.timer >= self.INPUT_TIME and self.timer <= self.INPUT_TIME + period):                 
                 #print("************ time for indirect command ***************")	                 
                 #self.display_selected_drone_info(self.random_drone)
                
                 for operator in self.operator_list: 
                     for drone in self.drone_list:                        
                         dx = (drone.center_x - operator.center_x)
                         dy = (drone.center_y - operator.center_y)
                         if(math.sqrt(dx * dx + dy * dy) <= self.operator_vision_radius):
                             self.send_gradual_indirect_command(self.normal_command, drone, self.alpha)
                                   
                 #self.display_selected_drone_info(self.random_drone) 
             '''
             if (self.timer == self.INPUT_TIME + period + 1):
                 print("displaying 101")
                 #self.display_selected_drone_info(self.random_drone)                                   
             if (self.timer == self.INPUT_TIME + period + 50):
                 print("displaying 150")
                 #self.display_selected_drone_info(self.random_drone)                 
             if (self.timer == self.INPUT_TIME + period + 100 ):
                 print("displaying 300")
                 #self.display_selected_drone_info(self.random_drone)                             
             '''
        message_sum_succ = 0
        message_sum_fail = 0
        if self.timer > self.run_time - 1:
            for drone in self.drone_list :
                message_sum_succ +=drone.message_count_succ
                message_sum_fail +=drone.message_count_fail
                # print ("succ for drone ", drone.message_count_succ)
                # print ("fail for drone ", drone.message_count_fail)
                
            # print ("SUCC: drone message noise: ", str(1 - self.drone_list[0].simulation.communication_noise_prob) +", result: " +str(message_sum_succ / len(self.drone_list)))
            # print ("FAILED: drone message noise: ", str(1 - self.drone_list[0].simulation.communication_noise_prob) +", result: " +str(message_sum_fail / len(self.drone_list)))
        
        # Save the time it took to do this.
        self.processing_time = timeit.default_timer() - start_time

        # Total time program has been running
        total_program_time = int(timeit.default_timer() - self.program_start_time)

        # Print out stats, or add more sprites
        # if total_program_time > self.last_fps_reading:
        #     self.last_fps_reading = total_program_time

        #     if total_program_time > 5:

        #         if total_program_time % 2 == 1:

        #             # Take timings
        #             output = f"{total_program_time}, {self.fps.get_fps():.1f}, " \
        #                      f"{self.processing_time:.4f}, {self.draw_time:.4f}\n"

        #             self.results_file.write(output)
                    
        #             # print(output, end="")

        #             self.fps_list.append(round(self.fps.get_fps(), 1))
        #             self.processing_time_list.append(self.processing_time)
        #             self.drawing_time_list.append(self.draw_time)
        
    def send_gradual_indirect_command(self, where, drone, alpha = 10):        
        if where == 'boundary':
            for i in range(self.GRID_Y):
                for j in range(self.GRID_X):
                    self.operator_list[0].confidence_map[i][j] = (j/self.GRID_X - 1)*alpha     
        elif where == 'corner':
            for i in range(self.GRID_Y):
                for j in range(self.GRID_X):
                    self.operator_list[0].confidence_map[i][j] = (j/self.GRID_X - 1)*alpha + (i/self.GRID_Y - 1)*alpha            
        elif where == 'center attract':
            value = 0
            for i in range(self.GRID_Y):
                for j in range(self.GRID_X):                    
                    r = np.sqrt((j-self.GRID_X/2)**2 + (i-self.GRID_Y/2)**2)                    
                    r = 1 - r/(self.GRID_X/2)                    
                    self.operator_list[0].confidence_map[i][j] = -alpha*r  
                    value += self.operator_list[0].confidence_map[i][j]        
        elif where == 'center repel':
            value = 0
            for i in range(self.GRID_Y):
                for j in range(self.GRID_X):                    
                    r = np.sqrt((j-self.GRID_X/2)**2 + (i-self.GRID_Y/2)**2) 
                    r = 1 - r/(self.GRID_X/2)
                    self.operator_list[0].confidence_map[i][j] = alpha*r
                    value += self.operator_list[0].confidence_map[i][j]                        
        elif where == 'small center repel':
            for i in range(self.GRID_Y):
                for j in range(self.GRID_X):                    
                    r = np.sqrt((j-self.GRID_X/2)**2 + (i-self.GRID_Y/2)**2)                      
                    if r <= 5:     
                        r = 1 - r/(self.GRID_X/2)
                        self.operator_list[0].confidence_map[i][j] = alpha*r
        elif where == 'disaster_attract':
            # y_max, x_max = np.unravel_index(self.operator_list[0].internal_map.argmax(), self.operator_list[0].internal_map.shape)

            #if self.operator_list[0].internal_map[y_max][x_max] > 0:
            for i in range(self.GRID_Y):
                for j in range(self.GRID_X):                    
                        #r = np.sqrt((j-x_max)**2 + (i-y_max)**2)
                    val = (-(i-self.GRID_Y/2)**2-(j-self.GRID_X/2)**2)/400

                    if j < self.GRID_X/2:
                        self.operator_list[0].confidence_map[i][j] = (-val -2)/alpha
                    else:
                        self.operator_list[0].confidence_map[i][j] = val/alpha
    
        else:
            print('wrong command')
        
        #self.display_selected_drone_info(self.operator_list[0])
        #np.savetxt('constant_repulsion_map.csv', self.operator_list[0].confidence_map, delimiter=",")        
        #self.display_selected_drone_info(drone)
        drone.communicate(self.operator_list[0]) #exchange message with drone        
        #self.display_selected_drone_info(drone)    
    
    def display_selected_drone_info(self, selected_drone):        
        '''
        if (self.exp_type == 5):
            self.save_one_heatmap(selected_drone.confidence_map, 'conf_' + str(self.timer), "5th")

        if (self.exp_type == 6):
            self.save_one_heatmap(selected_drone.internal_map, 'belief_' + str(self.timer), "6th")
        '''
        data_conf = np.asarray(selected_drone.confidence_map)
        data_internal =  np.asarray(selected_drone.internal_map)        
        data_global = np.asarray(self.global_map)
                
        # Rescale to 0-255 and convert to uint8 
        rescaled_conf = (255.0 * (data_conf - data_conf.min()) / (data_conf.max() - data_conf.min())).astype(np.uint8)
        rescaled_internal = (255.0 * (data_internal - data_internal.min())/ (data_internal.max() - data_internal.min())).astype(np.uint8)
        rescaled_global = (255.0 * (data_global - data_global.min()) / (data_global.max() - data_global.min())).astype(np.uint8)
        
        ax = []
        fig = plt.figure(figsize=(10,10))
        
        '''
        fig.suptitle('{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}'
                     .format(selected_drone.name, selected_drone.min_conf, selected_drone.min_x, 
                             selected_drone.min_y, selected_drone.grid_pos_x, selected_drone.grid_pos_y, 
                             selected_drone.change_x, selected_drone.change_y), fontsize=16)
        '''
        ax.append( fig.add_subplot(1, 3, 1) )
        #cmap = matplotlib.cm.gist_stern
        ax[-1].set_title(" Confidence")  # set title
        plt.imshow(rescaled_conf, cmap='coolwarm', interpolation='nearest')
        # im = plt.imshow(rescaled_conf, cmap='coolwarm', interpolation='nearest')
        
        #plt.imshow(rescaled_conf, cmap = cm.gnuplot2_r,interpolation='nearest')
        ax.append( fig.add_subplot(1, 3, 2) )
        ax[-1].set_title(" Internal Map")  # set title
        plt.imshow(rescaled_internal, cmap='coolwarm', interpolation='nearest')

        #plt.imshow(rescaled_conf, cmap = cm.gnuplot2_r,interpolation='nearest')
        ax.append( fig.add_subplot(1, 3, 3) )
        ax[-1].set_title(" Global Map")  # set title
        plt.imshow(rescaled_global, cmap='coolwarm', interpolation='nearest')        
        plt.show()
     
    def plot_heatmaps(self, maps, title):
        ax = []
        fig = plt.figure(figsize=(20,20))
        fig.suptitle(title, fontname="Arial", fontsize=12)        
        
        for i in range(min(len(maps), 10)):            
            data_map = np.asarray(maps[i])

            data_map = np.clip(data_map, 0, 1)
            
            rescaled_map = (255.0  * (data_map - data_map.min())/ (data_map.max()- data_map.min())).astype(np.uint8)                        
            ax.append( fig.add_subplot(2, 5, i + 1) )            
            plt.imshow(rescaled_map, cmap='coolwarm', interpolation='nearest')            
        plt.show()
    
    def save_image_plot_heatmaps(self, maps, title, directory='temp'):
        ax = []
        fig = plt.figure(figsize=(20,20))
        fig.suptitle(title)        
        
        for i in range(min(len(maps), 10)):            
            data_map = np.asarray(maps[i])

            data_map = np.clip(data_map, 0, 1)
            
            rescaled_map = (255.0  * (data_map - data_map.min())/ (data_map.max()- data_map.min())).astype(np.uint8)                        
            ax.append( fig.add_subplot(2, 5, i + 1) )            
            plt.imshow(rescaled_map, cmap='coolwarm', interpolation='nearest')            
        plt.savefig(directory + '/map_images/' + title + '.png')
        plt.close()

    def save_image_plot_boxplots(self, maps, title, directory):        
        #np.savetxt('maps_{0}.csv'.format(time.time()), maps, delimiter=",")
        #maps.to_csv('maps_{0}.csv'.format(time.time()))
        fig, ax = plt.subplots()
        #plt.yticks(np.arange(0, 800, 200))
        ax.set_xlabel('Simulation steps')
        ax.set_xticks(np.arange(0, 1001, 100))
        #ax.set_xticklabels(['0', '300', '600', '900', '1200'])        
        ax.set_yticks(np.arange(0, 1601, 400))
        #ax.set_yticklabels(['0', '400', '800', '1200', '1600'])        
        ax.boxplot(maps)        
        plt.savefig(directory + '/map_images/' + title + '.png')
        plt.close()
    
    def plot_boxplots(self, maps, title):        
        #np.savetxt('maps_{0}.csv'.format(time.time()), maps, delimiter=",")
        #maps.to_csv('maps_{0}.csv'.format(time.time()))
        fig, ax = plt.subplots()
        #plt.yticks(np.arange(0, 800, 200))
        ax.set_xlabel('Simulation steps', fontname="Arial", fontsize=12)
        ax.set_xticks(np.arange(0, 1001, 100))
        ax.set_xticklabels(['0', '300', '600', '900', '1200'], fontname="Arial", fontsize=12)        
        ax.set_yticks(np.arange(0, 1601, 400))
        ax.set_yticklabels(['0', '400', '800', '1200', '1600'], fontname="Arial", fontsize=12)        
        ax.boxplot(maps)        
        plt.show()
     
    def plot_positions(self, sim):
         fig, ax = plt.subplots()
         #plt.yticks(np.arange(0, 800, 200))
         ax.set_xlabel('Mission Zone', fontname="Arial", fontsize=12)
         ax.set_xticks([0,20,39])
         ax.set_xticklabels(['0', '20', '40'], fontname="Arial", fontsize=12)         
         ax.set_yticks(np.arange(0, 20, 39))
         ax.set_yticklabels(['0', '20', '40'], fontname="Arial", fontsize=12)            
         #data_map = np.rot90(sim.drones_positions)                
         data_map = sim.drones_positions                 
         rescaled_map = (255.0  * (data_map - data_map.min())/(data_map.max() - data_map.min())).astype(np.uint8)         
         #rescaled_map.to_csv('swarm_distribution.csv')
         np.savetxt('swarm_distribution_{0}.csv'.format(time.time()), rescaled_map, delimiter=",")         
         plt.imshow(rescaled_map, cmap='coolwarm', interpolation='nearest')                
         plt.show()
        
    def save_one_heatmap(self, drone_map, title, directory="temp"):    
        #for i in range(min(len(maps), 10)):        
        data_map = np.asarray(drone_map)
        rescaled_map = (255.0  * (data_map - data_map.min())/ (data_map.max()- data_map.min())).astype(np.uint8)
        #heatmaps.append(rescaled_map)    
        np.savetxt(directory + '/' + title +'.csv', rescaled_map, delimiter=",") 
        #np.savetxt(title + '_{0}.csv'.format(time.time()), heatmaps, delimiter=",")

    def save_heatmaps(self, maps, title, directory="temp"):            
        for i in range(min(len(maps))):        
            self.save_one_heatmap(maps[i], title + str(i), directory)
    
    def save_boxplots(self, maps, title, directory="temp"):
        #maps_reverse = np.rot90(maps)
        '''
        maps_reverse = np.array([[0.0 for i in range(len(maps[1]))] for j in range(len(maps[0]))])
        for i in range (len(maps[0])):
            for j in range (len(maps[1])):
                maps_reverse[j][i] = maps[i][j]
        '''
        '''
        map_precentiles = np.array([[0.0 for i in range(4)] for j in range(len(maps))])
        for i in range (len(maps)):
    	     map_precentiles [i][0] = i+1
    	     map_precentiles [i][1] = np.percentile(maps[i], 25)
    	     map_precentiles [i][2] = np.percentile(maps[i], 50)
    	     map_precentiles [i][3] = np.percentile(maps[i], 75)
    	     
        np.savetxt(directory + '/' + title+'.csv', map_precentiles, delimiter=",")
        '''
        np.savetxt(directory + '/' + title+'.csv', maps, delimiter=",")
    
    def save_positions(self, sim, directory="temp"):
         data_map = sim.drones_positions        
         rescaled_map = (255.0  * (data_map - data_map.min())/(data_map.max() - data_map.min())).astype(np.uint8)
         np.savetxt(directory + '/' + 'swarm_distribution.csv', rescaled_map, delimiter=",")
    
    # drag a drone
    def on_mouse_press(self, x, y, button, modifiers):
        if button == arcade.MOUSE_BUTTON_LEFT:
            if self.exp_type == "user_study":
                self.c_count += 1
                for drone in self.drone_list:
					# dx^2 + dy^2 < 2 * r^2 to catch the area around drone as well
                    if ((drone.center_x-x)*(drone.center_x-x) + (drone.center_y-y)*(drone.center_y-y) < drone.width*drone.width*4):
                        self.picked_drone = drone
                        #self.display_selected_drone_info(self.picked_drone)
                        break
                if(self.picked_drone==None):
                    return
    
    # drop the drone to a target position
    def on_mouse_release(self, x, y, button, modifiers):
        if button == arcade.MOUSE_BUTTON_LEFT:
            if self.exp_type == "user_study":
                x_gr = 0
                y_gr = 0
                if self.picked_drone:
                    # j = math.trunc((x * (self.GRID_X -1)/self.ARENA_WIDTH))
                    # i = math.trunc((y * (self.GRID_X -1)/self.ARENA_WIDTH))
                    x_grid = 40 - math.trunc((x * (self.GRID_X -1)/self.ARENA_WIDTH))
                    y_grid = 40 - math.trunc((y * (self.GRID_Y -1)/self.ARENA_HEIGHT))
                    x_gr = x_grid
                    y_gr = y_grid
                    #set the confidence of the drone 
                    #self.picked_drone.confidence_map = np.array([[0.5 for i in range(self.GRID_X)] for j in range(self.GRID_Y)])
                    
                    #self.picked_drone.confidence_map[j][j] = -100
                    
                    for i in range(self.GRID_Y):
                        for j in range(self.GRID_X):                    
                            r = np.sqrt((i-x_grid)**2 + (j-y_grid)**2)                    
                            r = 1 - r/(self.GRID_X/2)                    
                            self.picked_drone.confidence_map[j][self.GRID_X-1-i] = self.picked_drone.confidence_map[j][self.GRID_X-1-i] - 2 * r    
                             
                    print("{} to position ({},{})".format(self.picked_drone.name.title(), x_grid, y_grid))
                    # self.display_selected_drone_info(self.picked_drone)
                    self.picked_drone = None
                    
                else:
                    print("Select a drone first!")
                    
                c_i = None
                for click in self.click_map:
                    if click[1] == x_gr and click[2] == y_gr:
                        c_i = self.click_map.index(click)
                if c_i != None:
                    self.click_map[c_i] = (self.click_map[c_i][0] + 1, x_gr, y_gr)
                else:
                    self.click_map.append((1, x_gr, y_gr))
    def on_draw(self):
        # Start timing how long this takes
        draw_start_time = timeit.default_timer()
        
        arcade.start_render()
        self.operator_list.draw()
        self.disaster_list.draw()
        self.drone_list.draw()

        if self.maze != None:
            self.obstacle_list.draw()

        self.draw_time = timeit.default_timer() - draw_start_time
        self.fps.tick()
        if self.exp_type != "user_study":
            arcade.draw_text("Timesteps: {}/{}".format(self.timer, self.run_time), self.ARENA_WIDTH/2,
                                self.ARENA_HEIGHT - 40, arcade.color.ASH_GREY, 15, anchor_x='center')
        if self.exp_type == "normal_network":
            arcade.draw_text("Online", self.ARENA_WIDTH/2,
                            self.ARENA_HEIGHT - 60, arcade.color.ASH_GREY, 10, anchor_x='center')
        elif self.exp_type == "user_study_2":
            arcade.draw_text("User study 2", self.ARENA_WIDTH/2,
                            self.ARENA_HEIGHT - 55, arcade.color.ASH_GREY, 10, anchor_x='center')
        elif self.exp_type == "user_study":
            if self.timer > 1:
                t_now_s = int(self.u_timer) % 60
                t_now_m = int(self.u_timer) // 60
                arcade.draw_text(f"{t_now_m}m:{t_now_s}s elapsed", self.ARENA_WIDTH/2,
                                self.ARENA_HEIGHT - 40, arcade.color.ASH_GREY, 20, anchor_x='center')
            else:
                pass
            arcade.draw_text("User study 1", self.ARENA_WIDTH/2,
                            self.ARENA_HEIGHT - 55, arcade.color.ASH_GREY, 10, anchor_x='center')
            if self.picked_drone:
                arcade.draw_text("{} selected, now point to the location and release the button!".format(self.picked_drone.name.title()), self.ARENA_WIDTH/2,
                                20, arcade.color.GO_GREEN, 9, anchor_x='center')
            elif self.timer == 1:
                arcade.draw_rectangle_filled(self.ARENA_WIDTH/2, self.ARENA_HEIGHT/2 + 8, self.ARENA_WIDTH, 40, arcade.color.ASH_GREY)
                arcade.draw_text("Please enter your name in the terminal to start.", self.ARENA_WIDTH/2,
                                self.ARENA_HEIGHT/2, arcade.color.BLACK, 15, anchor_x='center')
            else:
                arcade.draw_text("Select a drone by right-clicking and not releasing button!", self.ARENA_WIDTH/2,
                                20, arcade.color.RED, 9, anchor_x='center')
        else:
            arcade.draw_text("Stand-alone", self.ARENA_WIDTH/2,
                            self.ARENA_HEIGHT - 60, arcade.color.ASH_GREY, 10, anchor_x='center')
        
    def send_data(self, operator):
        """
        Send simulation information to web via websocket
        :return: Json object
        """
        api_server = 'http://localhost:8000'
        conf_map_ls = operator.confidence_map.tolist()
        belief_map_ls = operator.internal_map.tolist()

        data = {
            "config":{
                "confidence": conf_map_ls,
                "belief": [[[r, c], belief_map_ls[r][c]] for r in range(len(belief_map_ls))
                           for c in range(len(belief_map_ls[r])) if belief_map_ls[r][c] > 0],
            },
        }

        r = requests.post(api_server + '/api/v1/simulations/' + self.sim_net_id + '/timestep/' + str(self.timer), json=data)
        
        return r.status_code

    def network_command(self, operation, x=0, y=0):        
        if operation == "attract":
            nbs = find_nbs(self.operator_list[0].confidence_map, [x, y])
            for node in nbs:
                a = node[0]
                b = node[1]
                self.operator_list[0].confidence_map[a][b] = 0
            self.operator_list[0].confidence_map[x][y] = 0
        elif operation == "deflect":
            nbs = find_nbs(self.operator_list[0].confidence_map, [x, y])
            for node in nbs:
                a = node[0]
                b = node[1]
                self.operator_list[0].confidence_map[a][b] += 1
            self.operator_list[0].confidence_map[x][y] += 1
        else:
            print("Operation not defined!")

    def on_map_click(self, event):
        if self.exp_type == "user_study_2":
            if event.inaxes:
                self.c_count += 1
                x_r = 40 - math.trunc(event.xdata)
                y_r = math.trunc(event.ydata)
                self.u2_warning = "({}, {})".format(math.trunc(event.xdata), y_r)
                
                for i in range(self.GRID_Y):
                    for j in range(self.GRID_X):                    
                        r = np.sqrt((i-x_r)**2 + (j-y_r)**2)                    
                        r = 1 - r/(self.GRID_X/2)                    
                        self.operator_list[0].confidence_map[j][self.GRID_X-1-i] = self.operator_list[0].confidence_map[j][self.GRID_X-1-i] - 2 * r    
                
                c_i = None
                for click in self.click_map:
                    if click[1] == x_r and click[2] == y_r:
                        c_i = self.click_map.index(click)
                if c_i != None:
                    self.click_map[c_i] = (self.click_map[c_i][0] + 1, x_r, y_r)
                else:
                    self.click_map.append((1, x_r, y_r))

                print("Clicked on {}, {} inside confidence map".format(x_r, y_r))      
            else:
                self.u2_warning = "Only click on map areas!"
                print("Clicked on the confidence map please!")      

    def save_click_map(self):
        import csv
        directory = self.directory
        if directory == '':
            directory = 'temp'
        with open(self.directory + '/' + "click_map" +'.csv','w') as f:
            csv_o=csv.writer(f)
            csv_o.writerow(['count','x','y'])
            for row in self.click_map:
                csv_o.writerow(row)

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

def main(SIM_ID, ARENA_WIDTH, ARENA_HEIGHT, name_of_experiment, SWARM_SIZE, run_time,
         INPUT_TIME, GRID_X, GRID_Y, disaster_size, disaster_location, operator_size,
         operator_location,reliability_1, reliability_2, unreliability_percentage, moving_disaster,
         communication_noise, alpha, normal_command, command_period, constant_repulsion,
         operator_vision_radius,communication_range, vision_range, velocity_weight_coef, boundary_repulsion,
         aging_factor, gp, gp_step, maze, through_walls,communication_noise_strength, communication_noise_prob,
         positioning_noise_strength, positioning_noise_prob,sensing_noise_strength, sensing_noise_prob):

    sim_net_id = SIM_ID

    sim = SwarmSimulator(ARENA_WIDTH, ARENA_HEIGHT, name_of_experiment,
                                    SWARM_SIZE, run_time, INPUT_TIME, GRID_X, GRID_Y, "normal_network")
    sim.setup(disaster_size, disaster_location, operator_size, operator_location, reliability_1,
                reliability_2, unreliability_percentage, moving_disaster, communication_noise,
                alpha, normal_command, command_period, constant_repulsion, operator_vision_radius,
                communication_range, vision_range, velocity_weight_coef, boundary_repulsion,
                aging_factor, gp, gp_step, maze, through_walls,communication_noise_strength,
                communication_noise_prob, positioning_noise_strength, positioning_noise_prob,
                sensing_noise_strength, sensing_noise_prob, sim_net_id)

    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    if (not os.path.isdir('outputs/' + name_of_experiment)):
        os.mkdir('outputs/' + name_of_experiment)
    if (not os.path.isdir('outputs/' + name_of_experiment + "/" + str(EXP_D_T))):
        os.mkdir('outputs/' + name_of_experiment + "/" + str(EXP_D_T))
    if (not os.path.isdir('outputs/' + name_of_experiment + "/" + str(EXP_D_T) + '/performance_test')):
        os.mkdir('outputs/' + name_of_experiment + "/" + str(EXP_D_T) + '/performance_test')

    sim.directory = str('outputs/' + name_of_experiment + "/" + str(EXP_D_T))
    directory = sim.directory
    sim.log_setup(directory)
    
    arcade.run()
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-size', type = int, default = 15) #swarm_size
    parser.add_argument('-name', type = str, default = "General") #experiment_name
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
    parser.add_argument('-aging', type = float, default = 0.9999) #aging factor
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
    parser.add_argument('-communication_noise_strength', type = float, default = 0) 
    parser.add_argument('-communication_noise_prob', type = float, default = 0) # comm rate
    parser.add_argument('-positioning_noise_strength', type = float, default = 0) 
    parser.add_argument('-positioning_noise_prob', type = float, default = 0) 
    parser.add_argument('-sensing_noise_strength', type = float, default = 0) 
    parser.add_argument('-sensing_noise_prob', type = float, default = 0) 
    parser.add_argument('-sim_id', type = str, default = "0") 

    args = parser.parse_args()
    
    disasters_locations = merge(args.d_xs, args.d_ys)
    operators_locations = merge(args.op_xs, args.op_ys)
    
    if args.d_size > len(args.d_xs):
        disasters_locations += [('random', 'random')]*(args.d_size - len(args.d_xs))
        
    if args.op_size > len(args.op_xs):
        operators_locations += [('random', 'random')]*(args.op_size - len(args.op_xs))

    main(args.sim_id, args.width, args.height, args.name, args.size, args.run_time, args.input_time, args.grid_x, args.grid_y, args.d_size, 
         disasters_locations, args.op_size, operators_locations, args.r_min, args.r_max, args.r_perc, args.d_move, 
         args.noise, args.alpha, args.cmd, args.cmd_t, args.const_repel, args.hum_r, args.comm_range, args.vis_range, args.w, 
         args.bound, args.aging, args.gp, args.gp_step, args.maze, args.walls, args.communication_noise_strength, args.communication_noise_prob,
         args.positioning_noise_strength, args.positioning_noise_prob, args.sensing_noise_strength, args.sensing_noise_prob)
