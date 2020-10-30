import os
import pandas as pd
import numpy as np


for sub in ['data']:
    belief_error = pd.DataFrame()
    confidence = pd.DataFrame()
    operator_belief_error = pd.DataFrame()
    operator_confidence = pd.DataFrame()

    for filename in os.listdir(os.getcwd() + '/' + sub):
        for file in os.listdir(os.getcwd() + '/' + sub + '/' + filename):
            
            if file == 'belief_error.csv':                
                print(sub + '/' + filename  + '/' + file )
                curr = pd.read_csv(sub + '/' + filename  + '/' + file, sep=',',header=None)
                belief_error = pd.concat([belief_error, curr], axis=1)

            if file == 'confidence_time.csv':                
                print(sub + '/' + filename  + '/' + file )
                curr = pd.read_csv(sub + '/' + filename  + '/' + file, sep=',',header=None)
                confidence = pd.concat([confidence, curr], axis=1)
                
            elif file == 'operator_belief_error.csv':
                print(sub + '/' + filename  + '/' + file )
                curr = pd.read_csv(sub + '/' + filename  + '/' + file, sep=',',header=None)
                operator_belief_error = pd.concat([operator_belief_error, curr], axis=1)
                
            elif file == 'operator_confidence_time.csv':
                print(sub + '/' + filename  + '/' + file )
                curr = pd.read_csv(sub + '/' + filename  + '/' + file, sep=',',header=None)
                operator_confidence = pd.concat([operator_confidence, curr], axis=1)
                
                
    #print(belief_error.shape)
    #print(confidence.shape)
    
    belief_error.to_csv('belief_error_combined.csv', sep=',')
    confidence.to_csv('confidence_combined.csv', sep=',')
	
    operator_belief_error.to_csv('operator_belief_error_combined.csv', sep=',')
    operator_confidence.to_csv('operator_confidence_combined.csv', sep=',')
