import pandas as pd
import numpy as np
import os


for sub in ['data']:

    for combined_map_name in ['belief_error_combined', 'confidence_combined', 'operator_belief_error_combined', 'operator_confidence_combined']:

        combined_map = np.array(pd.read_csv(combined_map_name + '.csv', sep=',',header=None).values)
                    
        map_precentiles = np.array([[0.0 for i in range(4)] for j in range(len(combined_map))])
        for i in range (len(combined_map)):
            map_precentiles[i][0] = i + 1
            map_precentiles[i][1] = np.percentile(combined_map[i][1:], 25)
            map_precentiles[i][2] = np.percentile(combined_map[i][1:], 50)
            map_precentiles[i][3] = np.percentile(combined_map[i][1:], 75)
                                     
        np.savetxt(combined_map_name + '_percentiles.csv', map_precentiles, delimiter = ",")

        print('done')
