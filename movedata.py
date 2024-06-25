# iterate over files in datasets/train folder

import os
import shutil
from pprint import pprint

# Path to the datasets folder
path = 'datasets/trainNEW/'

counter = {"CNN2":{}, "CNN3":{}, "CNN4":{}}

# Iterate over files in the datasets folder
for filename in os.listdir(path):
    net_layers = filename.split('_')[0]
    net_couples = filename.split('_')[2]
    net_accuracy = float(filename.split('_')[3])

    if net_accuracy < 0.5:
        # remove that file
        print(filename)
        # os.remove(path+filename)

    # if net_couples not in counter[net_layers]:
    #     counter[net_layers][net_couples] = 0
    
    # if counter[net_layers][net_couples] < 20:
    #     if net_accuracy >= 0.9:
    #         shutil.copy(path+filename, 'datasets/trainSMALLER/'+filename)
    #         counter[net_layers][net_couples] += 1


pprint(counter)