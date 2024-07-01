# iterate over files in datasets/train folder

import os
import shutil
from pprint import pprint

# Path to the datasets folder
path = 'datasets/trainMLP/'

counterCNN = {"CNN2":{}, "CNN3":{}, "CNN4":{}}
counterMLP = {"MLP2":{}, "MLP3":{}, "MLP4":{}}

# Iterate over files in the datasets folder
for filename in os.listdir(path):
    net_layers = filename.split('_')[0]
    net_couples = filename.split('_')[2]
    net_accuracy = float(filename.split('_')[3])

    if net_accuracy < 0.5:
        # remove that file
        print(filename)
        # os.remove(path+filename)

    if net_couples not in counterMLP[net_layers]:
        counterMLP[net_layers][net_couples] = 0
    
    if counterMLP[net_layers][net_couples] < 20:
        if net_accuracy >= 0.9:
            shutil.copy(path+filename, 'datasets/trainMLPSUBSET/'+filename)
            counterMLP[net_layers][net_couples] += 1


pprint(counterMLP)