# iterate over files in datasets/train folder

import os
import shutil
from pprint import pprint

# Path to the datasets folder
path = 'datasets/trainNEWSMALLER/'
destination_path = 'datasets/trainNEWSMALLER1pertypecouple/'

# create destination folder
if not os.path.exists(destination_path):
    os.makedirs(destination_path)

counter = {"CNN2":{}, "CNN3":{}, "CNN4":{}}

# Iterate over files in the datasets folder
for filename in os.listdir(path):
    net_layers = filename.split('_')[0]
    net_couples = filename.split('_')[2]
    net_accuracy = float(filename.split('_')[3])

    if net_couples not in counter[net_layers]:
        counter[net_layers][net_couples] = 0
    
    if counter[net_layers][net_couples] < 1:
        if net_accuracy >= 0.9:
            shutil.copy(path+filename, destination_path+filename)
            counter[net_layers][net_couples] += 1


pprint(counter)