# iterate over files in datasets/train folder

import os
import shutil
from pprint import pprint
import numpy as np

# Path to the datasets folder
path = 'datasets/trainVAR/'
destination_path = 'datasets/validVAR/'
not_used = 'datasets/not_usedVAR/'

# create destination folder
if not os.path.exists(destination_path):
    os.makedirs(destination_path)
if not os.path.exists(not_used):
    os.makedirs(not_used)

# Iterate over files in the datasets folder
samples = os.listdir(path)

dic = {}
for sample in samples:
    n_layer = sample.split('_')[0]
    if n_layer not in dic:
        dic[n_layer] = []
    else:
        dic[n_layer].append(sample)

# take 1% of the samples for each key in dic
names_to_move = {}
for key,value in dic.items():
    random_indices = np.random.choice(len(value), int(len(value)*0.01), replace=False)
    for i in random_indices:
        current_sample = value[i]
        short_name = current_sample.split(".")[0][:-2]
        cnn_type = short_name.split('_')[0]
        if cnn_type not in names_to_move:
            names_to_move[cnn_type] = []
        else:
            names_to_move[cnn_type].append(short_name)
        # shutil.move(path+current_sample, destination_path+current_sample)

print("NAMES TO MOVE")
for key,value in names_to_move.items():
    print(key, len(value))

# now search for duplicates of names_to_move
duplicates = {}
for item in os.listdir(path):
    short_name = item.split(".")[0][:-2]
    cnn_type = short_name.split('_')[0]
    if short_name in names_to_move[cnn_type]:
        if cnn_type not in duplicates:
            duplicates[cnn_type] = []
        else:
            duplicates[cnn_type].append(short_name)
        # shutil.move(path+item, not_used+item)


print("DUPLICATES")
for key,value in duplicates.items():
    print(key, len(value))