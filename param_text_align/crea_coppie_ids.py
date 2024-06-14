'''import itertools
import random

# Vettore delle classi
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Generazione di tutte le possibili coppie
pairs = list(itertools.combinations(classes, 2))
p= ['linear_784_50', 'relu', 'linear_50_25', 'relu', 'linear_25_10']

# Lista delle righe di comando
commands = [
    f'python train.py -cfg configs/train.yaml MODEL.LAYER_LIST "{p}" DATASETS.CLASS_IDS "[{x},{y}]"\n'
    for (x, y) in pairs
]

# Mescolare le righe di comando
random.shuffle(commands)

# Scrittura del file bash
with open('train_all_pairs.sh', 'w') as f:
    f.write("#!/bin/bash\n\n")
    f.writelines(commands)

print("File 'train_all_pairs.sh' creato con successo!")'''

import itertools
import random

# Parametri
hidden_layers = [16, 32, 64, 128, 256, 512]
class_ids = [[0,1], [1,4], [2,7], [4,9], [5,7], [8,9], [2,4], [4,6]]
num_layers = [2, 3, 4, 5, 6, 7]
output_layer = 10

# Apertura del file bash per la scrittura
with open('generate_commands.sh', 'w') as f:
    # Generazione delle combinazioni
    for gen in range(500):
        next_hidden = random.choice(hidden_layers)
        # Inizio della riga
        line = "python train.py -cfg configs/train.yaml MODEL.LAYER_LIST \"['linear_784_{}'".format(next_hidden)
        
        # Aggiunta degli hidden layer intermedi
        num_layers_choice = random.choice(num_layers)
        for i in range(num_layers_choice - 1):
            previous_hidden = next_hidden
            next_hidden = random.choice(hidden_layers)
            if i==num_layers_choice-2:
                line += ", 'relu', 'linear_{}_{}'".format(previous_hidden, output_layer)
            else:
                line += ", 'relu', 'linear_{0}_{1}'".format(previous_hidden, next_hidden)
        
        class_id = random.choice(class_ids)
        # Aggiunta delle classi del dataset
        line += "]\" DATASETS.CLASS_IDS \"{}\" MODEL.TYPE \"MLP{}\"\n".format(class_id, num_layers_choice)
        
        # Scrittura della riga nel file
        f.write(line)

print("File 'generate_commands.sh' creato con successo.")


