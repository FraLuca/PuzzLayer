# Read additional_conv_trainset.sh and split into 8 different files

import os

file = './additional_conv_trainset.sh'
with open(file) as f:
    lines = f.readlines()
# Remove blank lines
lines = [line for line in lines if line.strip()]

# Split the lines into 8 different files
num_lines = len(lines)
num_files = 8
lines_per_file = num_lines // num_files
resto = num_lines % num_files

configs_dir = '../configs/configs_for_generations/'
list_configs = os.listdir(configs_dir)
# Remove if not .yaml
list_configs = [config for config in list_configs if config.endswith('.yaml')]
# Sort
list_configs.sort()

for i in range(num_files):
    start = i * lines_per_file
    end = start + lines_per_file
    if i == num_files - 1:
        end += resto
    with open(f'./splitted_conv/additional_conv_trainset_{i}.sh', 'w') as f:
        lines_to_write = lines[start:end]
        # cange train.yaml with train_{i}.yaml in list_configs
        config_to_use = 'configs/configs_for_generations/'+list_configs[i]
        new_lines = []
        for j, line in enumerate(lines_to_write):
            if 'train.yaml' in line:
                line = line.replace('configs/train.yaml', config_to_use)
            new_lines.append(line)
        f.writelines(new_lines)

# Check the number of lines in each file
total = 0
for i in range(num_files):
    with open(f'./splitted_conv/additional_conv_trainset_{i}.sh') as f:
        lines = f.readlines()
    print(f'File {i}: {len(lines)} lines')
    total += len(lines)

print(f'Total: {total} lines')


