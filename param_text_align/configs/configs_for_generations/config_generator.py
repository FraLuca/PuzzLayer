# Create 8 different config based on the original config, change GPUS and OUTPUT_DIR accordingly to gpus
import yaml
# Read the original config


# Create 8 different config based on the original config, change GPUS and OUTPUT_DIR accordingly to gpus
for i in range(8):
    file = './configs/train.yaml'
    with open(file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config['SOLVER']['GPUS'] = [i]
        config['OUTPUT_DIR'] = f'results/output_{i}/train'
        config['TEMP_DIR'] = f'results/output_{i}/temp'
    # Create a new config file
    with open(f'./configs/configs_for_generations/train_{i}.yaml', 'w') as f:
        yaml.dump(config, f)

