import random

# Read lines from the original file
sufix='32maps_alpha'
folder=f'data/microstate_{sufix}_2sE_dataset_pretokenize'
with open(f'{folder}/output_{sufix}_2sE.txt', 'r') as file:
    lines = file.readlines()

# Shuffle the lines randomly
random.shuffle(lines)

# Calculate the number of lines for each set
total_lines = len(lines)
train_lines = int(total_lines * 0.6)
test_lines = int(total_lines * 0.2)

# Split the lines into train, test, and validation sets
train_set = lines[:train_lines]
test_set = lines[train_lines:train_lines + test_lines]
valid_set = lines[train_lines + test_lines:]

# Write the train set to a file
with open(f'{folder}/output_{sufix}_2sE.train.txt', 'w') as file:
    file.writelines(train_set)

# Write the test set to a file
with open(f'{folder}/output_{sufix}_2sE.test.txt', 'w') as file:
    file.writelines(test_set)

# Write the validation set to a file
with open(f'{folder}/output_{sufix}_2sE.valid.txt', 'w') as file:
    file.writelines(valid_set)
