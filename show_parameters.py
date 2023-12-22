import os
import json
import pandas as pd

# Define the path to the directory containing the json files
dir_path = './model'

# Initialize a list to hold the contents of all config.json files
config_contents = []

# Walk through the directory
for root, dirs, files in os.walk(dir_path):
    for file in files:
        # Check if the current file is a config.json
        if file == 'config.json':
            # Construct the full path to the file
            file_path = os.path.join(root, file)
            # Open and read the config.json file
            with open(file_path, 'r') as json_file:
                config_data = json.load(json_file)
                # Add the file name to the dictionary
                config_data['file_name'] = file_path
                # Append the data to the list
                config_contents.append(config_data)

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(config_contents)

# Split the resize into two separate columns for height and width
df['resize_height'], df['resize_width'] = zip(*df['resize'])
df.drop('resize', axis=1, inplace=True)  # drop the original resize column
