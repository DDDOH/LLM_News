# finetune Llama, Bert and finetune ChatGPT using API
import pandas as pd

import time
import os
import shutil
import argparse

import utils
import jsonlines
import progressbar


DEVICE = 'cuda:0' # 'cuda:0' or 'cuda:1'
LORA = True
DEBUG = False # if debug, use only few data

MODEL = 'llama3' # bert or llama3
# bert for distilbert-base-uncased
# llama for meta-llama/Meta-Llama-3-8B

## Data Config
# DATA = 'news' # 'imdb' or 'news'
SPLITION = 'strict' # 'strict' or 'loose'
# if strict, use Hua's code, if loose, use my original code

parser = argparse.ArgumentParser()
parser.add_argument("--ratio", type=float, default=0, help="Ratio for training data")
args = parser.parse_args()


RATIO = args.ratio
# the 'winner-all.csv' contains all pairs.
# We first construct the fixed test set:
# set p threshold to 0.05, keep all pairs with p < 0.05.
# then select 20% of the pairs as the test set. Use random_state = 42.
# then we remove the test set from winner-all.csv, call this as ALL_TRAIN
# ratio means the ratio of the ALL_TRAIN for training.

result_dir = 'tmp_results'
time_str = time.strftime("%Y%m%d-%H%M%S")
result_dir = result_dir + '/' + time_str
finetuned_model_dir = os.path.join(result_dir, 'finetuned_model')
print('result_dir', result_dir)

# create folder
os.makedirs(result_dir)
os.makedirs(finetuned_model_dir)

# copy current python file to the result folder
shutil.copy('main.py', result_dir)
shutil.copy('utils.py', result_dir)
# shutil.copy('README.md', result_dir)


log_file_path = 'Split_{} Model_{} Lora_{} RATIO{}.pkl'.format(SPLITION, MODEL, LORA, RATIO)
log_file_path = os.path.join(result_dir, log_file_path)


all_data = utils.load_data(SPLITION, RATIO, DEBUG)
# size of test set should be 5624

SYSTEM = "You are an editor tasked with choosing the catchier one from two drafted headlines for the same content. Please return either 1 or 2."
USER = "You are presented with two headlines. Which one is catchier?\n1. {}\n2. {}"

# Create a list to store the recordings
recordings = []

# Iterate over your data and create the recordings
train, test = all_data['train'], all_data['test']




# # Constants for the system and user prompts
# SYSTEM = "System prompt content"
# USER = "User prompt with placeholders {} {}"

# Example result directory, update as needed
result_dir = "results"
DEBUG = False

def process_record(data, i):
    headline1 = data['headline_1'][i]
    headline2 = data['headline_2'][i]
    label = data['label'][i]
    
    # Create the recording dictionary
    recording = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(headline1, headline2)},
            {"role": "assistant", "content": str(label)}
        ]
    }
    return recording

def save_json(data, file_path, batch_size=20):
    # Initialize the progress bar
    n_recording = len(data) if not DEBUG else 50
    progress = progressbar.ProgressBar(maxval=n_recording).start()
    
    # Define the path to the jsonl file
    # jsonl_file_path = os.path.join(result_dir, file_path)
    
    # Checkpoint to resume from
    start_index = 0
    if os.path.exists(file_path):
        with jsonlines.open(file_path, mode='r') as reader:
            for start_index, _ in enumerate(reader):
                pass
        start_index += 1  # Start from the next record
    
    with jsonlines.open(file_path, mode='a') as writer:
        for i in range(start_index, n_recording, batch_size):
            end_index = min(i + batch_size, n_recording)
            indices = range(i, end_index)
            
            # Process records
            recordings = [process_record(data, j) for j in indices]
            
            # Write the recordings to the jsonl file
            writer.write_all(recordings)
            
            # Update the progress bar
            progress.update(end_index)
    
    # Close the progress bar
    progress.finish()
    
    print(f"Saved {n_recording} recordings to {file_path}")
    return file_path

# def save_json(data, file_path):

#     n_recording = len(data) if not DEBUG else 50
#     for i in progressbar.progressbar(range(n_recording)):
#         headline1 = data['headline_1'][i]
#         headline2 = data['headline_2'][i]
#         label = data['label'][i]
        
#         # Create the recording dictionary
#         recording = {
#             "messages": [
#                 {"role": "system", "content": SYSTEM},
#                 {"role": "user", "content": USER.format(headline1, headline2)},
#                 {"role": "assistant", "content": str(label)}
#             ]
#         }
        
#         # Append the recording to the list
#         recordings.append(recording)

#     # Define the path to the jsonl file
#     jsonl_file_path = os.path.join(result_dir, file_path)

#     # Write the recordings to the jsonl file
#     with jsonlines.open(jsonl_file_path, mode='w') as writer:
#         writer.write_all(recordings)
#         print(f"Saved {n_recording} recordings to {jsonl_file_path}")

#     return jsonl_file_path

save_json(train, 'train.jsonl')
save_json(test, 'test.jsonl')

