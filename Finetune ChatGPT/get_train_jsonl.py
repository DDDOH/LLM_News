# Finetune ChatGPT using API
import pandas as pd
import os
import numpy as np

import jsonlines
import progressbar
from get_prompt import prompt_with_label
from common.utils import significance_test_one_news
from check_data import check_data


DEBUG = False # if debug, use only few data


# Create a list to store the recordings
recordings = []


# Load data
train = pd.read_csv('Code Dataset/LoRA_CTR_train.csv')
test_all = pd.read_csv('Code Dataset/LoRA_CTR_test.csv')

# remove existing jsonl file



# get significant news in test
test_ids = test_all['test_id'].unique()
significant_test_ids = []
for test_id in test_ids:
    df = test_all[test_all['test_id'] == test_id]
    impressions = df['impressions'].values.tolist()
    clicks = df['clicks'].values.tolist()
    CTRs = df['CTR'].values.tolist()
    if significance_test_one_news(impressions, clicks, CTRs):
        significant_test_ids.append(test_id)

DEBUG = False

def save_json(data, file_path):
    # remove existing jsonl file
    if os.path.exists(file_path):
        os.remove(file_path)

    # Initialize the progress bar
    unique_test_ids = data['test_id'].unique()
    n_recording = len(unique_test_ids) if not DEBUG else 50
    progress = progressbar.ProgressBar(maxval=n_recording).start()

    # delete the file if it already exists
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # writing into jsonl file
    with jsonlines.open(file_path, mode='a') as writer:
        for i, test_id in enumerate(unique_test_ids):
            if i >= n_recording:
                break
            df = data[data['test_id'] == test_id]
            recordings = prompt_with_label(df['headline'].values.tolist(), np.argmax(df.CTR.values) + 1)
            writer.write(recordings)
            progress.update(i+1)

    progress.finish()
    print(f"Saved {n_recording} recordings to {file_path}")

    check_data(file_path)
    return file_path


save_json(train, 'Finetune ChatGPT/train.jsonl')

# from openai import OpenAI
# client = OpenAI(api_key="OPENAIKEY")

# # upload the file to OpenAI
# client.files.create(
#   file=open('Finetune ChatGPT/train.jsonl', "rb"),
#   purpose="fine-tune"
# )

# # create a fine-tuning job
# client.fine_tuning.jobs.create(
#   training_file="file-abc123",
#   model="gpt-4o-mini-2024-07-18"
# )