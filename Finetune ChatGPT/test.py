# https://platform.openai.com/docs/guides/batch/getting-started
# 24-7-25: Bug here.

from openai import OpenAI
import progressbar
import pandas as pd
from get_prompt import prompt_without_label
# from common.get_prompt import get_prompt_without_example
import numpy as np
from common.utils import significance_test_one_news
from concurrent.futures import ThreadPoolExecutor


client = OpenAI(api_key="OPENAIKEY")
test_df = pd.read_csv('Code Dataset/LoRA_CTR_test.csv')
test_ids = np.unique(test_df['test_id'])

RERUN_TEST = False

if RERUN_TEST:
    result_rec = {test_id: {'real label': None, 'predicted label': None} for test_id in test_ids}
else:
    # check if existing result_rec is available
    try:
        result_rec = np.load('Finetune ChatGPT/result_rec.npy', allow_pickle=True).item()
    except FileNotFoundError:
        result_rec = {test_id: {'real label': None, 'predicted label': None} for test_id in test_ids}


# for each row in test_df that has not been predicted, use the headline_1 and headline_2 to generate a prompt

significant_test_ids = []

for test_id in test_ids:
    df = test_df[test_df['test_id'] == test_id]
    impressions = df['impressions'].values.tolist()
    clicks = df['clicks'].values.tolist()
    CTRs = df['CTR'].values.tolist()
    if significance_test_one_news(impressions, clicks, CTRs):
        significant_test_ids.append(test_id)


def process_test_id(test_id):
    if result_rec[test_id]['predicted label'] is not None:
        return

    df = test_df[test_df['test_id'] == test_id]
    impressions = df['impressions'].values.tolist()
    clicks = df['clicks'].values.tolist()
    CTRs = df['CTR'].values.tolist()

    message = prompt_without_label(df.headline.to_list())
    ground_truth = np.argmax(CTRs) + 1
    response = client.chat.completions.create(
        model="ft:gpt-4o-2024-08-06:uw-mib-llm::AImiGZwJ",
        messages=message,
        stream=False,
    )
    prediction = int(response.choices[0].message.content)

    result_rec[test_id]['real label'] = ground_truth
    result_rec[test_id]['predicted label'] = prediction

    return test_id

with ThreadPoolExecutor(max_workers=10) as executor:
    for idx, test_id in enumerate(progressbar.progressbar(test_ids)):
        future = executor.submit(process_test_id, test_id)
        if idx % 50 == 0:
            np.save('Finetune ChatGPT/result_rec.npy', result_rec)
            future.result()  # Ensure the task is completed before saving

# save finally
np.save('Finetune ChatGPT/result_rec.npy', result_rec)


# summarize the results
n_correct = 0
n_sig_correct = 0
for test_id in test_ids:
    if result_rec[test_id]['predicted label'] == result_rec[test_id]['real label']:
        n_correct += 1
    if test_id in significant_test_ids:
        if result_rec[test_id]['predicted label'] == result_rec[test_id]['real label']:
            n_sig_correct += 1

print(f"Accuracy on all data: {n_correct / len(test_ids)}")
print(f"Accuracy on significant data: {n_sig_correct / len(significant_test_ids)}")