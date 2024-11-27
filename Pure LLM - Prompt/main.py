import pandas as pd
import numpy as np
from common.get_prompt import *
from openai import OpenAI
from scipy.stats import fisher_exact
import progressbar
import os
from concurrent.futures import ThreadPoolExecutor
import time


DEBUG = False

# load dataset
all_train = pd.read_csv("Code Dataset/LoRA_CTR_train.csv")
all_test = pd.read_csv("Code Dataset/LoRA_CTR_test.csv")

result_path = 'Pure LLM - Prompt/prompt-results temperature zero/result_gpt.npy'
# if folder does not exist, create it
if not os.path.exists(os.path.dirname(result_path)):
    os.makedirs(os.path.dirname(result_path))


# convert pd dataframe to dict
def df_to_dict(df):
    test_id_ls = df["test_id"].unique()
    dict_df = {}
    for test_id in test_id_ls:
        dict_df[test_id] = df[df["test_id"] == test_id]
    return dict_df


all_train = df_to_dict(all_train)
all_test = df_to_dict(all_test)
if DEBUG:
    all_test = {k: all_test[k] for k in list(all_test)[:20]}



# split test data into significant and insignificant
# for each news in the test data, run significance test
def significance_test_one_news(impressions, clicks, CTRs):
    """
    Run significance test for one news. We compare the headline with the highest CTR with the rest of the headlines. If all such pairs are significant, we return True, otherwise False.

    :param impressions: list of impressions for each headline
    :param clicks: list of clicks for each headline
    :param CTRs: list of CTRs for each headline
    :return: True if all pairs are significant, False otherwise
    """

    # find the headline with the highest CTR.
    highest_CTR_index = CTRs.index(max(CTRs))

    # run pairwise significance test between the headline with the highest CTR and the rest of the headlines
    for i in range(len(CTRs)):
        if i == highest_CTR_index:
            continue
        pair_clicks = [clicks[highest_CTR_index], clicks[i]]
        pair_non_clicks = [
            impressions[highest_CTR_index] - clicks[highest_CTR_index],
            impressions[i] - clicks[i],
        ]
        contingency_table = [pair_clicks, pair_non_clicks]
        odds_ratio, p_value = fisher_exact(contingency_table)
        if p_value < 0.05:  # this pair is significant
            continue
        else:
            return False
    return True


significant_test = {}
significant_mask = []
# insignificant_test = {}
print("Running significance test for each news in the test data...")
for test_id in all_test.keys():
    one_news = all_test[test_id]
    impressions = one_news["impressions"].values.tolist()
    clicks = one_news["clicks"].values.tolist()
    CTRs = one_news["CTR"].values.tolist()
    significant = significance_test_one_news(impressions, clicks, CTRs)
    if significant:
        significant_test[test_id] = one_news
        significant_mask.append(True)
    else:
        significant_mask.append(False)
print("Significance test done!")
print('Among all {} test data, {} are significant.'.format(len(all_test), len(significant_test)))

# save significant_mask
result_dir = os.path.dirname(result_path)
np.save(os.path.join(result_dir, 'significant_mask.npy'), significant_mask)

n_headline = 0
n_correct_expected = 0
for test_id in significant_test.keys():
    one_news = significant_test[test_id]
    n_headline += len(one_news)
    n_correct_expected += 1/len(one_news)
print('Average number of headlines per news for significant news: ', n_headline / len(significant_test))
print('Random guess accuracy on significant test data: ', n_correct_expected/len(significant_test))

n_headline = 0
n_correct_expected = 0
for test_id in all_test.keys():
    one_news = all_test[test_id]
    n_headline += len(one_news)
    n_correct_expected += 1/len(one_news)
print('Average number of headlines per news for all test data: ', n_headline / len(all_test))
print('Random guess accuracy on all test data: ', n_correct_expected/len(all_test))

# run Prompt Based Method on both significant and insignificant test data


test_combination = [
    # model name, is_flip, n_demo, test_sig
    # is_flip: whether we give the correct answer to the model or not. 0 for correct, 1 for incorrect
    ["gpt-3.5-turbo", 0, 0],  # zero-shot learning with significant samples
    ["gpt-3.5-turbo", 0, 2],  # in-context learning
    ["gpt-3.5-turbo", 1, 2],
    ["gpt-3.5-turbo", 0, 5],
    ["gpt-3.5-turbo", 1, 5],
    ["gpt-4-turbo", 0, 0],
    ["gpt-4-turbo", 0, 2],
    ["gpt-4-turbo", 1, 2],
    ["gpt-4-turbo", 0, 5],
    ["gpt-4-turbo", 1, 5],
    ["gpt-4o-2024-08-06", 0, 0],
    ["gpt-4o-2024-08-06", 0, 2],
    ["gpt-4o-2024-08-06", 1, 2],
    ["gpt-4o-2024-08-06", 0, 5],
    ["gpt-4o-2024-08-06", 1, 5],
]


client = OpenAI(
    api_key="OPENAIKEY"
)  # need API key to acess GPT

def predict_parallel(is_flip, model_name, n_demo):
    true_labels, pred_labels = [], []

    def process_one_news(one_news):
        if n_demo != 0:
            gpt_input, true_label = get_prompt_with_examples(demo_examples, one_news, is_flip)
        else:
            gpt_input, true_label = get_prompt_without_example(one_news)
        
        while True:
            try:
                completion = client.chat.completions.create(
                    model=model_name, messages=gpt_input, temperature=0.0
                )
                break
            except:
                time.sleep(1)
        pred_label = completion.choices[0].message.content
        return true_label, pred_label

    with ThreadPoolExecutor() as executor:
        results = list(progressbar.progressbar(executor.map(process_one_news, all_test.values()), max_value=len(all_test)))

    for true_label, pred_label in results:
        true_labels.append(true_label)
        pred_labels.append(pred_label)

    return true_labels, pred_labels



# load previous results
try:
    result_gpt = list(np.load(result_path, allow_pickle=True))
except:
    result_gpt = []

for model_name, is_flip, n_demo in test_combination:
    print("--------------------------")
    print("Parameter setting: {} n_demo {} is_flip {}".format(model_name, n_demo, is_flip))

    # check if this combination has been tested before
    tested_before = False
    for result in result_gpt:
        if result['model_name'] == model_name and result['is_flip'] == is_flip and result['n_demo'] == n_demo:
            print("This combination has been tested before.")
            # print('Model name: {}, is_flip: {}, n_demo: {}'.format(result['model_name'], result['is_flip'], result['n_demo']))
            print("Accuracy on all:{:.2%}\t Accuracy on significant subset: {:.2%}".format(result['acc'], result['acc_sig']))   
            tested_before = True

    if tested_before:
        continue
            

    pred_label = []

    # if test_sig == 0:
    #     test_samples = all_test
    #     true_label = all_train
    # elif test_sig == 1:
    #     test_samples = significant_test
    #     true_label = all_train

    if n_demo > 0:
        # sample n_demo samples from the training data (all_train)
        train_ids = list(all_train.keys())
        np.random.seed(12) # ensure for different test_combination, we have the same demo samples
        demo_ids = np.random.choice(train_ids, n_demo, replace=False)
        demo_examples = []
        for demo_id in demo_ids:
            one_sample = all_train[demo_id]
            demo_examples.append(one_sample)

    # for each news in test_samples, ask gpt to select the best headline
    true_label, pred_label = predict_parallel(is_flip, model_name, n_demo)
    whether_correct = [1 if x == y else 0 for x, y in zip(true_label, pred_label)]
    n_correct_all = sum(whether_correct)
    accur_all = n_correct_all / len(all_test)
    n_correct_sig = np.array(whether_correct)[significant_mask].sum()
    accur_sig = n_correct_sig / len(significant_test)

    print("Accuracy on all data: {:.2%}".format(accur_all))
    print("Accuracy on significant subset: {:.2%}".format(accur_sig))
    result_gpt.append({'model_name': model_name,
                        'is_flip': is_flip,
                        'n_demo': n_demo,
                        'pred_label': pred_label,
                        'true_label': true_label,
                        'acc': accur_all,
                        'acc_sig': accur_sig
                        })
    # save result_gpt
    np.save(result_path, result_gpt)


# result_gpt_df.to_csv("Pure LLM - Prompt/prompt-results/result_gpt_prompt.csv", index=False)

