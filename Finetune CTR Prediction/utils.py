from datasets import Dataset, DatasetDict
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os
import numpy as np
from common.utils import significance_test_one_news


def load_data(DEBUG, split_id):
    # split_id: if 0, use the original split. If 1 or 2 or 3..., check if the split exists, if not create a new split

    # Load the dataset from the specified CSV file
    # dataset = load_dataset('csv', data_files=file_path)
    if split_id == 0:
        print('Using original split')
        train_df = pd.read_csv('Code Dataset/LoRA_CTR_train.csv')
        test_df = pd.read_csv('Code Dataset/LoRA_CTR_test.csv')
    else:
        split_file_path = f'Finetune CTR Prediction/train_test_split/split_{split_id}'
        if os.path.exists(split_file_path):
            print('Loading existing split')
            train_df = pd.read_csv(f'{split_file_path}/train.csv')
            test_df = pd.read_csv(f'{split_file_path}/test.csv')
        else:
            print('Creating new split')
            ori_train_df = pd.read_csv('Code Dataset/LoRA_CTR_train.csv')
            ori_test_df = pd.read_csv('Code Dataset/LoRA_CTR_test.csv')
            n_news_train = len(ori_train_df['test_id'].unique())
            n_news_test = len(ori_test_df['test_id'].unique())
            # split the data
            all_df = pd.concat([ori_train_df, ori_test_df], ignore_index=True)
            all_test_id = all_df['test_id'].unique()
            new_test_ids = np.random.choice(all_test_id, n_news_test, replace=False)
            new_train_ids = np.setdiff1d(all_test_id, new_test_ids)

            train_df = all_df[all_df['test_id'].isin(new_train_ids)]
            test_df = all_df[all_df['test_id'].isin(new_test_ids)]

            train_df = train_df.sort_values(by='test_id')
            test_df = test_df.sort_values(by='test_id')

            os.makedirs(split_file_path)
            train_df.to_csv(f'{split_file_path}/train.csv', index=False)
            test_df.to_csv(f'{split_file_path}/test.csv', index=False)
            print('New split created')

    train_df = train_df[['test_id', 'headline', 'CTR', 'impressions', 'clicks']]
    test_df = test_df[['test_id', 'headline', 'CTR', 'impressions', 'clicks']]
        
    print('train_df.shape', train_df.shape)
    print('test_df.shape', test_df.shape)

    # drop rows with missing values
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    print('train_df.shape', train_df.shape)
    print('test_df.shape', test_df.shape)

    if DEBUG:
        # take a subset of the data
        train_df = train_df.head(100)
        test_df = test_df.head(100)


    # get significant test_id
    test_ids = test_df['test_id'].unique()
    significant_test_ids = []
    for test_id in test_ids:
        df = test_df[test_df['test_id'] == test_id]
        impressions = df['impressions'].values.tolist()
        clicks = df['clicks'].values.tolist()
        CTRs = df['CTR'].values.tolist()
        if significance_test_one_news(impressions, clicks, CTRs):
            significant_test_ids.append(test_id)



    # remove unused columns
    train_df = train_df[['test_id', 'headline', 'CTR']]
    test_df = test_df[['test_id', 'headline', 'CTR']]

    # scale labels to mean 0 and std 1
    mean = train_df['CTR'].mean()
    std = train_df['CTR'].std()
    train_df['labels'] = (train_df['CTR'] - mean) / std
    test_df['labels'] = (test_df['CTR'] - mean) / std

    def rescale_labels(x):
        return x * std + mean
    
    # data: train and test,
    # train: headline, CTR
    # test: headline, CTR

    data = DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'test': Dataset.from_pandas(test_df),
    })

    print("Among all {} test news, {} are significant".format(len(test_ids), len(significant_test_ids)))

    return data, rescale_labels, significant_test_ids





def check_df_1_in_df_2(df_1, df_2):
    # check if each row of df_1 is in df_2

    # Merge df_1 and df_2 with an indicator
    merged_df = df_1.merge(df_2, on=df_1.columns.tolist(), how='left', indicator=True)

    # Add a new column to df_1 to indicate if each row is also in df_2
    df_1['in_df2'] = merged_df['_merge'] == 'both'

    # Check if all rows in df_1 are also in df_2
    # show rows that are in df_1 but not in df_2
    # print('Number of rows in test_df that are not in all_df', len(df_1[df_1['in_df2'] == False]))

    all_rows_in_df_1_in_df_2 = len(df_1[df_1['in_df2'] == False]) == 0

    return all_rows_in_df_1_in_df_2

def df_2_minus_df_1(df_1, df_2):
    # return rows that are in df_2 but not in df_1

    # Merge the DataFrames with an indicator
    merged_df = pd.merge(df_1, df_2, how='outer', indicator=True)

    # Filter to get rows that are only in df_2
    df_unique_to_2 = merged_df[merged_df['_merge'] == 'right_only']

    # Drop the merge indicator column if you no longer need it
    df_unique_to_2 = df_unique_to_2.drop(columns=['_merge'])

    return df_unique_to_2
