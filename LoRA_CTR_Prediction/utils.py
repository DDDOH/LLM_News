
from datasets import Dataset, DatasetDict
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os


def load_data(DEBUG):

    # Load the dataset from the specified CSV file
    # dataset = load_dataset('csv', data_files=file_path)

    train_df = pd.read_csv('data/LoRA_CTR_train.csv')
    test_df = pd.read_csv('data/LoRA_CTR_test.csv')

    # keep only test_id, headline, CTR columns
    train_df = train_df[['test_id', 'headline', 'CTR']]
    test_df = test_df[['test_id', 'headline', 'CTR']]

    # rename CTR to label
    # train_df = train_df.rename(columns={'CTR': 'labels'})
    # test_df = test_df.rename(columns={'CTR': 'labels'})


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

    return data, rescale_labels





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
