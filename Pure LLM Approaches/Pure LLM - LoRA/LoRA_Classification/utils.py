
from datasets import Dataset, DatasetDict
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os


def load_data(SPLITION, RATIO, DEBUG):
    # if DATA == 'imdb':
    #     imdb = load_dataset("imdb")
    #     return imdb
    # elif DATA == 'news':
    if SPLITION == 'strict':
        def load_train_test():
            # create the fixed test set first.
            # if fixed_test_set.csv already exists, skip this step
            if os.path.exists('data/fixed_test_set.csv'):
                print('fixed_test_set.csv already exists, skip creating fixed test set')
                test_df = pd.read_csv('data/fixed_test_set.csv')
            else:
                file_path = 'data/selected_pairs_df_005_256.csv'
                df = pd.read_csv(file_path)
                # From Hua's dataset_extraction function, do not modify
                random_state = 42
                test_size = 0.2
                gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
                df['news_id'] = df.groupby(['clickability_test_id', 'eyecatcher_id']).ngroup() + 1
                train_idx, test_idx = next(gss.split(df, groups=df['news_id']))
                train_df = df.iloc[train_idx]
                test_df_original = df.iloc[test_idx]
                train_headlines = set(train_df['headline_1'].tolist() + train_df['headline_2'].tolist())
                test_df = test_df_original[~test_df_original['headline_1'].isin(train_headlines)]
                test_df = test_df[~test_df['headline_2'].isin(train_headlines)]
                # print(f'Filtered Num Compared to the Begining_test_size_{test_size}', (len(test_df_original)-len(test_df)))
                # print(f'Remaining samples of the test set:', len(test_df))
                # print(f'Filter Ratio Compared to the Begining_test_size_{test_size}', (len(test_df_original)-len(test_df))/len(test_df_original))
                ########################################################
                # keep only headline_1, headline_2, higher_CTR columns in train_df and test_df
                # train_df = train_df[['headline_1', 'headline_2', 'higher_CTR']]
                test_df = test_df[['news_id', 'clickability_test_id', 'eyecatcher_id', 'headline_1', 'headline_2','higher_CTR', 'CTR_1', 'CTR_2', 'Size_1', 'Size_2', 'p_value']]
                # save test_df to csv
                test_df.to_csv('data/fixed_test_set.csv', index=False)


            file_path = 'data/winner-all.csv' # all pairs
            all_df = pd.read_csv(file_path)

            # rows with same news_id belongs to the same news, different news_id means different news
            all_df['news_id'] = all_df.groupby(['clickability_test_id', 'eyecatcher_id']).ngroup() + 1

            

            # show number of None in df['headline_1']
            # print('Number of None in df[headline_1]', all_df['headline_1'].isnull().sum())
            # print('Number of None in df[headline_2]', all_df['headline_2'].isnull().sum())

            # remove rows with None values
            # print('Number of rows before removing None', len(all_df))
            all_df = all_df.dropna()
            # print('Number of rows after removing None', len(all_df))

            # check if each row of test_df is in all_df
            check_df_1_in_df_2(test_df, all_df)

            all_train = df_2_minus_df_1(test_df, all_df)

            # sort all_train by the increasing order of p_value
            all_train = all_train.sort_values(by='p_value', ascending=True)

            # remove rows in all_train that contains headline in test_df
            # print('Number of rows before removing rows in all_train that contains headline in test_df', len(all_train))
            all_train = all_train[~all_train['headline_1'].isin(test_df['headline_1'])]
            all_train = all_train[~all_train['headline_2'].isin(test_df['headline_2'])]
            # print('Number of rows after removing rows in all_train that contains headline in test_df', len(all_train))

            # keep only the first RATIO of all_train
            if RATIO == 0:
                train_df = all_train[all_train['p_value'] <= 0.05]

            else:
                train_df = all_train[:int(len(all_train)*RATIO)]

            # keep only headline_1, headline_2, higher_CTR columns in train_df and test_df
            train_df = train_df[['headline_1', 'headline_2', 'higher_CTR']]
            test_df = test_df[['headline_1', 'headline_2', 'higher_CTR']]

            if DEBUG:
                train_df = train_df.head()
                test_df = test_df.head()

            print("Number of rows in train_df", len(train_df))
            print("Number of rows in test_df", len(test_df))

            return train_df, test_df

        train_df, test_df = load_train_test()
        train_df['label'] = [int(label) - 1 for label in train_df['higher_CTR']]
        test_df['label'] = [int(label) - 1 for label in test_df['higher_CTR']]
        train_df = train_df[['headline_1', 'headline_2', 'label']]
        test_df = test_df[['headline_1', 'headline_2', 'label']]

        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        test_dataset = Dataset.from_pandas(test_df, preserve_index=False)

        news = DatasetDict({
            'train': train_dataset,
            'test': test_dataset,
        })
    elif SPLITION == 'loose':
        # allow headlines of the same news to be put into both training and test set, might cause information leakage
        def load_and_prepare_dataset(file_path):
            # Load the dataset from the specified CSV file
            # dataset = load_dataset('csv', data_files=file_path)

            data = pd.read_csv(file_path)
            # keep only headline_1, headline_2, higher_CTR columns
            data = data[['headline_1', 'headline_2', 'higher_CTR']]
            # data['text'] = data['headline_1'] + " [SEP] " + data['headline_2']
            data['label'] = [int(label) - 1 for label in data['higher_CTR']]

            # keep only text and labels columns
            data = data[['headline_1', 'headline_2', 'label']]

            # remove rows with None values
            data = data.dropna()

            dataset = Dataset.from_pandas(data, preserve_index=False)

            # split into train, test, validation
            split_dataset = dataset.train_test_split(test_size=0.2)
            
            # test if Trainer eval_accuracy is on train or test         split_dataset['test'] = split_dataset['test'].remove_columns("label").add_column("label", [0]*len(split_dataset['test'])).cast(split_dataset['test'].features)

            final_dataset = DatasetDict({
                'train': split_dataset['train'],
                'test': split_dataset['test'],
            })
            return final_dataset
        
        news = load_and_prepare_dataset("winner_005.csv")
    return news



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
