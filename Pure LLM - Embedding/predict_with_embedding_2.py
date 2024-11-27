# load train set and test set
import numpy as np
import pandas as pd
import torch
import progressbar
from common.utils import significance_test_one_news
from model import train_mlp, train_linear
import os

MODEL = 'OpenAI' # OpenAI or Word2Vec256 or Word2Vec3072 or Llama4096 or Bert
DIM = 256 # Only useful for OpenAI, other models dimension are given in MODEL
ALGO = 'Linear' # MLP or Linear

# Effect of Training and Test Data Size on Performance
TRAINING_SIZE_TEST = True

SPLIT_BY_TIME = False # if True, always use Pure LLM - Embedding/saved_embedding/OpenAI_split_by_time, SPLIT_ID is not useful anymore

# use new split of train and test
SPLIT_ID = 0 # if 0, use original split, otherwise use new split


if TRAINING_SIZE_TEST:
    # rewrite the model and algo settings.
    MODEL = 'OpenAI'
    DIM = 256
    ALGO = 'Linear'

if SPLIT_BY_TIME:
    MODEL = 'OpenAI'
    DIM = 256
    ALGO = 'Linear'
    SPLIT_ID = 0 # set to 0 to avoid re-splitting the data


print("Loading data...")
if SPLIT_BY_TIME:
    all_data = np.load(os.path.join("Pure LLM - Embedding/saved_embedding/OpenAI_split_by_time", "all_data.npy"), allow_pickle=True)
    # train = np.load(os.path.join("Pure LLM - Embedding/saved_embedding/OpenAI_split_by_time", "train.npy"), allow_pickle=True)
    # test = np.load(os.path.join("Pure LLM - Embedding/saved_embedding/OpenAI_split_by_time", "test.npy"), allow_pickle=True)
    columns = np.load(os.path.join("Pure LLM - Embedding/saved_embedding/OpenAI_split_by_time", "columns.npy"), allow_pickle=True)
    all_data = pd.DataFrame(all_data, columns=columns)
    # sort by created_at
    all_data = all_data.sort_values(by='created_at')

    # get train news number
    _ = np.load('Pure LLM - Embedding/saved_embedding/OpenAI/test.npy', allow_pickle=True)
    _columns = np.load('Pure LLM - Embedding/saved_embedding/OpenAI/columns.npy', allow_pickle=True)
    _ = pd.DataFrame(_, columns=_columns)
    n_test_article = len(np.unique(_['test_id']))

    # split the data
    all_test_ids = all_data['test_id'].unique()

    start_idx = len(all_test_ids) - n_test_article
    id_for_test = all_test_ids[start_idx:start_idx+n_test_article]
    # id_for_train is the rest of the test ids
    id_for_train = np.concatenate([all_test_ids[0:start_idx], all_test_ids[start_idx+n_test_article:]])

    train = all_data[all_data['test_id'].isin(id_for_train)]
    test = all_data[all_data['test_id'].isin(id_for_test)]

    # assert no overlap
    assert len(set(train['test_id'].values).intersection(set(test['test_id'].values))) == 0
else:
    train = np.load(os.path.join("Pure LLM - Embedding/saved_embedding", MODEL, "train.npy"), allow_pickle=True)
    test = np.load(os.path.join("Pure LLM - Embedding/saved_embedding", MODEL, "test.npy"), allow_pickle=True)
    columns = np.load(os.path.join("Pure LLM - Embedding/saved_embedding", MODEL, "columns.npy"), allow_pickle=True)

if SPLIT_ID != 0:
    # compute original test ratio
    n_test_article = len(np.unique(test[:, np.where(columns=='test_id')[0][0]]))
    # combine train and test
    data = np.concatenate([train, test])
    np.random.seed(SPLIT_ID) # ensure SPLIT_ID corresponds to the same split
    article_ids = np.unique(data[:, np.where(columns=='test_id')[0][0]])
    # randomly select n_test_article articles
    test_article_ids = np.random.choice(article_ids, n_test_article, replace=False)

    # split the data
    test = data[np.isin(data[:, np.where(columns=='test_id')[0][0]], test_article_ids)]
    train = data[~np.isin(data[:, np.where(columns=='test_id')[0][0]], test_article_ids)]

    # fix the following settings for new split
    MODEL = 'OpenAI'
    DIM = 256
    ALGO = 'Linear'


train = pd.DataFrame(train, columns=columns)
test = pd.DataFrame(test, columns=columns)

train = train.sort_values(by='test_id')
test = test.sort_values(by='test_id')

# find the line in train that embedding columns is length 1
nan_idx = train[train['embedding'].apply(len) == 1].index
# get the news_id of the news that has nan embedding
news_id_to_remove = train.loc[nan_idx, 'test_id'].values
# remove the news that has nan embedding from train
train = train[~train['test_id'].isin(news_id_to_remove)]

if MODEL == 'OpenAI':
    train['embedding'] = train['embedding'].apply(lambda x: x[:DIM])
    test['embedding'] = test['embedding'].apply(lambda x: x[:DIM])
else:
    DIM = len(train['embedding'].values[0])


# find significant news in test
sig_test_id = []
print('Finding significant news...')
for news_id in progressbar.progressbar(test['test_id'].unique()):
    news = test[test['test_id'] == news_id]
    impressions = news['impressions'].values.tolist()
    clicks = news['clicks'].values.tolist()
    CTRs = news['CTR'].values.tolist()
    if significance_test_one_news(impressions, clicks, CTRs):
        sig_test_id.append(news_id)
print("Among all {} news, {} news are significant.".format(len(test['test_id'].unique()), len(sig_test_id)))
# SPLIT_BY_TIME True: Among all 3322 news, 329 news are significant.
# SPLIT_BY_TIME False, SPLIT_ID 0: Among all 3263 news, 614 news are significant.



def run_experiment(ALGO, train, test, sig_test_id):
    if ALGO == "Linear":
        # use linear regression to predict the CTR
        print("Training Linear Regression...")
        X_train = np.array(train['embedding'].values.tolist())
        y_train = train['CTR'].values
        X_test = np.array(test['embedding'].values.tolist())
        y_test = test['CTR'].values
        news_id_train = train['test_id'].values.tolist()
        news_id_test = test['test_id'].values.tolist()

        accuracy = train_linear(X_train, y_train, X_test, y_test, news_id_train, news_id_test, sig_test_id)
        return accuracy
    else:
        print("Convert data to tensor...")
        # prepare training and testing data
        X_train = torch.tensor(train['embedding'].values.tolist())
        y_train = torch.tensor(train['CTR'].values.tolist())
        news_id_train = train['test_id'].values.tolist()

        X_test = torch.tensor(test['embedding'].values.tolist())
        y_test = torch.tensor(test['CTR'].values.tolist())
        news_id_test = test['test_id'].values.tolist()

        print("Training MLP...")
        model = train_mlp(X_train, y_train, X_test, y_test, news_id_train, news_id_test, sig_test_id, hidden_size=2048, hidden_layer_num=1, dropout_rate=0.1, rate=4, DIM=DIM, lr=0.01, batch_size=100, epochs=1000)


if TRAINING_SIZE_TEST:
    ratio_ls = np.linspace(0.05, 1, 20)
    acc_all = []
    acc_sig = []
    acc_train = []

    all_train_id = train['test_id'].unique()
    # randomly shuffle the train data
    all_train_id = np.random.permutation(all_train_id)
    for ratio in ratio_ls:
        print(f"Training size ratio: {ratio}")
        train_size = int(len(all_train_id) * ratio)
        # get the subset of the train data
        id_train_sub = all_train_id[:train_size]
        train_sub = train[train['test_id'].isin(id_train_sub)]
        train_sub = train[:train_size]
        acc = run_experiment(ALGO, train_sub, test, sig_test_id)
        acc_all.append(acc['acc_all'])
        acc_sig.append(acc['acc_sig'])
        acc_train.append(acc['acc_train'])

    # Create the plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.plot(ratio_ls, np.array(acc_train) * 100, marker='x', linestyle='-', color='C0', label='Accuracy on Training Set')
    plt.plot(ratio_ls, np.array(acc_all) * 100, marker='o', linestyle='-', color='C1', label='Accuracy on Test Set')
    # plt.plot(ratio_ls, np.array(acc_sig) * 100, marker='o', linestyle='-', color='C2', label='Accuracy on Significant Subset')

    # Add labels and title
    plt.xlabel('Training Set Ratio')
    plt.ylabel('Accuracy (%)')
    # plt.ylim(0, 100)
    plt.xticks(ratio_ls, rotation=45)  # Set the x-ticks to show all values in ratio_ls and rotate them by 45 degrees
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Pure LLM - Embedding/results_proportion_ratio_regression_256.pdf')

else:
    run_experiment(ALGO, train, test, sig_test_id)


##### OpenAI
# DIM = 3072
# ALGO = MLP
# Train Accuracy: 0.3950707070707071, Test Accuracy all: 0.46123199509653695 Test Accuracy sig: 0.754071661237785
# Train Accuracy: 0.3927272727272727, Test Accuracy all: 0.4529574011645725 Test Accuracy sig: 0.7312703583061889
# Train Accuracy: 0.3945858585858586, Test Accuracy all: 0.4410052099295127 Test Accuracy sig: 0.750814332247557

# ALGO = Linear
# Train Accuracy: 0.39385858585858585, Test Accuracy all: 0.430585350904076 Test Accuracy sig: 0.7247557003257329


# DIM = 256
# MLP
# Train Accuracy: 0.38294949494949493, Test Accuracy all: 0.4535703340484217 Test Accuracy sig: 0.739413680781759
# Train Accuracy: 0.3808484848484848, Test Accuracy all: 0.4413116763714373 Test Accuracy sig: 0.7312703583061889
# Train Accuracy: 0.3768888888888889, Test Accuracy all: 0.43640821330064355 Test Accuracy sig: 0.7019543973941368

# Linear
# SPLIT_ID = 0
# Train Accuracy: 0.389979797979798, Test Accuracy all: 0.46276432730615996 Test Accuracy sig: 0.745928338762215
# SPLIT_ID = 1
# Train Accuracy: 0.39975757575757576, Test Accuracy all: 0.40851976708550414 Test Accuracy sig: 0.6086021505376344
# SPLIT_ID = 2
# Train Accuracy: 0.4050909090909091, Test Accuracy all: 0.40851976708550414 Test Accuracy sig: 0.6348195329087049
# SPLIT_ID = 3
# rain Accuracy: 0.4063838383838384, Test Accuracy all: 0.398406374501992 Test Accuracy sig: 0.6227180527383367
# SPLIT_ID = 4
# Train Accuracy: 0.405010101010101, Test Accuracy all: 0.40913269996935336 Test Accuracy sig: 0.6535269709543569
# SPLIT_ID = 5
# Train Accuracy: 0.4024242424242424, Test Accuracy all: 0.40330983757278577 Test Accuracy sig: 0.6263048016701461

if False:
    import matplotlib.pyplot as plt
    plt.figure()
    # plot for split_id 1,2,3,4,5
    train_acc_ls = np.array([0.39975757575757576, 0.4050909090909091, 0.4063838383838384, 0.405010101010101, 0.4024242424242424])
    test_acc_all_ls = np.array([0.40851976708550414, 0.40851976708550414, 0.398406374501992, 0.40913269996935336, 0.40330983757278577])
    test_acc_sig_ls = np.array([0.6086021505376344, 0.6348195329087049, 0.6227180527383367, 0.6535269709543569, 0.6263048016701461])
    plt.plot(range(5), 100 * train_acc_ls, label='Train Accuracy')
    plt.plot(range(5), 100 * test_acc_all_ls, label='Test Accuracy on All Data')
    plt.plot(range(5), 100 * test_acc_sig_ls, label='Test Accuracy on Significant Subset')
    plt.legend()
    plt.xlabel('Split ID')
    plt.ylabel('Accuracy (%)')
    plt.xticks(range(5))
    plt.tight_layout()
    plt.savefig('Pure LLM - Embedding/linear_regression_split_id.pdf')


##### Word2Vec
# DIM = 256
# MLP
# Train Accuracy: 0.290158371040724, Test Accuracy all: 0.3610174685871897 Test Accuracy sig: 0.5765472312703583
# Train Accuracy: 0.30066257272139624, Test Accuracy all: 0.3797119215445909 Test Accuracy sig: 0.5863192182410424
# Train Accuracy: 0.30502585649644476, Test Accuracy all: 0.36806619675145574 Test Accuracy sig: 0.6091205211726385

# Linear
# Train Accuracy: 0.32853910795087266, Test Accuracy all: 0.4106650321789764 Test Accuracy sig: 0.6677524429967426

# DIM = 3072
# MLP
# Train Accuracy: 0.30235940530058175, Test Accuracy all: 0.3781795893349678 Test Accuracy sig: 0.6107491856677525
# Train Accuracy: 0.27319004524886875, Test Accuracy all: 0.3775666564511186 Test Accuracy sig: 0.6042345276872965
# Train Accuracy: 0.2887039431157078, Test Accuracy all: 0.36224333435488815 Test Accuracy sig: 0.6009771986970684

# Linear
# Train Accuracy: 0.32449903038138334, Test Accuracy all: 0.40269690468893654 Test Accuracy sig: 0.6547231270358306


##### Llama4096
# MLP
# Train Accuracy: 0.2748868778280543, Test Accuracy all: 0.3650015323322096 Test Accuracy sig: 0.6009771986970684
# Train Accuracy: 0.2974305106658048, Test Accuracy all: 0.35856573705179284 Test Accuracy sig: 0.5553745928338762
# Train Accuracy: 0.2979961215255333, Test Accuracy all: 0.3573398712840944 Test Accuracy sig: 0.5553745928338762
# Linear
# Train Accuracy: 0.3860698125404008, Test Accuracy all: 0.4278271529267545 Test Accuracy sig: 0.7068403908794788


# Bert
# Linear
# Train Accuracy: 0.35310277957336783, Test Accuracy all: 0.4232301562978854 Test Accuracy sig: 0.6661237785016286
# MLP
# Train Accuracy: 0.2748868778280543, Test Accuracy all: 0.3337419552558995 Test Accuracy sig: 0.5765472312703583
# Train Accuracy: 0.3165804783451842, Test Accuracy all: 0.3935029114311983 Test Accuracy sig: 0.6254071661237784
# Train Accuracy: 0.2793309631544926, Test Accuracy all: 0.34937174379405456 Test Accuracy sig: 0.5912052117263844
