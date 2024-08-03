# This code demonstrates classification using embeddings with logits and MLP. To run it, follow these steps:
# 
# 1. Run the first two sections: Initialization and Data Preparation, to obtain and preprocess the data.
# 2. Depending on your requirements, run the corresponding sections.
# 
# Note that we provide the embeddings from OpenAI stored in CSV, the corresponding code to return OpenAI embedding is provided in Appendix.

# # Initialization

from llama_embedding import get_embedding

import pandas as pd
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import time, datetime
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, train_test_split, cross_val_score
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader, TensorDataset
import pickle
from joblib import dump, load
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.stats import fisher_exact
import copy
import random
import scipy.stats as stats
import ast
import copy

import matplotlib.font_manager as fm
# import seaborn as sns

_font_size = 38

class Net_test(nn.Module):
    def __init__(self, hidden_size=128, hidden_layer_num=1, dropout_rate=0.5, rate=2, DIM=256):
        super(Net_test, self).__init__()
        self.rate = rate
        self.hidden_layer_num = hidden_layer_num
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(DIM*2, self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate) 
        num_max = int(self.hidden_size/self.rate)           
        self.fc_hidden = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_size, self.get_hidden_size()),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ) for _ in range(min(self.hidden_layer_num - 1, num_max-1))
        ])
        self.fc2 = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
   
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        for fc in self.fc_hidden:
            x = fc(x)
        x = self.fc2(x)
        return self.sigmoid(x)
    
    def get_hidden_size(self):
        self.hidden_size = int(self.hidden_size/self.rate)
        return self.hidden_size

def loss_computation(net_, testloader, loss_history_test):
    criterion = nn.BCELoss()
    with torch.no_grad():
        running_loss_test = 0
        for data in testloader:
            inputs, labels = data  
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            outputs = net_(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))      
            running_loss_test += loss.item()
        loss_history_test.append(running_loss_test / len(testloader))
    return loss_history_test
 

        
def acc_computation(net_, X_test, y_test):
    with torch.no_grad(): 
        outputs = net_(X_test.cuda())
        predicted = torch.round(outputs.cuda())
        accuracy_ = (predicted.squeeze().cuda() == y_test.cuda()).sum().item() / y_test.cuda().size(0)
    return accuracy_
        
        
def prediction_w_NN(X_train, y_train, X_test, y_test, DIM=256, proportion=0.2, lr=0.01, batch_size=10, EPOCH=1000, early_stopping_rounds=None, tol=None, hidden_size=128, hidden_layer_num=1, dropout_rate=0.5, weight_decay = 0.01, rate=2, fin=False, optim_name='sgd', loss_history_val=None):    
    net = Net_test(hidden_size=hidden_size, hidden_layer_num=hidden_layer_num, dropout_rate=dropout_rate, rate=rate, DIM=DIM)
    if torch.cuda.is_available():
        net = net.cuda()
    
    criterion = nn.BCELoss()
    if optim_name == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(net.parameters(), lr=lr, weight_decay=weight_decay)
    
    trainset = TensorDataset(X_train, y_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = TensorDataset(X_test, y_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    loss_history_train = []
    loss_history_test = []
    acc_history_train = []
    acc_history_test = []
    best_loss = float('inf')  
    early_stopping_counter = 0 
    print(f'Start Training Neural Network_hidden_size_{proportion}:hidden_size_{hidden_size}_hidden_layer_num:{hidden_layer_num}_dropout_rate:{dropout_rate}_weight_decay:{weight_decay}')
    
    start_time = time.time()
    epoch_1 = 0
    for epoch in range(EPOCH):
        epoch_1 = epoch
        acc_samples = 0
        running_loss = 0.0
        net.train()
        for index, data in enumerate(trainloader, 0):
            inputs, labels = data  
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            optimizer.zero_grad()
            outputs = net(inputs)  # Slowest
            predicted = torch.round(outputs)  
            acc_samples += (predicted.squeeze() == labels).sum().item()
            
            loss = criterion(outputs, labels.unsqueeze(1))      
            loss.backward()     
            optimizer.step()                 
            running_loss += loss.item()
        if epoch % (EPOCH // 10) == 0:
            eta = (time.time() - start_time) / (epoch + 1) * (EPOCH - epoch - 1)
            print('[%d] loss: %.3f, eta: %s' % (epoch + 1, running_loss / (index + 1), str(datetime.timedelta(seconds=eta))))
        loss_history_train.append(running_loss / len(trainloader))  
        acc_history_train.append(acc_samples/y_train.size(0))
    
        loss_history_test = loss_computation(net, testloader, loss_history_test)
        net.eval()
        acc_history_test.append(acc_computation(net, X_test=X_test, y_test=y_test))
        #print(acc_computation(net, X_test=X_test, y_test=y_test))
   
        if early_stopping_rounds is not None:  
            if tol is None:
                tol = np.finfo(np.float32).eps
            if best_loss - loss_history_val[-1] > tol: 
                best_loss = loss_history_val[-1]
                early_stopping_counter = 0
                if fin and epoch >= 10:
                    torch.save(net.state_dict(), f'Best_models_for_hyperparameter_search_proportion_{proportion}/NN_{hidden_size}_hidden_layer_num_{hidden_layer_num}_dropout_rate_{dropout_rate}_weight_decay_{weight_decay}_rate_{rate}_fin_{fin}.pth')
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_rounds:
                    print("Best val_loss at epochs:", epoch-early_stopping_rounds+1)
                    break
        
    if early_stopping_rounds is None:
        early_stopping_rounds = 0
    accuracy_train = acc_history_train[epoch_1-early_stopping_rounds]
    accuracy_test = acc_history_test[epoch_1-early_stopping_rounds]
    
    print('train_accuracy:', accuracy_train)
    print('test_accuracy:', accuracy_test)
    print('Finished Training')
    print('Total period: ',time.time()-start_time)
    
    return net, loss_history_train, loss_history_test, acc_history_train, acc_history_test, accuracy_train, accuracy_test


# # Data Preparation
# Generate and save the embeddings needed for the following codes

def embed_generate_save():
    # if embedding/embed_1 and embedding/embed_2 are already saved, you can load them directly
    if os.path.exists('embedding/embed_1') and os.path.exists('embedding/embed_2'):
        print('Loading pre-computed embedding')
        embed_1 = torch.load('embedding/embed_1')
        embed_2 = torch.load('embedding/embed_2')
    else:
        print('get data from csv')
        df = pd.read_csv(f'data/selected_pairs_df_005_256.csv')
        embed_1 = get_embedding(df['headline_1'].tolist())
        embed_2 = get_embedding(df['headline_2'].tolist())
        if not os.path.exists('embedding'):
            os.mkdir('embedding')
        torch.save(embed_1, 'embedding/embed_1')
        torch.save(embed_2, 'embedding/embed_2')

    return embed_1, embed_2, embed_1.shape[1]
    # embed_1 = torch.tensor(np.array([eval(s) for s in df['embedding_1']])).to(torch.float32)
    # embed_2 = torch.tensor(np.array([eval(s) for s in df['embedding_2']])).to(torch.float32)
    
    # directory_path = 'embedding/'
    # embed_1_path = f'{directory_path}embed_1_{DIM}.pt'
    # embed_2_path = f'{directory_path}embed_2_{DIM}.pt'
    # torch.save(embed_1, embed_1_path)
    # torch.save(embed_2, embed_2_path)


# embed_1, embed_2 = embed_generate_save()

directory_name = 'stored_embeddings_significant'
if not os.path.exists(directory_name):
    os.mkdir(directory_name)
print(f"Directory '{directory_name}' is ready.")

# DIM = 256
print('get data from csv')
file_path = 'data/selected_pairs_df_005_256.csv'
df_embedding_256 = pd.read_csv(file_path)
df_embedding_256['test_id'] = df_embedding_256.groupby(['clickability_test_id', 'eyecatcher_id']).ngroup() + 1

# remove embedding_2 and embedding_1 columns
df_embedding_256 = df_embedding_256.drop(columns=['embedding_1', 'embedding_2'])

# DIM = 3072
# print('get data from csv')
# file_path = 'stored_embeddings_significant/selected_pairs_df_005_' + str(DIM) + '.csv'
# df_embedding_3072 = pd.read_csv(file_path)
# df_embedding_3072['test_id'] = df_embedding_3072.groupby(['clickability_test_id', 'eyecatcher_id']).ngroup() + 1


# Define directory path and file names for saving individual components with variables included
def embed_get(DIM, directory_path='stored_embeddings_significant/train_val_test_split/'):
    embed_1_path = f'{directory_path}embed_1_{DIM}.pt'
    embed_2_path = f'{directory_path}embed_2_{DIM}.pt'
    embed_1 = torch.load(embed_1_path)
    embed_2 = torch.load(embed_2_path)
    return embed_1, embed_2


def dataset_extraction(df, test_size, rs = 42):
    random_state = rs
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(df, groups=df['test_id']))
    train_df = df.iloc[train_idx]
    test_df_original = df.iloc[test_idx]

    train_headlines = set(train_df['headline_1'].tolist() + train_df['headline_2'].tolist())
    test_df = test_df_original[~test_df_original['headline_1'].isin(train_headlines)]
    test_df = test_df[~test_df['headline_2'].isin(train_headlines)]
    print(f'Filtered Num Compared to the Begining_test_size_{test_size}', (len(test_df_original)-len(test_df)))
    print(f'Remaining samples of the test set:', len(test_df))
    print(f'Filter Ratio Compared to the Begining_test_size_{test_size}', (len(test_df_original)-len(test_df))/len(test_df_original))

    # embed_1, embed_2 = embed_get(DIM=DIM)
    embed_1, embed_2, DIM = embed_generate_save()
    X_train = torch.cat([embed_1[train_df.index], embed_2[train_df.index]], dim=1)
    y_train = torch.tensor(train_df['higher_CTR'].values).to(torch.float32) - 1
    X_test = torch.cat([embed_1[test_df.index], embed_2[test_df.index]], dim=1)
    y_test = torch.tensor(test_df['higher_CTR'].values).to(torch.float32) - 1
    
    return X_train, y_train, X_test, y_test, DIM


# %%
df = df_embedding_256
X_train, y_train, X_test, y_test, DIM = dataset_extraction(df=df, test_size=0.2)
# regression
print('Run Logistic Regression')
clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
print('Simple Linear Model Performance + gpt embedding')
accur = clf.score(X_train, y_train)
total = len(X_train)
print('Accuracy on training data:', accur)


accur = clf.score(X_test, y_test)
total = len(X_test)
print('Accuracy on test data:', accur)






# %%
lr=0.01
batch_size=10
EPOCH=1000

# %%
weight_decay = 0.0001
hidden_size=256
hidden_layer_num=1
dropout_rate=0.5
rate=2

best_epochs = 74  # # The optimal epochs are 32

# %%
torch.manual_seed(0)
net_fin, loss_history_train, loss_history_test, acc_history_train, acc_history_test, accuracy_train, accuracy_test = prediction_w_NN(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, DIM=DIM, hidden_layer_num=hidden_layer_num, dropout_rate=dropout_rate, hidden_size=hidden_size, lr=lr, batch_size=batch_size, EPOCH=best_epochs, early_stopping_rounds=None, tol=0.0001, weight_decay=weight_decay, rate=rate, fin=True)
