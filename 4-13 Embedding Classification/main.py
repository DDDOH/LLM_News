'''
In this file, we will use the embeddings from OpenAI to train a simple linear model and a simple neural network to predict the winner of the headline pair.

We get the embeddings from OpenAI first. To save API calls, we will save the embeddings to a file and load them from the file if the file exists.

Then we will train a simple linear model and a simple neural network to predict the winner of the headline pair.

We temporarily use the first 1000 rows of the dataset for a illustration purpose. The performance of simple linear model on test set is around 0.62, and the performance of simple neural network is around 0.66. The performance is nice for a simple model, but let's see the performance on the whole dataset and try to improve the performance by tuning the parameters.

Things to do next:
- Use the whole dataset. You might need some parallel processing to speed up the embedding generation process. Try to search "embedding model parallel", "langchain parallel", "pandas apply parallel", etc. It's already cost quite some time to generate embeddings for 1000 rows.
- Try different parameters, for example, DIM (embedding dimension), MODEL (which model to use), EPOCH (number of epochs for training the neural network), learning rate, batch size, etc.
'''
import pandas as pd
from openai import OpenAI
import torch
import numpy as np
import os
# import progressbar # pip install progressbar2
import time, datetime


# which model to use: https://platform.openai.com/docs/guides/embeddings/embedding-models
# text-embedding-3-large is the one with best performance on MTEB eval, so let's try this model first
MODEL = "text-embedding-3-large"
# use smaller embedding dimension
DIM = 256
# try first 1000 rows first
N_ROWS = 1000




# try to load from cache
file_name = os.path.join('4-13 Embedding Classification/stored_embeddings', MODEL, 'embeddings_N_ROW_{}_DIM_{}.pkl'.format(DIM, N_ROWS))
if os.path.exists(file_name):
    print('loading embeddings from cache')
    df = pd.read_pickle(file_name)
else:
    print('get embeddings from OpenAI')
    # load paired-winner-datasets/winner-all.csv with pandas
    df = pd.read_csv('paired-winner-datasets/winner-all.csv')

    client = OpenAI(api_key='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') # this is api key for you to use, please do not abuse and share it.

    text = df.iloc[0]['headline_1']
    print('default embedding dimension:', len(client.embeddings.create(input = [text], model=MODEL).data[0].embedding))

    df = df.iloc[:N_ROWS]
    print('getting embeddings for headline_1')
    df['embedding_1'] = df['headline_1'].apply(lambda x: client.embeddings.create(input = [x], model=MODEL, dimensions=DIM).data[0].embedding)
    print('getting embeddings for headline_2')
    df['embedding_2'] = df['headline_2'].apply(lambda x: client.embeddings.create(input = [x], model=MODEL, dimensions=DIM).data[0].embedding)



    print('save embeddings')
    # save the embeddings, if folder does not exist, create it
    if not os.path.exists(os.path.join('stored_embeddings', MODEL)):
        os.makedirs(os.path.join('stored_embeddings', MODEL))
    df.to_pickle(file_name)


# predict the winner
embed_1 = torch.tensor(np.array(df['embedding_1'].tolist())).to(torch.float32)
embed_2 = torch.tensor(np.array(df['embedding_2'].tolist())).to(torch.float32)
target = torch.tensor(df['higher_CTR'].values).to(torch.float32) - 1 # 0 for headline_1, 1 for headline_2

# split the data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(torch.cat([embed_1, embed_2], dim=1), target, test_size=0.2, random_state=42)


# train a simple linear model
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
print('Simple Linear Model Performance')
print('Accuracy on training data:', clf.score(X_train, y_train))
print('Accuracy on test data:', clf.score(X_test, y_test))



# train a simple neural network
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Net(nn.Module):
      def __init__(self):
         super(Net, self).__init__()
         self.fc1 = nn.Linear(DIM*2, 128)
         self.fc2 = nn.Linear(128, 1)
         self.sigmoid = nn.Sigmoid()
   
      def forward(self, x):
         x = torch.relu(self.fc1(x))
         x = self.fc2(x)
         return self.sigmoid(x)
      
net = Net()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

trainset = TensorDataset(X_train, y_train)
trainloader = DataLoader(trainset, batch_size=10, shuffle=True)

EPOCH = 1000
# for epoch in progressbar.progressbar(range(EPOCH)):
print('Start Training Neural Network')
start_time = time.time()
for epoch in range(EPOCH):
      running_loss = 0.0
      for i, data in enumerate(trainloader, 0):
         inputs, labels = data
         optimizer.zero_grad()
         outputs = net(inputs)
         loss = criterion(outputs, labels.unsqueeze(1))
         loss.backward()
         optimizer.step()
         running_loss += loss.item()
      if epoch % (EPOCH // 10) == 0:
         eta = (time.time() - start_time) / (epoch + 1) * (EPOCH - epoch - 1)
         print('[%d] loss: %.3f, eta: %s' % (epoch + 1, running_loss / (i + 1), str(datetime.timedelta(seconds=eta))))
      #    print('[%d] loss: %.3f' % (epoch + 1, running_loss / (i + 1)))
      # print loss and eta formatted
            

print('Finished Training')

print('Simple Neural Network Performance')
with torch.no_grad():
      outputs = net(X_test)
      predicted = torch.round(outputs)
      total = y_test.size(0)
      correct = (predicted.squeeze() == y_test).sum().item()
print('Accuracy on test data:', correct / total)
with torch.no_grad():
      outputs = net(X_train)
      predicted = torch.round(outputs)
      total = y_train.size(0)
      correct = (predicted.squeeze() == y_train).sum().item()
print('Accuracy on training data:', correct / total)
    



