import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import progressbar

class MLP(nn.Module):
    def __init__(self, hidden_size=128, hidden_layer_num=1, dropout_rate=0.5, rate=2, DIM=256):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.hidden_layer_num = hidden_layer_num
        self.dropout_rate = dropout_rate
        self.DIM = DIM
        self.fc1 = nn.Linear(DIM, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(hidden_layer_num)])
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout(x)
        for i in range(self.hidden_layer_num):
            x = self.hidden_layers[i](x)
            x = nn.functional.relu(x)
            x = self.dropout(x)
        x = self.fc2(x)
        return x
    

# train MLP to predict the CTR
def train_mlp(X_train, y_train, X_test, y_test, X_cal, y_cal, news_id_train, news_id_test, news_id_cal, sig_test_id, hidden_size=128, hidden_layer_num=1, dropout_rate=0.5, rate=2, DIM=256, lr=0.01, batch_size=10, epochs=1000):
    model = MLP(hidden_size=hidden_size, hidden_layer_num=hidden_layer_num, dropout_rate=dropout_rate, rate=rate, DIM=DIM)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    trainset = TensorDataset(X_train, y_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = TensorDataset(X_test, y_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            break

        if epoch % 500 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/batch_size}')
            with torch.no_grad():
                y_pred_train = model(X_train).squeeze().detach().numpy()
                y_pred_test = model(X_test).squeeze().detach().numpy()
                acc_train = evaluate_select_acc(y_pred_train, y_train, news_id_train, sig_test_id, eval_sig=False)
                acc_test, sig_test_acc = evaluate_select_acc(y_pred_test, y_test, news_id_test, sig_test_id, eval_sig=True)
                print(f'Train Accuracy: {acc_train}, Test Accuracy all: {acc_test}', f'Test Accuracy sig: {sig_test_acc}')
                
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for inputs, labels in testloader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            test_loss += loss.item()
        print(f'Test Loss: {test_loss/len(testloader)}')
        with torch.no_grad():
            y_pred_train = model(X_train).squeeze().detach().numpy()
            y_pred_test = model(X_test).squeeze().detach().numpy()
            y_pred_cal = model(X_cal).squeeze().detach().numpy()
            acc_train = evaluate_select_acc(y_pred_train, y_train, news_id_train, sig_test_id, eval_sig=False)
            acc_test, sig_test_acc = evaluate_select_acc(y_pred_test, y_test, news_id_test, sig_test_id, eval_sig=True)
            acc_cal = evaluate_select_acc(y_pred_cal, y_cal, news_id_cal, sig_test_id, eval_sig=False)
            print(f'Train Accuracy: {acc_train}, Test Accuracy all: {acc_test}',
                  f'Test Accuracy sig: {sig_test_acc}',
                  f'Calibration Accuracy: {acc_cal}')
        
    
    return model



def evaluate_select_acc(y_pred, y_true, news_id, sig_test_id, eval_sig=False):
    n_correct = 0
    # assert news_id is in non-decreasing order
    assert all(news_id[i] <= news_id[i+1] for i in range(len(news_id)-1))
    news_id_uni = np.sort(np.unique(news_id))
    start = 0
    if eval_sig:
        n_sig_correct = 0

    for id in progressbar.progressbar(news_id_uni):
        # find the last index of the same news_id
        # end = np.where(news_id == id)[0][-1] + 1
        end = np.searchsorted(news_id, id, side='right')
        best_news_pred = np.argmax(y_pred[start:end])
        best_news_true = np.argmax(y_true[start:end])
        if best_news_pred == best_news_true:
            n_correct += 1
            if eval_sig and id in sig_test_id:
                n_sig_correct += 1
        start = end
    accuracy = n_correct / len(news_id_uni)
    if eval_sig:
        sig_accuracy = n_sig_correct / len(sig_test_id)
        return accuracy, sig_accuracy
    else:
        return accuracy



def train_linear(X_train, y_train, X_test, y_test, X_cal, y_cal, news_id_train, news_id_test, news_id_cal, sig_test_id):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_cal = model.predict(X_cal)
    acc_train = evaluate_select_acc(y_pred_train, y_train, news_id_train, sig_test_id, eval_sig=False)
    acc_test, sig_test_acc = evaluate_select_acc(y_pred_test, y_test, news_id_test, sig_test_id, eval_sig=True)
    acc_cal = evaluate_select_acc(y_pred_cal, y_cal, news_id_cal, sig_test_id, eval_sig=False)
    print(f'Train Accuracy: {acc_train}, Test Accuracy all: {acc_test}',
          f'Test Accuracy sig: {sig_test_acc}',
          f'Calibration Accuracy: {acc_cal}')
    return {'acc_all': acc_test, 'acc_sig': sig_test_acc, 'acc_train': acc_train, 'acc_cal': acc_cal, 'model': model}