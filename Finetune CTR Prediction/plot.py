# plot better looking figure using saved results
import numpy as np
import pickle
import matplotlib.pyplot as plt
from common.utils import significance_test_one_news

train_loss_rec_ls = []
test_loss_rec_ls = []
train_acc_rec_ls = []
test_acc_rec_ls = []
test_acc_rec_sig_ls = []
epoch_rec_ls = []
global_step_rec_ls = []


n_epoch = 5
pkl = 'Finetune CTR Prediction/saved_results/Model_llama3 Lora_True Split_id_0.pkl'

with open(pkl, 'rb') as f:
    data = pickle.load(f)
train_loss_rec_ls.append(data['train_loss_rec'][:n_epoch])
test_loss_rec_ls.append(data['test_loss_rec'][:n_epoch])
train_acc_rec_ls.append(data['train_accuracy_rec'][:n_epoch])
test_acc_rec_ls.append(data['test_accuracy_rec_all'][:n_epoch])
test_acc_rec_sig_ls.append(data['test_accuracy_rec_significant'][:n_epoch])
epoch_rec_ls.append(data['epoch_rec'][:n_epoch])
global_step_rec_ls.append(data['global_step_rec'][:n_epoch])

def get_CI(rec_ls):
    rec_ls = np.array(rec_ls)
    mean = np.mean(rec_ls, axis=0)
    std = np.std(rec_ls, axis=0)
    ub = mean + 1.96 * std / np.sqrt(len(rec_ls))
    lb = mean - 1.96 * std / np.sqrt(len(rec_ls))
    return ub, lb, mean

plt.figure(figsize=(8, 6))
plt.subplot(2, 1, 1)
x_max = max(epoch_rec_ls[0])
ub, lb, mean = get_CI(train_loss_rec_ls)
# plt.fill_between(epoch_rec_ls[0], ub, lb, alpha=0.5, label='Train Loss')
plt.plot(epoch_rec_ls[0], mean, label='Training Loss')

ub, lb, mean = get_CI(test_loss_rec_ls)
# plt.fill_between(epoch_rec_ls[0], ub, lb, alpha=0.5, label='Test Loss')
# plt.semilogy(epoch_rec_ls[0], mean, label='Test Loss')
plt.plot(epoch_rec_ls[0], mean, label='Test Loss')
plt.xticks(np.arange(0, x_max + 1, 1))

plt.ylabel('Mean Squared Error')
plt.legend()

plt.subplot(2, 1, 2)
ub, lb, mean = get_CI(train_acc_rec_ls)
# plt.fill_between(epoch_rec_ls[0], ub * 100, lb * 100, alpha=0.5, label='Train Accuracy')
plt.plot(epoch_rec_ls[0], mean * 100, label='Training Accuracy')
plt.text(epoch_rec_ls[0][-1], mean[-1] * 100, f'{mean[-1] * 100:.2f}%', ha='right', va='bottom')

ub, lb, mean = get_CI(test_acc_rec_ls)
# plt.fill_between(epoch_rec_ls[0], ub * 100, lb * 100, alpha=0.5, label='Test Accuracy')
plt.plot(epoch_rec_ls[0], mean * 100, label='Test Accuracy')
plt.text(epoch_rec_ls[0][-1], mean[-1] * 100, f'{mean[-1] * 100:.2f}%', ha='right', va='bottom')


# ub, lb, mean = get_CI(test_acc_rec_sig_ls)
# # plt.fill_between(epoch_rec_ls[0], ub * 100, lb * 100, alpha=0.5, label='Test Accuracy (Significant News)')
# plt.plot(epoch_rec_ls[0], mean * 100, label='Test Accuracy on Significant Subset')

# text the test accuracy at the last epoch
# plt.text(epoch_rec_ls[0][-1], ub[-1] * 100, f'[{lb[-1] * 100:.2f}%, {ub[-1] * 100:.2f}%]', ha='right', va='bottom')
plt.text(epoch_rec_ls[0][-1], mean[-1] * 100, f'{mean[-1] * 100:.2f}%', ha='right', va='bottom')

# get x-axis range using plt
x_min = min(epoch_rec_ls[0])
x_max = max(epoch_rec_ls[0])

# plt.ylim(75, 95)

# plt.hlines(39.08, x_min, x_max, colors='C0', linestyles='dashed', label='OpenAI Embedding Training Accuracy (39.08%)')
# plt.hlines(46.28, x_min, x_max, colors='C1', linestyles='dashed', label='OpenAI Embedding Test Accuracy (46.28%)')
# plt.hlines(33.02, x_min, x_max, colors='g', linestyles='dashed', label='Random Guess (33.02%)')
# draw horizontal grid lines
plt.grid(axis='y')

# legend at right bottom
plt.legend(loc='upper left')

plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
# xtick on only integer values
plt.xticks(np.arange(0, x_max + 1, 1))
plt.tight_layout()
plt.savefig('Finetune CTR Prediction/finetune_result.pdf')


# compute accuracy on significant subset

import pandas as pd
ori_test = pd.read_csv('Code Dataset/LoRA_CTR_test.csv')
predicted_result = pd.read_csv('Finetune CTR Prediction/saved_results/result.csv')
assert len(ori_test) == len(predicted_result)

unique_test_ids = predicted_result['test_id'].unique()
correct_ls = []
for test_id in unique_test_ids:
    df = predicted_result[predicted_result['test_id'] == test_id]
    correct_ls.append(np.argmax(df['real_CTR'].values.tolist()) == np.argmax(df['predictions'].values.tolist()))

ratio = 0.1
# randomly select ratio of the correct_ls, compute the accuracy
n_rep = 5000
acc_ls = []
for i in range(n_rep):
    selected = np.random.choice(correct_ls, int(ratio * len(correct_ls)), replace=False)
    acc_ls.append(np.mean(selected))

alpha = 0.95
ub = np.quantile(acc_ls, 1 - (1 - alpha) / 2)
lb = np.quantile(acc_ls, (1 - alpha) / 2)
print(f'Accuracy: [{lb * 100:.2f}%, {ub * 100:.2f}%]')
# Accuracy: [41.10%, 51.23%]

mean, lb, ub = get_CI(acc_ls)
print(f'Accuracy: [{lb * 100:.2f}%, {ub * 100:.2f}%]')
