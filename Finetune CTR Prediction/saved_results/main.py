# allow DISABLE LORA
# modified from https://huggingface.co/docs/transformers/en/tasks/sequence_classification

# pip install datasets
# pip install peft
# pip install evaluate
# pip install transformers -U
# pip install -U scikit-learn
# pip install -U matplotlib
# pip install progressbar2

# pip install -U "huggingface_hub[cli]"
# echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
# huggingface-cli login
# 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

# change HF_HOME variable to other directory
# export HF_HOME=/root/autodl-tmp/huggingface_hub

# we always use both significant and insignificant data for training
# but we report test performance on 1. all data 2. significant data

# TODO
# evaluate on both all test and significant data

from datasets import load_dataset
import pandas as pd

import torch
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from peft import (
        get_peft_model, 
        prepare_model_for_kbit_training, 
        LoraConfig
    )
from transformers import TrainerCallback
import time
import pickle
import os
import shutil
import evaluate
import numpy as np
import matplotlib.pyplot as plt
import argparse

import utils



DEVICE = 'cuda:0' # 'cuda:0' or 'cuda:1'
LORA = True
DEBUG = False # if debug, use only few data

MODEL = 'llama3' # bert or llama3
# bert for distilbert-base-uncased
# llama for meta-llama/Meta-Llama-3-8B


result_dir = '/root/autodl-tmp/tmp_results'
time_str = time.strftime("%Y%m%d-%H%M%S")
result_dir = result_dir + '/' + time_str
finetuned_model_dir = os.path.join(result_dir, 'finetuned_model')
print('result_dir', result_dir)

# create folder
os.makedirs(result_dir)
os.makedirs(finetuned_model_dir)

# copy current python file to the result folder
shutil.copy('Finetune CTR Prediction/main.py', result_dir)
shutil.copy('Finetune CTR Prediction/utils.py', result_dir)
# shutil.copy('README.md', result_dir)

# split_id: if 0, use the original split. If 1 or 2 or 3..., check if the split exists, if not create a new split
parser = argparse.ArgumentParser(description="Fine-tune CTR Prediction Model")
parser.add_argument("--split_id", type=int, default=0, help="Split ID for data loading")
args = parser.parse_args()

split_id = args.split_id
data, rescale_func, significant_test_ids = utils.load_data(DEBUG, split_id)

log_file_path = 'Model_{} Lora_{} Split_id_{}.pkl'.format(MODEL, LORA, split_id)
log_file_path = os.path.join(result_dir, log_file_path)

# ## Preprocess
# The next step is to load a DistilBERT tokenizer to preprocess the `text` field:
from transformers import AutoTokenizer
if MODEL == 'bert':
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
elif MODEL == 'llama3':
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    tokenizer.pad_token = tokenizer.eos_token


# Create a preprocessing function to tokenize `text` and truncate sequences to be no longer than DistilBERT's maximum input length:
def preprocess_function(examples):
    return tokenizer(examples["headline"], truncation=True)

encoded_dict = preprocess_function(data['train'][0])
print(encoded_dict)
print(tokenizer.decode(encoded_dict['input_ids']))

# To apply the preprocessing function over the entire dataset, use 🤗 Datasets [map](https://huggingface.co/docs/datasets/main/en/package_reference/main_classes#datasets.Dataset.map) function. You can speed up `map` by setting `batched=True` to process multiple elements of the dataset at once:

# if DATA == 'imdb':
#     tokenized_imdb = data.map(preprocess_function, batched=True)
# elif DATA == 'news':
tokenized_news = data.map(preprocess_function, batched=True)

# Now create a batch of examples using [DataCollatorWithPadding](https://huggingface.co/docs/transformers/main/en/main_classes/data_collator#transformers.DataCollatorWithPadding). It's more efficient to *dynamically pad* the sentences to the longest length in a batch during collation, instead of padding the whole dataset to the maximum length.
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# ## Evaluate
# Including a metric during training is often helpful for evaluating your model's performance. You can quickly load a evaluation method with the 🤗 [Evaluate](https://huggingface.co/docs/evaluate/index) library. For this task, load the [accuracy](https://huggingface.co/spaces/evaluate-metric/accuracy) metric (see the 🤗 Evaluate [quick tour](https://huggingface.co/docs/evaluate/a_quick_tour) to learn more about how to load and compute a metric):


# accuracy = evaluate.load("accuracy")

# Then create a function that passes your predictions and labels to [compute](https://huggingface.co/docs/evaluate/main/en/package_reference/main_classes#evaluate.EvaluationModule.compute) to calculate the accuracy:



# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     predictions = np.argmax(predictions, axis=1)
#     return accuracy.compute(predictions=predictions, references=labels)

# Your `compute_metrics` function is ready to go now, and you'll return to it when you setup your training.

# ## Train
# <Tip>
# 
# If you aren't familiar with finetuning a model with the [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer), take a look at the basic tutorial [here](https://huggingface.co/docs/transformers/main/en/tasks/../training#train-with-pytorch-trainer)!
# 
# </Tip>
# 
# You're ready to start training your model now! Load DistilBERT with [AutoModelForSequenceClassification](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoModelForSequenceClassification) along with the number of expected labels, and the label mappings:


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer



if MODEL == 'bert':
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=1, # id2label=id2label, label2id=label2id
    )
elif MODEL == 'llama3':
    model = AutoModelForSequenceClassification.from_pretrained(
        "meta-llama/Meta-Llama-3-8B", num_labels=1,#  id2label=id2label, label2id=label2id,
        device_map = DEVICE,
        from_pretrained=False
    )
    model.config.pad_token_id = model.config.eos_token_id


    if LORA:
        # the config for two pkls in 24-5-17 folder:
        # config = LoraConfig(
        #     r=16, # 32 oob
        #     lora_alpha=32, # 64 oob
        #     target_modules=["q_proj", "v_proj"],
        #     lora_dropout=0.05,
        #     bias="none",
        #     task_type="CAUSAL_LM"
        # )
        config = LoraConfig(
            r=4, # 32 oob
            lora_alpha=4, # 64 oob
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.3,
            bias="none",
            task_type="CAUSAL_LM"
        )

        model = get_peft_model(model, config)

        model.print_trainable_parameters()
    else:
        print('SKIP LORA')


# # run the model on a few examples to see the predictions before training
# if DATA == 'imdb':
#     outputs = model(**tokenized_imdb["train"][:2])
# elif DATA == 'news':
#     outputs = model(**tokenized_news["train"][:2])
#     # model(**preprocess_function(news['train'][:3]))



# print output before training, set to inference mode
model.eval()
inputs = tokenizer("Your input text goes here", "Your input text goes here", return_tensors="pt", padding=True, truncation=True)

# get device of the model
device = model.device
inputs = {key: value.to(device) for key, value in inputs.items()}

outputs = model(**inputs)

# Extract the logits or classification result
# logits = outputs.logits
# predictions = torch.argmax(logits, dim=-1)

print(f"Predicted value: {outputs.logits}")
# print(f"Predicted class: {predictions.item()}")


# At this point, only three steps remain:
# 
# 1. Define your training hyperparameters in [TrainingArguments](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments). The only required parameter is `output_dir` which specifies where to save your model. You'll push this model to the Hub by setting `push_to_hub=True` (you need to be signed in to Hugging Face to upload your model). At the end of each epoch, the [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) will evaluate the accuracy and save the training checkpoint.
# 2. Pass the training arguments to [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) along with the model, dataset, tokenizer, data collator, and `compute_metrics` function.
# 3. Call [train()](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.train) to finetune your model.


if MODEL == 'bert':
    training_args = TrainingArguments(
        output_dir=finetuned_model_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=3,
        # label_names=['CTR'],
    )
elif MODEL == 'llama3':
    training_args = TrainingArguments(
        output_dir=finetuned_model_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=16, # 不加lora的话，调到1了还是会显存爆炸
        per_device_eval_batch_size=16,
        auto_find_batch_size=False,
        fp16=True, # speed up significantly
        num_train_epochs=10,
        weight_decay=0.05,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        push_to_hub=False,
        save_total_limit=3,
        # label_names=['CTR'],
    )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_news["train"],
    eval_dataset=tokenized_news["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    # compute_metrics=compute_metrics,
)

# print trainer learning rate schedule
print(trainer.lr_scheduler)

self_logger = {'train_loss_rec': [],
    'test_loss_rec': [],
    'train_accuracy_rec': [],
    'test_accuracy_rec_all': [],
    'test_accuracy_rec_significant': [],
    'time_stamp_rec': [],
    'epoch_rec': [],
    'global_step_rec': []}

def evaluate(tokenized, test_id, real_CTR, part_test_id=None):
    # tokenized: Dataset
    # test_id: list of test_id, in non-decreasing order. For example, two articles with test_id 1 and 20. The first one has five headlines, the second one has three headlines. Then test_id will be [1, 1, 1, 1, 1, 20, 20, 20]
    # real_CTR: list of real CTR for each headlines. The order is the same as test_id.

    # part_test_id: if not None, also report the accuracy on the subset of test_id.
    # No need to repeat for each headline of the same news.
    
    output = trainer.predict(tokenized)
    predictions = output.predictions
    loss = output.metrics['test_loss']

    assert len(test_id) == len(predictions)
    # assert test_id is of increasing order
    assert all(test_id[i] <= test_id[i+1] for i in range(len(test_id)-1))

    # for each news (all headlines with the same test_id), we select the one with largest predicted CTR as the final prediction
    # use same method but for real CTR to get the real label
    # then we calculate the accuracy
    
    test_id_unique = np.unique(test_id)
    final_predictions = []
    final_label = []

    start_idx = 0
    for _ in test_id_unique:
        end_idx = np.where(test_id == _)[0][-1] + 1
        pred = predictions[start_idx:end_idx]
        final_predictions.append(np.argmax(pred))

        real = real_CTR[start_idx:end_idx]
        final_label.append(np.argmax(real))

        start_idx = end_idx

    accuracy = sum(np.array(final_predictions) == np.array(final_label)) / len(final_predictions)

    if part_test_id is not None:
        # assert part_test_id is a subset of test_id
        assert all(_ in test_id for _ in part_test_id)
        assert len(final_predictions) == len(test_id_unique)
        mask = np.array([_ in part_test_id for _ in test_id_unique])
        part_final_predictions = np.array(final_predictions)[mask]
        part_final_label = np.array(final_label)[mask]
        part_accuracy = sum(part_final_predictions == part_final_label) / len(part_final_predictions)
        return {'eval_loss': loss, 'eval_accuracy': accuracy, 'eval_accuracy_part': part_accuracy}
    else:
        return {'eval_loss': loss, 'eval_accuracy': accuracy}


def plot_log(fig_file_path, data):
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(data['epoch_rec'], data['train_loss_rec'], label='train_loss')
    plt.plot(data['epoch_rec'], data['test_loss_rec'], label='test_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(212)
    plt.plot(data['epoch_rec'], data['train_accuracy_rec'], label='train_acc', c='C0')
    plt.plot(data['epoch_rec'], data['test_accuracy_rec_all'], label='test_acc all', c='C1')
    plt.plot(data['epoch_rec'], data['test_accuracy_rec_significant'], label='test_acc significant', c='C1', linestyle='dashed')
    for i in range(len(data['epoch_rec'])):
        plt.text(data['epoch_rec'][i], data['train_accuracy_rec'][i], f"{data['train_accuracy_rec'][i]:.3f}", ha='center', va='bottom')

        plt.text(data['epoch_rec'][i], data['test_accuracy_rec_all'][i], f"{data['test_accuracy_rec_all'][i]:.3f}", ha='center', va='bottom')

        plt.text(data['epoch_rec'][i], data['test_accuracy_rec_significant'][i], f"{data['test_accuracy_rec_significant'][i]:.3f}", ha='center', va='bottom')

    plt.hlines(0.3302, 0, len(data['epoch_rec'])-1, colors='C2', label='33.02% All Test Random Guess', alpha=0.5)
    plt.hlines(0.5367, 0, len(data['epoch_rec'])-1, linestyles='dashed', colors='C2', label='53.67% Significant Test Random Guess', alpha=0.5)

    plt.hlines(0.4628, 0, len(data['epoch_rec'])-1, colors='C3', label='46.282% All Test Embedding', alpha=0.5)
    plt.hlines(0.7460, 0, len(data['epoch_rec'])-1, linestyles='dashed', colors='C3', label='74.60% Significant Test Embedding', alpha=0.5)

    # plt.ylim(0, 0.5)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(fig_file_path)
    plt.close('all')

print('\nperformance before training ###########')
print('Performance on test')
_ = evaluate(tokenized_news['test'], data['test']['test_id'], data['test']['labels'], significant_test_ids)
# _ = trainer.evaluate(tokenized_news["test"])
self_logger['test_loss_rec'].append(_['eval_loss'])
self_logger['test_accuracy_rec_significant'].append(_['eval_accuracy_part'])
self_logger['test_accuracy_rec_all'].append(_['eval_accuracy'])
print(_)

print('Performance on train')
_ = evaluate(tokenized_news['train'], data['train']['test_id'], data['train']['labels'])
# _ = trainer.evaluate(tokenized_news["train"])
self_logger['train_loss_rec'].append(_['eval_loss'])
self_logger['train_accuracy_rec'].append(_['eval_accuracy'])
print(_)

self_logger['time_stamp_rec'].append(time.time())
self_logger['epoch_rec'].append(0)
self_logger['global_step_rec'].append(0)

plot_log(log_file_path.replace('.pkl', '.png'), self_logger)

class TrainingCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        logs = {}
        print('Now running self defined callback')

        print('Now evaluate on train dataset')
        # train_metrics = trainer.evaluate(eval_dataset=tokenized_news["train"])
        train_metrics = evaluate(tokenized_news['train'], data['train']['test_id'], data['train']['labels'])

        print('Now evaluate on test dataset')
        # test_metrics = trainer.evaluate(eval_dataset=tokenized_news["test"])
        test_metrics = evaluate(tokenized_news['test'], data['test']['test_id'], data['test']['labels'], significant_test_ids)

        self_logger['train_loss_rec'].append(train_metrics['eval_loss'])
        self_logger['test_loss_rec'].append(test_metrics['eval_loss'])
        self_logger['train_accuracy_rec'].append(train_metrics['eval_accuracy'])

        self_logger['test_accuracy_rec_all'].append(test_metrics['eval_accuracy'])
        self_logger['test_accuracy_rec_significant'].append(test_metrics['eval_accuracy_part'])
        self_logger['time_stamp_rec'].append(time.time())
        self_logger['epoch_rec'].append(state.epoch)
        self_logger['global_step_rec'].append(state.global_step)


        logs['train_loss'] = train_metrics['eval_loss']
        logs['train_accuracy'] = train_metrics['eval_accuracy']
        logs['test_loss'] = test_metrics['eval_loss']
        logs['test_accuracy'] = test_metrics['eval_accuracy']
        trainer.log(logs)
        print('the log is printed as', logs)

        # save self_logger to pickle
        with open(log_file_path, 'wb') as f:
            pickle.dump(self_logger, f)

        # plot the log
        plot_log(log_file_path.replace('.pkl', '.png'), self_logger)



trainer.add_callback(TrainingCallback())

print('\ntraining model ###########')
trainer.train()

os.mkdir(os.path.join(finetuned_model_dir, 'final'))
# trainer.save_model(os.path.join(finetuned_model_dir, 'final'))
model.save_pretrained(os.path.join(finetuned_model_dir, 'final'))


# run inference on test set, using the latest model
model.eval()
model = model.to(DEVICE)
inputs = tokenizer(data['test']['headline'], return_tensors="pt", padding=True, truncation=True)
inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

predictions = []
with torch.no_grad():
    for i in range(0, len(data['test']['headline']), 16):
        batch_inputs = {key: value[i:i+16] for key, value in inputs.items()}
        batch_outputs = model(**batch_inputs).logits.flatten().cpu().numpy().tolist()
        # batch_logits = batch_outputs.logits
        # batch_predictions = torch.argmax(batch_logits, dim=-1).cpu().numpy().tolist()
        predictions += batch_outputs

# n_matched = sum(np.array(data['test']['label']) == np.array(predictions))
# print('####### matched ratio:', n_matched / len(data['test']['label']))

predictions = np.array(predictions)
rescale_predictions = rescale_func(predictions)

# outputs = model(**inputs)
# logits = outputs.logits
# predictions = torch.argmax(logits, dim=-1)

# save the predictions to a csv file
result = pd.DataFrame({'headline': data['test']['headline'], 'real_CTR': data['test']['CTR'], 'predictions': rescale_predictions, 'test_id': data['test']['test_id']})

# save result to csv
result.to_csv(os.path.join(result_dir, 'result.csv'))