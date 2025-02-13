Code for paper LOLA: LLM-Assisted Online Learning Algorithm for Content Experiments.


# Python Environment
We recommend using conda and pip to manage the environment. To set up the environment:

``` python
conda create --name lola
conda activate lola
conda install pip
pip install datasets
pip install peft
pip install evaluate
pip install transformers -U
pip install -U scikit-learn
pip install -U matplotlib
pip install progressbar2
pip install openai

# to download the Llama-3 model (only needed for fine-tuning Llama-3), register on huggingface for access to the model and then run the following command
pip install -U "huggingface_hub[cli]"
huggingface-cli login
# type in your huggingface credentials
```


# Reproducing the Results
We save some intermediate result to reproduce the results in the paper. Use these intermediate results can save time for OpenAI API calls, finetuning the model.

## For Prompt Engineering Method
run `Pure LLM - Prompt/visualize_result.py`, this will generate mean_differences_heatmap_multiple.pdf and p_values_heatmap_multiple.pdf.

## For Embedding Method
run `Pure LLM - Embedding/predict_with_embedding.py`

## For Finetuning
run `Finetune CTR Prediction/plot.py`



Below are the steps to run all the code, from getting intermediate results to getting final result.

# Dataset and Code

The original dataset we used is https://osf.io/jd64p/.

The pre-processed dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/shuffleofficial/lola-llm-assisted-online-learning-algorithm), or use the kaggle CLI command:
`kaggle datasets download -d shuffleofficial/lola-llm-assisted-online-learning-algorithm`


- For data processing
	- Code Path `Upworthy Data Processing.ipynb`
	- Running this code will generate a csv file named `winner-all.csv`
	- Data used:  `upworthy-archive-holdout-packages-03.12.2020.csv`,  `upworthy-archive-exploratory-packages-03.12.2020.csv` and `upworthy-archive-confirmatory-packages-03.12.2020.csv` (these data are downloaded from https://osf.io/jd64p/)
- For Prompt Engineering Method
	- Code Path `Pure LLM Approaches/Pure LLM - Prompt/Prompt-based Approaches.ipynb`
	- Data used `winner-all.csv`
- For Classification using OpenAI and Word2Vec Embedding
	- Code Path `Pure LLM Approaches/Pure LLM - Embedding/Embedding (OpenAI&Word2Vec) Classification.ipynb`
	- Data used: `selected_pairs_df_005_256.csv` and `selected_pairs_df_005_3072.csv`
- For Predicting CTR using OpenAI Embedding
	- Code Path `LOLA/LOLA - Regret Minimize/LOLA_regret_minimize.ipynb`
	- Data used: `all_test_headline_embed_3072.csv`


- Survey Results
	- Code and data path `Survey`

