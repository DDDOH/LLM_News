Code for paper LOLA: LLM-Assisted Online Learning Algorithm for Content Experiments.

We recommend using conda and pip to manage the environment. To set up the environment:

``` python
conda create --name lola
conda install pip
pip install datasets
pip install peft
pip install evaluate
pip install transformers -U
pip install -U scikit-learn
pip install -U matplotlib
pip install progressbar2
pip install openai

# to download the Llama3 model, register on huggingface for access to the model and then run the following command
pip install -U "huggingface_hub[cli]"
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc # for faster downloads in China
huggingface-cli login
# type in your huggingface credentials
```


The original dataset we used is https://osf.io/jd64p/.

The pre-processed dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/shuffleofficial/lola-llm-assisted-online-learning-algorithm), or use the kaggle CLI command:
`kaggle datasets download -d shuffleofficial/lola-llm-assisted-online-learning-algorithm`




