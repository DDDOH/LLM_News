
# get embedding and save it to a npy file.
# might stop in the middle due to OpenAI rate limit, but the progress will be saved in the file.
import pandas as pd
import numpy as np
import progressbar # pip install progressbar2
import os
import concurrent.futures

MODEL = 'OpenAI' # OpenAI or Word2Vec256 or Word2Vec3072 or Llama4096 or Bert

SPLIT_BY_TIME = False # if True, load the original data from the LoRA_CTR_test and LoRA_CTR_train csv files
# otherwise, load test_order_by_time.csv and train_order_by_time.csv


DEBUG = False # if True, only process the first 1000 headlines

if MODEL == 'OpenAI':
    # to access OpenAI API in China, open VPN first and then set the proxy
    import os
    
    # run the following code in the terminal use python3
    # export http_proxy=http://127.0.0.1:1087;export https_proxy=http://127.0.0.1:1087;
    os.environ["http_proxy"] = "http://127.0.0.1:1087"
    os.environ["https_proxy"] = "http://127.0.0.1:1087"


if SPLIT_BY_TIME:
    result_dir = os.path.join('Pure LLM - Embedding/saved_embedding', MODEL + '_split_by_time')
else:
    result_dir = os.path.join('Pure LLM - Embedding/saved_embedding', MODEL)

if not os.path.exists(result_dir):
    # load headlines
    if SPLIT_BY_TIME:
        # ori_test = pd.read_csv('Code Dataset/test_order_by_time.csv')
        # ori_train = pd.read_csv('Code Dataset/train_order_by_time.csv')
        ctr_all = pd.read_csv('Code Dataset/ctr-all.csv')

        df = ctr_all.sort_values(by='created_at').reset_index(drop=True)
        df['test_id'] = df.groupby(['clickability_test_id', 'eyecatcher_id']).ngroup()

        # drop clickability_test_id, eyecatcher_id columns
        df = df.drop(columns=['clickability_test_id', 'eyecatcher_id'])

        df['embedding'] = np.empty((len(df), 0)).tolist()


        # try to load saved embedding from file (saved_embedding/OpenAI)
        train_df = np.load('Pure LLM - Embedding/saved_embedding/OpenAI/train.npy', allow_pickle=True)
        test_df = np.load('Pure LLM - Embedding/saved_embedding/OpenAI/test.npy', allow_pickle=True)
        all_df = np.concatenate([train_df, test_df], axis=0)

        # add embedding column to the dataframe
        for i in progressbar.progressbar(range(len(df))):
            headline = df.loc[i, 'headline']
            # find the embedding for the headline
            if headline in all_df[:, 1]:
                df.at[i, 'embedding'] = all_df[all_df[:, 1] == headline, -1].tolist()[0]

        # print the number of headlines that have not been embedded
        mask = np.array([len(_) == 0 for _ in df['embedding']])
        n_embedded = len(df) - np.sum(mask)
        print(f'{n_embedded} among {len(df)} headlines have been embedded.')

        # save dataframe to result_dir as a npy file
        columns = df.columns.tolist()
        os.mkdir(result_dir)
        np.save(os.path.join(result_dir, 'all_data.npy'), df)
        np.save(os.path.join(result_dir, 'columns.npy'), columns)

    else:
        ori_test = pd.read_csv('Code Dataset/LoRA_CTR_test.csv')
        ori_train = pd.read_csv('Code Dataset/LoRA_CTR_train.csv')
        ori_calibration = pd.read_csv('Code Dataset/LoRA_CTR_calibration.csv')
        

        # remove Unnamed: 0 column from the dataframe
        ori_test = ori_test.drop(columns=['Unnamed: 0'])
        ori_train = ori_train.drop(columns=['Unnamed: 0'])
        # ori_calibration = ori_calibration.drop(columns=['Unnamed: 0'])

        # drop rows with nan values
        ori_test = ori_test.dropna()
        ori_train = ori_train.dropna()
        ori_calibration = ori_calibration.dropna()

        # add a title_id column to the dataframe
        ori_test['title_id'] = ori_test.index
        ori_train['title_id'] = ori_train.index
        ori_calibration['title_id'] = ori_calibration.index

        # add embedding column to the dataframe
        ori_test['embedding'] = np.empty((len(ori_test), 0)).tolist()
        ori_train['embedding'] = np.empty((len(ori_train), 0)).tolist()
        ori_calibration['embedding'] = np.empty((len(ori_calibration), 0)).tolist()

        # save dataframe to result_dir as a npy file
        columns = ori_test.columns.tolist()

        os.makedirs(result_dir)
        np.save(os.path.join(result_dir, 'test.npy'), ori_test)
        np.save(os.path.join(result_dir, 'train.npy'), ori_train)
        np.save(os.path.join(result_dir, 'calibration.npy'), ori_calibration)
        np.save(os.path.join(result_dir, 'columns.npy'), columns)




def get_embedding(headline):
    # assert headline is a string
    if isinstance(headline, str):
        response = client.embeddings.create(
        input=headline,
        model="text-embedding-3-large"
        )
        return response.data[0].embedding
    else:
        return [np.nan]
    
def get_embedding_parallel(data_name):
    # data_name: train or test
    # try to load saved embedding from file
    print(f'Getting embeddings for {data_name}...')
    file_path = os.path.join(result_dir, data_name + '.npy')
    df = np.load(file_path, allow_pickle=True)
    if DEBUG:
        df = df[:1000]
    columns = np.load(os.path.join(result_dir, 'columns.npy'), allow_pickle=True)
    headline_index = np.where(columns == 'headline')[0][0]

    # count the number of headlines that have not been embedded
    mask = np.array([len(_) == 0 for _ in df[:,-1]])
    n_embedded = len(df) - np.sum(mask)
    print(f'{n_embedded} among {len(df)} headlines have been embedded.')

    batch_size = 500

    while np.sum(mask) > 0:
        mask_cumsum = np.cumsum(mask)
        mask_batch = (mask_cumsum < batch_size) & mask
        headlines = df[mask_batch]

        # get the embeddings for the headlines parallelly
        with concurrent.futures.ThreadPoolExecutor() as executor:
            embeddings = list(progressbar.progressbar(executor.map(get_embedding, headlines[:, headline_index]), max_value=len(headlines)))

        # update the dataframe with the new embeddings
        df[mask_batch, -1] = embeddings

        # save the updated dataframe back to the file
        np.save(file_path, df)

        # find the fload in df[:,-1]
        is_fload = [isinstance(_, float) for _ in df[:,-1]]
       

        # update the mask
        mask = np.array([len(_) == 0 for _ in df[:,-1]])

        # count the number of headlines that have been embedded
        n_embedded = len(df) - np.sum(mask)
        print(f'{n_embedded} among {len(df)} headlines have been embedded.')

    # save the final dataframe to a file
    np.save(file_path, df)
    print(f'{data_name} embedding saved.')


def get_one_word2vec_embedding(text, model):
    np.random.seed(42)
    words = word_tokenize(text.lower())
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)


def get_embedding_word2vec(data_name):
    # data_name: train or test
    # try to load saved embedding from file
    file_path = os.path.join(result_dir, data_name + '.npy')
    df = np.load(file_path, allow_pickle=True)

    # for each line in df, get the embedding
    for i in progressbar.progressbar(range(len(df))):
        headline = df[i, 1]
        embedding = get_one_word2vec_embedding(headline, model_word2vec)
        df[i, -1] = embedding

    # save the final dataframe to a file
    np.save(file_path, df)
    print(f'{data_name} embedding saved.')


# export HF_ENDPOINT=https://hf-mirror.com

def get_embedding_llama(data_name):
    # data_name: train or test
    # try to load saved embedding from file
    file_path = os.path.join(result_dir, data_name + '.npy')
    df = np.load(file_path, allow_pickle=True)

    # for all titles, get embedding
    headlines = df[:,1].tolist()

    l2v = LLM2Vec.from_pretrained(
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
        peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )
    
    embedding = l2v.encode(headlines)
    df[:, -1] = list(embedding.numpy())
    np.save(file_path, df)
    print(f'{data_name} embedding saved.')
    
if MODEL == 'OpenAI':
    # get openai embedding for each headline
    from openai import OpenAI
    client = OpenAI(api_key="YOUR_API_KEY")

    if SPLIT_BY_TIME:
        get_embedding_parallel('all_data')
    else:
        get_embedding_parallel('test')
        get_embedding_parallel('train')
        get_embedding_parallel('calibration')

elif MODEL in ['Word2Vec256', 'Word2Vec3072']:
    # for Word2Vec
    from gensim.models import Word2Vec # pip install --upgrade gensim
    from nltk.tokenize import word_tokenize  # pip install --user -U nltk
    import nltk

    nltk.download('punkt')
    nltk.download('punkt_tab')


    train_headlines = np.load(os.path.join(result_dir, 'train.npy'), allow_pickle=True)
    test_headlines = np.load(os.path.join(result_dir, 'test.npy'), allow_pickle=True)

    # combine train and test headlines
    texts = list(train_headlines[:, 1]) + list(test_headlines[:, 1])

    tokenized_texts = [word_tokenize(text.lower()) for text in texts]
    if MODEL == 'Word2Vec3072':
        DIM = 3072
    elif MODEL == 'Word2Vec256':
        DIM = 256
    model_word2vec = Word2Vec(tokenized_texts, vector_size=DIM, window=5, min_count=1, workers=4)

    get_embedding_word2vec('test')
    get_embedding_word2vec('train')
elif MODEL == 'Llama4096':
    # for Llama
    import torch
    from llm2vec import LLM2Vec # pip install llm2vec

    get_embedding_llama('test')
    get_embedding_llama('train')

elif MODEL == 'Bert':
    from transformers import BertTokenizer, BertModel
    import torch
    def get_embedding_bert(data_name):
        # data_name: train or test
        # try to load saved embedding from file
        file_path = os.path.join(result_dir, data_name + '.npy')
        df = np.load(file_path, allow_pickle=True)

        # Load BERT tokenizer and model
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        # Move model to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)


        BATCH_SIZE = 1024
        index = 0
        while index < len(df):
            # get the headlines that have not been embedded of size BATCH_SIZE
            headlines = df[index:index+BATCH_SIZE, 1].tolist()

            # Tokenize and encode text using batch_encode_plus
            # The function returns a dictionary containing the token IDs and attention masks
            encoding = tokenizer.batch_encode_plus(
                headlines,                # List of input texts
                padding=True,             # Pad to the maximum sequence length
                truncation=True,          # Truncate to the maximum sequence length if necessary
                return_tensors='pt',      # Return PyTorch tensors
                add_special_tokens=True   # Add special tokens CLS and SEP
            )

            input_ids = encoding['input_ids'].to(device)  # Token IDs
            attention_mask = encoding['attention_mask'].to(device)  # Attention mask

            word_embeddings = model(input_ids, attention_mask=attention_mask).last_hidden_state
            sentence_embedding = word_embeddings.mean(dim=1)  # Average pooling along the sequence length dimension

            # Update the dataframe with the new embeddings
            # df[unembedded_mask][:BATCH_SIZE, -1] = sentence_embedding.cpu().detach().numpy().tolist()

            df[index:index+BATCH_SIZE, -1] = sentence_embedding.cpu().detach().numpy().tolist()

            index += BATCH_SIZE

        # Save the final dataframe to a file
        np.save(file_path, df)
        print(f'{data_name} embedding saved.')


    get_embedding_bert('test')
    get_embedding_bert('train')

    # Output the shape of the sentence embedding



    

