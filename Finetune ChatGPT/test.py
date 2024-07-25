# https://platform.openai.com/docs/guides/batch/getting-started
# 24-7-25: Bug here.

from openai import OpenAI
import progressbar
import pandas as pd


client = OpenAI(api_key="YOUR API KEY HERE")

# response = client.chat.completions.create(
#     model="ft:gpt-3.5-turbo-0125:nb-org:llm-news:9noa1jMr",
#     messages=[{"role": "user", "content": "Say this is a test"}],
#     stream=False,
# )


test_set = "data/fixed_test_set.csv"
# load test set
test_df = pd.read_csv(test_set)


SYSTEM = "You are an editor tasked with choosing the catchier one from two drafted headlines for the same content. Please return either 1 or 2."
USER = "You are presented with two headlines. Which one is catchier?\n1. {}\n2. {}"
RERUN_TEST = False

# check if predicted_label is one column in test_df
if "predicted_label" not in test_df.columns or RERUN_TEST:
    test_df["predicted_label"] = None
    n_predicted = 0
    n_correct = 0
else:
    n_predicted = len(test_df[test_df["predicted_label"].notna()])
    n_correct = len(test_df[test_df["predicted_label"] == test_df["higher_CTR"]])
    
if n_predicted == len(test_df):
    print(n_predicted)
    print(n_correct)
    print("Predicted {} rows".format(n_predicted))
    print("Accuracy: {:.2f}%".format(n_correct / n_predicted * 100))
    exit()

# for each row in test_df that has not been predicted, use the headline_1 and headline_2 to generate a prompt



progress = progressbar.ProgressBar(maxval=len(test_df)).start()

for index, row in test_df.iterrows():
    if row["predicted_label"] is None:
        prompt = USER.format(row["headline_1"], row["headline_2"])
        response = client.chat.completions.create(
            model="ft:gpt-3.5-turbo-0125:nb-org:llm-news-full:9oFf1b8H",
            messages=[
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": USER.format(row["headline_1"], row["headline_2"]),
                },
            ],
            stream=False,
        )
        prediction = int(response.choices[0].message.content) + 1
        test_df.loc[index, "predicted_label"] = prediction
        # print("predicted: {}, real: {}".format(test_df.loc[index, "predicted_label"], test_df.loc[index, "higher_CTR"]))
        n_predicted += 1
        if test_df.loc[index, "predicted_label"] == test_df.loc[index, "higher_CTR"]:
            n_correct += 1

    if n_predicted % 10 == 0:
        print("Predicted {} rows".format(n_predicted))
        print("Accuracy: {:.2f}%".format(n_correct / n_predicted * 100))
        
    if n_predicted % 50 == 0:
        # save test_df to csv
        test_df.to_csv('data/fixed_test_set.csv', index=False)
        
    progress.update(n_predicted)
    
if n_predicted == len(test_df):
    print("Predicted {} rows".format(n_predicted))
    print("Accuracy: {:.2f}%".format(n_correct / n_predicted * 100))
    test_df.to_csv('data/fixed_test_set.csv', index=False)
    

    