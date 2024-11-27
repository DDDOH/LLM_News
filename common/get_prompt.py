# TODO seems some small 瑕疵 in the prompt.
import random
import numpy as np
import pandas as pd

prompt_part_1 = """
You are an expert in digital media marketing and content strategy, specializing in optimizing headlines for maximum click-through rates (CTR) and increasing ad revenue. Your deep understanding of reader behavior, SEO, and digital engagement allows you to evaluate headlines for their potential to capture attention and drive clicks.

I have a news article, and I need to choose the catchiest headline from the following list. By “catchiest,” I mean the headline that is most likely to generate the highest CTR, thereby increasing ad revenue for the news company. The selected headline should:
	1.	Capture readers’ attention immediately.
	2.	Appeal to emotions, curiosity, or urgency.
	3.	Be engaging enough to make readers want to click on the article.
	4.	Follow best practices for digital news, considering SEO, shareability, and intrigue.
"""

prompt_example_1 = """I will also provide examples of multiple headline sets that have performed well in the past, along with the best-performing headline index for each set to guide your selection.
"""

prompt_example_2 = """Here are examples of headlines that have worked well before:"""


prompt_part_2 = {
    "letter": """Please review the headlines and return only the letter before the headline that is most likely to generate more clicks. **No explanation is needed. No need to return the headline, only the letter.**

Here are the headlines I need you to evaluate:""",
    "number": """Please review the headlines and return only the number before the headline that is most likely to generate more clicks. **No explanation is needed. No need to return the headline, only the number.**

Here are the headlines I need you to evaluate:""",
    "symbol": """
Please review the headlines and return only the symbol (e.g., !, @, #, $, %, etc.) before the headline that is most likely to generate more clicks. **No explanation is needed. No need to return the headline, only the symbol.**

Here are the headlines I need you to evaluate:""",
}

marker_types = ["letter", "number", "symbol"]
marker_ls = {
    "symbol": [
        "!",
        "@",
        "#",
        "$",
        "%",
        "^",
        "&",
        "*",
        "(",
        ")",
        "-",
        "_",
        "+",
        "=",
        "{",
        "}",
        "[",
        "]",
        "|",
        "\\",
        ":",
        ";",
        "'",
        '"',
        "<",
        ">",
        ",",
        ".",
        "?",
        "/",
    ],
    "number": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],
    "letter": [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ],
}

def get_prompt_with_label(headlines, best_headline_id):
    # randomly choose one marker type
    marker_type = random.choice(marker_types)
    conversation = [
        {"role": "system", "content": prompt_part_1 + prompt_part_2[marker_type]}
    ]

    content = ""
    for i, headline in enumerate(headlines):
        content += f"{marker_ls[marker_type][i]}. {headline}\n"

    conversation.append({"role": "system", "content": content})
    ground_truth = marker_ls[marker_type][best_headline_id]
    conversation.append({"role": "assistant", "content": ground_truth})
    return conversation

def get_prompt_without_example(one_news_to_test):
    # randomly choose one marker type
    marker_type = random.choice(marker_types)
    conversation = [
        {"role": "system", "content": prompt_part_1 + prompt_part_2[marker_type]}
    ]

    # randomly shuffle the rows of one_news_to_test
    one_news_to_test = one_news_to_test.sample(frac=1)
    headlines = one_news_to_test["headline"].tolist()
    best_headline_id = np.argmax(one_news_to_test["CTR"].tolist())

    content = ""
    for i, headline in enumerate(headlines):
        content += f"{marker_ls[marker_type][i]}. {headline}\n"

    conversation.append({"role": "system", "content": content})
    ground_truth = marker_ls[marker_type][best_headline_id]
    return conversation, ground_truth


def get_prompt_with_examples(examples, one_news_to_test, is_flip):
    # is_flip: if 1, in the example, we give a wrong answer
    marker_type = random.choice(marker_types)
    conversation = [
        {"role": "system", "content": prompt_part_1 + prompt_example_1 + prompt_part_2[marker_type] + prompt_example_2}
    ]

    for i in range(len(examples)):
        content = "Example " + str(i + 1) + ":\n"
        # conversation.append({"role": "system", "content": "Example " + str(i + 1) + ":"})
        # randomly shuffle the rows of the example
        example = examples[i].sample(frac=1)
        headlines = example["headline"].tolist()
        best_headline_id = np.argmax(example["CTR"].tolist())
        for j, headline in enumerate(headlines):
            content += f"{marker_ls[marker_type][j]}. {headline}\n"
        if is_flip == 0:
            content += "Best-performing headline: {}".format(marker_ls[marker_type][best_headline_id])
        else:
            # randomly choose a wrong answer
            wrong_anwers = [k + 1 for k in range(len(headlines)) if k != best_headline_id]
            content += "Best-performing headline: {}".format(marker_ls[marker_type][random.choice(wrong_anwers)])

        conversation.append({"role": "system", "content": content})

    conversation.append({"role": "system", "content": "Here are the headlines I need you to evaluate:"})
        
    # randomly shuffle the rows of one_news_to_test
    one_news_to_test = one_news_to_test.sample(frac=1)
    headlines = one_news_to_test["headline"].tolist()
    best_headline_id = np.argmax(one_news_to_test["CTR"].tolist())

    content = ""
    for i, headline in enumerate(headlines):
        content += f"{marker_ls[marker_type][i]}. {headline}\n"

    conversation.append({"role": "system", "content": content})
    ground_truth = marker_ls[marker_type][best_headline_id]
    return conversation, ground_truth


def print_conversation(conversation):
    for i in range(len(conversation)):
        print(conversation[i]["role"] + ": " + conversation[i]["content"])

if __name__ == "__main__":
    one_news_to_test = pd.DataFrame(columns=["headline", "CTR"])
    one_news_to_test["headline"] = ["headline " + str(i) for i in range(3)]
    one_news_to_test["CTR"] = [0.1, 0.2, 0.3]

    examples = [pd.DataFrame(columns=["headline", "CTR"]) for _ in range(2)]

    for i in range(2):
        examples[i]["headline"] = ["example " + str(i) + " headline " + str(j) for j in range(3)]
        examples[i]["CTR"] = [0.1, 0.2, 0.3]

    print("""### get_prompt_with_examples""")
    conversation, ground_truth = get_prompt_with_examples(examples, one_news_to_test, 0)
    print_conversation(conversation)

    print("""### get_prompt_with_label""")
    conversation = get_prompt_with_label(one_news_to_test["headline"].tolist(), 2)
    print_conversation(conversation)