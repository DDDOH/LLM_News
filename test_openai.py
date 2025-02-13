# # export http_proxy=http://127.0.0.1:1087;export https_proxy=http://127.0.0.1:1087;
import os

os.environ["http_proxy"] = "http://127.0.0.1:1087"
os.environ["https_proxy"] = "http://127.0.0.1:1087"

from openai import OpenAI
client = OpenAI(api_key="YOUR_API_KEY")


response = client.embeddings.create(
        input="hello",
        model="text-embedding-3-large"
        )

print(response)



