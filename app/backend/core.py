import os
from typing import Any

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
# import pinecone
#
# from consts import INDEX_NAME
#
# pinecone.init(
#     api_key=os.environ["PINECONE_API_KEY"],
#     environment=os.environ["PINECONE_ENVIRONMENT_REGION"],
# )

import boto3
import json
import os
import argparse

client = boto3.client('bedrock-runtime')

parser = argparse.ArgumentParser()
parser.add_argument("--file", type=str, required=True, help="Prompt for text generation")
parser.add_argument("--modelid", type=str, required=True, help="Model ID for generation")
args = parser.parse_args()

modelId = args.modelid
accept = 'application/json'
contentType = 'application/json'

# Read the prompt from a text file
with open(args.file, "r") as file:
    prompt_data = file.read().strip()

# Format the prompt to start with "Human:" and end with "Assistant:"
formatted_prompt = """
    given the complaint information from a customer {complaint},
    I want you to create:
    1. A short summary
    2. two interesting facts about them 

    Use both information from twitter and Linkedin
    \n{format_instructions}
    """

claude_input = json.dumps({
    "prompt": formatted_prompt,
    "max_tokens_to_sample": 500,
    "temperature": 0.5,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": []
})

response = client.invoke_model(body=claude_input, modelId=modelId, accept=accept, contentType=contentType)
response_body = json.loads(response.get('body').read())
print(response_body['completion'])


def run_llm(query: str) -> Any:
    embeddings = OpenAIEmbeddings()
    docsearch = Pinecone.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOpenAI(verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain?"))

