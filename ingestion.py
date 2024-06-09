import os

import boto3
from langchain_community.document_loaders import TextLoader, S3FileLoader
from langchain_text_splitters import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import Bedrock

if __name__ == '__main__':
    print("Ingesting...")
    loader = S3FileLoader(bucket='docs-legalcase-hackathon', region_name='eu-west-1', key='section75_process.txt')
    document = loader.load()

    region_name = 'us-east-1'
    bedrock_client = boto3.client('bedrock-agent-runtime', region_name=region_name)

    response = bedrock_client.retrieve_and_generate(
        input={"text": 'hi'},
        retrieveAndGenerateConfiguration={
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": 'ANBV0LAIO5',
                "modelArn": f"arn:aws:bedrock:us-east1::foundation-model/anthropic.claude-v2"
            },
            "type": "KNOWLEDGE_BASE"
        }
    )

    print(response['output']['text'])

