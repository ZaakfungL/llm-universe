import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from config import QWEN3_EMBEDDING_MODEL_PATH

# from embedding.zhipuai_embedding import ZhipuAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from llm.call_llm import parse_llm_api_key

def get_embedding(embedding: str, embedding_key: str=None):
    if embedding == 'qwen3':
        return HuggingFaceEmbeddings(model_name=QWEN3_EMBEDDING_MODEL_PATH)
    if embedding_key == None:
        embedding_key = parse_llm_api_key(embedding)
    if embedding == "openai":
        return OpenAIEmbeddings(openai_api_key=embedding_key)
    # elif embedding == "zhipuai":
    #     return ZhipuAIEmbeddings(zhipuai_api_key=embedding_key)
    else:
        raise ValueError(f"embedding {embedding} not support ")