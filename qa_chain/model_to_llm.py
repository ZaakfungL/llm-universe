import sys

sys.path.append("../llm")
from langchain_openai import ChatOpenAI
from langchain_community.llms import Tongyi
from llm.call_llm import parse_llm_api_key


def model_to_llm(model: str = None, temperature: float = 0.0, api_key: str = None):
    """
    返回 LangChain 兼容的 LLM 对象
    """
    if model in [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k-0613",
        "gpt-3.5-turbo-0613",
        "gpt-4",
        "gpt-4-32k",
    ]:
        if api_key == None:
            api_key = parse_llm_api_key("openai")
        llm = ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)
    elif model in ["qwen-plus", "qwen-turbo"]:
        if api_key == None:
            api_key = parse_llm_api_key("dashscope")
        llm = Tongyi(model=model, temperature=temperature, dashscope_api_key=api_key)
    else:
        raise ValueError(f"model {model} not support!!!")
    return llm
