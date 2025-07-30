from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging
import re

from qa_chain.get_vectordb import get_vectordb
from qa_chain.model_to_llm import model_to_llm


class QA_chain_self:
    """
    问答链类，用于基于向量数据库的知识问答
    """

    def __init__(
        self,
        temperature: float = 0.9,
        top_k: int = 4,
        file_path: str = None,
        persist_path: str = None,
        embedding: str = "openai",
        embedding_key: str = None,
        model: str = "qwen-plus",
        template: str = None,
    ):
        """
        初始化问答链对象
        :param temperature: 温度参数，控制回答的随机性
        :param top_k: 向量数据库检索的前K个结果
        :param file_path: 知识库文件路径
        :param persist_path: 向量数据库持久化路径
        :param embedding: 使用的嵌入模型
        :param embedding_key: 嵌入模型的密钥
        :param model: 大语言模型名称
        :param template: 自定义提示词模板
        """
        self.temperature = temperature
        self.top_k = top_k
        self.file_path = file_path
        self.persist_path = persist_path
        self.embedding = embedding
        self.embedding_key = embedding_key
        # 设置提示词模板，若外部未提供则使用默认模板
        self.template = template or (
            "请根据以下上下文内容，结合知识库回答用户问题：\n"
            "上下文：{context}\n"
            "问题：{question}\n"
            "回答："
        )

        # 初始化向量数据库
        try:
            self.vectordb = get_vectordb(
                file_path=self.file_path,
                persist_path=self.persist_path,
                embedding=self.embedding,
                embedding_key=self.embedding_key,
            )
            logging.info("向量数据库初始化成功")
        except Exception as e:
            logging.error(f"向量数据库初始化失败: {e}")
            raise

        self.llm = model_to_llm(
            model=model,
            temperature=temperature,
        )
        self.QA_CHAIN_PROMPT = PromptTemplate(
            input_variables=["context", "question"], template=self.template
        )
        self.retriever = self.vectordb.as_retriever(
            search_type="similarity", search_kwargs={"k": self.top_k}
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": self.QA_CHAIN_PROMPT},
        )

    def answer(
        self,
        question: str,
        temperature: float = None,
        top_k: int = None,
    ) -> str:
        """
        调用问答链
        :param question: 用户提问
        :param temperture: 温度参数，控制回答的随机性
        :param top_k: 向量数据库检索的前K个结果
        :return 大模型回答文本
        """
        # 使用默认参数或按需覆盖 LLM 温度与检索数量
        if temperature is None:
            temperature = self.temperature
        else:
            # 动态调整 llm 的温度
            self.llm.temperature = temperature
        if top_k is None:
            top_k = self.top_k
        else:
            # 动态调整 retriever 的检索数量
            self.retriever.search_kwargs["k"] = top_k

        # 执行问答链，只传入用户提问
        result = self.qa_chain({"query": question})
        answer_text = result["result"]
        answer_text = re.sub(r"\\n", "<br/>", answer_text)
        return answer_text
