from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb
from llm.call_llm import get_completion


class Chat_QA_chain_self:
    """ "
    带历史记录的问答链
    - model：调用的模型名称
    - temperature：温度系数，控制生成的随机性
    - top_k：返回检索的前k个相似文档
    - chat_history：历史记录，输入一个列表，默认是一个空列表
    - history_len：控制保留的最近 history_len 次对话
    - file_path：建库文件所在路径
    - persist_path：向量数据库持久化路径
    - embeddings：使用的embedding模型
    - embedding_key：使用的embedding模型的秘钥
    - streaming：是否开启流式输出
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        top_k: int = 4,
        chat_history: list = [],
        file_path: str = None,
        persist_path: str = None,
        embedding="openai",
        embedding_key: str = None,
        streaming: bool = False,
        use_knowledge_base: bool = True,
    ):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.chat_history = chat_history
        # self.history_len = history_len
        self.file_path = file_path
        self.persist_path = persist_path
        self.embedding = embedding
        self.embedding_key = embedding_key
        self.streaming = streaming
        self.use_knowledge_base = use_knowledge_base
        self.callbacks = [StreamingStdOutCallbackHandler()] if streaming else []

        if use_knowledge_base and file_path:
            print(f"[DEBUG] 初始化向量数据库 | file_path: {file_path} | persist_path: {persist_path}")
            self.vectordb = get_vectordb(
                self.file_path, self.persist_path, self.embedding, self.embedding_key
            )
            print(f"[DEBUG] 向量数据库初始化完成 | 状态: {'成功' if self.vectordb else '失败'}")
        else:
            print(f"[DEBUG] 跳过向量数据库初始化 | use_knowledge_base: {use_knowledge_base} | file_path: {file_path}")
            self.vectordb = None

    def clear_history(self):
        "清空历史记录"
        return self.chat_history.clear()

    def change_history_length(self, history_len: int = 1):
        """
        保存指定对话轮次的历史记录
        输入参数：
        - history_len ：控制保留的最近 history_len 次对话
        - chat_history：当前的历史对话记录
        输出：返回最近 history_len 次对话
        """
        n = len(self.chat_history)
        return self.chat_history[n - history_len :]

    def answer(self, question: str = None, temperature=None, top_k=4, use_knowledge_base=None):
        """ "
        核心方法，调用问答链
        arguments:
        - question：用户提问
        - temperature：温度参数
        - top_k：检索数量
        - use_knowledge_base：是否使用知识库
        """

        if len(question) == 0:
            return "", self.chat_history

        if temperature == None:
            temperature = self.temperature

        if use_knowledge_base == None:
            use_knowledge_base = self.use_knowledge_base

        llm = model_to_llm(
            self.model,
            temperature
        )

        # self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        if use_knowledge_base and self.vectordb:
            # 使用知识库问答
            retriever = self.vectordb.as_retriever(
                search_type="similarity", search_kwargs={"k": top_k}
            )
            
            # 检索相关文档
            docs = retriever.get_relevant_documents(question)
            
            # 构建包含检索上下文的prompt
            context = "\n".join([doc.page_content for doc in docs])
            rag_prompt = self.format_rag_prompt(question, context, self.chat_history)
            
            # 使用统一的生成方法
            answer = get_completion(
                rag_prompt, 
                self.model, 
                temperature=temperature,
                streaming=self.streaming
            )
        else:
            # 纯LLM对话，使用历史记录
            formatted_prompt = self.format_chat_prompt(question, self.chat_history)
            answer = get_completion(
                formatted_prompt, 
                self.model, 
                temperature=temperature,
                streaming=self.streaming
            )
        
        answer = re.sub(r"\\n", "<br/>", answer)
        self.chat_history.append((question, answer))  # 更新历史记录
        return self.chat_history  # 返回更新后的历史记录

    def answer_stream(self, question: str = None, temperature=None, top_k=4, use_knowledge_base=None):
        """
        流式问答方法，返回生成器用于 Gradio 流式显示
        arguments:
        - question：用户提问
        - temperature：温度参数
        - top_k：检索数量
        - use_knowledge_base：是否使用知识库
        """
        if len(question) == 0:
            return

        if temperature == None:
            temperature = self.temperature

        if use_knowledge_base == None:
            use_knowledge_base = self.use_knowledge_base

        if use_knowledge_base and self.vectordb:
            # RAG模式
            print(f"[DEBUG] 启动 RAG 模式 | top_k: {top_k}")
            retriever = self.vectordb.as_retriever(
                search_type="similarity", search_kwargs={"k": top_k}
            )
            
            # 检索相关文档
            docs = retriever.get_relevant_documents(question)
            print(f"[DEBUG] 检索到 {len(docs)} 个相关文档")
            
            # 构建包含检索上下文的prompt
            context = "\n".join([doc.page_content for doc in docs])
            rag_prompt = self.format_rag_prompt(question, context, self.chat_history)
            print(f"[DEBUG] RAG Prompt 长度: {len(rag_prompt)}")
            
            # 使用流式生成
            full_answer = ""
            for chunk in self._stream_completion(rag_prompt, temperature):
                full_answer += chunk
                yield full_answer
            
            # 处理最终答案
            full_answer = re.sub(r"\\n", "<br/>", full_answer)
            self.chat_history.append((question, full_answer))
        else:
            # 纯LLM对话
            print(f"[DEBUG] 启动纯 LLM 模式 | 向量DB状态: {self.vectordb is not None}")
            formatted_prompt = self.format_chat_prompt(question, self.chat_history)
            
            # 使用流式生成
            full_answer = ""
            for chunk in self._stream_completion(formatted_prompt, temperature):
                full_answer += chunk
                yield full_answer
            
            # 处理最终答案
            full_answer = re.sub(r"\\n", "<br/>", full_answer)
            self.chat_history.append((question, full_answer))

    def _stream_completion(self, prompt: str, temperature: float):
        """
        内部方法：实现流式文本生成
        """
        if self.model in ["qwen-plus", "qwen-turbo"]:
            yield from self._stream_dashscope(prompt, temperature)
        elif self.model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"]:
            yield from self._stream_openai(prompt, temperature)
        else:
            # 不支持流式的模型，直接返回完整结果
            answer = get_completion(prompt, self.model, temperature=temperature, streaming=False)
            yield answer

    def _stream_dashscope(self, prompt: str, temperature: float):
        """通义千问流式生成"""
        import dashscope
        from llm.call_llm import parse_llm_api_key
        
        api_key = parse_llm_api_key("dashscope")
        dashscope.api_key = api_key
        
        response = dashscope.Generation.call(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            stream=True,
        )
        
        full_response = ""
        for chunk in response:
            if hasattr(chunk, 'output') and hasattr(chunk.output, 'text'):
                current_text = chunk.output.text
                if current_text and len(current_text) > len(full_response):
                    new_content = current_text[len(full_response):]
                    full_response = current_text
                    yield new_content

    def _stream_openai(self, prompt: str, temperature: float):
        """OpenAI 流式生成"""
        from openai import OpenAI
        from llm.call_llm import parse_llm_api_key
        
        api_key = parse_llm_api_key("openai")
        client = OpenAI(api_key=api_key)
        
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content

    def format_chat_prompt(self, message, chat_history):
        """
        格式化聊天 prompt
        """
        prompt = ""
        for turn in chat_history:
            user_message, bot_message = turn
            prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
        prompt = f"{prompt}\nUser: {message}\nAssistant:"
        return prompt

    def format_rag_prompt(self, question, context, chat_history):
        """
        格式化RAG prompt，包含检索到的上下文
        """
        # 构建历史对话
        history_text = ""
        for turn in chat_history:
            user_message, bot_message = turn
            history_text = f"{history_text}\nUser: {user_message}\nAssistant: {bot_message}"
        
        # 构建包含上下文的prompt
        prompt = f"""请基于以下上下文信息回答用户问题。如果上下文中没有相关信息，请诚实说明。

上下文信息：
{context}

历史对话：{history_text}

User: {question}
Assistant:"""
        return prompt
