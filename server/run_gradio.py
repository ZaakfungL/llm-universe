import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import re
import io
import gradio as gr
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from loguru import logger
import time

from config import KNOWLEDGE_DB_DIR, VECTOR_DB_DIR
from llm.call_llm import get_completion
from database.create_db import create_db_info
from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self
from qa_chain.QA_chain_self import QA_chain_self

# 配置 loguru
def setup_logger():
    """配置 loguru 日志"""
    # 移除默认的控制台输出
    logger.remove()
    
    # 添加控制台输出（带颜色）
    logger.add(
        sys.stderr, 
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 添加文件输出
    logger.add(
        "logs/chatomni_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="INFO",
        rotation="1 day",  # 每天轮转
        retention="7 days",  # 保留7天
        compression="zip",  # 压缩旧日志
        encoding="utf-8"
    )
    
    # 添加错误日志文件
    logger.add(
        "logs/chatomni_error_{time:YYYY-MM-DD}.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        rotation="1 day",
        retention="30 days",  # 错误日志保留更久
        compression="zip",
        encoding="utf-8"
    )

setup_logger()

_ = load_dotenv(find_dotenv())
LLM_MODEL_DICT = {
    "openai": [
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k-0613",
        "gpt-3.5-turbo-0613",
        "gpt-4",
        "gpt-4-32k",
    ],
    "dashscope": ["qwen-turbo", "qwen-plus"],
    "deepseek": ["deepseek-r1-distill-qwen-7b"],
}

LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()), [])
INIT_LLM = "qwen-plus"
EMBEDDING_MODEL_LIST = ["openai", "qwen3"]
INIT_EMBEDDING_MODEL = "qwen3"

DEFAULT_DB_PATH = KNOWLEDGE_DB_DIR
DEFAULT_PERSIST_PATH = VECTOR_DB_DIR

AIGC_AVATAR_PATH = "./figures/aigc_avatar.png"
DATAWHALE_AVATAR_PATH = "./figures/datawhale_avatar.png"
AIGC_LOGO_PATH = "./figures/aigc_logo.png"
DATAWHALE_LOGO_PATH = "./figures/datawhale_logo.png"


def get_model_by_platform(platform):
    return LLM_MODEL_DICT.get(platform, "")


class Model_center:
    """
    存储问答 Chain 的对象

    - chat_qa_chain_self: 以 (model, embedding) 为键存储的带历史记录的问答链。
    - qa_chain_self: 以 (model, embedding) 为键存储的不带历史记录的问答链。
    """

    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}

    def chat_qa_chain_self_answer(
        self,
        question: str,
        chat_history: list = [],
        model: str = "openai",
        embedding: str = "openai",
        temperature: float = 0.0,
        top_k: int = 4,
        history_len: int = 3,
        file_path: str = DEFAULT_DB_PATH,
        persist_path: str = DEFAULT_PERSIST_PATH,
    ):
        """
        调用带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            # 转换消息格式为元组格式供链使用
            history_tuples = []
            for i in range(0, len(chat_history), 2):
                if i+1 < len(chat_history):
                    user_msg = chat_history[i].get("content", "") if chat_history[i].get("role") == "user" else ""
                    assistant_msg = chat_history[i+1].get("content", "") if chat_history[i+1].get("role") == "assistant" else ""
                    if user_msg and assistant_msg:
                        history_tuples.append((user_msg, assistant_msg))
            
            if (model, embedding) not in self.chat_qa_chain_self:
                self.chat_qa_chain_self[(model, embedding)] = Chat_QA_chain_self(
                    model=model,
                    temperature=temperature,
                    top_k=top_k,
                    chat_history=history_tuples,
                    file_path=file_path,
                    persist_path=persist_path,
                    embedding=embedding,
                )
            chain = self.chat_qa_chain_self[(model, embedding)]
            result_history = chain.answer(
                question=question, temperature=temperature, top_k=top_k
            )
            
            # 转换回消息格式
            if result_history and len(result_history) > 0:
                latest_qa = result_history[-1]
                chat_history.append({"role": "user", "content": latest_qa[0]})
                chat_history.append({"role": "assistant", "content": latest_qa[1]})
            
            return "", chat_history
        except Exception as e:
            return e, chat_history

    def qa_chain_self_answer(
        self,
        question: str,
        chat_history: list = [],
        model: str = "qwen3",
        embedding="qwen3",
        temperature: float = 0.0,
        top_k: int = 4,
        file_path: str = DEFAULT_DB_PATH,
        persist_path: str = DEFAULT_PERSIST_PATH,
    ):
        """
        调用不带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding) not in self.qa_chain_self:
                self.qa_chain_self[(model, embedding)] = QA_chain_self(
                    model=model,
                    temperature=temperature,
                    top_k=top_k,
                    file_path=file_path,
                    persist_path=persist_path,
                    embedding=embedding,
                )
            chain = self.qa_chain_self[(model, embedding)]
            answer = chain.answer(question, temperature, top_k)
            
            # 使用新的消息格式
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": answer})
            
            return "", chat_history
        except Exception as e:
            return e, chat_history

    def clear_history(self):
        if len(self.chat_qa_chain_self) > 0:
            for chain in self.chat_qa_chain_self.values():
                chain.clear_history()

    def unified_answer(
        self,
        question: str,
        chat_history: list = [],
        model: str = "openai",
        embedding: str = "openai",
        temperature: float = 0.0,
        top_k: int = 4,
        history_len: int = 3,
        use_knowledge_base: bool = True,
        enable_streaming: bool = False,
        file_path: str = DEFAULT_DB_PATH,
        persist_path: str = DEFAULT_PERSIST_PATH,
    ):
        """
        统一问答接口，优化显示逻辑
        """
        if question == None or len(question) < 1:
            return "", chat_history

        # 立即将用户消息添加到对话历史，先显示用户消息
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": "思考中..."})

        # 记录请求开始
        logger.info("问答请求开始 | 问题: {} | 模型: {} | 知识库: {} | 流式: {}", 
                   question[:50] + "..." if len(question) > 50 else question,
                   model, use_knowledge_base, enable_streaming)
        
        start_time = time.time()
        
        try:
            # 统一使用 Chat_QA_chain_self，无论是否使用知识库
            if (model, embedding) not in self.chat_qa_chain_self:
                logger.debug("创建新的问答链 | 模型: {} | 嵌入: {}", model, embedding)
                # 注意：这里传入的chat_history不包含当前问题，避免重复
                # 转换为元组格式供 Chat_QA_chain_self 使用
                history_tuples = []
                # 处理除了最后两条消息（当前问答对）的历史记录
                valid_history = chat_history[:-2] if len(chat_history) >= 2 else []
                
                for i in range(0, len(valid_history), 2):
                    if i+1 < len(valid_history):
                        user_msg = valid_history[i]
                        assistant_msg = valid_history[i+1]
                        
                        # 确保消息是字典格式且包含必要字段
                        if (isinstance(user_msg, dict) and user_msg.get("role") == "user" and
                            isinstance(assistant_msg, dict) and assistant_msg.get("role") == "assistant"):
                            history_tuples.append((user_msg["content"], assistant_msg["content"]))
                
                self.chat_qa_chain_self[(model, embedding)] = Chat_QA_chain_self(
                    model=model,
                    temperature=temperature,
                    top_k=top_k,
                    chat_history=history_tuples,
                    file_path=file_path if use_knowledge_base else None,
                    persist_path=persist_path if use_knowledge_base else None,
                    embedding=embedding,
                    streaming=enable_streaming,
                    use_knowledge_base=use_knowledge_base,
                )

            chain = self.chat_qa_chain_self[(model, embedding)]
            
            # 更新chain的历史记录（不包含当前问题）
            history_tuples = []
            valid_history = chat_history[:-2] if len(chat_history) >= 2 else []
            
            for i in range(0, len(valid_history), 2):
                if i+1 < len(valid_history):
                    user_msg = valid_history[i]
                    assistant_msg = valid_history[i+1]
                    
                    # 确保消息是字典格式且包含必要字段
                    if (isinstance(user_msg, dict) and user_msg.get("role") == "user" and
                        isinstance(assistant_msg, dict) and assistant_msg.get("role") == "assistant"):
                        history_tuples.append((user_msg["content"], assistant_msg["content"]))
            
            chain.chat_history = history_tuples
            
            # 更新chain的流式设置（如果用户改变了设置）
            chain.streaming = enable_streaming
            chain.callbacks = (
                [StreamingStdOutCallbackHandler()] if enable_streaming else []
            )

            # 调用问答，获取答案
            result_history = chain.answer(
                question=question, 
                temperature=temperature, 
                top_k=top_k, 
                use_knowledge_base=use_knowledge_base
            )
            
            # 从result_history中提取最新的答案
            if result_history and len(result_history) > 0:
                latest_answer = result_history[-1][1]  # 获取最新问答对的答案部分
                # 更新chat_history中的最后一个答案
                chat_history[-1] = {"role": "assistant", "content": latest_answer}
            else:
                chat_history[-1] = {"role": "assistant", "content": "抱歉，没有获得有效回答"}
            
            # 记录成功
            elapsed_time = time.time() - start_time
            logger.success("问答请求完成 | 耗时: {:.2f}s | 答案长度: {}", 
                          elapsed_time, len(str(latest_answer)) if 'latest_answer' in locals() else 0)
            
            return "", chat_history

        except Exception as e:
            # 记录详细错误信息
            elapsed_time = time.time() - start_time
            logger.error("问答请求失败 | 耗时: {:.2f}s | 错误: {}", elapsed_time, str(e))
            logger.exception("详细错误堆栈:")  # 自动记录完整的异常堆栈
            
            # 记录关键参数用于调试
            logger.debug("错误时的参数 | 问题: {} | 模型: {} | 温度: {} | top_k: {}", 
                        question, model, temperature, top_k)
            
            # 更新最后一个答案为错误信息
            chat_history[-1] = {"role": "assistant", "content": f"处理失败: {str(e)}"}
            return "", chat_history


model_center = Model_center()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):

        with gr.Column(scale=2):
            gr.Markdown(
                """<h1><center>Chatomni</center></h1>
                <center>created by 瓦坎达老广</center>
                """
            )

    with gr.Row():
        # 主对话区域
        with gr.Column():
            chatbot = gr.Chatbot(
                height=400,
                show_copy_button=True,
                show_share_button=True,
                avatar_images=(AIGC_AVATAR_PATH, DATAWHALE_AVATAR_PATH),
                type='messages',  # 使用 OpenAI 风格的消息格式
            )

            # 输入框
            msg = gr.Textbox(
                label="输入你的问题",
                placeholder="请在这里输入问题...",
                lines=2,
                max_lines=5,
            )

            # 控制面板
            with gr.Row():
                with gr.Column(scale=2):
                    # 基本功能开关
                    with gr.Group():
                        gr.Markdown("### 功能配置")
                        use_knowledge_base = gr.Checkbox(
                            label="启用知识库",
                            value=False,
                            info="是否使用知识库进行问答",
                        )
                        enable_streaming = gr.Checkbox(
                            label="启用流式问答",
                            value=True,
                            info="开启后将实时显示回答过程",
                        )

                with gr.Column(scale=2):
                    # 模型选择
                    with gr.Group():
                        gr.Markdown("### 模型配置")
                        llm = gr.Dropdown(
                            LLM_MODEL_LIST,
                            label="语言模型",
                            value=INIT_LLM,
                            interactive=True,
                        )
                        embeddings = gr.Dropdown(
                            EMBEDDING_MODEL_LIST,
                            label="向量化模型",
                            value=INIT_EMBEDDING_MODEL,
                        )

                with gr.Column(scale=2):
                    # 参数配置
                    with gr.Group():
                        gr.Markdown("### 参数设置")
                        temperature = gr.Slider(
                            0,
                            1,
                            value=0.01,
                            step=0.01,
                            label="创造性(Temperature)",
                            interactive=True,
                        )
                        top_k = gr.Slider(
                            1,
                            10,
                            value=3,
                            step=1,
                            label="检索数量(Top-K)",
                            interactive=True,
                        )
                        history_len = gr.Slider(
                            0, 5, value=3, step=1, label="历史长度", interactive=True
                        )

            # 操作按钮
            with gr.Row():
                submit_btn = gr.Button("发送", variant="primary", scale=2)
                clear_btn = gr.Button("清空对话", scale=1)

            # 知识库管理（可折叠）
            with gr.Accordion("知识库管理", open=False):
                file = gr.File(
                    label="选择知识库文件/目录",
                    file_count="directory",
                    file_types=[".txt", ".md", ".docx", ".pdf"],
                )
                init_db = gr.Button("向量化知识库")

        # 设置统一的问答事件 - 实现真正的流式输出
        def submit_with_streaming(question, chat_history, model, embedding, temperature, top_k, history_len, use_knowledge_base, enable_streaming):
            """支持真正流式输出的问答函数"""
            if not question or len(question.strip()) < 1:
                yield "", chat_history
                return
            
            # 立即显示用户消息
            chat_history.append({"role": "user", "content": question})
            chat_history.append({"role": "assistant", "content": ""})
            yield "", chat_history
            
            # 记录请求开始
            logger.info("流式问答请求开始 | 问题: {} | 模型: {} | 知识库: {} | 流式: {}", 
                       question[:50] + "..." if len(question) > 50 else question,
                       model, use_knowledge_base, enable_streaming)
            
            try:
                # 获取或创建问答链
                if (model, embedding) not in model_center.chat_qa_chain_self:
                    logger.debug("创建新的问答链 | 模型: {} | 嵌入: {}", model, embedding)
                    # 转换为元组格式供 Chat_QA_chain_self 使用
                    history_tuples = []
                    # 处理 chat_history，确保跳过当前新添加的消息对
                    valid_history = chat_history[:-2] if len(chat_history) >= 2 else []
                    
                    for i in range(0, len(valid_history), 2):
                        if i+1 < len(valid_history):
                            user_msg = valid_history[i]
                            assistant_msg = valid_history[i+1]
                            
                            # 确保消息是字典格式且包含必要字段
                            if (isinstance(user_msg, dict) and user_msg.get("role") == "user" and
                                isinstance(assistant_msg, dict) and assistant_msg.get("role") == "assistant"):
                                history_tuples.append((user_msg["content"], assistant_msg["content"]))
                    
                    model_center.chat_qa_chain_self[(model, embedding)] = Chat_QA_chain_self(
                        model=model,
                        temperature=temperature,
                        top_k=top_k,
                        chat_history=history_tuples,
                        file_path=DEFAULT_DB_PATH if use_knowledge_base else None,
                        persist_path=DEFAULT_PERSIST_PATH if use_knowledge_base else None,
                        embedding=embedding,
                        streaming=enable_streaming,
                        use_knowledge_base=use_knowledge_base,
                    )

                chain = model_center.chat_qa_chain_self[(model, embedding)]
                # 更新历史记录
                history_tuples = []
                valid_history = chat_history[:-2] if len(chat_history) >= 2 else []
                
                for i in range(0, len(valid_history), 2):
                    if i+1 < len(valid_history):
                        user_msg = valid_history[i]
                        assistant_msg = valid_history[i+1]
                        
                        # 确保消息是字典格式且包含必要字段
                        if (isinstance(user_msg, dict) and user_msg.get("role") == "user" and
                            isinstance(assistant_msg, dict) and assistant_msg.get("role") == "assistant"):
                            history_tuples.append((user_msg["content"], assistant_msg["content"]))
                
                chain.chat_history = history_tuples
                chain.streaming = enable_streaming
                
                if enable_streaming:
                    # 使用流式输出（支持知识库和非知识库模式）
                    try:
                        for partial_answer in chain.answer_stream(
                            question=question, 
                            temperature=temperature, 
                            top_k=top_k, 
                            use_knowledge_base=use_knowledge_base
                        ):
                            # 实时更新最后一条消息的AI回复
                            chat_history[-1] = {"role": "assistant", "content": partial_answer}
                            yield "", chat_history
                    except Exception as stream_e:
                        logger.error("流式输出过程中出错: {}", str(stream_e))
                        # 如果流式输出失败，回退到非流式模式
                        result_history = chain.answer(
                            question=question, 
                            temperature=temperature, 
                            top_k=top_k, 
                            use_knowledge_base=use_knowledge_base
                        )
                        
                        if result_history and len(result_history) > 0:
                            latest_answer = result_history[-1][1]
                            chat_history[-1] = {"role": "assistant", "content": latest_answer}
                        else:
                            chat_history[-1] = {"role": "assistant", "content": "抱歉，没有获得有效回答"}
                        
                        yield "", chat_history
                else:
                    # 非流式输出
                    result_history = chain.answer(
                        question=question, 
                        temperature=temperature, 
                        top_k=top_k, 
                        use_knowledge_base=use_knowledge_base
                    )
                    
                    if result_history and len(result_history) > 0:
                        latest_answer = result_history[-1][1]
                        chat_history[-1] = {"role": "assistant", "content": latest_answer}
                    else:
                        chat_history[-1] = {"role": "assistant", "content": "抱歉，没有获得有效回答"}
                    
                    yield "", chat_history
                
                logger.success("流式问答请求完成")
                
            except Exception as e:
                logger.error("流式问答请求失败 | 错误: {}", str(e))
                logger.exception("详细错误堆栈:")
                chat_history[-1] = {"role": "assistant", "content": f"处理失败: {str(e)}"}
                yield "", chat_history

        submit_btn.click(
            submit_with_streaming,
            inputs=[
                msg,          # 这是输入的问题文本
                chatbot,      # 聊天历史
                llm,          # 模型选择
                embeddings,   # 嵌入模型
                temperature,  # 温度参数
                top_k,        # top_k参数
                history_len,  # 历史长度
                use_knowledge_base,  # 是否使用知识库
                enable_streaming,    # 是否启用流式输出
            ],
            outputs=[msg, chatbot],
            show_progress="minimal",
        )

        # Enter键提交
        msg.submit(
            submit_with_streaming,
            inputs=[
                msg,          # 这是输入的问题文本
                chatbot,      # 聊天历史
                llm,          # 模型选择
                embeddings,   # 嵌入模型
                temperature,  # 温度参数
                top_k,        # top_k参数
                history_len,  # 历史长度
                use_knowledge_base,  # 是否使用知识库
                enable_streaming,    # 是否启用流式输出
            ],
            outputs=[msg, chatbot],
            show_progress="hidden",
        )

        # 清空对话
        clear_btn.click(lambda: ([], None), outputs=[chatbot, msg])
        clear_btn.click(model_center.clear_history)

        # 初始化知识库
        init_db.click(create_db_info, inputs=[file, embeddings], outputs=[msg])


# threads to consume the request
gr.close_all()

logger.info("启动 Chatomni 服务 | 端口: 8001")
logger.info("可用模型: {}", LLM_MODEL_LIST)
logger.info("默认模型: {} | 默认嵌入模型: {}", INIT_LLM, INIT_EMBEDDING_MODEL)

demo.launch(server_port=8001)
