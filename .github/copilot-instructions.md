# LLM Universe 项目结构说明

## 项目概述
这是一个基于大语言模型的知识管理系统，主要用于处理GitHub仓库的README文件，生成知识库和摘要。

## 项目文件结构

```
llm-universe/
├── .env                          # 环境变量配置文件，包含API密钥等敏感信息
├── config.py                     # 项目配置文件
├── .github/
│   └── instruction               # 项目说明文档（本文件）
├── .vscode/
│   └── launch.json               # VSCode调试配置
├── database/                     # 数据库相关模块
│   ├── create_db.py              # 数据库创建脚本
│   ├── test_get_all_repo.py      # GitHub仓库数据获取工具
│   ├── text_summary_readme.py    # README文件摘要生成工具
│   └── readme_db/                # 存储从GitHub获取的README文件
│       ├── repositories.txt      # 仓库列表文件
│       └── {repo_name}/          # 各个仓库的README文件目录
│           └── README.md         # 具体仓库的README文件
├── embedding/                    # 向量化处理模块
│   └── call_embedding.py         # 文本向量化处理工具
├── knowledge_db/                 # 知识库存储目录
├── llm/                          # 大语言模型调用模块
│   └── call_llm.py               # 大语言模型API调用工具
├── qa_chain/                     # 问答链模块
│   ├── Qa_chain_self.py          # 问答链处理工具
│   ├── Chat_Qa_chain_self.py     # 带历史记录的问答链处理工具
│   └── get_vectordb.py           # 向量数据库处理工具
└── README.md                     # 项目说明文档
```

## 主要模块说明

### 1. database/ 目录
- **create_db.py**: 数据库创建和初始化脚本
- **test_get_all_repo.py**: 
  - 从GitHub API获取指定组织的所有仓库信息
  - 下载各仓库的README文件到本地
  - 生成仓库列表文件
- **text_summary_readme.py**: 
  - 使用大语言模型对README文件进行摘要生成
  - 支持批量处理多个仓库的README文件
  - 生成中文摘要并保存到指定目录
- **readme_db/**: 存储从GitHub下载的README文件和仓库信息

### 2. llm/ 目录
- **call_llm.py**: 
  - 封装大语言模型API调用逻辑
  - 支持多种模型(如通义千问、OpenAI等)
  - 提供统一的调用接口

### 3. embedding/ 目录
- **call_embedding.py**: 
  - 文本向量化处理工具
  - 支持将文本转换为向量表示
  - 用于后续的语义搜索和相似度计算

### 4. knowledge_db/ 目录
- 存储处理后的知识库文件
- 包含向量化后的文档数据

## 主要功能流程

1. **数据获取**: 使用`test_get_all_repo.py`从GitHub获取组织的所有仓库README文件
2. **文本处理**: 使用`text_summary_readme.py`对README文件进行清洗和摘要生成
3. **向量化**: 使用`call_embedding.py`将文本转换为向量表示
4. **知识库构建**: 将处理后的数据存储到知识库中
5. **API调用**: 通过`call_llm.py`调用大语言模型进行各种NLP任务


## 使用说明

1. 配置环境变量文件`.env`
2. 运行`test_get_all_repo.py`获取GitHub仓库数据
3. 运行`text_summary_readme.py`生成README摘要
4. 使用其他模块进行向量化和知识库构建
