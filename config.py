import os

PROJECT_ROOT = os.getcwd()

DATABASE_DIR = os.path.join(PROJECT_ROOT, 'database')
README_DB_DIR = os.path.join(DATABASE_DIR, 'readme_db')
KNOWLEDGE_DB_DIR = os.path.join(PROJECT_ROOT, 'knowledge_db')
SUMMARY_DIR = os.path.join(KNOWLEDGE_DB_DIR, 'readme_summary')
VECTOR_DB_DIR = os.path.join(PROJECT_ROOT, 'vector_db')

HF_MODEL_DIR = "/root/Code/hf_model"
QWEN3_EMBEDDING_MODEL_PATH = os.path.join(HF_MODEL_DIR, "Qwen3-Embedding-8B")
