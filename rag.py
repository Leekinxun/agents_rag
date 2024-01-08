# %pip install pyautogen[retrievechat] langchain "chromadb<0.4.15" cohere -q

import autogen # 大家可以使用gpt-4 或其它，我这里用的是3.5, 还能用。
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter
from autogen.agentchat.contrib.retrieve_assistant_agent import RetrieveAssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent

config_list = [ { 'model': 'gpt-3.5-turbo-16k', 'api_key': '' }]
llm_config={ "seed": 42, #为缓存做的配置
            "config_list": config_list }

# 从chromadb数据库中引入embedding_functions
# 调用OpenAIEmbeddingFunction
huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key="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",
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\r", "\t"])

llm_config = {
   # "request_timeout": 600,
    "config_list": config_list,
    "temperature": 0
}

assistant = RetrieveAssistantAgent(
    name="assistant",
    system_message="You are a helpful assistant.",
    llm_config=llm_config,
)

rag_agent = RetrieveUserProxyAgent(
    human_input_mode="ALWAYS",
    retrieve_config={
        "task": "qa",
        "docs_path": "/content/DPO.pdf",
        "collection_name": "rag_collection",
        "embedding_function": huggingface_ef,
        "custom_text_split_function": text_splitter.split_text,
    },
)

assistant.reset()
rag_agent.initiate_chat(assistant, problem="这篇论文讲了什么?", n_results=2)
