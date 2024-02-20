# LlamaIndex-Study
LlamaIndex study note.

# 1_LlamaIndex入门
LlamaIndex是一个人工智能数据框架，旨在帮助开发者通过大型语言模型（LLM）来摄取、构建和访问私有或特定领域的数据。它允许用户将外部数据源与LLM相结合，以便在处理文本数据时获得更好的性能和效率。

该框架的核心组件包括数据连接器、数据指标、引擎、数据代理、应用程序集成。数据连接器负责从多种原生数据源中读取数据，如API、PDF文件、SQL数据库等。一旦数据被连接器获取，接下来就是构建索引的过程，这涉及到将数据转换为一种中间表示形式，这种形式既便于LLMs处理，又能保持高效的性能。

LlamaIndex支持多种索引类型，包括向量索引、列表索引和树形索引等，每种索引都针对特定的数据类型和查询需求进行了优化。此外，它还提供了强大的查询接口，使得用户能够通过自然语言与数据交互。这意味着用户不仅可以查询索引，还可以自定义提示组合方式，以实现更精确和个性化的数据检索。

除了作为数据处理工具之外，LlamaIndex还提供了类似于LangChain的功能，但更加专注于数据处理。它可以帮助开发者构建诸如文档问答系统这样的应用，让用户能够根据问题自动查找相关文档并生成答案，这在知识管理、在线客服和自动化问答等领域具有广泛应用。

LlamaIndex为开发者提供了一个强大而灵活的工具集，使他们能够利用大型语言模型的强大功能来构建和处理各种自然语言处理应用程序。

## v0.10最大的变化（v0.10版本大概200万行代码）
* 创建了"llama-index-core"包，这个包包含了所有的核心抽象
* 将所有集成(integrations)和模板(templates)拆分为单独的包，更加解耦、干净且不易损坏
* 弃用ServiceContext，它笨重且不透明，不知道默认参数是什么
* 其他分离的包：
  - llama-index-integrations（集成的所有第三方包）
  - llama-index-finetuning（微调）
  - llama-index-experimental
  - llama-index-cli（LlamaIndex工具）
  - llama-index-legacy（v0.9的代码都在这个包里，以防用错）
  - llama-index-packs
 
## 文件夹结构
LlamaIndex的文件夹结构比较独特，在导入时保留了LlamaIndex的namespace

## LlamaIndex生态系统（ecosystem）
* 150+ data loaders
* 35+ agent tools
* 50+ LlamaPack templates
* 50+ LLMs
* 25+ embeddings
* 40+ vector stores

但是，也有一些痛点，比如所有的集成都缺乏正确的测试，这些集成都被纳入了主包，依赖项中的任何更新，您都必须更新版本

# Usage Example
```
import os

os.environ["REPLICATE_API_TOKEN"] = "YOUR_REPLICATE_API_TOKEN"

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
from transformers import AutoTokenizer

# set the LLM
llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
Settings.llm = Replicate(
    model=llama2_7b_chat,
    temperature=0.01,
    additional_kwargs={"top_p": 1, "max_new_tokens": 300},
)

# set tokenizer to match LLM
Settings.tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf"
)

# set the embed model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

documents = SimpleDirectoryReader("YOUR_DATA_DIRECTORY").load_data()
index = VectorStoreIndex.from_documents(
    documents,
)
query_engine = index.as_query_engine()
query_engine.query("YOUR_QUESTION")
```
# 2_LlamaIndex高级
## 检索增强生成 (RAG) 
![image](https://github.com/txjlrk/LlamaIndex-Study/assets/8086669/6bb120da-e052-4297-b1ca-b690ff2fb644)
RAG 的主要用途是为了给生成式 AI 输出的文本提供支撑。
换言之，RAG 就是通过事实、自定义数据以减少 LLM 幻觉。具体而言，在 RAG 中，我们可以使用可靠、可信的自定义数据文本，如产品文档，随后从向量数据库中检索相似结果。然后，将准确的文本答案作为“上下文”和“问题”一起插入到“Prompt”中，并将其输入到诸如 OpenAI 的 ChatGPT 之类的 LLM 中。最终，LLM 生成一个基于事实的聊天答案。
