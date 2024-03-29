# LlamaIndex-Study
LlamaIndex study note.

# 1_LlamaIndex入门
LlamaIndex是一个应用于LLM应用程序的数据框架，旨在帮助开发者通过大型语言模型（LLM）来摄取、构建和访问私有或特定领域的数据。它允许用户将外部数据源与LLM相结合，以便在处理文本数据时获得更好的性能和效率。

该框架的核心组件包括数据连接器（Data Connectors）、数据索引（Data Indexs）、数据代理（Data Agents）、引擎（Engines）、应用程序集成（Application Integrations）。

LlamaIndex支持多种索引类型，包括向量索引、列表索引和树形索引等，每种索引都针对特定的数据类型和查询需求进行了优化。此外，它还提供了强大的查询接口，使得用户能够通过自然语言与数据交互。这意味着用户不仅可以查询索引，还可以自定义提示组合方式，以实现更精确和个性化的数据检索。

除了作为数据处理工具之外，LlamaIndex还提供了类似于LangChain的功能，但更加专注于数据处理。它可以帮助开发者构建诸如文档问答系统这样的应用，让用户能够根据问题自动查找相关文档并生成答案，这在知识管理、在线客服和自动化问答等领域具有广泛应用。

LlamaIndex为开发者提供了一个强大而灵活的工具集，使他们能够利用大型语言模型的强大功能来构建和处理各种自然语言处理应用程序。

## v0.10最大的变化（v0.10版本大概200万行代码）
* 创建了"llama-index-core"包，此文件夹包含所有核心 LlamaIndex 抽象
* 将所有集成(integrations)和模板(templates)拆分为单独的包，更加解耦、干净且不易损坏
* 弃用ServiceContext，它笨重且不透明，不知道默认参数是什么
* 其他分离的包：
  - llama-index-core（此文件夹包含所有核心 LlamaIndex 抽象）
  - llama-index-integrations（此文件夹包含 19 个 LlamaIndex 抽象的第三方集成。这包括数据加载器、LLM、嵌入模型、向量存储等）
  - llama-index-finetuning（包含 LlamaIndex 微调抽象。这些仍然是相对实验性的）
  - llama-index-experimental（包含实验性功能。目前基本上未使用（外部参数调整））
  - llama-index-cli（LlamaIndex工具）
  - llama-index-legacy（包含旧版 LlamaIndex 代码，以防用错）
  - llama-index-packs（此文件夹包含我们的 50 多个 LlamaPack，这些模板旨在启动用户的应用程序）

## 文件夹结构
LlamaIndex的文件夹结构比较独特，在导入时保留了LlamaIndex的namespace。
integrations和中的子目录packs代表各个包。文件夹的名称与包名称相对应。例如llama-index-integrations/llms/llama-index-llms-gemini对应llama-index-llms-geminiPyPI包。
在每个包文件夹中，源文件排列在您用于导入它们的相同路径中。例如，在 Gemini LLM 包中，您将看到一个名为 的文件夹，其中llama_index/llms/gemini包含源文件。此文件夹结构允许您在导入期间保留顶级llama_index命名空间。对于 Gemini LLM，您可以 pip 安装llama-index-llms-gemini，然后使用from llama_index.llms.gemini import Gemini
这些子文件夹中的每一个还具有打包所需的资源：pyproject.toml、poetry.lock和 a Makefile，以及自动创建包的脚本。

## 集成
所有第三方集成现在都在llama-index-integrations.这里有 19 个文件夹。主要集成类别有：
* llms
* embeddings
* multi_modal_llms
* readers
* tools
* vector_stores
为了完整起见，这里列出了所有其他类别：agent, callbacks, evaluation, extractors, graph_stores, indices, output_parsers, postprocessor, program, question_gen, response_synthesizers, retrievers, storage, tools。

## LlamaIndex生态系统（ecosystem）
* 150+ data loaders
* 35+ agent tools
* 50+ LlamaPack templates
* 50+ LLMs
* 25+ embeddings
* 40+ vector stores

但是，也有一些痛点，比如所有的集成都缺乏正确的测试，这些集成都被纳入了主包，依赖项中的任何更新，您都必须更新版本

## 五大核心组件
包括数据连接器（Data Connectors）、数据索引（Data Indexs）、数据代理（Data Agents）、引擎（Engines）、应用程序集成（Application Integrations）
* Data Connectors: 负责从多种原生数据源中读取数据，如API、PDF文件、SQL数据库等注入LlamaIndex
* Data Indexs: 将原生数据转换为不同的索引类型（Summary、Vector Store、Tree等），这种索引类型既便于LLMs处理，又能保持高效的性能
* Data Agents: 
* Engines: 一种是Query引擎、一种是Chat引擎
* Application Integrations: 对接生态中的框架、应用、接口等，比如Langchain、ChatGPT

## Usage Example
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

## RAG 中的五个关键阶段
RAG 中有五个关键阶段，这将成为您构建任何大型应用程序的一部分：
* 加载：这是指将数据从其所在位置（无论是文本文件、PDF、其他网站、数据库还是 API）获取到您的管道中。LlamaHub提供数百种连接器可供选择。
* 索引：这意味着创建一个允许查询数据的数据结构。对于LLMs来说，这几乎总是意味着创建向量嵌入、数据含义的数字表示，以及许多其他元数据策略，以便轻松准确地找到上下文相关的数据。
* 存储：一旦数据被索引，您几乎总是希望存储索引以及其他元数据，以避免重新索引。
* 查询：对于任何给定的索引策略，您可以通过多种方式利用 LLM 和 LlamaIndex 数据结构进行查询，包括子查询、多步查询和混合策略。
* 评估：任何管道中的关键步骤是检查它相对于其他策略的有效性，或者何时进行更改。评估提供客观衡量您对查询的答复的准确性、忠实度和速度的程度。
![image](https://github.com/txjlrk/LlamaIndex-Study/assets/8086669/a335f526-1d54-4865-9554-48090a044be1)

## 构建生产级 RAG 的通用技术
* 解耦用于检索的块与用于合成的块
* 较大文档集的结构化检索
* 根据您的任务动态检索块
* 优化上下文嵌入

### 通用技术1：解耦用于检索的块与用于合成的块
将用于检索的块与用于合成的块分离。
![image](https://github.com/txjlrk/LlamaIndex-Study/assets/8086669/ce489174-d779-4a7a-93dc-4c294da87a61)
#### 为什么要这么搞？ 
用于检索的最佳块表示可能与用于合成的最佳考虑不同。例如，原始文本块可能包含LLMs所需的详细信息，以根据查询合成更详细的答案。然而，它可能包含可能使嵌入表示产生偏差的填充词/信息，或者它可能缺乏全局上下文，并且当相关查询出现时根本不会被检索。
#### 关键技术
1. 嵌入文档摘要，该摘要链接到与文档关联的块。
这可以帮助在检索块之前检索高级相关文档，而不是直接检索块（可能在不相关的文档中）。
2. 嵌入一个句子，然后链接到该句子周围的窗口。
这允许更细粒度地检索相关上下文（嵌入巨大的块会导致“中间丢失”问题），但也确保了 LLM 合成有足够的上下文。

### 通用技术2：较大文档集的结构化检索
![image](https://github.com/txjlrk/LlamaIndex-Study/assets/8086669/6d420fab-fe70-4cc5-9e5c-4b9bf6e33ad5)
#### 为什么要这么搞？ 
标准 RAG 堆栈（top-k 检索 + 基本文本分割）的一个大问题是，它随着文档数量的增加而表现不佳 - 例如，如果您有 100 个不同的 PDF。在此设置中，给定一个查询，您可能希望使用结构化信息来帮助更精确的检索；例如，如果您提出一个仅与两个 PDF 相关的问题，请使用结构化信息来确保返回的这两个 PDF 超出了与块的原始嵌入相似性。
#### 关键技术
有几种方法可以为生产质量的 RAG 系统执行更结构化的标记/检索，每种方法都有自己的优缺点。
1. 元数据过滤器+自动检索 用元数据标记每个文档，然后存储在矢量数据库中。在推理期间，使用 LLM 推断正确的元数据过滤器，以除了语义查询字符串之外还查询向量数据库。
优点 ✅：主要矢量数据库支持。可以通过多个维度过滤文档。
缺点🚫：很难定义正确的标签。标签可能没有包含足够的相关信息以进行更精确的检索。此外，标签表示文档级别的关键字搜索，不允许语义查找。
2. 存储文档层次结构（摘要 -> 原始块）+递归检索 嵌入文档摘要并映射到每个文档的块。首先在文档级别获取，然后在块级别获取。
优点 ✅：允许在文档级别进行语义查找。
缺点🚫：不允许通过结构化标签进行关键字查找（可以比语义搜索更精确）。自动生成摘要的成本也可能很高。

### 通用技术3：根据您的任务动态检索块
![image](https://github.com/txjlrk/LlamaIndex-Study/assets/8086669/75e8da1a-2e42-430d-893d-92fc08f70943)
#### 为什么要这么搞？ 
RAG 不仅仅是针对特定事实进行问答，还针对其中的 top-k 相似度进行了优化。用户可能会提出各种各样的问题。由幼稚的 RAG 堆栈处理的查询包括询问具体事实的查询，例如“告诉我这家公司 2023 年的 D&I 计划”或“叙述者在 Google 期间做了什么”。但查询还可以包括摘要，例如“您能给我一份该文档的高级概述吗”，或比较“您可以比较/对比 X 和 Y”吗？所有这些用例可能需要不同的检索技术。
#### 关键技术
LlamaIndex 提供了一些核心抽象来帮助您进行特定于任务的检索。这包括我们的路由器模块以及数据代理模块。这还包括一些高级查询引擎模块。这还包括连接结构化和非结构化数据的其他模块。
您可以使用这些模块进行联合问答和摘要，甚至将结构化查询与非结构化查询结合起来。
#### 核心模块资源
* 查询引擎
* 代理商
* 路由器
#### 详细指南资源
* 子问题查询引擎
* 联合 QA 摘要查询引擎
* 递归检索器+文档代理
* 路由器查询引擎
* OpenAI Agent + 查询引擎实验手册
* OpenAI 代理查询规划

### 通用技术4：优化上下文嵌入
#### 为什么要这么搞？ 
这与上面“解耦用于检索与合成的块”中描述的动机有关。我们希望确保嵌入经过优化，以便更好地检索您的特定数据集。预先训练的模型可能无法捕获与您的用例相关的数据的显着属性。
#### 关键技术
除了上面列出的一些技术之外，我们还可以尝试微调嵌入模型。实际上，我们可以通过无标签的方式在非结构化文本语料库上做到这一点。

## Usage Example
### Sub Question Query Engine
使用子问题查询引擎来解决使用多个数据源回答复杂查询的问题。
它首先将复杂的查询分解为每个相关数据源的子问题，然后收集所有中间响应并合成最终响应。
```
# load data
pg_essay = SimpleDirectoryReader(input_dir="./data/paul_graham/").load_data()

# build index and query engine
vector_query_engine = VectorStoreIndex.from_documents(
    pg_essay,
    use_async=True,
).as_query_engine()

# setup base query engine as tool
query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="pg_essay",
            description="Paul Graham essay on What I Worked On",
        ),
    ),
]

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    use_async=True,
)

response = query_engine.query(
    "How was Paul Grahams life different before, during, and after YC?"
)

print(response)

### 运行后会生成3个问题
Generated 3 sub questions.
[pg_essay] Q: What did Paul Graham work on before YC?
[pg_essay] Q: What did Paul Graham work on during YC?
[pg_essay] Q: What did Paul Graham work on after YC?
[pg_essay] A: After YC, Paul Graham worked on starting his own investment firm with Jessica.
[pg_essay] A: During his time at YC, Paul Graham worked on various projects. He wrote all of YC's internal software in Arc and also worked on Hacker News (HN), which was a news aggregator initially meant for startup founders but later changed to engage intellectual curiosity. Additionally, he wrote essays and worked on helping the startups in the YC program with their problems.
[pg_essay] A: Paul Graham worked on writing essays and working on YC before YC.
```

### Finetune Embeddings微调嵌入带来的性能提升
我们可以使用LlamaIndex的generate_qa_embedding_pairs把数据集存储到EmbeddingQAFinetuneDataset
```
# 生成语料库
from llama_index.finetuning import generate_qa_embedding_pairs
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.llms.openai import OpenAI

train_dataset = generate_qa_embedding_pairs(
    llm=OpenAI(model="gpt-3.5-turbo"), nodes=train_nodes
)
val_dataset = generate_qa_embedding_pairs(
    llm=OpenAI(model="gpt-3.5-turbo"), nodes=val_nodes
)

train_dataset.save_json("train_dataset.json")
val_dataset.save_json("val_dataset.json")

# 加载数据
train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

from llama_index.finetuning import SentenceTransformersFinetuneEngine
finetune_engine = SentenceTransformersFinetuneEngine(
    train_dataset,
    model_id="BAAI/bge-small-en",
    model_output_path="llm_model_ft",
    val_dataset=val_dataset,
)
finetune_engine.finetune()
embed_model = finetune_engine.get_finetuned_model()

# 评估器
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
from pathlib import Path


def evaluate_st(
    dataset,
    model_id,
    name,
):
    corpus = dataset.corpus
    queries = dataset.queries
    relevant_docs = dataset.relevant_docs

    evaluator = InformationRetrievalEvaluator(
        queries, corpus, relevant_docs, name=name
    )
    model = SentenceTransformer(model_id)
    output_path = "results/"
    Path(output_path).mkdir(exist_ok=True, parents=True)
    return evaluator(model, output_path=output_path)

# 运行评估
ada = OpenAIEmbedding()
ada_val_results = evaluate(val_dataset, ada)
df_ada = pd.DataFrame(ada_val_results)
hit_rate_ada = df_ada["is_hit"].mean()
hit_rate_ada
0.8779904306220095

bge = "local:BAAI/bge-small-en"
bge_val_results = evaluate(val_dataset, bge)
df_bge = pd.DataFrame(bge_val_results)
hit_rate_bge = df_bge["is_hit"].mean()
hit_rate_bge
0.7930622009569378

# Finetuned
finetuned = "local:llm_model_ft"
val_results_finetuned = evaluate(val_dataset, finetuned)
df_finetuned = pd.DataFrame(val_results_finetuned)
hit_rate_finetuned = df_finetuned["is_hit"].mean()
hit_rate_finetuned
evaluate_st(val_dataset, "llm_model_ft", name="finetuned")
```

### Setence Windows Retrieval （句子窗口检索）带来的性能提升
LlamaIndex框架中的SentenceWindowNodeParser是一种用于将文档分割成句子的解析器，其底层原理基于自然语言处理（NLP）技术，特别是分句技术。这种解析器通常利用正则表达式、自然语言处理库（例如NLTK或spaCy）等方法来识别文本中的句子边界。
在具体的实现过程中，SentenceWindowNodeParser通过分析文本中的标点符号、特殊字符以及单词的排列来判定句子的结束位置。例如，句号（.）、问号（？）、感叹号（！）等标点符号通常是句子结束的标识。除此之外，该解析器还可能考虑到半角和全角的标点符号使用差异，因为在中文文档中常常使用全角的标点符号，而在英文或其他西方语言中通常使用半角的标点符号。
SentenceWindowNodeParser不仅将文本分割成句子，还将每个句子视为一个独立的“节点”（Node），每个节点包含了自身的文本内容和一些元数据信息，如句子的ID、父节点的引用、前置和后续的句子等。这些信息有助于在检索和生成回复时，能够提供更大的上下文环境。
​LlamaIndex的从小到大的检索中我们都是按指定的块大小(chunk_size)来对文档进行切割的，然而“句子-窗口检索”方法中我们将不再按chunk_size来切割文档，而是按完整的句子来切割文档即每一个句子切割成一个文档，然而如何识别出文本中的句子呢？在LlamaIndx中采样的是通过句尾的标点符号如句号(.), 问号(?), 感叹号(!)等来识别句子。
Setence Windows Retrieval根据查询找到最相关的句子，然后根据这个句子周围的窗口添加上下文，然后将该上下文内容用于LLM，因为我们最想找的信息很有可能就在这个上下文中。
```
from llama_index.node_parser import SentenceWindowNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
 
#定义句子解析器
# create the sentence window node parser w/ default settings
self.node_parser = SentenceWindowNodeParser.from_defaults(
    window_size=3,
    window_metadata_key="window",
    original_text_metadata_key="original_text",
)

self.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
self.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-mpnet-base-v2", max_length=512
)
self.service_context = ServiceContext.from_defaults(
    llm=self.llm,
    embed_model=self.embed_model,
)
# extract nodes
nodes = self.node_parser.get_nodes_from_documents(docs)
self.sentence_index = VectorStoreIndex(
    nodes, service_context=self.service_context
)
self.postprocessor = MetadataReplacementPostProcessor(
    target_metadata_key="window"
)
self.query_engine = self.sentence_index.as_query_engine(
    similarity_top_k=2,
    # the target key defaults to `window` to match the node_parser's default
    node_postprocessors=[self.postprocessor],
)

```










