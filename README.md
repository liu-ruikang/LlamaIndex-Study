# LlamaIndex-Study
LlamaIndex study note.

# 1_LlamaIndex概述
LlamaIndex是一个数据框架，旨在帮助开发者通过大型语言模型（LLM）来摄取、构建和访问私有或特定领域的数据。它允许用户将外部数据源与LLM相结合，以便在处理文本数据时获得更好的性能和效率。

该框架的核心组件包括数据连接器、数据指标、引擎、数据代理、应用程序集成。数据连接器负责从多种原生数据源中读取数据，如API、PDF文件、SQL数据库等。一旦数据被连接器获取，接下来就是构建索引的过程，这涉及到将数据转换为一种中间表示形式，这种形式既便于LLMs处理，又能保持高效的性能。

LlamaIndex支持多种索引类型，包括向量索引、列表索引和树形索引等，每种索引都针对特定的数据类型和查询需求进行了优化。此外，它还提供了强大的查询接口，使得用户能够通过自然语言与数据交互。这意味着用户不仅可以查询索引，还可以自定义提示组合方式，以实现更精确和个性化的数据检索。

除了作为数据处理工具之外，LlamaIndex还提供了类似于LangChain的功能，但更加专注于数据处理。它可以帮助开发者构建诸如文档问答系统这样的应用，让用户能够根据问题自动查找相关文档并生成答案，这在知识管理、在线客服和自动化问答等领域具有广泛应用。

LlamaIndex为开发者提供了一个强大而灵活的工具集，使他们能够利用大型语言模型的强大功能来构建和处理各种自然语言处理应用程序。

v0.10最大的变化：
* 创建了"llama-index-core"包
* 将所有集成(integrations)和模板(templates)拆分为单独的包，更加解耦、干净且不易损坏
* 弃用ServiceContext
