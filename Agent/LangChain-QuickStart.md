# LangChain / LangGraph 入门实践 - 面试版

本文整理了 LangChain/LangGraph **最常用的基础用法和代码示例**，应付面试中"给个简单例子"这类问题足够用了。

---

## 目录

## LangChain 部分（全部放前面）
- [环境安装](#环境安装)
- [第一个 LangChain 程序](#第一个-langchain-程序)
- [核心概念快速过](#核心概念快速过)
- [Prompt 模板](#prompt-模板)
- [输出解析器](#输出解析器)
- [LCEL 链式调用](#lcel-链式调用)
- [Memory 记忆组件](#memory-记忆组件)
- [工具调用 & Agent 创建](#工具调用--agent-创建)
- [简单 RAG 示例](#简单-rag-示例)
- [面试常考代码片段](#面试常考代码片段)
- [常用工具列表](#常用工具-list)

## LangGraph 部分（全部放后面）
- [LangGraph 核心概念](#langgraph-核心概念)
- [最简单的 Agent 例子](#最简单的-agent-例子)
- [多 Agent 协作示例](#多-agent-协作示例)
- [状态持久化 & 断点恢复](#状态持久化--断点恢复)
- [人类在环（Human-in-the-loop）](#人类在环human-in-the-loop)
- [工具调用循环完整例子](#工具调用循环完整例子)

## 其他
- [面试回答套路](#面试回答套路)

---

---

## 环境安装

```bash
# 安装核心 LangChain
pip install langchain

# 安装核心接口（Runnable 等）
pip install langchain-core

# 安装 OpenAI 支持
pip install langchain-openai

# 安装 LangGraph
pip install langgraph

# 安装向量库（示例用 Chroma）
pip install langchain-community chromadb

# 安装文档加载
pip install unstructured
```

---

## 第一个 LangChain 程序

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 初始化大模型
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 2. 创建 Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个专业的{role}，请用简洁语言回答问题。"),
    ("human", "{question}")
])

# 3. LCEL 组装链
chain = prompt | llm | StrOutputParser()

# 4. 运行
result = chain.invoke({
    "role": "Python讲师",
    "question": "什么是装饰器？"
})

print(result)
```

**这就是最基本的 LangChain 程序**：`Prompt → LLM → OutputParser`，用 `|` 拼接就是 LCEL。

---

## 核心概念快速过

| 概念 | 作用 |
|------|------|
| `Runnable` | 所有组件的抽象接口，有 `invoke/ainvoke/stream/batch` 方法 |
| `LCEL` | LangChain 表达式语言，用 `|` 把多个 Runnable 拼成一个链 |
| `ChatPromptTemplate` | Prompt 模板，动态填充变量 |
| `OutputParser` | 把大模型输出转成结构化格式（JSON、列表等） |
| `Memory` | 记忆组件，存储对话历史 |
| `Tool` | 封装大模型可以调用的工具 |
| `AgentExecutor` | LangChain 运行 Agent 的执行器 |
| `StateGraph` | LangGraph 的核心图结构 |

---

## Prompt 模板

**用法：** 动态生成 Prompt，填充变量。

```python
from langchain_core.prompts import ChatPromptTemplate

# 从消息创建
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{job}，回答要符合身份。"),
    ("user", "请回答：{question}")
])

# 填充变量
messages = prompt.format_messages(job="客服", question="我的订单在哪里？")
```

**还有几种写法：**
```python
# 从模板字符串创建
prompt = ChatPromptTemplate.from_template(
    "给我讲一个关于{topic}的笑话"
)

# 消息占位符（适合放对话历史）
prompt = ChatPromptTemplate.from_messages([
    ("system", "你好"),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])
```

---

## 输出解析器

**用法：** 把大模型的文本输出转成结构化数据。

**1. 简单字符串输出：**
```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
# 直接得到字符串
result = parser.invoke(llm_response)
```

**2. JSON 输出（Pydantic）：**
```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class UserInfo(BaseModel):
    name: str = Field(description="用户名")
    age: int = Field(description="年龄")

parser = PydanticOutputParser(pydantic_object=UserInfo)

# 在 prompt 里告诉模型输出格式
prompt = ChatPromptTemplate.from_template("""
提取用户信息。
{format_instructions}
用户输入：{user_input}
""").partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
result = chain.invoke({"user_input": "我叫张三，25岁"})
# result 就是 UserInfo 对象，可以 result.name 这样用
```

**3. 列表输出：**
```python
from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()
# 输出转成 Python 列表
```

---

## LCEL 链式调用

LCEL（LangChain Expression Language）是 LangChain 推荐的组件组装方式，用 `|` 符号把多个组件串起来，数据从左流到右。

下面给几个完整可运行的例子，看清楚每一步流程：

---

**1. 最基本的顺序链（最常用）**

完整流程：用户输入 → Prompt 模板填充 → 大模型调用 → 输出解析 → 得到结果

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 第一步：定义每个组件
llm = ChatOpenAI(model="gpt-3.5-turbo")
prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
parser = StrOutputParser()

# 第二步：用 | 组装成链，数据从左往右流
# 输入：{"topic": "猫"} → prompt → llm → parser → 输出
chain = prompt | llm | parser

# 第三步：调用
result = chain.invoke({"topic": "猫"})
print(result)  # 直接输出字符串笑话
```

**每一步发生了什么：**
1. `invoke({"topic": "猫"})` 把输入传给 `prompt`
2. `prompt` 填充变量，得到完整 Prompt：`"给我讲一个关于猫的笑话"`
3. `llm` 调用大模型，得到 ChatResult
4. `parser` 从 ChatResult 中提取出字符串
5. 返回最终结果

---

**2. 并行调用：同时多个分支，拿到所有结果**

场景：同一个输入，同时生成笑话和诗歌，两个任务互相独立，可以并行执行节省时间。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

# 先定义两个子链
joke_prompt = ChatPromptTemplate.from_template("给我讲一个关于{topic}的笑话")
joke_chain = joke_prompt | ChatOpenAI() | StrOutputParser()

poem_prompt = ChatPromptTemplate.from_template("给我写一首关于{topic}的短诗")
poem_chain = poem_prompt | ChatOpenAI() | StrOutputParser()

# RunnableParallel 让两个链并行执行
parallel_chain = RunnableParallel(
    joke=joke_chain,
    poem=poem_chain
)

# 调用一次，同时得到两个结果
result = parallel_chain.invoke({"topic": "程序员"})
print(result["joke"])  # 笑话
print(result["poem"])  # 诗
```

**好处：** 两个大模型调用同时进行，总时间 ≈ 慢的那个，比顺序调用快一倍。

---

**3. 分支条件：根据输入判断走哪个分支**

场景：用户让你生成内容，用户说要"笑话"就走笑话链，说要"诗歌"就走诗歌链，默认走通用回答。

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch

# 定义各个分支
joke_prompt = ChatPromptTemplate.from_template("讲一个关于{topic}的笑话")
joke_chain = joke_prompt | ChatOpenAI() | StrOutputParser()

poem_prompt = ChatPromptTemplate.from_template("写一首关于{topic}的诗")
poem_chain = poem_prompt | ChatOpenAI() | StrOutputParser()

default_prompt = ChatPromptTemplate.from_template("回答问题：{question}")
default_chain = default_prompt | ChatOpenAI() | StrOutputParser()

# RunnableBranch 定义条件分支
branch_chain = RunnableBranch(
    # (条件, 分支)：条件为 True 就走这个分支
    (lambda x: x["type"] == "joke", joke_chain),
    (lambda x: x["type"] == "poem", poem_chain),
    # 默认分支，所有条件都不满足走这里
    default_chain
)

# 测试：type 是 joke，走笑话分支
result1 = branch_chain.invoke({
    "type": "joke",
    "topic": "猫",
    "question": "..."
})
# 结果：笑话

# 测试：type 是 poem，走诗歌分支
result2 = branch_chain.invoke({
    "type": "poem",
    "topic": "猫",
    "question": "..."
})
# 结果：诗歌

# 测试：其他类型，走默认
result3 = branch_chain.invoke({
    "type": "other",
    "question": "什么是LCEL",
})
# 结果：默认回答
```

---

**4. RunnablePassthrough：同时传参和处理**

这是 RAG 场景最常用的写法，解释一下每一步：

```python
from langchain_core.runnables import RunnablePassthrough

# 假设已经有了 retriever（向量检索器）
def format_docs(docs):
    """把检索到的多个文档拼成一个字符串"""
    return "\n\n".join(doc.page_content for doc in docs)

# RAG 经典 LCEL 写法
rag_chain = (
    # 第一步：字典并行处理
    # - "question": RunnablePassthrough() 意思是：把输入原封不动传给这里
    # - "context": 输入先传给 retriever 检索，再用 format_docs 格式化
    {"question": RunnablePassthrough(), "context": retriever | format_docs}
    # 第二步：把上面得到的 {"question": ..., "context": ...} 传给 prompt 填充
    | prompt
    # 第三步：prompt 结果传给 llm
    | ChatOpenAI()
    # 第四步：解析输出
    | StrOutputParser()
)

# 调用
answer = rag_chain.invoke("RAG 解决了什么问题？")
```

**整个流程分解：**
1. 输入：`"RAG 解决了什么问题？"`（字符串）
2. `RunnablePassthrough()` 把输入直接赋值给 `question` → `question = "RAG 解决了什么问题？"`
3. 同一个输入也传给 `retriever` → `retriever` 检索出相关文档 → `format_docs` 拼成字符串 → 赋值给 `context`
4. 现在有了 `{"question": "...", "context": "..."}`，传给 `prompt` 填充变量
5. `prompt` 输出传给 `llm`，`llm` 输出传给 `parser`
6. 得到最终回答

这样写非常简洁，一行都不浪费，同时完成了两件事：传参 + 检索。

---

**5. 管道式数据处理**

任何函数都可以放到 LCEL 管道里，不需要包装成 Runnable：

```python
def add_extra_info(text):
    return text + "\n\n---\n以上回答由AI生成"

chain = prompt | llm | StrOutputParser() | add_extra_info
result = chain.invoke({"topic": "猫"})
# 输出：笑话 + 你加的 extra 信息
```

**LCEL 自动把普通函数转成 Runnable，非常方便。**

---

## Memory 记忆组件

**用法：** 存储对话历史，让 Agent 能记住之前说了什么。

### 完整例子：带记忆的对话机器人

```python
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 1. 初始化记忆
# return_messages=True 返回 messages 格式，适配 ChatPromptTemplate
memory = ConversationBufferMemory(return_messages=True)

# 2. 组装带记忆的链
# 注意 prompt 里要有 {history} 占位符放对话历史
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个友好的聊天助手，请记住用户说的信息，回答要自然。"),
    ("placeholder", "{history}"),
    ("human", "{input}"),
])

chain = (
    RunnablePassthrough.assign(
        # 从 memory 加载对话历史到 history 字段
        history=memory.load_memory_variables
    )
    | prompt
    | ChatOpenAI()
    | StrOutputParser()
)

# 3. 多轮对话
# 第一轮
response1 = chain.invoke({"input": "你好，我叫张三，我喜欢打篮球"})
print("助手:", response1)
# 保存到记忆
memory.save_context({"input": "你好，我叫张三，我喜欢打篮球"}, {"output": response1})

# 第二轮，模型能记住之前的对话
response2 = chain.invoke({"input": "我喜欢什么运动？我叫什么名字？"})
print("助手:", response2)
# 输出会回答：你叫张三，你喜欢打篮球
```

### 几种常见 Memory 类型对比，带例子：

**1. `ConversationBufferMemory` - 保存完整对话（默认最常用）**
```python
from langchain.memory import ConversationBufferMemory
# 保存所有对话历史，不截断
memory = ConversationBufferMemory(return_messages=True)
```
✅ 优点：完整记住所有内容  
❌ 缺点：对话长了会占满上下文窗口  
**适用：** 对话轮次少（10轮以内）

---

**2. `ConversationBufferWindowMemory` - 只保存最近 k 轮**
```python
from langchain.memory import ConversationBufferWindowMemory
# k=2 只保留最近两轮对话
memory = ConversationBufferWindowMemory(k=2, return_messages=True)
```
✅ 优点：控制上下文长度，不会爆窗  
**适用：** 对话轮次中等，只需要最近上下文

---

**3. `ConversationSummaryMemory` - 总结对话保存摘要**
```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

# 每次对话后用大模型总结成摘要保存
memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    return_messages=True
)
```
✅ 优点：对话很长也只占很少上下文空间  
❌ 缺点：每次都要调用大模型做总结，有额外成本，可能丢细节  
**适用：** 对话很长，但不需要完整细节，只要核心信息

---

**4. `VectorStoreRetrieverMemory` - 向量存储，检索相关记忆**
```python
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
memory = VectorStoreRetrieverMemory(retriever=retriever)
```
✅ 优点：非常长的对话，只检索相关内容进上下文  
❌ 缺点：实现复杂，有 embedding 成本  
**适用：** 超级长对话（几十上百轮）

---

### 要点总结：
- `memory.save_context(input, output)` 每轮对话后要调用保存
- `RunnablePassthrough.assign` 把记忆加载进输入
- prompt 一定要留 `{history}` 占位符放对话历史

---

## 工具调用 & Agent 创建

LangChain 有多种创建 Agent 的方式，这里对比几种常见写法（都是**最新 v0.2+ API**）：

| 创建方式 | 简洁程度 | 推荐场景 |
|----------|----------|----------|
| `create_openai_tools_agent` + 手动拼 Prompt | 最啰嗦，需要自己拼 Prompt | 教学/自定义程度高，面试常考 |
| `create_openai_agent` | 简洁，内部帮你拼好 Prompt | 日常开发，OpenAI 模型，推荐 |
| `create_tool_calling_agent` | 通用简洁，不绑定 OpenAI | 任何支持工具调用的模型，通用推荐 |
| `initialize_agent` | 最简短，旧版写法 | 老项目，现在不推荐 |

下面每个方式都给完整例子：

---

### 方式一：`create_openai_tools_agent`（手动拼 Prompt，最啰嗦，面试常考）

需要你自己组装 Prompt，需要 `agent_scratchpad` 占位符，适合学习原理：

```python
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate

# 1. 定义工具，@tool 装饰器即可
@tool
def get_weather(city: str) -> str:
    """Get the weather of a city. Input is city name."""
    # 这里调用真实天气 API
    return f"{city} 今天晴天，25度"

@tool
def calculate(expression: str) -> float:
    """Calculate a math expression. Input is the expression string."""
    return eval(expression)

tools = [get_weather, calculate]

# 2. 自己手动拼 Prompt，必须要有 {agent_scratchpad}
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有用的助手，可以调用工具回答问题。"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # 必须有这个占位符，放中间步骤
])

# 3. 创建 Agent
agent = create_openai_tools_agent(llm, tools, prompt)

# 4. 创建执行器并运行
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True
)
result = agent_executor.invoke({
    "input": "北京今天天气多少度？帮我算一下 32 + 8 等于多少"
})
print(result["output"])
```

**特点：** 完全可控，你可以完全定制 Prompt，但需要自己记得加占位符，比较啰嗦。

---

### 方式二：`create_openai_agent`（简洁，OpenAI 模型推荐，日常用这个）

**这个最简洁，内部帮你拼好了默认 Prompt，开箱即用：**

```python
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_agent

# 定义工具和上面一样
@tool
def get_weather(city: str) -> str:
    """Get the weather of a city"""
    return f"{city} 今天晴天，25度"

@tool
def calculate(expression: str) -> float:
    """Calculate a math expression"""
    return eval(expression)

tools = [get_weather, calculate]

# 一句话创建，不用自己拼 Prompt！
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_agent(llm, tools)

# 执行
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
result = agent_executor.invoke({
    "input": "北京今天天气多少度？帮我算一下 32 + 8 等于多少"
})
print(result["output"])
```

**特点：**
- ✅ 代码非常干净，只有几行
- ✅ 内部默认 Prompt 已经调优过了，不容易错
- ✅ 只针对 OpenAI 模型优化，是 OpenAI 官方工具调用格式
- 确实是日常用最好的选择，干净简单不容易错

---

### 方式三：`create_tool_calling_agent`（通用简洁，任何模型都能用，通用推荐）

如果你用的不是 OpenAI，是 Anthropic Claude 或者其他支持工具调用的模型，用这个：

```python
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

@tool
def get_weather(city: str) -> str:
    """Get the weather of a city"""
    return f"{city} 今天晴天，25度"

tools = [get_weather]

llm = ChatOpenAI(model="gpt-3.5-turbo")
# 还是需要你自己拼 Prompt，但比第一种简单
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    ("user", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm, tools, prompt)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools)
result = agent_executor.invoke({"input": "北京天气"})
```

**特点：** 不绑定 OpenAI，任何模型只要支持工具调用就能用，比第一种简洁一点。

---

### 方式四：`initialize_agent`（最简短，旧版写法，现在不推荐）

这是 LangChain v0.1 之前的旧写法，现在还能用来但不推荐了：

```python
from langchain.agents import initialize_agent, Tool
from langchain_openai import ChatOpenAI

# 旧版工具定义方式
def get_weather(city: str):
    return f"{city} 今天晴天，25度"

tools = [
    Tool(
        name="get_weather",
        func=get_weather,
        description="Get the weather of a city"
    )
]

llm = ChatOpenAI(temperature=0)
# 一句话初始化，选 Agent 类型
agent = initialize_agent(
    tools, llm, 
    agent="zero-shot-react-description",
    verbose=True
)

result = agent.run("北京今天天气怎么样？")
```

**为什么不推荐：**
- 用 `AgentType` 字符串选类型，不够灵活
- 旧版链式写法，不符合现在 LCEL 风格
- 现在维护少了，新功能都加在新 API 上

---

### 总结推荐

| 场景 | 选哪个 |
|------|--------|
| 面试要写完整例子，考察你对原理理解 | `create_openai_tools_agent` 手动拼 |
| 日常开发用 OpenAI 模型 | **`create_openai_agent`** 干净简单不容易错 |
| 用 Claude 或其他支持工具调用的模型 | `create_tool_calling_agent` |
| 维护老项目 | `initialize_agent` |
| 开新项目 | 不推荐 `initialize_agent`，用上面几种新的 |

**要点：**
- `@tool` 装饰器把普通 Python 函数变成 LangChain 工具，**函数的 docstring 一定要写清楚**，大模型靠这个理解工具用途
- `verbose=True` 会打印出每一步调用过程，开发调试非常方便

**内置常用工具：** LangChain 社区已经有很多现成工具：
- `SerpAPIWrapper`：谷歌搜索
- `PythonREPL`：执行 Python 代码
- `WikipediaQueryRun`：维基百科查询
- 还有连接数据库、Notion、Slack 等等很多

---

## 简单 RAG 示例

**最简化 RAG 代码，面试写得出来：**

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

# 1. 加载文档
loader = TextLoader("test.txt")
documents = loader.load()

# 2. 分割文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# 3. 存入向量库
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()

# 4. Prompt 模板
prompt = ChatPromptTemplate.from_template("""
根据以下上下文回答问题：
{context}

问题：{question}
""")

# 5. 组装 RAG 链
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | ChatOpenAI()
    | StrOutputParser()
)

# 6. 查询
result = rag_chain.invoke("文档里说了什么？")
print(result)
```

**步骤记住：** `加载文档 → 分割 → Embedding → 存向量库 → 检索 → 拼Prompt → 生成回答`

---

## 面试常考代码片段

### 1. 自定义工具

```python
from langchain.tools import tool

@tool
def my_function(arg1: str, arg2: int) -> str:
    """这是工具的描述，大模型能看到，一定要写清楚
    Args:
        arg1: 第一个参数的作用
        arg2: 第二个参数的作用
    """
    # 你的逻辑
    return f"结果是...{arg1} {arg2}"
```

**要点：** docstring 很重要，大模型靠这个理解工具做什么的。

---

### 2. 自定义 Runnable

```python
from langchain_core.runnables import Runnable

class MyRunnable(Runnable):
    def invoke(self, input, config=None):
        # 处理 input
        return processed_input
```

简单函数直接用：
```python
def my_process(input):
    return input + " processed"

# 直接就能放到 LCEL 链里
chain = prompt | my_process | llm
```

---

## 常用工具列表

LangChain 社区提供了很多开箱即用的工具，面试能说出来几个加分：

| 工具 | 作用 |
|------|------|
| `PythonREPL` | 在沙箱里执行 Python 代码，解决计算问题 |
| `SerpAPI` / `GoogleSearchResults` | 网络搜索，获取实时信息 |
| `WikipediaQueryRun` | 维基百科搜索 |
| `SQLDatabaseToolkit` | 连接 SQL 数据库，自然语言转 SQL 查询 |
| `ArxivQueryRun` | arXiv 论文搜索 |
| `PubmedQueryRun` | 医学文献搜索 |
| `BingSearch` / `DuckDuckGoSearchResults` | 搜索引擎 |
| `ZapierNLA` | 连接 Zapier 自动化工作流 |

---

## 常用内置工具使用例子

### 1. PythonREPL - 执行 Python 代码

让 Agent 能执行 Python 代码计算，解决数学问题：

```python
from langchain.tools import PythonREPLTool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_agent

# 直接实例化就能用
tool = PythonREPLTool()

# 给 Agent 使用
tools = [tool]
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_agent(llm, tools)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)

# Agent 会自己写代码计算执行
result = agent_executor.invoke({
    "input": "计算 1-100 的累加和是多少？"
})
print(result["output"])
```

**原理：** Agent 生成 Python 代码，PythonREPL 执行它，返回结果。非常适合计算、数据处理问题。

---

### 2. DuckDuckGoSearch - 免费网络搜索

不需要 API Key 就能用的开源搜索：

```python
from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

# 直接搜索
result = search.run("LangGraph 最新版本号是多少？")
print(result)

# 放到 Agent 里用
from langchain.agents import AgentExecutor, create_openai_agent
from langchain_openai import ChatOpenAI

tools = [DuckDuckGoSearchRun()]
llm = ChatOpenAI()
agent = create_openai_agent(llm, tools)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools)

result = agent_executor.invoke({
    "input": "今天北京天气预报是什么？"
})
```

**优点：** 免费，不需要申请 API Key，开发测试非常方便。

---

### 3. Wikipedia - 维基百科搜索

```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# 初始化
api_wrapper = WikipediaAPIWrapper(top_k_results=1)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# 查询
result = wiki.run("LangChain")
print(result)
```

适合查询知识型内容，维基百科内容比较靠谱。

---

### 4. SQLDatabaseToolkit - 自然语言查询 SQL

让 Agent 能自然语言转 SQL 查询数据库：

```python
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

# 连接数据库（这里以 SQLite 为例）
db = SQLDatabase.from_uri("sqlite:///mydb.db")

# 创建 toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=ChatOpenAI())

# 创建 SQL Agent，开箱即用
agent_executor = create_sql_agent(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    toolkit=toolkit,
    verbose=True
)

# 自然语言查询，Agent 自己转 SQL 执行
result = agent_executor.invoke({
    "input": "查询上个月销售额 top 3 的商品是哪三个？"
})
print(result)
```

**非常方便：** Agent 会自己找表、写 SQL、执行、解释结果。

---

### 5. Arxiv - 论文搜索

```python
from langchain_community.tools import ArxivQueryRun
from langchain_community.utilities import ArxivAPIWrapper

arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
result = arxiv.run("LangChain")
print(result)
```

---

## 使用内置工具要点：
1. 绝大多数内置工具都在 `langchain_community.tools` 里
2. 需要额外安装依赖，按照提示 `pip install ...` 就行
3. 可以直接用，也可以放到 Agent 里让 Agent 决定什么时候用

---

---

# LangGraph 部分

---

## LangGraph 核心概念

| 概念 | 作用 |
|------|------|
| **State（状态）** | 整个图执行过程中共享的数据结构，所有节点都能读写 |
| **Node（节点）** | 一个执行单元，接收当前状态，更新状态 |
| **Edge（边）** | 定义控制流，从一个节点跳转到另一个节点 |
| **Conditional Edge（条件边）** | 根据状态判断下一步跳哪 |
| **Checkpointer（检查点）** | 保存状态快照，支持中断恢复、时间旅行 |

---

## 最简单的 Agent 例子

**LangGraph 实现一个能循环调用工具的 Agent，核心代码：**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 1. 定义 State（共享状态）
class AgentState(TypedDict):
    user_input: str
    intermediate_steps: list
    final_answer: str

# 2. 定义工具
@tool
def search(query: str) -> str:
    """Search for information on the web"""
    return f"搜索结果：关于 {query} ..."

tools = [search]

# 3. 定义节点（Node）
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def agent_node(state: AgentState):
    # 这里简化：大模型决定要不要调用工具，还是直接回答
    # 实际开发需要组装 prompt 给大模型，解析输出
    if need_search(state["user_input"]):
        # 需要搜索，跳转到工具节点
        return {"intermediate_steps": []}
    else:
        # 直接回答，结束
        answer = llm.invoke(state["user_input"]).content
        return {"final_answer": answer}

def tool_node(state: AgentState):
    # 执行工具调用，结果放到 state
    query = extract_query(state)
    result = search(query)
    return {"intermediate_steps": state["intermediate_steps"] + [result]}

# 4. 条件边：决定下一步
def should_continue(state: AgentState) -> str:
    if "final_answer" in state and state["final_answer"]:
        return "end"
    else:
        return "continue"

# 5. 构建图
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tool", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tool",
        "end": END
    }
)
workflow.add_edge("tool", "agent")

# 6. 编译运行
graph = workflow.compile()
result = graph.invoke({"user_input": "北京天气怎么样？"})
print(result["final_answer"])
```

**核心记住：**
- State 是全局共享字典，所有节点都能读写
- Node 是函数，接收 state 返回更新后的 state
- Edge 定义跳转，条件边根据 state 决定下一步去哪
- 循环就是 `Agent → 判断 → 工具 → Agent` 这样构成

---

## 多 Agent 协作示例

**两个 Agent 协作，一个写代码一个 review：**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo")

class MultiAgentState(TypedDict):
    requirement: str    # 用户需求
    code: str            # 生成的代码
    review_comments: str # review 意见
    done: bool           # 是否完成

# 写代码的 Agent 节点
def coder_node(state):
    prompt = f"根据需求写代码：{state['requirement']}\n之前的review意见：{state.get('review_comments', '')}"
    code = llm.invoke(prompt).content
    return {"code": code, "done": False}

# Review 代码的 Agent 节点
def reviewer_node(state):
    prompt = f"""
    需求：{state['requirement']}
    代码：{state['code']}
    请给出评审意见，如果没问题就说'approved'
    """
    comments = llm.invoke(prompt).content
    if "approved" in comments.lower():
        return {"review_comments": comments, "done": True}
    else:
        return {"review_comments": comments, "done": False}

# 条件判断：是否完成
def check_done(state):
    if state["done"]:
        return "end"
    else:
        return "coder"

# 构图
workflow = StateGraph(MultiAgentState)
workflow.add_node("coder", coder_node)
workflow.add_node("reviewer", reviewer_node)

workflow.set_entry_point("coder")
workflow.add_edge("coder", "reviewer")
workflow.add_conditional_edges("reviewer", check_done, {
    "coder": "coder",
    "end": END
})

graph = workflow.compile()
result = graph.invoke({"requirement": "写一个快速排序算法"})
print(result["code"])
```

---

## 状态持久化 & 断点恢复

LangGraph 内置了状态持久化，用 checkpointer 保存每个步骤的状态，可以在服务重启后恢复，也支持人类介入。

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# 1. 创建 checkpointer，SQLite 持久化
conn = sqlite3.connect("checkpoints.sqlite")
checkpointer = SqliteSaver(conn)

# 2. 编译图的时候传入 checkpointer
graph = workflow.compile(checkpointer=checkpointer)

# 3. 调用的时候需要传 thread_id
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke({"user_input": "...", config})

# 4. 如果中断了，服务重启了，可以通过 thread_id 恢复状态
#    从最近的检查点继续执行
current_state = graph.get_state(config)
print(current_state.values)
```

**特点：**
- 每一步执行完都会保存检查点
- 支持**时间旅行**：可以回到任意历史检查点重新执行，方便调试
- 除了 SQLite，也支持 PostgreSQL 等其他后端

---

## 人类在环（Human-in-the-loop）

需要人类审核确认后再继续执行，LangGraph 用 **`interrupt_before`** 在指定节点前设置断点，写法非常简洁：

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict

class BookingState(TypedDict):
    flight_info: dict
    confirmed: bool = False

# 节点1：查询航班
def book_flight_node(state):
    # 查询航班信息...
    return {"flight_info": {"flight_id": "CA123", "price": 1000}}

# 节点2：支付（需要人类确认后才能执行）
def pay_node(state):
    print(f"支付 {state['flight_info']['price']} 元成功")
    return {}

# 构建工作流
workflow = StateGraph(BookingState)
workflow.add_node("book_flight", book_flight_node)
workflow.add_node("pay", pay_node)

# ✅ 核心：在 pay 节点之前设置断点，执行到这里会自动中断等待人类确认
workflow.add_edge("book_flight", "pay", interrupt_before=["pay"])
workflow.set_entry_point("book_flight")
workflow.add_edge("pay", END)

# 必须使用 checkpointer 才能支持断点持久化
with SqliteSaver.from_conn_string(":memory:") as saver:
    graph = workflow.compile(checkpointer=saver)
    config = {"configurable": {"thread_id": "booking-123"}}

    # 第一次运行：到 pay 节点前自动中断
    # 执行完 book_flight 就停下来了
    snapshot = graph.invoke({"user_input": "订CA123"}, config)
    
    # ✅ 此时：你可以把航班信息展示给用户，等待用户确认
    
    # ✅ 用户确认后，更新状态（把 confirmed 设为 True）
    graph.update_state(config, {"confirmed": True})
    
    # ✅ 从断点继续执行，直接走 pay 节点
    final_result = graph.invoke(None, config)
```

**写法要点：**
- `interrupt_before=["pay"]` 在 `add_edge` 时设置，意思是"进入 `pay` 节点之前先中断"
- 必须传入 `checkpointer` 才能保存状态支持中断恢复
- 中断后通过 `graph.update_state()` 更新状态，再 `graph.invoke(None, config)` 继续执行

**常见场景：**
- 支付前需要用户确认
- 生成代码后需要人类 review
- 关键决策需要人拍板
- 模型不确定的时候问人

---

## 工具调用循环完整例子（带状态持久化）

这是一个完整可运行的 LangGraph Agent 循环调用工具例子：

```python
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
import operator
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langgraph.prebuilt import ToolNode

# 1. 定义状态
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

# 2. 定义工具
@tool
def search(query: str) -> str:
    """搜索网络获取信息"""
    return f"搜索结果：{query}..."

tools = [search]
tool_node = ToolNode(tools)

# 3. Agent 节点
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = llm.bind_tools(tools)

def agent_node(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    # 返回新加到 messages 里
    return {"messages": [response]}

# 4. 条件判断：要不要继续调用工具
def should_continue(state):
    last_message = state["messages"][-1]
    # 如果有工具调用，继续去工具节点
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "continue"
    # 否则结束
    return "end"

# 5. 构建图
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "agent")

# 6. 编译，带持久化
checkpointer = SqliteSaver.from_conn_string(":memory:")
graph = workflow.compile(checkpointer=checkpointer)

# 7. 运行
config = {"configurable": {"thread_id": "conv-1"}}
result = graph.invoke(
    {"messages": [("human", "北京今天天气怎么样？")]},
    config
)

# 打印最终回答
final_message = result["messages"][-1]
print(final_message.content)
```

**这个例子用了 LangGraph 预构建的 `ToolNode`，不用自己写工具节点，更简洁。**

---

---

## 面试回答套路

**如果面试官问："说说你用 LangChain 做过什么项目？"**

可以这么说：
> 我用 LangChain + LangGraph 做了一个 [领域] Agent 系统：
> 1. 用 LCEL 组装了 RAG 检索流程，从企业知识库检索相关文档
> 2. 用 LangGraph 做了 Agent 的控制流，支持多轮工具调用、人类在环确认
> 3. 自定义了几个业务工具（比如查询订单、取消订单）
> 4. 用 LangSmith 调试追踪，评估回答准确率，不断优化 Prompt
> 5. 上线后用 LangSmith 监控错误率和延迟

这样说就很完整，能体现你真的用过。

---

> 本文整理来源：
> - LangChain 官方文档入门教程
> - LangGraph 官方文档示例
> - 社区实践总结

---
