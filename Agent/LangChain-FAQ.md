# LangChain / LangGraph / LangSmith 面试常见问题 FAQ

整理自公开资料和 Agent 岗位面试常见问题，适合 LLM/Agent 开发岗位面试复习

---

## 目录

- [LangChain 基础篇](#langchain-基础篇)
- [LangChain 核心组件篇](#langchain-核心组件篇)
- [LangGraph 篇](#langgraph-篇)
- [LangSmith 篇](#langsmith-篇)
- [总结速记](#总结速记)

---

## LangChain 基础篇

### 1. 什么是 LangChain？它解决了什么问题？

**Q:** 为什么需要 LangChain 这样的框架？它不是只是包装了 OpenAI API 吗？

**A:**

**LangChain** 是一个**开源的大语言模型应用开发框架**，帮助开发者快速搭建 LLM 应用（特别是 Agent 和 RAG 系统）。

它解决的核心问题是：
> 开发 LLM 应用有很多重复的模板代码，LangChain 把这些通用模块帮你写好了，你只需要组装就行，不用从零造轮子。

**LangChain 提供了什么：**
1. **组件封装**：封装了大模型调用、向量库、工具调用、检索器等常用组件，统一接口
2. **链式调用**：提供了链式组装的能力，把多个组件串起来完成复杂流程
3. **生态丰富**：集成了上百种大模型、向量库、工具、数据源，拿来就能用
4. **高级能力**：内置了 Agent、RAG、记忆等常见架构，不用自己从零实现

它不是简单包装 OpenAI API，而是一个**应用开发框架**，让你能快速组装出复杂的 LLM 应用。

---

### 2. LangChain 的核心设计思想是什么？什么是 Chain（链）？

**Q:** LangChain 里的"链"是什么概念？为什么要用链？

**A:**

LangChain 的核心设计思想是：**把大模型应用看作多个组件的串联流水线**，每个组件做一件事，组件之间通过明确接口传递数据。

**Chain（链）** 就是串联多个组件的执行单元：
- 输入从第一个组件进，出来后传给下一个组件，一直到最后输出结果
- 每个链只做一件明确的事情，能组合嵌套
- 比如 `RetrievalQA` 链就是：**用户问题 → 检索 → 把检索结果拼进 Prompt → 调用大模型 → 返回回答**

**为什么用链的设计：**
- 清晰，每个模块职责单一，好调试好维护
- 灵活，你可以替换其中任何一个组件（比如把 OpenAI 换成 Claude，把 Pinecone 换成 Chroma）
- 复用，相同组件能在不同链中复用

---

### 3. LangChain 和原生 OpenAI SDK 有什么区别？什么时候选哪个？

**Q:** 我直接用 OpenAI Python SDK 不行吗？为什么要加 LangChain 一层？

**A:**

| 对比 | 原生 OpenAI SDK | LangChain |
|------|-----------------|-----------|
| **定位** | 简单 API 调用封装 | 完整 LLM 应用开发框架 |
| **复杂应用支持** | RAG/Agent 需要你自己写所有组装代码 | RAG/Agent 开箱即用，官方已经实现好 |
| **生态集成** | 只调用 OpenAI API | 集成了上百种模型、向量库、工具 |
| **代码量** | 简单调用代码少 | 复杂应用代码少很多 |

**选择建议：**
- ✅ **选原生 SDK**：非常简单的应用，就只是调用一下大模型，不需要 RAG/Agent
- ✅ **选 LangChain**：要做 RAG、Agent、需要集成多个数据源/工具，需要可扩展的架构

---

## LangChain 核心组件篇

### 4. LangChain 有哪些核心组件？各自作用是什么？

**Q:** LangChain 主要包含哪些模块？说说每个模块做什么？

**A:**

LangChain 的核心组件可以分为几类：

| 组件类别 | 代表组件 | 作用 |
|----------|----------|------|
| **模型层** | `LLM` / `ChatModel` | 统一封装不同大模型（OpenAI、Anthropic、开源模型等），统一调用接口 |
| **提示词** | `PromptTemplate` | 提示词模板化，动态填充变量 |
| **检索链** | `Document` / `Retriever` | 封装文档检索，统一接口，不同向量库能互换 |
| **链** | `Chain` / `Runnable` | 组装多个组件成可执行流程 |
| **Agent** | `Agent` / `Tool` | Agent 框架，支持多种 Agent 类型，工具调用 |
| **记忆** | `Memory` | 存储对话历史，支持不同存储方式 |
| **输出解析** | `OutputParser` | 把大模型输出解析成结构化格式（JSON、列表等） |

核心设计思路：**每个组件都有抽象接口，你可以轻松替换实现**。比如你想把 OpenAI 换成 Anthropic Claude，只需要换一行代码，其他不用改。

---

### 5. 什么是 LCEL？它解决了什么问题？

**Q:** LangChain Expression Language 是什么？为什么需要它？

**A:**

**LCEL（LangChain Expression Language）** 是 LangChain 推出的**声明式组件组合语言**，让你能通过简单的语法把多个组件组合成链。

**它解决了旧版 Chain 的什么问题：**
- 旧版 Chain 继承式写法太笨重，要写很多模板代码
- 调试困难，不容易看清楚组件之间怎么连
- 并行、分支这些高级流程写起来复杂

**LCEL 优点：**
1. **声明式**：你只需要说清楚组件怎么拼，不用管调度细节
   ```python
   # LCEL 写法，非常简洁
   chain = prompt | llm | output_parser
   ```
2. **原生支持并行、批处理、流输出**
3. **自动追踪**：集成 LangSmith，每一步输入输出都能看到
4. **灵活性强**：支持分支、循环、条件判断

现在 LangChain 官方推荐用 LCEL 写法代替旧版 Chain。

---

### 6. LangChain 中的 Memory 组件有哪些类型？怎么选？

**Q:** LangChain 记忆组件怎么分类？各自适用场景？

**A:**

常见 Memory 类型：

| Memory 类型 | 特点 | 适用场景 |
|-------------|------|----------|
| `ConversationBufferMemory` | 把完整对话都存在内存里 | 对话轮次少，对完整上下文需要 |
| `ConversationBufferWindowMemory` | 只保留最近 k 轮对话 | 防止上下文太长，窗口滑动 |
| `ConversationSummaryMemory` | 把历史对话总结成摘要，只存摘要 | 对话长，节省上下文空间 |
| `VectorStoreRetrieverMemory` | 把历史存在向量库，需要时检索相关记忆 | 非常长的对话，只检索相关内容 |

**选型思路：**
- 对话短 → BufferMemory 最简单
- 对话中等长度 → BufferWindowMemory
- 对话很长 → Summary 或者 VectorStoreRetrieverMemory

---

### 7. LangChain 中怎么实现自定义组件？什么时候需要自定义？

**Q:** 官方提供的组件满足不了需求，怎么写自定义组件？

**A:**

如果用 LCEL，自定义组件非常简单：
- 如果是函数，直接就能用，`lambda x: ...` 就行
- 如果需要保存状态或者复杂逻辑，继承 `Runnable` 基类，实现 `invoke` 方法就行

**什么时候需要自定义组件：**
- 官方没有支持的特殊数据源/模型/工具
- 你的业务有特殊逻辑，官方组件不满足
- 需要对现有组件做特殊定制

---

### 8. 什么是 Runnables？LangChain 中 Runnable 是什么？

**Q:** LCEL 中的 Runnable 概念是什么？

**A:**

**Runnable** 是 LangChain 对**可调用单元**的抽象，任何能处理输入输出的组件都是 Runnable：
- 大模型是 Runnable
- Prompt 模板是 Runnable
- 检索器是 Runnable
- OutputParser 是 Runnable
- 你自己写的函数也能变成 Runnable

Runnable 提供了统一的调用接口：
- `invoke()`：同步调用
- `ainvoke()`：异步调用
- `stream()`：流式输出
- `batch()`：批量处理

可以通过 `|` 把多个 Runnable 组合成更大的 Runnable，这就是 LCEL 的基础。

---

## LangGraph 篇

### 9. 什么是 LangGraph？它和 LangChain 是什么关系？解决了什么问题？

**Q:** LangGraph 不是 LangChain 同一个公司出的吗？为什么需要它？它和原生 LangChain Agent 有什么区别？

**A:**

**LangGraph** 是 LangChain 团队开源的**用于构建有状态、多步骤大模型工作流的库**，基于 LangChain 但更加强大。

**关系：**
- LangChain：基础框架，提供组件封装和 LCEL 简单链式组装
- LangGraph：扩展 LangChain，支持**循环、条件分支、持久化状态**，适合构建复杂 Agent

**解决了 LangChain 原生 Chain/Agent 的什么问题：**
- LangChain 原生 Chain 是**线性的**，不能做循环、不能条件分支
- 原生 Agent 状态管理弱，LangGraph 把状态显示化管理
- LangGraph 能轻松实现：`人类参与循环审核`、`多个 Agent 轮询协作`、`条件跳转` 这些复杂流程

一句话：简单线性流程用 LangChain LCEL 足够，**需要循环、分支、多 Agent 协作、持久化状态就用 LangGraph**。

---

### 10. LangGraph 的核心概念是什么？State、Node、Edge 分别是什么？

**Q:** LangGraph 编程模型中几个核心概念解释一下。

**A:**

LangGraph 基于有状态图模型，三个核心概念：

| 概念 | 作用 |
|------|------|
| **State（状态）** | 整个图执行过程中共享的数据结构，存储对话历史、中间结果、用户信息等，所有节点都能读写 |
| **Node（节点）** | 一个执行单元，接收当前状态，更新状态，相当于函数。比如：调用大模型、调用工具、人类输入 |
| **Edge（边）** | 定义控制流，从一个节点跳转到另一个节点。支持条件边：根据状态判断下一个节点是谁 |

典型的 Agent 图结构：
```
START → Agent 节点 → 条件判断 → 是不是要调用工具 → YES → 工具节点 → 回到 Agent 节点
                                      → NO → END
```

State 是核心，所有节点都修改同一个 State，比传参更清晰。

---

### 11. LangGraph 怎么实现多 Agent 协作？举个例子？

**Q:** 多个 Agent 用 LangGraph 怎么协作？

**A:**

LangGraph 天生适合做多个 Agent 协作，典型模式：

**例子：生成器 + 检测器**
- Node1: `generator_agent` 生成回答
- Node2: `checker_agent` 检查回答对不对
- 条件边：如果检查不通过，回到 generator 重写，如果通过就结束
- 状态共享：两个 Agent 能看到对方生成的内容和检查意见

**另一个例子：多个专业 Agent 协作**
- 每个 Agent 是一个 Node，处理自己专长领域
- Supervisor 节点决定下一步该哪个 Agent 执行
- 所有 Agent 共享同一个 State，能拿到之前所有步骤结果

LangGraph 让多 Agent 的控制流变得非常清晰，比原生 LangChain 好维护太多。

---

### 12. LangGraph 怎么实现人类在环（Human-in-the-loop）？

**Q:** 需要人类审核确认步骤，LangGraph 怎么处理？

**A:**

LangGraph 原生支持中断和恢复，实现人类在环非常简单：

**做法：**
1. 在需要人类确认的节点，设置 **breakpoint**（断点）
2. 图执行到断点会主动暂停，等待人类输入
3. 人类审核后，可以：
   - 批准继续执行
   - 修改状态后再继续
   - 否决回退到上一步
4. 调用 `graph.resume(..., human_input)` 恢复执行

**典型场景：**
- Agent 生成了代码，需要人类 review 后再执行
- 预订支付，需要用户确认后再扣款
- 关键决策步骤，需要人拍板

---

### 13. LangGraph 状态怎么持久化？

**Q:** 停掉服务之后 State 会丢吗？LangGraph 怎么持久化？

**A:**

LangGraph 内置了状态持久化，你只需要传一个 `checkpointer` 进去：

```python
from langgraph.checkpoint.sqlite import SqliteSaver

checkpointer = SqliteSaver(conn)
graph = workflow.compile(checkpointer=checkpointer)
```

**特性：**
- 每一次状态更新都会保存到检查点
- 服务重启后能从检查点恢复
- 能支持时间旅行：回到任意一个历史检查点重新执行
- 除了 SQLite，也支持其他持久化后端（比如 PostgreSQL）

---

## LangSmith 篇

### 14. 什么是 LangSmith？它和 LangChain 是什么关系？解决了什么痛点？

**Q:** LangSmith 是做什么的？为什么需要它？开发 LLM 应用调试难在哪？

**A:**

**LangSmith** 是 LangChain 团队推出的**LLM 应用开发平台**，专门用来调试、测试、评估、监控 LLM 应用。

**关系：**
- LangChain：开发框架，写代码用
- LangSmith：观测调试平台，帮助你开发上线后监控，和 LangChain 深度集成但也能独立用

**解决的核心痛点：**
传统软件开发你能 debug 每一行代码，但**LLM 应用是黑盒**：
- 到底哪一步错了？Prompt 对不对？检索出来的文档对不对？
- 大模型输出不稳定，怎么评估不同版本哪个好？
- 线上出问题了，怎么复现怎么排查？

LangSmith 把每一步的输入输出、trace 都给你存下来，可视化让你能清楚看到整个流程，方便调试。

---

### 15. LangSmith 核心功能有哪些？

**Q:** LangSmith 主要能做什么？说几个核心功能。

**A:**

四大核心功能：

1. **Tracing（链路追踪）**
   - 自动记录每一次请求的完整执行链
   - 每个组件输入输出都能看到，prompt 是什么，检索结果是什么，大模型输出是什么
   - 可视化整个调用流程，哪一步慢了哪一步错了一眼能看到

2. **Prompt 调试**
   - 在线编辑 Prompt，立即运行看结果，对比不同 Prompt 效果
   - 不用来回改代码重启

3. **Datasets & Evaluation（数据集和评估）**
   - 创建测试数据集，批量跑你的 LLM 应用
   - 用指标或者大模型打分评估效果
   - 对比不同版本，看改了之后效果变好还是变坏

4. **Monitoring（线上监控）**
   - 上线后监控 latency、错误率、用户反馈
   - 发现异常告警

简单说就是：**开发阶段帮你调试，上线阶段帮你监控评估**。

---

### 16. LangSmith 怎么对比不同 Prompt 或不同 Chain 效果？

**Q:** 我改了 Prompt，怎么知道改完是不是比之前好？

**A:**

LangSmith 提供了**实验对比**功能：

1. 你创建一个共享的测试数据集（一堆用户提问+标准答案）
2. 分别跑 A 版本（旧 Prompt）和 B 版本（新 Prompt）
3. LangSmith 自动批量运行，计算指标（准确率、延迟等）
4. 可以用大模型作为裁判，给两个版本打分对比
5. 输出对比表格，告诉你哪个版本更好

这样你改完不用靠感觉，靠数据说话。

---

### 17. LangSmith 一定要用吗？什么时候需要什么时候不需要？

**Q:** 小项目一定要上 LangSmith 吗？

**A:**

**不需要，分情况：**

✅ **推荐用 LangSmith：**
- 项目超过原型阶段，需要调试维护
- 需要评估不同版本效果
- 上线后需要监控
- 团队协作开发，需要共享调试信息

❌ **可以不用：**
- 非常简单的原型demo，就几行代码
- 完全本地开发，不需要云端共享
- 数据敏感不能出网，可以用离线替代方案

LangSmith 有免费额度，一般中小项目免费额度够用了。

---

### 18. LangChain / LangGraph / LangSmith 三者关系总结一下？

**Q:** 这三个都是 LangChain 公司出的，分别定位是什么？怎么配合使用？

**A:**

一句话总结：

| 产品 | 定位 | 什么时候用 |
|------|------|------------|
| **LangChain** | 基础开发框架，组件封装，LCEL 链式组装 | 开发任何 LLM 应用都需要 |
| **LangGraph** | 有状态工作流框架，支持循环分支多 Agent | 需要复杂控制流、Agent、多 Agent 协作的时候用 |
| **LangSmith** | 开发调试评估监控平台 | 开发阶段调试，上线后监控，不管用不用 LangGraph 都能用 |

**配合流程：**
1. 用 LangChain 组件 + LCEL 搭基础流程
2. 需要循环/多Agent就用 LangGraph 做控制流
3. 开发调试用 LangSmith 追踪评估，上线后用 LangSmith 监控

---

## 工程实践篇

### 19. 使用 LangChain 开发过程中遇到过哪些坑？怎么避免？

**Q:** 说说你踩过的 LangChain 坑？

**A:**

常见坑和经验：

1. **版本变化太快，API  breaking change 多**
   - 解决：锁版本，看清楚文档对应版本，升级前看 changelog

2.** 回调嵌套调试难 **- 解决：开 LangSmith 追踪，每一步输入输出都能看到

3. **过度封装，自定义复杂逻辑有时候不如原生写方便**
   - 解决：简单直接优先，官方组件满足不了就写自定义 Runnable，不难

4.** 太多抽象概念，新手入门陡峭 **- 解决：先把 LCEL 玩明白，Runnable 理解了，其他组件都是 Runnable 拼起来

5. **不同组件兼容性问题**
   - 解决：尽量用官方推荐的组合，第三方集成先测清楚

---

### 20. LangChain 性能瓶颈在哪？怎么优化？

**Q:** LangChain 会不会比原生调用慢很多？怎么优化？

**A:**

**主要瓶颈：**
- 多组件组装确实有一点额外 overhead，但一般不大（几ms到几十ms）
- 真正的瓶颈还是在大模型调用和网络请求，那才是占 99% 时间的地方
- 复杂 Agent 多轮调用，自然就慢，不是 LangChain 的锅

**优化方法：**
- 能并行就并行，LCEL 原生支持并行
- 缓存大模型结果，相同请求直接返回缓存
- 不必要的组件去掉，不要为了用框架而用框架
- 异步调用，提高并发能力

总的来说：LangChain 额外 overhead 对绝大多数应用可以忽略，不用太担心。

---

## 总结速记

| 考点 | 核心结论 |
|------|---------|
| 什么是 LangChain | LLM 应用开发框架，封装通用组件，让你快速搭建 RAG/Agent 应用 |
| 什么是 LCEL | LangChain 声明式组件组合语言，`chain = prompt | llm | output_parser`，简洁灵活 |
| 什么是 Runnable | LCEL 的基础抽象，任何可调用单元都是 Runnable，统一接口 |
| LangChain 核心组件 | 模型 / Prompt 模板 / 检索器 / 链 / Agent / Memory / 输出解析器 |
| 什么是 LangGraph | LangChain 团队出品的有状态工作流框架，支持循环分支多 Agent，解决原生 LangChain 线性流程限制 |
| LangGraph 核心概念 | State（共享状态）+ Node（执行单元）+ Edge（控制流），支持条件边、断点 |
| 什么是 LangSmith | LLM 应用调试评估监控平台，和 LangChain 深度集成，解决 LLM 应用黑盒难调试问题 |
| LangSmith 核心功能 | Tracing 链路追踪 / Prompt 调试 / 数据集评估 / 线上监控 |
| 三者关系 | LangChain = 开发框架，LangGraph = 复杂工作流，LangSmith = 调试监控 |
| 什么时候用 LangGraph | 需要循环、条件分支、多 Agent 协作、人类在环的时候用 |
| 常见坑 | API 变化快，版本锁死；调试难开 LangSmith；过度封装不如自定义 |

---

> 本文整理来源：
> - LangChain 官方文档
> - 掘金/CSDN/牛客网 LangChain 面经
> - LangGraph 官方文档

---
