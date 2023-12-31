---
layout:     draft
title:      Langchain-101
date:       2023-12-18
categories: 
- nlp
---

[LangChain](https://python.langchain.com/docs/get_started/introduction) 是一个基于语言模型的 APP 开发框架，可以把提示、示例及内容等上下文内容与 LM 结合。核心理念是为各种 LLMs 实现通用的接口，把 LLMs 相关的组件“链接”在一起，简化 LLMs 应用的开发难度，方便开发者快速地开发复杂的 LLMs 应用。


对于用户而言，首先一个问题，为什么需要 LangChain？

笔者认为最核心的有两点：
1. **统一接口**，LLM 种类太多了，GPT、PaLM、Gemini 等等，几乎每个项目开发者都有一套自己的接口，同样的“聊天”接口，有的叫 `complete(...)`，有的叫 `start_chat(...)`，参数更是五花八门。到开发者这里，就需要熟悉各种 LLMs 的接口及参数，开发成本高。LangChain 做的事情就是把这些差异性给屏蔽掉，开发者只需要关心 LangChain 的接口，就可以使用各种 LLMs。
2. **数据感知**，LLM 本身存储的知识是有限的、滞后的，有时候需要结合外部数据，比如知识库、知识图谱，LangChain 能把 LLM 跟外部数据连接起来，实现更丰富的上下文交互。

其他的还有**可视化、与外部系统的连接、多语言支持**等等。

LangChain 有几个核心的模块：
- 模型输入输出 / Model I/O。提供与模型交互的接口，让交互更加方便。
- 检索 / Retrieval。
- 代理 / Agents。



# 教程与资料
- 由 [Harrison Chase](https://dlj.one/8g4dq3) 和 Andrew Ng 联合开发的两门短课程，[https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)，前者是 LangChain 创始人。
- [LangChain 中文网](https://www.langchain.asia/getting_started/getting_started)
