+++
date = '2025-06-20T16:02:13+08:00'
draft = true
title = 'LangGraph'
series =[ '学习笔记']
series_weight=4
showTableOfContents='article.showTableOfContents'

+++

# LangGraph

LangGraph，它是一个Python库 。其作用是构建有状态、多操作的大语言模型（LLM）应用程序，用于创建智能体（agent）和组合智能体（multi-agent）流程。 与其他LLM应用框架相比，LangGraph有核心优势：

- 持久执行：构建能够在出现故障时持续运行并长时间工作的智能体，可自动从停止的位置精确恢复运行。
-  人工介入：通过在执行过程中的任何时刻检查和修改智能体状态，无缝融入人工监督。
-  全面记忆：创建真正有状态的智能体，具备用于持续推理的短期工作记忆和跨会话的长期持久记忆。 
- 使用LangSmith进行调试：借助可视化工具深入了解复杂的智能体行为，这些工具可追踪执行路径、捕捉状态转换并提供详细的运行时指标。 
- 可投入生产的部署：利用专为应对有状态、长时间运行的工作流程所面临的独特挑战而设计的可扩展基础设施，自信地部署复杂的智能体系统。

## 初始化

安装：

```
pip install -U langgraph
```

然后，使用预构建组件创建一个智能体：

```
# 安装必要库
# pip install dashscope langchain langchain-community langgraph

from langgraph.prebuilt import create_react_agent
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
from langchain.agents import Tool
import os
def get_weather(city: str) -> str:
    """获取指定城市的天气信息"""
    return f"{city}的天气是晴天，25℃！"

# 创建千问模型实例
model = ChatTongyi(
    model_name="qwen-turbo",   # 也可以使用 qwen-plus 或 qwen-max
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY") # 替换为真实的API密钥
)

# 创建代理
agent = create_react_agent(
    model=model,
    tools=[Tool(
        name="get_weather",
        func=get_weather,
        description="获取城市的天气信息"
    )],
    prompt="你是一个有用的天气助手"
)

# 运行代理
response = agent.invoke({
    "messages": [
        HumanMessage(content="上海的天气怎么样？")
    ]
})

# 打印响应结果
print("最终回答:", response["messages"][-1].content)
```

## 快速入门

### StateGraph

首先stategraph是用来描述整个图的，图中的状态会随着多个agent的工作不断的更新，节点node就是用来更新状态的如何来定义一张图中的状态

```
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)#State必须。定义状态数据的 Pydantic 模型，描述图中的数据结构
```

我们的图现在可以处理两个关键任务： 

- 每个node都可以接收当前State作为输入，并输出对该状态的更新。 
- 由于使用了带有Annotated语法的预构建add_messages函数，对messages的更新将追加到现有列表中，而不是覆盖它。

### Nodes

**Nodes** 代表工作单元，通常是普通的 Python 函数。

首先选择一个模型：

```
from langchain_community.chat_models.tongyi import ChatTongyi 
llm = ChatTongyi(
    model_name="qwen-turbo",  # 也可使用 qwen-plus 或 qwen-max
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")  # 替换为真实的API密钥
)
```

将聊天模型整合到一个简单的节点中：

```
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
```

chatbot 节点函数接收当前的 `State`（状态）作为输入，并返回一个字典。这个字典包含一个更新后的 `messages` 列表，该列表位于键 "messages" 之下。这就是所有 LangGraph 节点函数的基本模式。

 `State`（状态）中的 `add_messages` 函数会将大型语言模型（LLM）的响应消息附加到状态中已有的任何消息之后。

### entry

添加一个entry点，以便每次运行图时告知图从何处开始工作：

```
graph_builder.add_edge(START, "chatbot")
```

### Compile

在运行图之前，我们需要对其进行编译。我们可以通过在图构建器上调用compile()来实现。这将创建一个CompiledGraph，我们可以在状态上调用它。

```
graph = graph_builder.compile()
```

### 可视化

```
from IPython.display import Image, display

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass
```

![image-20250620171634220](https://s2.loli.net/2025/06/20/CoB3VsReNgDtSm4.png)

### 运行

```
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
```

>流式输出：
>`stream()` 方法返回一个可迭代的 `event` 流，每个 `event` 代表系统响应的一部分（可能是逐字生成的文本）。通过 `for event in ...` 循环，可以逐次处理这些流式事件。

## 添加工具

### 定义搜索工具

```
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SerpAPIWrapper
from langchain_openai import ChatOpenAI

# 定义Tool
# 需要定义环境变量 export GOOGLE_API_KEY="", 在网站上注册并生成API Key: https://serpapi.com/searches

search = SerpAPIWrapper()
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to answer questions about current events",
    )
]
```

### 定义语言模型

```
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.messages import HumanMessage
from langchain.agents import Tool
import os

# 创建千问模型实例
llm = ChatTongyi(
    model_name="qwen-turbo",   # 也可以使用 qwen-plus 或 qwen-max
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY") # 替换为真实的API密钥
)
```

### 定义图

bind_tools：这使得大语言模型（LLM）知道如果它想使用搜索引擎，应该使用的正确JSON格式。

```
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

# Modification: tell the LLM which tools it can call
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)
```

创建一个函数来运行工具
现在，如果工具被调用，创建一个函数来运行这些工具。具体做法是，将这些工具添加到一个名为BasicToolNode的新节点中，该节点会检查状态中最新的消息，如果消息中包含tool_calls，就会调用工具。这依赖于大语言模型（LLM）的tool_calling支持，Anthropic、OpenAI、谷歌Gemini以及其他一些大语言模型提供商都提供这种支持。 

```
import json

from langchain_core.messages import ToolMessage


class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


tool_node = BasicToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
```

也可以使用LangGraph预先构建好的 ToolNode

```
from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

tool = TavilySearch(max_results=2)
tools = [tool]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=[tool])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
```

### 定义条件边

添加了工具节点后，现在你可以定义条件边了。

边将控制流从一个节点导向下一个节点。条件边从单个节点出发，通常包含 “if” 语句，以便根据当前图状态导向不同的节点。这些函数接收当前图状态，并返回一个字符串或字符串列表，指示接下来要调用哪个（哪些）节点。

接下来，定义一个名为 route_tools 的路由函数，用于检查聊天机器人输出中的工具调用。通过调用 add_conditional_edges 将此函数提供给图，这会告诉图，每当聊天机器人节点完成时，检查此函数以确定接下来的走向。

该条件在存在工具调用时将导向工具，不存在时则导向 END。由于该条件可以返回 END，这次你无需显式设置 finish_point。

**add_conditional_edges参数**：

- source：起始节点。当退出此节点时，将运行此条件边。
- path：可调用对象，用于确定下一个节点或多个节点。如果未指定path_map，则该可调用对象应返回一个或多个节点名称。如果返回END，图将#停止执行。
- path_map：可选的路径到节点名称的映射。如果省略，path返回的路径应直接为节点名称。

```
def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


# The `tools_condition` function returns "tools" if the chatbot asks to use a tool, and "END" if
# it is fine directly responding. This conditional routing defines the main agent loop.
#add_conditional_edges参数：
#source：起始节点。当退出此节点时，将运行此条件边。
#path：可调用对象，用于确定下一个节点或多个节点。如果未指定path_map，则该可调用对象应返回一个或多个节点名称。如果返回END，图将#停止执行。
#path_map：可选的路径到节点名称的映射。如果省略，path返回的路径应直接为节点名称。
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
    # It defaults to the identity function, but if you
    # want to use a node named something else apart from "tools",
    # You can update the value of the dictionary to something else
    # e.g., "tools": "my_tools"
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()
```

得到以下模型

![image-20250622203633854](https://s2.loli.net/2025/06/22/2gSKdGwWBPFUO4f.png)

### 运行

```
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)

while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
```

## 添加记忆

LangGraph 通过持久化检查点保存上下文信息。如果你在编译图时提供一个检查点工具，并在调用图时提供一个线程 ID，LangGraph 会在每一步之后自动保存状态。当你使用相同的线程 ID 再次调用图时，图会加载其保存的状态，使聊天机器人能够从上次中断的地方继续。

### 创建`MemorySaver`检查点

创建基于内存的检查点

```
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
```

### 编译图



```
graph = graph_builder.compile(checkpointer=memory)
```



### 运行

选择一个线程作为此次对话的key：

```
config = {"configurable": {"thread_id": "1"}}
```

运行图：

```
user_input = "Hi there! My name is Will."

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
```

继续询问：

```
user_input = "Remember my name?"

# The config is the **second positional argument** to stream() or invoke()!
events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()
```

### 检查状态

快照包含当前状态值、相应的配置以及`next`要处理的节点。

```
snapshot = graph.get_state(config)
snapshot
```

> 如果在图形调用中获取状态，`snapshot.next` 会告知下一个将执行的节点

## 添加人工介入控制

智能体可能不可靠，可能需要人类输入才能成功完成任务。同样，对于某些操作，你可能希望在运行前获得人类批准，以确保一切按预期运行。 LangGraph的持久层支持人在回路工作流程，允许根据用户反馈暂停和恢复执行。此功能的主要接口是`interrupt`函数。在节点内部调用`interrupt`将暂停执行。通过传入一个`Command`，可以连同来自人类的新输入一起恢复执行。`interrupt`在使用上类似于Python的内置函数`input()`，但有一些注意事项。 

### 添加`human_assistance`工具

定义模型：

```
from langchain_community.chat_models.tongyi import ChatTongyi  # 替换为千问模型
llm = ChatTongyi(
    model_name="qwen-turbo",  # 也可使用 qwen-plus 或 qwen-max
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")  # 替换为真实的API密钥
)
```

添加 `human_assistance` ：

```
from typing import Annotated

from langchain_core.tools import tool, Tool
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langgraph.types import Command, interrupt

class State(TypedDict):
    messages: Annotated[list, add_messages]


from langchain_community.utilities import SerpAPIWrapper

# 定义Tool
# 需要定义环境变量 export GOOGLE_API_KEY="", 在网站上注册并生成API Key: https://serpapi.com/searches

# search = SerpAPIWrapper()
# tools = [
#     Tool(
#         name="Search",
#         func=search.run,
#         description="useful for when you need to answer questions about current events",
#     )
# ]
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]
@tool
def Search(query: str) -> str:
    "Perform information retrieval based on user queries."
    tool = SerpAPIWrapper()
    return tool.run(query)
    
    
tools = [Search, human_assistance]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
```

### 编译图

```
memory = MemorySaver()

graph = graph_builder.compile(checkpointer=memory)
```

### 向聊天机器人提问

向聊天机器人提出一个能调用的uman_assistance工具的问题：

```
user_input = "I need some expert guidance for building an AI agent. Could you request assistance for me?"
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

输出：

```
================================[1m Human Message [0m=================================

I need some expert guidance for building an AI agent. Could you request assistance for me?
==================================[1m Ai Message [0m==================================
Tool Calls:
  human_assistance (call_8fded5dd235844db9ea6eb)
 Call ID: call_8fded5dd235844db9ea6eb
  Args:
    query: I need some expert guidance for building an AI agent.

```

 聊天机器人生成了一个工具调用，但随后执行被中断。如果你检查图状态，会发现它在工具节点处停止：

```
snapshot = graph.get_state(config)
snapshot.next
```

> 与Python的内置input()函数类似，在工具内部调用interrupt将暂停执行。进度会根据checkpointer进行持久化；因此，如果使用Postgres进行持久化，只要数据库处于运行状态，就可以随时恢复执行。在此示例中，它使用内存中的checkpointer进行持久化，只要Python内核正在运行，就可以随时恢复。 

### 恢复执行

要恢复执行，请传递一个包含工具所需数据的Command对象。此数据的格式可根据需要自定义。在本示例中，使用一个键为"data"的字典：

```
human_response = (
    "We, the experts are here to help! We'd recommend you check out LangGraph to build your agent."
    " It's much more reliable and extensible than simple autonomous agents."
)

human_command = Command(resume={"data": human_response})

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

## 自定义状态

在本教程中，将向状态添加更多字段，以便在不依赖消息列表的情况下定义复杂行为。聊天机器人将使用其搜索工具查找特定信息，并将其转发给人工进行审核。

### 向状态添加键

 通过向状态添加name和birthday键，更新聊天机器人以查询实体的生日：

```
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str
    birthday: str
```

### 更新工具内部的状态

在`human_assistance`工具内部填充状态键。这使得人类可以在信息存储到状态中之前对其进行审查。使用[`Command`](https://langchain-ai.github.io/langgraph/concepts/low_level/#using-inside-tools)从工具内部发出状态更新。

```
from typing import Annotated
from langchain_core.messages import ToolMessage
from langchain_core.tools import InjectedToolCallId, tool
from langgraph.types import Command

@tool
def human_assistance(
    name: str, birthday: str, tool_call_id: Annotated[str, InjectedToolCallId]
) -> Command:
    """Request assistance from a human."""
    print(f"--- Entering human_assistance tool ---")
    print(f"Initial name: {name}, birthday: {birthday}")
    from  langgraph.types import  interrupt
    human_response = interrupt(
        {
            "question": "Is this correct?",
            "name": name,
            "birthday": birthday,
        },
    )
    print(f"Human response: {human_response}")

    if human_response.get("correct", "").lower().startswith("y"):
        verified_name = name
        verified_birthday = birthday
        response = "Correct"
        print("Human confirmed correctness.")
    else:
        verified_name = human_response.get("name", name)
        verified_birthday = human_response.get("birthday", birthday)
        response = f"Made a correction: {human_response}"
        print(f"Human made corrections. New name: {verified_name}, new birthday: {verified_birthday}")

    state_update = {
        "name": verified_name,
        "birthday": verified_birthday,
        "messages": [ToolMessage(response, tool_call_id=tool_call_id)],
    }
    print(f"State update being returned: {state_update}")
    print(f"--- Exiting human_assistance tool ---")
    return Command(update=state_update)
```

### 使用聊天机器人查询

提示聊天机器人查找LangGraph库的“诞生日期”，并指示聊天机器人在获取所需信息后调用human_assistance工具。通过在工具的参数中设置name和birthday，你可以迫使聊天机器人为这些字段生成建议。

```
user_input = (
    "Can you look up when LangGraph was released? "
    "When you have the answer, use the human_assistance tool for review."
)
config = {"configurable": {"thread_id": "1"}}

events = graph.stream(
    {"messages": [{"role": "user", "content": user_input}]},
    config,
    stream_mode="values",
)
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

输出：

```
================================[1m Human Message [0m=================================

Can you look up when LangGraph was released? When you have the answer, use the human_assistance tool for review.
==================================[1m Ai Message [0m==================================
Tool Calls:
  tavily_search (call_103316aa46a54bab99e235)
 Call ID: call_103316aa46a54bab99e235
  Args:
    query: When was LangGraph released?
=================================[1m Tool Message [0m=================================
Name: tavily_search

{"query": "When was LangGraph released?", "follow_up_questions": null, "answer": null, "images": [], "results": [{"title": "Releases · langchain-ai/langgraph - GitHub", "url": "https://github.com/langchain-ai/langgraph/releases", "content": "Releases · langchain-ai/langgraph · GitHub *   fix(langgraph): remove deprecated `output` usage in favor of `output_schema` (#5095) *   Remove Checkpoint.writes (#4822) *   Remove old checkpoint test fixtures (#4814) *   fix(langgraph): remove deprecated `output` usage in favor of `output_schema` (#5095) *   Remove support for node reading a single managed value *   Remove Checkpoint.writes (#4822) *   Remove Checkpoint.pending_sends (#4820) *   Remove old checkpoint test fixtures (#4814) Changes since checkpoint==2.0.26 *   langgraph-checkpoint 2.1.0 *   Preparation for 0.5 release: langgraph-checkpoint (#5124) *   Remove Checkpoint.writes *   Remove Checkpoint.pending_sends *   Remove Checkpoint.writes (#4822) *   Remove Checkpoint.pending_sends (#4820) *   Remove old checkpoint test fixtures (#4814) *   Remove postgres shallow checkpointer (#4813) *   Remove Checkpoint.writes *   Remove Checkpoint.pending_sends *   Remove old checkpoint test fixtures *   Remove postgres shallow checkpointer", "score": 0.98562, "raw_content": null}, {"title": "LangGraph - LangChain", "url": "https://www.langchain.com/langgraph", "content": "Design agent-driven user experiences with LangGraph Platform's APIs. Quickly deploy and scale your application with infrastructure built for agents. LangGraph sets the foundation for how we can build and scale AI workloads — from conversational agents, complex task automation, to custom LLM-backed experiences that 'just work'. The next chapter in building complex production-ready features with LLMs is agentic, and with LangGraph and LangSmith, LangChain delivers an out-of-the-box solution to iterate quickly, debug immediately, and scale effortlessly.” LangGraph sets the foundation for how we can build and scale AI workloads — from conversational agents, complex task automation, to custom LLM-backed experiences that 'just work'. LangGraph Platform is a service for deploying and scaling LangGraph applications, with an opinionated API for building agent UXs, plus an integrated developer studio.", "score": 0.98391, "raw_content": null}], "response_time": 1.65}
==================================[1m Ai Message [0m==================================
Tool Calls:
  human_assistance (call_0fb6aff5612246d9976f84)
 Call ID: call_0fb6aff5612246d9976f84
  Args:
    name: LangGraph
    birthday: 2023-10-01
--- Entering human_assistance tool ---
Initial name: LangGraph, birthday: 2023-10-01

```

### 添加人工介入

聊天机器人未能识别出正确日期，因此请为其提供相关信息：

```
human_command = Command(
    resume={
        "name": "LangGraph",
        "birthday": "Jan 17, 2024",
    },
)

events = graph.stream(human_command, config, stream_mode="values")
for event in events:
    if "messages" in event:
        event["messages"][-1].pretty_print()
```

输出：

```
==================================[1m Ai Message [0m==================================
Tool Calls:
  human_assistance (call_0fb6aff5612246d9976f84)
 Call ID: call_0fb6aff5612246d9976f84
  Args:
    name: LangGraph
    birthday: 2023-10-01
--- Entering human_assistance tool ---
Initial name: LangGraph, birthday: 2023-10-01
Human response: {'name': 'LangGraph', 'birthday': 'Jan 17, 2024'}
Human made corrections. New name: LangGraph, new birthday: Jan 17, 2024
State update being returned: {'name': 'LangGraph', 'birthday': 'Jan 17, 2024', 'messages': [ToolMessage(content="Made a correction: {'name': 'LangGraph', 'birthday': 'Jan 17, 2024'}", tool_call_id='call_0fb6aff5612246d9976f84')]}
--- Exiting human_assistance tool ---
=================================[1m Tool Message [0m=================================
Name: human_assistance

Made a correction: {'name': 'LangGraph', 'birthday': 'Jan 17, 2024'}
==================================[1m Ai Message [0m==================================

The release date of LangGraph is January 17, 2024. Let me know if you need any further assistance!

```

## 时间回溯

在典型的聊天机器人工作流程中，用户与机器人进行一次或多次交互以完成任务。记忆和人工介入可在图状态中设置检查点并控制未来的回复。 如果希望用户能够从先前的回复开始并探索不同的结果，或者，希望用户能够回溯聊天机器人的工作以纠正错误或尝试不同的策略可以使用LangGraph的内置时间回溯功能来创建此类功能。

### 回退Graph

通过使用图表的 `get_state_history` 方法获取检查点来回退graph。然后，你可以在这个先前的时间点恢复执行。

```

```

