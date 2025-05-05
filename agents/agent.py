from dotenv import load_dotenv

load_dotenv()


from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from tools.searchtools import wiki_search, web_search, arvix_search, vector_store
from tools.mathtools import multiply, add, subtract, divide, modulus


# load the system prompt from the file
with open("system_prompt.txt", "r") as f:
    system_prompt = f.read()

# System message
sys_msg = SystemMessage(content=system_prompt)

tools = [
    multiply,
    add,
    subtract,
    divide,
    modulus,
    wiki_search,
    web_search,
    arvix_search
]

# Build graph function
def build_graph():
    """Build the graph"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    def retriever(state: MessagesState):
        """Retriever node"""
        similar_question = vector_store.similarity_search(state["messages"][0].content)
        example_msg = HumanMessage(
            content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
        )
        return {"messages": [sys_msg] + state["messages"] + [example_msg]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges("assistant",
        tools_condition)
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()