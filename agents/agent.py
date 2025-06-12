from tools.mathtools import multiply, add, subtract, divide, modulus, power, square_root
from tools.documenttools import save_file, download_file_from_url, extract_text_from_image, analyze_csv_file, analyze_excel_file
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_groq import ChatGroq
from tools.imagetools import analyze_image, transform_image, draw_on_image, generate_simple_image, combine_images
from tools.codetools import execute_code_multilang
from tools.searchtools import wiki_search, web_search, arxiv_search, vector_store
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langgraph.graph import START, StateGraph, MessagesState
from dotenv import load_dotenv

load_dotenv()


# load the system prompt from the file
with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

# System message
sys_msg = SystemMessage(content=system_prompt)


tools = [
    web_search,
    wiki_search,
    arxiv_search,
    multiply,
    add,
    subtract,
    divide,
    modulus,
    power,
    square_root,
    save_file,
    download_file_from_url,
    extract_text_from_image,
    analyze_csv_file,
    analyze_excel_file,
    execute_code_multilang,
    analyze_image,
    transform_image,
    draw_on_image,
    generate_simple_image,
    combine_images,
]


# Build graph function
def build_graph():
    """Build the graph"""
    # Load environment variables from .env file
    llm = ChatGroq(model="qwen-qwq-32b", temperature=0)

    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Node
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def retriever(state: MessagesState):
        """Retriever node"""
        similar_question = vector_store.similarity_search(
            state["messages"][0].content)

        if similar_question:  # Check if the list is not empty
            example_msg = HumanMessage(
                content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
            )
            return {"messages": [sys_msg] + state["messages"] + [example_msg]}
        else:
            # Handle the case when no similar questions are found
            return {"messages": [sys_msg] + state["messages"]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    builder.add_edge("tools", "assistant")

    # Compile graph
    return builder.compile()
