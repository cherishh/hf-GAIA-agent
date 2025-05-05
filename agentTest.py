
from langchain_core.messages import HumanMessage
from agents.agent import build_graph


# test
if __name__ == "__main__":
    
    question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"
    # Build the graph
    graph = build_graph()
    # Run the graph
    messages = [HumanMessage(content=question)]
    messages = graph.invoke({"messages": messages})
    for m in messages["messages"]:
        m.pretty_print()
