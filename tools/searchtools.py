from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
import os
from supabase.client import Client, create_client
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from langchain.tools.retriever import create_retriever_tool

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2")  # dim=768

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
supabase: Client = create_client(supabase_url, supabase_key)
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="documents",
    query_name="match_documents",
)

question_retrieve_tool = create_retriever_tool(
    vector_store.as_retriever(),
    "Question_Retriever",
    "Find similar questions in the vector database for the given question.",
)


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia for a query and return maximum 2 results. If you are suggested to search wikipedia, you should use this tool.
    Args:
        query: The search query."""
    search_docs = WikipediaLoader(
        query=query, load_max_docs=1, doc_content_chars_max=19000).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return formatted_search_docs


@tool
def web_search(query: str) -> str:
    """Search Tavily for a query and return maximum 3 results.
    Args:
        query: The search query."""
    search_docs = TavilySearchResults(max_results=3).invoke({"query": query})
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.get("url", "")}" page=""/>\n{doc.get("content", "")}\n</Document>'
            for doc in search_docs
        ]
    )
    return formatted_search_docs


@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ]
    )
    return formatted_search_docs


@tool
def similar_question_search(question: str) -> str:
    """Search the vector database for similar questions and return the first results.

    Args:
        question: the question human provided."""
    matched_docs = vector_store.similarity_search(question, 3)
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in matched_docs
        ])
    return formatted_search_docs
