"""An explicit tool letting the agent deliberately search the user's
uploaded documents mid-plan, reusing the same retrieval pipeline
`ChatService` already uses automatically against the initial message.
"""

from src.rag.retriever import RagRetriever
from src.tools.base import Tool


def build_search_documents_tool(rag_retriever: RagRetriever) -> Tool:
    async def _search(arguments: dict) -> str:
        query = arguments.get("query", "")
        context = await rag_retriever.build_context(query)
        return context or "No relevant documents found."

    return Tool(
        name="search_documents",
        description=(
            "Search the user's uploaded documents for information relevant to a specific question."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for in the documents"},
            },
            "required": ["query"],
        },
        execute=_search,
    )
