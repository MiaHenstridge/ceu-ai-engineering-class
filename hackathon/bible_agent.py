import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from agents import Agent, FunctionTool, function_tool
from agents.mcp import MCPServerStreamableHttp

import dotenv
dotenv.load_dotenv()

MODEL = "litellm/bedrock/eu.amazon.nova-lite-v1:0"


def bedrock_tool(tool: dict) -> FunctionTool:
    """Converts an OpenAI Agents SDK function_tool to a Bedrock-compatible FunctionTool."""
    return FunctionTool(
        name=tool["name"],
        description=tool["description"],
        params_json_schema={
            "type": "object",
            "properties": {
                k: v for k, v in tool["params_json_schema"]["properties"].items()
            },
            "required": tool["params_json_schema"].get("required", []),
        },
        on_invoke_tool=tool["on_invoke_tool"],
    )


# ── ChromaDB setup ───────────────────────────────────────────────
chroma_path = Path(__file__).parent.parent / "chroma"

ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L12-v2"
)

chroma_client = chromadb.PersistentClient(path=str(chroma_path))
bible_db = chroma_client.get_collection(name="genz_bible", embedding_function=ef)


# ── RAG tool ─────────────────────────────────────────────────────
@function_tool
def bible_lookup_tool(query: str, max_results: int = 3) -> str:
    """
    Look up relevant Bible verses for a given question or topic.
    Searches using the original KJV text for accuracy, and returns
    the GenZ translation for the answer.

    Use this tool FIRST before falling back to Exa web search.
    If this tool returns 'NO_RELEVANT_RESULTS', use Exa to search
    the web for the answer and translate it into GenZ language yourself.

    Args:
        query: The question or topic to look up (e.g. 'love your enemies', 'creation of the world').
        max_results: The maximum number of verses to return.

    Returns:
        A string containing the relevant Bible verses in GenZ translation,
        or 'NO_RELEVANT_RESULTS' if nothing relevant was found.
    """
    results = bible_db.query(
        query_texts=[query],
        n_results=max_results,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"][0]:
        return "NO_RELEVANT_RESULTS"

    # Filter out low-relevance results using cosine distance threshold
    RELEVANCE_THRESHOLD = 1.0
    formatted_results = []
    for doc, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        if distance > RELEVANCE_THRESHOLD:
            continue
        reference = meta["reference"]
        genz_text = meta.get("genz_text", doc)  # fall back to KJV if genz_text missing
        formatted_results.append(f"{reference}: {genz_text}")

    if not formatted_results:
        return "NO_RELEVANT_RESULTS"

    return "Relevant Bible verses (GenZ translation):\n" + "\n".join(formatted_results)


# ── Exa MCP ──────────────────────────────────────────────────────
exa_search_mcp = MCPServerStreamableHttp(
    name="Exa Search MCP",
    params={
        "url": f"https://mcp.exa.ai/mcp?exaApiKey={os.environ.get('EXA_API_KEY')}",
        "timeout": 90,
    },
    client_session_timeout_seconds=90,
    cache_tools_list=True,
    max_retry_attempts=1,
)


# ── Agent ────────────────────────────────────────────────────────
bible_agent = Agent(
    name="GenZ Bible Assistant",
    instructions="""
    You are a Bible assistant who answers questions about the Bible using GenZ language translation.
    You are knowledgeable, engaging, and keep it real with the user. No cap.

    Follow this workflow for every question:
    1) ALWAYS try bible_lookup_tool first to find relevant verses from our database.
    2) If bible_lookup_tool returns 'NO_RELEVANT_RESULTS', fall back to Exa web search
       to find the answer from a trusted GenZ Bible website (e.g. https://genz.bible/).).
       Then translate the answer into GenZ language yourself before responding.
    3) Never skip step 1 — always check the RAG database before going to the web.

    When answering:
    - Answer using the GenZ translation from the tool, or the GenZ translation from https://genz.bible/ if using Exa
    - Keep answers concise but insightful
    - Use GenZ slang naturally (e.g. fr fr, no cap, bussin, slay, lowkey, vibe, it's giving)
    - Always cite the Bible reference (e.g. John 3:16) when quoting a verse in GenZ translation
    - If the question is not related to the Bible at all, let the user know that's not your vibe
    """,
    model=MODEL,
    tools=[bedrock_tool(bible_lookup_tool.__dict__)],
    mcp_servers=[exa_search_mcp],
)