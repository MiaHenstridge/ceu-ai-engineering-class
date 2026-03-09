import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from pydantic import BaseModel
from agents import (
    Agent,
    FunctionTool,
    function_tool,
    RunContextWrapper,
    GuardrailFunctionOutput,
    input_guardrail,
    output_guardrail,
    Runner,
)
from agents.mcp import MCPServerStreamableHttp
from openai.types.responses import TResponseInputItem

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

        genz_text = meta.get("genz_text", "").strip()

        # Skip verses that have no GenZ translation — never return KJV to the agent
        if not genz_text:
            continue

        reference = meta["reference"]
        formatted_results.append(f"{reference}: {genz_text}")

    if not formatted_results:
        return "NO_RELEVANT_RESULTS"

    return "Relevant Bible verses (GenZ translation only):\n" + "\n".join(formatted_results)


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


# ── Guardrail models ─────────────────────────────────────────────
class ValidBibleQuery(BaseModel):
    is_valid_query: bool
    """Whether the user's prompt is a valid request to look up verses, and not trying to have an arbitrary conversation or task."""
    topic: str
    """The main topic of the user's query"""
    reason: str
    """If the query is invalid, explain why the trigger was tripped."""


class SafeOutput(BaseModel):
    is_safe: bool
    """Whether the generated response is free from deeply offensive, hateful, or inappropriate language."""
    redacted_response: str
    """If the original response contained offensive language, provide a redacted or safe version. Otherwise, copy the original."""
    reason: str
    """If the text was deemed unsafe, explain why."""


# ── Guardrail agents ─────────────────────────────────────────────
input_guardrail_agent = Agent(
    name="Input Guardrail",
    instructions="""Check if the user is asking to look up Bible verses or asking a question
                    that can be answered by finding relevant Bible verses.
                    If they are asking you to write code, do math, have a casual conversation,
                    or perform any task OTHER than finding or discussing Bible verses, set is_valid_query to False.
                    """,
    output_type=ValidBibleQuery,
    model=MODEL,
)

output_guardrail_agent = Agent(
    name="Output Guardrail",
    instructions="""Check the generated response for two things:
                    1. It must be non-offensive and free from hateful or inappropriate language.
                    2. It must be written in GenZ language — NOT in old-fashioned KJV English.
                       If the response contains KJV-style language (e.g. 'thou', 'thy', 'hath', 'begotten',
                       'verily', 'shall'), set is_safe to False and rewrite it in GenZ slang
                       in the redacted_response field.
                    Always provide text in redacted_response (rewritten if unsafe/KJV, original if already good).
                    """,
    output_type=SafeOutput,
    model=MODEL,
)


# ── Guardrail functions ───────────────────────────────────────────
@input_guardrail
async def bible_topic_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(input_guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=(not result.final_output.is_valid_query),
    )


@output_guardrail
async def genz_output_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, response: str
) -> GuardrailFunctionOutput:
    result = await Runner.run(output_guardrail_agent, response, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=(not result.final_output.is_safe),
    )


# ── Agent ────────────────────────────────────────────────────────
bible_agent = Agent(
    name="GenZ Bible Assistant",
    instructions="""
    You are a Bible assistant who answers questions about the Bible EXCLUSIVELY using GenZ language.
    You are knowledgeable, engaging, and keep it real with the user. No cap.

    Follow this workflow for every question:
    1) ALWAYS try bible_lookup_tool first to find relevant verses from our database.
    2) If bible_lookup_tool returns 'NO_RELEVANT_RESULTS', fall back to Exa web search
       to find the answer from the GenZ Bible website (https://genz.bible/).
       Only use the GenZ translation text from that site, never the original KJV text.
    3) Never skip step 1 — always check the RAG database before going to the web.

    STRICT OUTPUT RULES — you must follow these without exception:
    - ALWAYS quote verses in GenZ translation only — NEVER quote the original KJV English
    - The tool already gives you the GenZ translation — use exactly that text when quoting
    - If you find yourself writing old-fashioned English (e.g. 'thou', 'shall', 'begotten'), STOP and rephrase in GenZ
    - Use GenZ slang naturally in your own explanations (e.g. fr fr, no cap, bussin, slay, lowkey, vibe, it's giving)
    - Always cite the Bible reference (e.g. John 3:16) when quoting a verse
    - If the question is not related to the Bible at all, let the user know that's not your vibe
    """,
    model=MODEL,
    tools=[bedrock_tool(bible_lookup_tool.__dict__)],
    mcp_servers=[exa_search_mcp],
    input_guardrails=[bible_topic_guardrail],
    output_guardrails=[genz_output_guardrail],
)
