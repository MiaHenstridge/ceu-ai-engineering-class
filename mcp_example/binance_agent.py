import asyncio
import sys

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp
from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent

# binance_mcp = MCPServerStreamableHttp(
#     name="Binance MCP",
#     params={"url": "http://localhost:8000/mcp"},
# )

binance_mcp = MCPServerStreamableHttp(
    name="Binance MCP",
    params={"url": "https://binance-mcp.onrender.com/mcp"},
)

agent = Agent(
    name="Crypto Assistant",
    instructions="You are a helpful crypto assistant. Use the available tools to answer questions about crypto prices.",
    model="litellm/bedrock/eu.amazon.nova-lite-v1:0",
    mcp_servers=[binance_mcp],
)


async def main():
    async with binance_mcp:
        result = Runner.run_streamed(agent, "What's the current price of Bitcoin?")

        # From this point on all the boilerplace you see is there  
        # just to print all the agent events
        
        async for event in result.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                data = event.data
                if hasattr(data, "type") and data.type == "response.output_text.delta":
                    sys.stdout.write(data.delta)
                    sys.stdout.flush()

            elif isinstance(event, RunItemStreamEvent):
                if event.name == "tool_called":
                    raw = event.item.raw_item
                    tool_name = getattr(raw, "name", "?")
                    tool_args = getattr(raw, "arguments", "")
                    print(f"\n[Tool Call] {tool_name}({tool_args})")

                elif event.name == "tool_output":
                    output = str(event.item.output)
                    print(f"[Tool Output] {output}\n")

        print()


asyncio.run(main())
