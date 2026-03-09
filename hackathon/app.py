import chainlit as cl
import dotenv
from openai.types.responses import ResponseTextDeltaEvent

from agents import Runner, SQLiteSession
from bible_agent import bible_agent, exa_search_mcp

dotenv.load_dotenv()


@cl.on_chat_start
async def on_chat_start():
    # Connect Exa MCP once per chat session
    await exa_search_mcp.connect()

    # Create a SQLite memory session per user
    session = SQLiteSession("bible_conversation_history")
    cl.user_session.set("agent_session", session)

    # Welcome message
    await cl.Message(
        content="Yo, welcome to the GenZ Bible Assistant! 📖🔥 Ask me anything about the Bible and I'll keep it real with you. No cap."
    ).send()


@cl.on_chat_end
async def on_chat_end():
    # Clean up MCP connection when chat ends
    await exa_search_mcp.cleanup()


@cl.on_message
async def on_message(message: cl.Message):
    session = cl.user_session.get("agent_session")

    result = Runner.run_streamed(
        bible_agent,
        message.content,
        session=session,
    )

    msg = cl.Message(content="")
    async for event in result.stream_events():
        # Stream final message text to screen
        if event.type == "raw_response_event" and isinstance(
            event.data, ResponseTextDeltaEvent
        ):
            await msg.stream_token(token=event.data.delta)

        # Show tool calls as steps in the UI
        elif (
            event.type == "raw_response_event"
            and hasattr(event.data, "item")
            and hasattr(event.data.item, "type")
            and event.data.item.type == "function_call"
            and len(event.data.item.arguments) > 0
        ):
            with cl.Step(name=f"{event.data.item.name}", type="tool") as step:
                step.input = event.data.item.arguments

    await msg.update()