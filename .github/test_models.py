"""
Quick smoke test: send "say hello" to every Bedrock model students have access to.
Run manually:  python .github/test_models.py
Requires:      AWS creds + OPENAI_API_KEY in env (or .env file)
"""

import asyncio
import dotenv
import os

dotenv.load_dotenv()
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

from agents import Agent, Runner

MODELS = [
    "litellm/bedrock/eu.amazon.nova-lite-v1:0",
    "litellm/bedrock/eu.amazon.nova-micro-v1:0",
    "litellm/bedrock/mistral.mistral-7b-instruct-v0:2",
    "litellm/bedrock/anthropic.claude-3-haiku-20240307-v1:0",
]


async def test_model(model_id: str) -> None:
    short_name = model_id.split("/")[-1]
    try:
        agent = Agent(
            name="Test",
            instructions="Respond with exactly: hello",
            model=model_id,
        )
        result = await Runner.run(agent, "Say hello.")
        print(f"  ✅ {short_name}: {result.final_output[:60]}")
    except Exception as e:
        print(f"  ❌ {short_name}: {e}")


async def main():
    print("Testing Bedrock models via Agents SDK...\n")
    for model in MODELS:
        await test_model(model)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
