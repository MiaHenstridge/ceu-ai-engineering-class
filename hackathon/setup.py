from agents import FunctionTool
import dotenv
import os

dotenv.load_dotenv()

_required = {
    "AWS credentials": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
    "OpenAI credentials": ["OPENAI_API_KEY"],
    "EXA credentials": ["EXA_API_KEY"],
}

for label, keys in _required.items():
    if all(os.getenv(k) for k in keys):
        print(f"✅ {label} found")
    else:
        missing = [k for k in keys if not os.getenv(k)]
        raise EnvironmentError(f"❌ {label} not found. Missing: {', '.join(missing)}")


def bedrock_tool(tool: dict) -> FunctionTool:
    """
    Converts an OpenAI tool to a Bedrock tool.
    """

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
