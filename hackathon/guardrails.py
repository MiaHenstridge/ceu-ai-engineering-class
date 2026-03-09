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

input_guardrail_agent = Agent(
    name="Input Guardrail",
    instructions="""Check if the user is asking to look up Bible verses or asking a question that can be answered by finding relevant verses.
                    If they are asking you to write code, do math, have a casual conversation, or perform any task OTHER than finding verses, set is_valid_query to False.
                    """,
    output_type=ValidBibleQuery,
    model="litellm/bedrock/eu.amazon.nova-lite-v1:0",
)

output_guardrail_agent = Agent(
    name="Output Guardrail",
    instructions="""Check the generated response. It must be non-offensive.
                    If the response contains deeply offensive, hateful, or inappropriate language, set is_safe to False.
                    Always provide the text in redacted_response (redacted if it was unsafe, or just the original text if it was safe).
                    """,
    output_type=SafeOutput,
    model="litellm/bedrock/eu.amazon.nova-lite-v1:0",
)

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
async def offensive_output_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, response: str
) -> GuardrailFunctionOutput:
    # Notice this takes the output text and modifies it if needed
    result = await Runner.run(output_guardrail_agent, response, context=ctx.context)
    
    # We trigger the tripwire if it's NOT safe
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=(not result.final_output.is_safe),
    )