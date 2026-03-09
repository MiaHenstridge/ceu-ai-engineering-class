# Hackathon: Build an AI Agent That Actually Helps Someone

> **DRAFT** — This document is a work in progress and may is going to before the hackathon.

## The Challenge

Build a working AI agent using the **OpenAI Agents SDK** and **AWS Bedrock** that solves a real problem for a specific user.

Your agent **must** use at least **3 of the following capabilities** you learned in class:

- **Tool calling** — custom function tools
- **RAG** — bring your own data via ChromaDB
- **MCP** — I.e. Exa
- **Guardrails** — input validation, topic enforcement
- **Memory** — multi-turn conversations
- **Multi-agent orchestration**

You may pick **any domain** — it does NOT have to be nutrition.

### Example ideas

- A travel planning agent that searches the web and remembers preferences
- A study buddy that answers questions from uploaded course notes (RAG)
- A personal finance advisor with guardrails against bad advice
- A customer support agent that routes to specialist sub-agents
- A fitness coach that looks up exercises and tracks your plan across turns

## Teams

- **5 teams of 4**
- Self-selected, but the instructor may rebalance

## Schedule (200 minutes total)

| Block               | Duration | What happens |
|----------------------|----------|--------------|
| Kickoff & teams      | 10 min   | Form teams, pick a topic, assign roles |
| Build sprint 1       | 40 min   | Work... |
| Check-in             | 5 min    | Quick stand-up: what works, what's blocked |
| Build sprint 2       | 40 min   | Work... |
| Check-in             | 5 min    | Quick stand-up: what works, what's blocked |
| **BREAK**            | 30 min   | Break |
| Build sprint 3       | 40 min   | Work... |
| Deployment           | 10 min   | Prepare a 5-minute live demo |
| Presentations        | 45 min   | 5 teams x 8 min (5 min demo + 3 min Q&A) |
| Wrap-up              | 5 min    | Feedback, pass criteria confirmation |

## What You Present (= Pass Criteria)

1. **Live demo** (5 min) — Show the agent **running in Chainlit**. Walk us through a real conversation that demonstrates your 2+ capabilities.
3. **Q&A** (3 min) — Be ready to explain your architecture choices.

**You pass** if your agent runs, demonstrates at least 3 capabilities, and your team can explain what it does.

## Available Models

The notebooks use **Amazon Nova Lite**, but you have access to other Bedrock models too. Use these litellm model IDs (same format as in the notebooks):

| Model | LiteLLM ID |
|-------|-----------|
| Amazon Nova Lite | `litellm/bedrock/eu.amazon.nova-lite-v1:0` |
| Amazon Nova Micro *(fastest, cheapest)* | `litellm/bedrock/eu.amazon.nova-micro-v1:0` |
| Mistral 7B Instruct | `litellm/bedrock/mistral.mistral-7b-instruct-v0:2` |
| Claude 3 Haiku *(most capable)* | `litellm/bedrock/anthropic.claude-3-haiku-20240307-v1:0` |

You can mix models — e.g. use Nova Micro for fast sub-agents and Claude 3 Haiku for the main reasoning agent.

## Tips

- **Get something working first.** A basic agent with one tool in the first 60 minutes. Make it smarter in the second half.
- **Deploy latest in sprint 2.** By the end of sprint two, you must be able to deploy the agent.
- **Don't spend all your time on data.** A 50-row CSV or a short text file is plenty for RAG. See `notebooks/4_rag_setup.ipynb` for an example.
- **Start from the chatbot examples.** The `chatbot/` folder has a working agent skeleton — swap the domain and data.
- **Assign roles.** One person on agent logic, one on data/RAG, one on prompt engineering/testing, one on the presentation.
