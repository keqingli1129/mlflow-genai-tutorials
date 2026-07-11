import os

import mlflow
from dotenv import load_dotenv

from utils.clnt_utils import (
    get_ai_gateway_model_names,
    get_databricks_ai_gateway_langchain_client,
    is_databricks_ai_gateway_client,
)


def setup():
    load_dotenv()

    # Configure MLflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    from langchain.chat_models import init_chat_model

    use_databricks_provider = is_databricks_ai_gateway_client()
    if use_databricks_provider:
        model_name = get_ai_gateway_model_names()[0]
        llm = get_databricks_ai_gateway_langchain_client(model_name)
    else:
        # Initialize the LLM - uses OpenAI by default
        model_name = "gpt-5-mini"
        llm = init_chat_model(f"openai:{model_name}")

    if not use_databricks_provider and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

    mlflow.set_experiment("11-deep-agents-langgraph")

    # Enable auto-tracing for LangChain/LangGraph (covers Deep Agents)
    mlflow.langchain.autolog()

    print("✅ Environment configured: using", "Databricks AI Gateway" if use_databricks_provider else "OpenAI", "client")
    print(f"   MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"   Experiment: {mlflow.get_experiment_by_name('11-deep-agents-langgraph').name}")
    print(f"   Using model: {model_name}")

    return llm, model_name, use_databricks_provider


# Define custom tools the agent can use to search a knowledge base.
# This is a simple knowledge base that we'll use to test the agent.
# In a real use case, this would be a much larger and more complex knowledge base, most
# likely stored in a database or a vector database.
# For this example, we'll just use a small knowledge base of 4 topics.

def search_knowledge_base(query: str) -> str:
    """Search an internal knowledge base for information about MLflow, GenAI, and Agentic Workflow topics.
    Returns relevant information snippets."""

    knowledge = {
        "mlflow tracing": (
            "MLflow Tracing provides observability for AI applications. It captures "
            "hierarchical spans showing LLM calls, tool invocations, and retrieval steps. "
            "Supports auto-tracing for OpenAI, LangChain, LlamaIndex, and LangGraph, and many more. "
            "Traces are stored in the MLflow tracking server and viewable in the UI. With Managved MLflow traces configured to be stores in Unity Catalog, you can also access them in Databricks SQL."
        ),
        "mlflow evaluation": (
            "MLflow GenAI Evaluation (mlflow.genai.evaluate) assesses AI application quality "
            "using built-in scorers (Correctness, RelevanceToQuery, Safety) and custom scorers. "
            "It integrates with tracing to evaluate end-to-end agent behavior. "
            "Supports LLM-as-a-judge and Agent-as-a-judge patterns for automated quality assessment."
        ),
        "deep agents": (
            "Deep Agents are LangChain's open-source agent harness built on LangGraph. "
            "They add planning (todo lists), file system tools, and sub-agent delegation "
            "to the standard tool-calling loop. Designed for long-running, multi-step tasks "
            "like research, coding, and analysis."
        ),
        "langgraph": (
            "LangGraph is a framework for building stateful, multi-actor AI applications "
            "using graph-based workflows. It supports conditional routing, cycles, "
            "checkpointing, and streaming. Agents built with LangGraph are automatically "
            "traced by MLflow when mlflow.langchain.autolog() is enabled."
        ),
    }
    # Simple keyword matching
    results = []
    for topic, info in knowledge.items():
        if any(word in query.lower() for word in topic.split()):
            results.append(f"[{topic.upper()}]: {info}")
    return "\n\n".join(results) if results else f"No results found for: {query}"


def get_latest_stats(category: str) -> str:
    """Get the latest statistics for a given MLflow/AI category."""
    stats = {
        "adoption": "MLflow has 20M+ monthly downloads, 100K+ GitHub stars, and 1000+ contributors.",
        "performance": "GPT-4o averages 250ms first-token latency. Claude Sonnet: 200ms.",
        "cost": "GPT-4o: $2.50/1M input tokens. Claude Sonnet: $3/1M input tokens.",
    }
    return stats.get(category.lower(), f"No stats available for: {category}")


def example_1_basic_deep_agent(llm):
    from deepagents import create_deep_agent

    print("Tools defined: search_knowledge_base, get_latest_stats")

    # Create a basic Deep Agent with custom tools
    research_agent = create_deep_agent(
        model=llm,  # The LLM to use for the agent, created above
        tools=[search_knowledge_base, get_latest_stats],
        system_prompt=(
            "You are a research assistant. When given a research topic:\n"
            "1. Use write_todos to plan your research steps\n"
            "2. Search the knowledge base for relevant information\n"
            "3. Gather supporting statistics\n"
            "4. Synthesize findings into a structured summary with sections: "
            "Overview, Key Findings, Statistics, and Conclusion"
        ),
    )

    print(f"Deep Agent created — type: {type(research_agent)}")

    # Run the research agent
    result = research_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    "Research how MLflow provides observability for AI agents. "
                    "Cover tracing capabilities, evaluation features, and "
                    "how it integrates with frameworks like LangGraph and Deep Agents."
                ),
            }
        ]
    })

    # Print the agent's final response
    print(result["messages"][-1].content)

    return research_agent


def example_2_filesystem(llm):
    import os
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    # Create a workspace directory with an initial draft document for the agent to work on
    # and expand on it in the next steps.
    workspace_dir = "./agent_workspace"
    os.makedirs(workspace_dir, exist_ok=True)

    draft_content = """# MLflow GenAI Platform Overview

MLflow is a tool for machine learning. It does tracking and stuff.

## Tracing
MLflow can trace things. It works with some frameworks.

## Evaluation
You can evaluate models with MLflow. It has some scorers.

## Conclusion
MLflow is good for ML.
"""

    with open(os.path.join(workspace_dir, "draft.md"), "w") as f:
        f.write(draft_content)

    print(f"Workspace created at: {workspace_dir}")
    print(f"Draft document written ({len(draft_content)} chars)")

    # Create a Deep Agent with file system backend
    editor_agent = create_deep_agent(
        model=llm,
        system_prompt=(
            "You are a technical editor. Your task:\n"
            "1. Read the draft document at /draft.md\n"
            "2. Identify 3 specific improvements (vague language, missing details, weak structure)\n"
            "3. Write your improvement plan to /edit_plan.md\n"
            "4. Apply each improvement by editing the draft\n"
            "5. Write the final polished version to /final.md\n\n"
            "Be specific and substantive in your edits. Replace vague statements with "
            "concrete technical details about MLflow's GenAI capabilities."
        ),
        backend=FilesystemBackend(
            root_dir=workspace_dir,
            virtual_mode=True,  # Restricts file access to the workspace
        ),
    )

    print("Editor agent created with FilesystemBackend")

    # Run the editor agent
    edit_result = editor_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": "Please review and improve the draft at /draft.md. It needs to be more specific and technically accurate.",
            }
        ]
    })

    print(edit_result["messages"][-1].content)

    # Check the files the agent created/modified
    for filename in ["edit_plan.md", "final.md"]:
        filepath = os.path.join(workspace_dir, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                content = f.read()
            print(f"\n{'='*60}")
            print(f"📄 {filename}")
            print(f"{'='*60}")
            print(content[:500])
            if len(content) > 500:
                print(f"... ({len(content)} chars total)")
        else:
            print(f"⚠️  {filename} not found — agent may have used different file names")

    return workspace_dir


def example_3_subagents(llm):
    from deepagents import create_deep_agent

    # Create a coordinator agent with specialized sub-agents
    coordinator_agent = create_deep_agent(
        model=llm,
        system_prompt=(
            "You are a Technical Report Coordinator. When asked to produce a report:\n"
            "1. Delegate research to the 'researcher' sub-agent\n"
            "2. Send the research findings to the 'analyst' sub-agent for analysis\n"
            "3. Send the analysis to the 'writer' sub-agent to produce the final report\n"
            "4. Review the final report and present it to the user\n\n"
            "Use the task() tool to delegate work to each sub-agent. "
            "Provide clear, specific instructions to each sub-agent."
        ),
        tools=[search_knowledge_base, get_latest_stats],
        subagents=[
            {
                "name": "researcher",
                "description": "Gathers information from the knowledge base and collects statistics",
                "system_prompt": (
                    "You are a research specialist. Use the available tools to gather "
                    "comprehensive information on the assigned topic. Return your findings "
                    "as a structured list of key facts and data points."
                ),
                "tools": [search_knowledge_base, get_latest_stats],
            },
            {
                "name": "analyst",
                "description": "Analyzes research findings and identifies key trends and insights",
                "system_prompt": (
                    "You are a data analyst. Given research findings, identify the top 3 trends, "
                    "key insights, and any gaps in the data. Structure your analysis with: "
                    "Trends, Insights, and Recommendations sections."
                ),
            },
            {
                "name": "writer",
                "description": "Produces polished technical reports from analysis",
                "system_prompt": (
                    "You are a technical writer. Given an analysis, produce a concise, "
                    "well-structured report with: Executive Summary, Detailed Findings, "
                    "and Actionable Recommendations. Use clear, professional language."
                ),
            },
        ],
    )

    print("Coordinator agent created with 3 sub-agents: researcher, analyst, writer")

    # Run the coordinator — it will delegate to sub-agents automatically
    coordinator_result = coordinator_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    "Produce a technical report on the state of AI observability. "
                    "Cover how MLflow tracing and evaluation help teams monitor and "
                    "improve their AI agents. Include adoption statistics."
                ),
            }
        ]
    })

    print(coordinator_result["messages"][-1].content)


def example_4_evaluation(research_agent):
    import pandas as pd
    from mlflow.genai.scorers import RelevanceToQuery, Safety, Guidelines

    # Define evaluation dataset — multiple queries for the research agent
    eval_data = pd.DataFrame({
        "inputs": [
            {"query": "What is MLflow tracing and how does it work?"},
            {"query": "How do Deep Agents compare to standard LangGraph agents?"},
            {"query": "What evaluation capabilities does MLflow provide for GenAI?"},
            {"query": "How does LangGraph enable stateful agent workflows?"},
        ],
    })

    print(f"Evaluation dataset: {len(eval_data)} queries")

    # Wrap the research agent as a predict function for evaluation
    def research_predict(query: str) -> str:
        """Run the research agent and return its response."""
        result = research_agent.invoke({
            "messages": [{"role": "user", "content": query}]
        })
        return result["messages"][-1].content

    # Define a custom Guidelines scorer for research quality
    research_quality_guidelines = Guidelines(
        name="research_completeness",
        guidelines=(
            "The response should be a well-structured research summary that includes: "
            "(1) An overview or introduction to the topic, "
            "(2) Specific technical details and facts (not vague generalizations), "
            "(3) Multiple aspects or dimensions of the topic covered, "
            "(4) A clear conclusion or synthesis. "
            "Responses that are too brief, overly vague, or miss key aspects should score lower."
        ),
    )

    print("Predict function and custom scorer defined")

    # Run evaluation
    eval_results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=research_predict,
        scorers=[
            RelevanceToQuery(),
            Safety(),
            research_quality_guidelines,
        ],
    )

    print("Evaluation complete!")
    print(eval_results.metrics)

    # View per-row results
    print(eval_results.tables["eval_results"])


def cleanup(workspace_dir):
    import shutil

    # Cleanup workspace
    shutil.rmtree(workspace_dir, ignore_errors=True)
    print("Workspace cleaned up.")


def main():
    llm, model_name, use_databricks_provider = setup()

    research_agent = example_1_basic_deep_agent(llm)
    workspace_dir = example_2_filesystem(llm)
    example_3_subagents(llm)
    example_4_evaluation(research_agent)
    cleanup(workspace_dir)


if __name__ == "__main__":
    main()
