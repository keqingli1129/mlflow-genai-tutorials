import os
import uuid
from typing import Any

import mlflow
from dotenv import load_dotenv

from utils.clnt_utils import (
    get_ai_gateway_model_names,
    get_databricks_ai_gateway_client,
    get_openai_client,
    is_databricks_ai_gateway_client,
)

EXPERIMENT_NAME = "07-agent-evaluation"


def setup():
    load_dotenv()

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Check if we are using a Databricks AI Gateway client
    use_databricks_provider = is_databricks_ai_gateway_client()
    if use_databricks_provider:
        client = get_databricks_ai_gateway_client()
        models = get_ai_gateway_model_names()
        JUDGE_MODEL = models[2]
        AGENT_MODEL = models[0]
    else:
        # Initialize as an OpenAI client
        client = get_openai_client()
        JUDGE_MODEL = "gpt-5.2"
        AGENT_MODEL = "gpt-5.2"

    if not use_databricks_provider and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

    # Enable autologging
    mlflow.openai.autolog()

    print("✅ Environment configured for agent evaluation")
    print(f"use_databricks_provider: {use_databricks_provider}")
    print(f"MLflow tracking: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Agent model : {AGENT_MODEL}")
    print(f"Judge model : {JUDGE_MODEL}")

    return client, JUDGE_MODEL, AGENT_MODEL, use_databricks_provider


class SimpleQAAgent:
    """
    A simple Q&A agent for demonstration.
    """

    def __init__(self, client: Any, model: str = "gpt-5-mini"):
        self.model = model
        self.client = client

    # create a manual trace for the agent with the span type AGENT
    @mlflow.trace(name="qa_agent", span_type="AGENT")
    def answer(self, question: str) -> str:
        """Answer a question."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": """You are a helpful Agent Assistant. Provide concise, accurate answers,
                no hallucinations, with a focus on MLflow and GenAI."""},
                {"role": "user", "content": question}
            ],
        )
        return response.choices[0].message.content


def create_and_test_agent(client, AGENT_MODEL):
    # Initialize agent
    agent = SimpleQAAgent(client=client, model=AGENT_MODEL)

    # Test the agent with a single question
    test_question = "What is MLflow's Gen AI and Agent platform capabilities?"
    test_response = agent.answer(test_question)
    print(f"Question: {test_question}")
    print(f"\nResponse: {test_response}")

    return agent


def quick_evaluation(use_databricks_provider, JUDGE_MODEL):
    from mlflow.genai.scorers import RelevanceToQuery

    # Configure judge model URI based on provider
    if use_databricks_provider:
        databricks_token = os.environ.get("DATABRICKS_TOKEN")
        ai_gateway_base_url = os.environ.get("AI_GATEWAY_BASE_URL")
        os.environ["OPENAI_API_KEY"] = databricks_token
        os.environ["OPENAI_API_BASE"] = ai_gateway_base_url
        judge_model_uri = f"openai:/{JUDGE_MODEL}"
    else:
        judge_model_uri = f"openai:/{JUDGE_MODEL}"

    # Grab the trace from our agent call above — it already has inputs and outputs
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())

    # Evaluate with built-in RelevanceToQuery scorer
    quick_scorer = RelevanceToQuery(model=judge_model_uri)

    print("🔄 Evaluating the test response with RelevanceToQuery scorer...\n")

    quick_result = mlflow.genai.evaluate(
        data=[trace],
        scorers=[quick_scorer]
    )

    # Show the result
    print("📊 Quick Evaluation Result:")
    print("-" * 40)
    score = quick_result.metrics.get("relevance_to_query/mean", "N/A")
    print(f"   Relevance Score: {score}")
    print("\n✅ The LLM judge evaluated our agent's response!")
    print("   Now let's learn about all the built-in scorers available...")


def build_builtin_scorers(use_databricks_provider, JUDGE_MODEL):
    from mlflow.genai.scorers import (
        RelevanceToQuery,
        Correctness,
        Guidelines,
        Safety
    )

    if use_databricks_provider:
        databricks_token = os.environ.get("DATABRICKS_TOKEN")
        ai_gateway_base_url = os.environ.get("AI_GATEWAY_BASE_URL")

        # Try configuring as OpenAI-compatible endpoint
        os.environ["OPENAI_API_KEY"] = databricks_token
        os.environ["OPENAI_API_BASE"] = ai_gateway_base_url

        judge_model_uri = f"databricks:/{JUDGE_MODEL}"
        print("🔧 Configured for Databricks AI Gateway")
        print(f"   Base URL: {ai_gateway_base_url}")
        print(f"   Model: {JUDGE_MODEL}")
    else:
        judge_model_uri = f"openai:/{JUDGE_MODEL}"

    print(f"🔧 Judge model URI: {judge_model_uri}")

    # Initialize built-in scorers
    relevance_scorer = RelevanceToQuery(model=judge_model_uri)
    correctness_scorer = Correctness(model=judge_model_uri)
    safety_scorer = Safety(model=judge_model_uri)

    #  Let's create a custom guidelines scorer, which will be used downstream to evaluate traces
    #  This guidelines judge, like a policy, dictates how the response should be formatted, and what the response
    # should contain or not contain.
    guidelines_scorer = Guidelines(
        model=judge_model_uri,
        guidelines=[ """
Response should be appropriately detailed:
- Simple factual questions: < 200 words
- Technical how-to questions: < 500 words
- Complex architectural questions: < 1000 words
"""
        ],
        name="custom_guidelines"
    )

    print("\n✅ Built-in scorers initialized:")
    print("   - RelevanceToQuery, Correctness, Safety, Guidelines")

    return relevance_scorer, correctness_scorer, safety_scorer, guidelines_scorer


def create_evaluation_dataset():
    # Evaluation dataset focused on MLflow for GenAI and Agent Observability
    # NOTE: Use 'expected_response' - this is the field name that MLflow's Correctness scorer expects
    from mlflow.genai import create_dataset
    eval_dataset = [
        {
            "inputs": {"question": "What is MLflow Tracing and why is it important for GenAI applications?"},
            "expectations": {"expected_response": "MLflow Tracing provides observability for GenAI applications by capturing the complete execution flow including LLM calls, retrieval steps, tool usage, and agent reasoning. It's important because it enables debugging, performance analysis, and understanding of complex AI pipelines."}
        },
        {
            "inputs": {"question": "How does MLflow help with prompt management in GenAI development?"},
            "expectations": {"expected_response": "MLflow's Prompt Registry allows you to version control prompts, tag and search prompt versions, link prompts to experiments, and collaborate with teams. This ensures reproducibility and systematic prompt engineering."}
        },
        {
            "inputs": {"question": "What are spans in MLflow Tracing and what types are available?"},
            "expectations": {"expected_response": "Spans are units of work captured during tracing. MLflow supports span types including LLM (for model calls), RETRIEVER (for RAG retrieval), TOOL (for function calls), AGENT (for agent orchestration), CHAIN (for sequential operations), and EMBEDDING (for vector operations)."}
        },
        {
            "inputs": {"question": "How can you evaluate LLM outputs using MLflow?"},
            "expectations": {"expected_response": "MLflow provides an evaluation framework with built-in scorers like RelevanceToQuery, Correctness, Guidelines, and Safety. You can also create custom scorers using the @scorer decorator or integrate third-party libraries like DeepEval and RAGAS."}
        },
        {
            "inputs": {"question": "What is the LLM-as-Judge pattern and how does MLflow support it?"},
            "expectations": {"expected_response": "LLM-as-Judge uses an LLM to evaluate outputs from another LLM, replacing brittle string matching with intelligent assessment. MLflow supports this through built-in scorers that use configurable judge models to provide scores and reasoning explanations."}
        },
        {
            "inputs": {"question": "How do you track costs and token usage in MLflow for GenAI?"},
            "expectations": {"expected_response": "MLflow automatically logs token usage (prompt tokens, completion tokens, total tokens) and can calculate costs based on model pricing. This data is captured in traces and experiment runs, enabling cost analysis and optimization across different models and configurations."}
        },
        {
            "inputs": {"question": "What frameworks does MLflow integrate with for GenAI auto-tracing?"},
            "expectations": {"expected_response": "MLflow provides auto-tracing for 40+ frameworks including OpenAI, Anthropic, LangChain, LlamaIndex, AWS Bedrock, Google Vertex AI, Cohere, Ollama, DSPy, AutoGen, and CrewAI. Auto-tracing automatically captures LLM calls without manual instrumentation."}
        },
        {
            "inputs": {"question": "How do you implement session-level tracing for multi-turn conversations?"},
            "expectations": {"expected_response": "Key concepts: (1) stable session identifier, (2) tag traces with session_id, (3) filter traces by session, (4) MLflow search capabilities"}
        },
    ]

    # Create and register the dataset. Registering the dataset allows you to use it in other experiments.
    # It's stored in the same experiment as the evaluation runs
    dataset  = create_dataset(
        name="regression_test_suite",
        experiment_id= mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id,
        tags={"type": "regression", "priority": "critical"},
    )

    dataset.merge_records(eval_dataset)

    print(f"✅ Evaluation dataset created with {len(eval_dataset)} examples")
    print("\n📋 Questions cover:")
    print("   - MLflow Tracing fundamentals")
    print("   - Prompt management")
    print("   - Span types and structure")
    print("   - LLM evaluation methods")
    print("   - LLM-as-Judge pattern")
    print("   - Cost and token tracking")
    print("   - Framework integrations")
    print("   - Session-level observability")

    return eval_dataset


def make_predict_fn(agent):
    def predict_fn(question: str) -> dict:
        """
        Prediction function wrapper for evaluation.

        Note: mlflow.genai.evaluate() unpacks the 'inputs' dict as keyword arguments,
        so the function signature must match the keys in your dataset's 'inputs' field.

        Dataset: {"inputs": {"question": "..."}}
        Called as: predict_fn(question="...")

        Args:
            question: The question string (unpacked from inputs dict)

        Returns:
            Dictionary with 'response' key
        """
        response = agent.answer(question)
        return {"response": response}

    print("✅ Prediction function defined")
    print("   Signature: predict_fn(question: str) -> dict")

    return predict_fn


def run_builtin_evaluation(eval_dataset, predict_fn, relevance_scorer, correctness_scorer, safety_scorer, guidelines_scorer):
    # Run evaluation
    print("🔄 Running evaluation with built-in scorers...\n")

    results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        scorers=[
            relevance_scorer,
            correctness_scorer,
            safety_scorer,
            guidelines_scorer
        ]
    )

    print("\n✅ Evaluation complete!")

    # Display metrics
    print("\n📊 Metrics Summary:")
    print("-" * 50)
    if results.metrics:
        for metric_name, value in results.metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.3f}")
            else:
                print(f"  {metric_name}: {value}")
    else:
        print("  No metrics returned")
        print("\n⚠️  Scorers returned None - this usually means the judge model call failed.")
        print("  Check the 'error_message' or similar columns above for details.")


def define_custom_scorers():
    from mlflow.genai import scorer

    @scorer
    def response_length_check(outputs: dict) -> bool:
        """
        Check if response is within acceptable length.
        Returns True if response is between 200 and 500 characters.
        """
        response = outputs.get("response", "")
        length = len(response)
        return 20 <= length <= 500

    @scorer
    def contains_keywords(outputs: dict, expectations: dict) -> bool:
        """
        Check if response contains key terms from expected answer.
        """
        response = outputs.get("response", "").lower()
        # Use 'expected_response' to match the dataset field name
        expected = expectations.get("expected_response", "").lower()

        # Extract key words (simple approach)
        key_words = [word for word in expected.split() if len(word) > 4]

        # Check if at least 30% of key words are present
        # If no keywords to check, fail conservatively (may indicate data issue)
        if not key_words:
            return False

        matches = sum(1 for word in key_words if word in response)
        return matches / len(key_words) >= 0.3

    @scorer
    def no_hallucination_markers(outputs: dict) -> bool:
        """
        Check for common hallucination markers.
        """
        response = outputs.get("response", "").lower()

        hallucination_markers = [
            "i think",
            "i believe",
            "probably",
            "might be",
            "i'm not sure",
            "as far as i know"
        ]

        return not any(marker in response for marker in hallucination_markers)

    print("✅ Custom scorers defined:")
    print("   - response_length_check")
    print("   - contains_keywords")
    print("   - no_hallucination_markers")

    return response_length_check, contains_keywords, no_hallucination_markers


def run_custom_evaluation(eval_dataset, predict_fn, response_length_check, contains_keywords, no_hallucination_markers):
    # Run evaluation with custom scorers
    print("🔄 Running evaluation with custom scorers...\n")

    custom_results = mlflow.genai.evaluate(
        data=eval_dataset,
        predict_fn=predict_fn,
        scorers=[
            response_length_check,
            contains_keywords,
            no_hallucination_markers
        ]
    )

    print("\n✅ Custom evaluation complete!")
    print("\nCustom Metrics Summary:")
    print("-" * 40)
    for metric_name, value in custom_results.metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.3f}")
        else:
            print(f"  {metric_name}: {value}")


def create_llm_judge(JUDGE_MODEL):
    from mlflow.genai.judges import make_judge

    # Create an LLM-as-a-Judge evaluator
    explanation_quality_judge = make_judge(
        name="explanation_quality",
        instructions=(
            "Analyze the agent's response in {{ trace }}.\n\n"
            "Evaluate the explanation quality on these criteria:\n"
            "1. **Clarity**: Is the response well-structured and easy to understand?\n"
            "2. **Completeness**: Does it cover the key aspects of the question?\n"
            "3. **Technical Accuracy**: Are technical terms used correctly?\n"
            "4. **Conciseness**: Is it appropriately detailed without unnecessary verbosity?\n\n"
            "Rate as 'yes' if the explanation meets all criteria, 'no' otherwise.\n"
            "Provide specific evidence from the response for your rating."
        ),
        model=f"openai:/{JUDGE_MODEL}",
    )

    print("✅ LLM-as-a-Judge created: 'explanation_quality'")
    print("   Judge model: openai:/{JUDGE_MODEL}")
    print("   Template variable: {{{{ trace }}}} (auto-populated with full trace)")

    return explanation_quality_judge


def run_judge_evaluation(agent, explanation_quality_judge):
    # Generate a fresh trace by calling the agent
    test_question_for_judge = "What is MLflow Tracing and why is it important for GenAI applications?"
    print(f"Question: {test_question_for_judge}\n")

    response = agent.answer(test_question_for_judge)
    print(f"Agent Response: {response[:500]} ...\n")

    # Retrieve the trace and evaluate using mlflow.genai.evaluate() — same pattern as cell 7
    trace = mlflow.get_trace(mlflow.get_last_active_trace_id())
    print(f"Trace ID: {trace.info.trace_id}")
    print("Running LLM-as-a-Judge evaluation...\n")

    judge_result = mlflow.genai.evaluate(
        data=[trace],
        scorers=[explanation_quality_judge]
    )

    print("=" * 50)
    print("Judge Evaluation Result:")
    print("-" * 50)
    for metric_name, value in judge_result.metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.3f}")
        else:
            print(f"  {metric_name}: {value}")
    print("=" * 50)


def build_deepeval_scorers(JUDGE_MODEL):
    from mlflow.genai.scorers.deepeval import (
        ConversationCompleteness,
        KnowledgeRetention,
        TopicAdherence,
        Toxicity,
    )

    # Initialize DeepEval scorers
    jude_model_uri = f"openai:/{JUDGE_MODEL}"

    completeness_scorer = ConversationCompleteness(model=jude_model_uri, threshold=0.7, include_reason=True)
    retention_scorer = KnowledgeRetention(model=jude_model_uri, threshold=0.7, include_reason=True)
    toxicity_scorer = Toxicity(model=jude_model_uri, threshold=0.7, include_reason=True)
    topic_scorer = TopicAdherence(model=jude_model_uri, threshold=0.7, include_reason=True, relevant_topics=["MLflow", "machine learning", "AI", "data science", "genai", "agent", "observability", "prompt engineering", "prompt management", "prompt registry", "experiment tracking"])

    print("✅ DeepEval scorers initialized:")
    print("   - ConversationCompleteness")
    print("   - KnowledgeRetention")
    print("   - TopicAdherence")
    print("   - Toxicity")
    print("\n Next, let's evaluate a multi-turn conversation using these DeepEval scorers")

    return completeness_scorer, retention_scorer, topic_scorer, toxicity_scorer


class ConversationalAgent:
    """
    An agent that maintains conversation history for multi-turn interactions.
    """

    def __init__(self, client: Any, model: str = "gpt-5-mini"):
        self.model = model
        self.client = client
        self.conversation_history = []
        self.session_id = str(uuid.uuid4())

    def reset(self):
        """Reset conversation history and start new session."""
        self.conversation_history = []
        self.session_id = str(uuid.uuid4())

    @mlflow.trace(name="conversational_agent", span_type="AGENT")
    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response, maintaining history.
        """
        # Tag trace with session ID for grouping
        mlflow.update_current_trace(metadata={
            "mlflow.trace.session": self.session_id,
            "turn_number": len(self.conversation_history) // 2 + 1
        })

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Prepare messages with system prompt. Add previous context to the conversation.
        messages = [
            {"role": "system",
            "content": """You are a helpful MLflow expert assistant. Answer questions about MLflow clearly, accurately,
                        concisely, without hallucinations, and accurately. Remember previous context in the conversation."""}
        ] + self.conversation_history

        # Get response from the Agent LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )

        assistant_message = response.choices[0].message.content

        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })

        return assistant_message


def simulate_conversation(client, AGENT_MODEL):
    print("✅ ConversationalAgent defined with session tracking")

    # Simulate a multi-turn conversation
    conv_agent = ConversationalAgent(client=client, model=AGENT_MODEL)

    conversation_turns = [
        "What is MLflow for GenAI?",
        "What are its main GenAI main components?",
        "Tell me more about the Tracing component.",
        "How does it compare to other tools?",
        "What is the difference between Tracing and Tracking?",
        "How do I get started with MLflow for GenAI?",
    ]

    print("🗣️ Multi-Turn Conversation\n")
    print("=" * 60)

    for i, user_msg in enumerate(conversation_turns, 1):
        print(f"\n[Turn {i}]")
        print(f"User: {user_msg}")
        response = conv_agent.chat(user_msg)
        print(f"Agent: {response}")

    print("\n" + "=" * 60)
    print(f"\n✅ Conversation complete (Session: {conv_agent.session_id[:8]}...)")

    return conv_agent


def search_session_traces(conv_agent):
    # Search for traces from this session
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    session_traces = mlflow.search_traces(
        locations=[experiment.experiment_id],
        filter_string=f"metadata.`mlflow.trace.session` = '{conv_agent.session_id}'"
    )

    print(f"📊 Found {len(session_traces)} traces for session {conv_agent.session_id[:8]}...")

    return session_traces


def run_deepeval_evaluation(session_traces, completeness_scorer, retention_scorer, topic_scorer, toxicity_scorer):
    with mlflow.start_run(run_name="DeepEval") as run:
        deepeval_results = mlflow.genai.evaluate(
            data= session_traces,
            # My DeepEval scorers defined above
            scorers=[
                completeness_scorer,
                retention_scorer,
                topic_scorer,
                toxicity_scorer
            ]
        )

    print("\n✅ DeepEval session evaluation complete!")
    print("\nDeepEval Metrics:")
    print("-" * 40)
    for metric_name, value in deepeval_results.metrics.items():
        if isinstance(value, float):
            print(f"  {metric_name}: {value:.3f}")
        else:
            print(f"  {metric_name}: {value}")


def main():
    client, JUDGE_MODEL, AGENT_MODEL, use_databricks_provider = setup()

    agent = create_and_test_agent(client, AGENT_MODEL)

    quick_evaluation(use_databricks_provider, JUDGE_MODEL)

    relevance_scorer, correctness_scorer, safety_scorer, guidelines_scorer = build_builtin_scorers(
        use_databricks_provider, JUDGE_MODEL
    )

    eval_dataset = create_evaluation_dataset()

    predict_fn = make_predict_fn(agent)

    run_builtin_evaluation(
        eval_dataset, predict_fn, relevance_scorer, correctness_scorer, safety_scorer, guidelines_scorer
    )

    response_length_check, contains_keywords, no_hallucination_markers = define_custom_scorers()

    run_custom_evaluation(
        eval_dataset, predict_fn, response_length_check, contains_keywords, no_hallucination_markers
    )

    explanation_quality_judge = create_llm_judge(JUDGE_MODEL)
    run_judge_evaluation(agent, explanation_quality_judge)

    completeness_scorer, retention_scorer, topic_scorer, toxicity_scorer = build_deepeval_scorers(JUDGE_MODEL)

    conv_agent = simulate_conversation(client, AGENT_MODEL)

    session_traces = search_session_traces(conv_agent)

    run_deepeval_evaluation(
        session_traces, completeness_scorer, retention_scorer, topic_scorer, toxicity_scorer
    )


if __name__ == "__main__":
    main()
