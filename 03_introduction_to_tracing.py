import os

import mlflow
from dotenv import load_dotenv

from utils.clnt_utils import (
    get_ai_gateway_model_names,
    get_databricks_ai_gateway_client,
    get_databricks_ai_gateway_langchain_client,
    get_langchain_chat_openai_client,
    get_openai_client,
    is_databricks_ai_gateway_client,
)


def setup():
    load_dotenv()
    mlflow.set_tracking_uri("http://localhost:5000")

    use_databricks_provider = is_databricks_ai_gateway_client()
    if use_databricks_provider:
        client = get_databricks_ai_gateway_client()
        model_name = get_ai_gateway_model_names()[0]
    else:
        client = get_openai_client()
        model_name = "gpt-5-mini"

    if not use_databricks_provider and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

    print("✅ Environment configured: using", "Databricks" if use_databricks_provider else "OpenAI", "client")
    print(f"   MLflow version: {mlflow.__version__}")
    print(f"   Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"   Using model: {model_name}")

    return client, model_name, use_databricks_provider


def call_without_tracing(client, model_name, prompt):
    # Create experiment for tracing examples
    mlflow.set_experiment("06-tracing-introduction")

    # Without tracing - basic call
    print("\n📝 Making LLM call WITHOUT tracing...\n")

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_completion_tokens=1000,
    )

    print(f"Response: {response.choices[0].message.content}")
    print("\n❌ What we DON'T know:")
    print("   - Exact timing of the call")
    print("   - Detailed request/response structure")
    print("   - Easy way to correlate with other operations")
    print("   - Visual representation of execution")

    # check the trace in the UI
    print("\n🔍 View trace in MLflow UI: http://localhost:5000")
    print("   Navigate to: Traces tab")


def call_with_tracing(client, model_name, prompt):
    # Enable OpenAI autologging - THIS IS THE MAGIC LINE!
    mlflow.openai.autolog()

    print("✅ OpenAI autologging enabled")
    print("   All OpenAI API calls will now be automatically traced!")

    # Make the same call - now it's automatically traced!
    print("\n🔍 Making LLM call WITH tracing...\n")

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_completion_tokens=1000,
    )

    print(f"Response: {response.choices[0].message.content}")
    print("\n✅ What we NOW know:")
    print("   ✓ Complete request details (model, messages, parameters)")
    print("   ✓ Response content and metadata")
    print("   ✓ Token usage (prompt, completion, total)")
    print("   ✓ Timing information (latency)")
    print("   ✓ All captured automatically!")
    print("\n🔗 View trace in MLflow UI: http://localhost:5000")
    print("   Navigate to: Traces tab")


def multi_step_workflow(client, model_name):
    # Simple multi-step workflow
    print("\n🔄 Multi-step workflow with automatic tracing...\n")

    # Step 1: Generate a topic
    print("Step 1: Generating topic...")
    topic_response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI expert who explains concepts clearly and concisely.",
            },
            {
                "role": "user",
                "content": "Suggest one interesting AI topic for a blog post.",
            },
        ],
        temperature=1.0,
        max_completion_tokens=1000,
    )

    # get the topic
    topic = topic_response.choices[0].message.content
    print(f"  Topic: {topic}")

    # Step 2: Generate an outline
    print("\nStep 2: Creating outline...")
    outline_response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": f"Create a 3-point outline for a blog post about: {topic}",
        }],
        temperature=1.0,
        max_completion_tokens=500,
    )
    outline = outline_response.choices[0].message.content
    print(f"  Outline: {outline[:100]}...")

    # Step 3: Write the introduction
    print("\nStep 3: Writing introduction...")
    intro_response = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "user",
            "content": f"Write a 4-sentence introduction paragraph for: {topic}",
        }],
        temperature=1.0,
        max_completion_tokens=2000,
    )
    intro = intro_response.choices[0].message.content
    print(f"  Introduction: {intro}")

    print("\n✅ All three steps completed!")
    print("\n📊 Total tokens used:")
    total_tokens = (topic_response.usage.total_tokens +
                    outline_response.usage.total_tokens +
                    intro_response.usage.total_tokens)
    print(f"   {total_tokens} tokens")

    print("\n🔍 View in MLflow UI:")
    print("   You'll see THREE separate traces, one for each call")
    print("   Each trace shows timing, tokens, and complete I/O")


def langchain_tracing(model_name, use_databricks_provider):
    from langchain_core.prompts import ChatPromptTemplate

    # Enable LangChain autologging
    mlflow.langchain.autolog()

    print("✅ LangChain autologging enabled")
    print("✅ Using", "Databricks AI Gateway" if use_databricks_provider else "OpenAI", "as provider")

    # Create a simple LangChain chain
    print("\n🔗 Creating and running LangChain chain with tracing...\n")

    # Define prompt template
    prompt_template = ChatPromptTemplate.from_template(
        "You are a {role}. Answer the following question: {question}"
    )

    # Create LangChain LLM object
    if use_databricks_provider:
        llm = get_databricks_ai_gateway_langchain_client(model_name, temperature=1.0)
    else:
        llm = get_langchain_chat_openai_client(model_name, temperature=1.0)

    # Create chain using the prompt template and the LLM.
    # The UNIX like Pipe operator | is used to chain the prompt template and the LLM.
    chain = prompt_template | llm

    # Run chain
    response = chain.invoke({
        "role": "helpful AI assistant",
        "question": "What are the benefits of tracing in GenAI applications?",
    })

    print(f"Response: {response.content}")
    print("\n✅ LangChain execution traced!")
    print("\n🔍 In the trace, you'll see:")
    print("   - Prompt template construction")
    print("   - Variable substitution")
    print("   - LLM invocation")
    print("   - All as separate spans in a hierarchy!")

    return llm


def error_scenario_invalid_role(client, model_name):
    # ── Error Scenario 1: Invalid Role in the Message List ──────────────────────
    # Valid OpenAI roles are: "system", "user", "assistant", "tool".
    # Passing an unknown role ("robot") triggers a BadRequestError.
    # Note: Databricks AI Gateway may surface a slightly different error message.

    print("=" * 60)
    print("🐛 Error Scenario 1: Invalid Message Role")
    print("=" * 60)
    print("Calling the API with role='robot' (not a valid OpenAI role)...\n")

    try:
        client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "robot", "content": "Classify this review as positive or negative."},
                {"role": "user", "content": "I love this product!"},
            ],
            max_completion_tokens=50,
        )
    except Exception as e:
        print(f"❌ {type(e).__name__}: {str(e)[:200]}")

    print("\n✅ Failed trace captured — open MLflow UI to inspect it")
    print("   Ask MLflow Assistant: 'Why did this request fail and how do I fix it?'")


def error_scenario_invalid_tool_name(client, model_name):
    # ── Error Scenario 2: Invalid Tool/Function Name ─────────────────────────────
    # OpenAI function names must match ^[a-zA-Z0-9_-]{1,64}$.
    # Using spaces (e.g. "get weather data") is a common mistake that triggers
    # a BadRequestError — captured automatically by mlflow.openai.autolog().

    print("=" * 60)
    print("🐛 Error Scenario 2: Invalid Tool/Function Name")
    print("=" * 60)
    print("Calling the API with a function name that contains spaces...\n")

    # Common mistake: using a natural-language name with spaces instead of
    # snake_case or kebab-case (e.g. 'get_weather' or 'get-weather').
    broken_tool = {
        "type": "function",
        "function": {
            "name": "get weather data",   # ← BUG: spaces are not allowed
                                            # Fix: use 'get_weather_data'
            "description": "Retrieve the current weather for a given city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    }

    try:
        client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[broken_tool],
            max_completion_tokens=100,
        )
    except Exception as e:
        print(f"❌ {type(e).__name__}: {str(e)[:200]}")

    print("\n✅ Failed trace captured — open MLflow UI to inspect it")
    print("   Ask MLflow Assistant: 'What is wrong with this tool definition?'")


def error_scenario_multi_step_pipeline(llm):
    # ── Error Scenario 3: Multi-Step Pipeline — Step 3 Fails ─────────────────────
    # A three-step LangChain LCEL pipeline:
    #   Step 1 – Generate a product review           (succeeds)
    #   Step 2 – Ask the LLM to rate the review      (succeeds, returns "X/10")
    #   Step 3 – Parse the rating as a bare integer  (FAILS – "8/10" → ValueError)
    #
    # mlflow.langchain.autolog() (enabled earlier) records the ENTIRE pipeline as
    # ONE hierarchical trace, so you can see each step's output even after the
    # pipeline fails — no @mlflow.trace or manual instrumentation needed.

    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    print("=" * 60)
    print("🐛 Error Scenario 3: Multi-Step Pipeline — Step 3 Fails")
    print("=" * 60)
    print("3-step pipeline: steps 1 & 2 succeed, step 3 crashes...\n")

    parser = StrOutputParser()
    product = "ultra-lightweight noise-cancelling wireless headphones"

    def parse_integer_rating(text: str) -> int:
        """BUG: LLM returns 'X/10' format, but int() cannot parse it."""
        return int(text.strip())  # ← raises ValueError when text is e.g. "8/10"

    # Build the full LCEL pipeline.
    # mlflow.langchain.autolog() captures it as a SINGLE hierarchical trace
    # with one span per step — you will see the successful spans for steps 1 & 2
    # even though step 3 fails.
    pipeline = (
        # Step 1: generate a 2-sentence product review (succeeds)
        ChatPromptTemplate.from_template(
            "Write a 2-sentence product review for: {product}"
        )
        | llm
        | parser
        # Step 2: rate the review — explicitly ask for "X/10" format (succeeds)
        | RunnableLambda(lambda review: {"review": review})
        | ChatPromptTemplate.from_template(
            "Rate this product review from 1 to 10. Reply in the format 'X/10':\n\n{review}"
        )
        | llm
        | parser
        # Step 3: parse the rating as a plain integer (FAILS)
        | RunnableLambda(parse_integer_rating)  # ← "8/10" → ValueError
    )

    try:
        result = pipeline.invoke({"product": product})
        print(f"Rating: {result}")
    except Exception as e:
        print(f"❌ Pipeline failed at Step 3 ({type(e).__name__}): {e}")
        print("\n💡 Steps 1 & 2 succeeded — their outputs are preserved in the trace")

    print("\n✅ Hierarchical trace captured — open MLflow UI to inspect it")
    print("   You will see:")
    print("   ✓ Step 1 span: product review generated  (Success)")
    print("   ✓ Step 2 span: LLM returned rating text  (Success)")
    print("   ✗ Step 3 span: integer parsing failed    (Error)")
    print("\n   Ask MLflow Assistant: 'Which step failed and what did each step return?'")


def main():
    client, model_name, use_databricks_provider = setup()

    prompt = "Explain what distributed tracing is in one sentence."
    call_without_tracing(client, model_name, prompt)
    call_with_tracing(client, model_name, prompt)

    multi_step_workflow(client, model_name)

    llm = langchain_tracing(model_name, use_databricks_provider)

    error_scenario_invalid_role(client, model_name)
    error_scenario_invalid_tool_name(client, model_name)
    error_scenario_multi_step_pipeline(llm)


if __name__ == "__main__":
    main()
