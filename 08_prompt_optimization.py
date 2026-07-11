import os

import mlflow
from dotenv import load_dotenv

from utils.clnt_utils import (
    get_ai_gateway_model_names,
    get_databricks_ai_gateway_client,
    get_openai_client,
    is_databricks_ai_gateway_client,
)


def setup():
    load_dotenv()

    # Configure MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("08-prompt-optimization")

    # Configure client and model based on provider
    use_databricks_provider = is_databricks_ai_gateway_client()
    if use_databricks_provider:
        client = get_databricks_ai_gateway_client()
        model_name = get_ai_gateway_model_names()[0]
        optimizer_model = f"databricks:/{model_name}"
    else:
        client = get_openai_client()
        model_name = "gpt-5-mini"
        optimizer_model = f"openai:/{model_name}"

    if not use_databricks_provider and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

    # Enable autologging
    mlflow.openai.autolog()

    print("✅ Environment configured")
    print(f"   Provider: {'Databricks AI Gateway' if use_databricks_provider else 'OpenAI'}")
    print(f"   Model: {model_name}")
    print(f"   Optimizer model: {optimizer_model}")
    print(f"   Tracking URI: {mlflow.get_tracking_uri()}")

    return client, model_name, use_databricks_provider, optimizer_model


def register_baseline_prompt():
    # Register the baseline prompt (same template as qa_simple from Notebook 1.5)
    baseline_prompt = mlflow.genai.register_prompt(
        name="gepa-qa-simple",
        template="Answer this question: {{ question }}",
        commit_message="Baseline prompt for GEPA optimization",
        tags={"author": "jules", "use_case": "Simple Q&A", "status": "baseline"}
    )

    print("✅ Baseline prompt registered")
    print(f"   Name: {baseline_prompt.name}")
    print(f"   Version: {baseline_prompt.version}")
    print(f"   URI: {baseline_prompt.uri}")
    print(f"   Template: '{baseline_prompt.template}'")

    return baseline_prompt


def prepare_optimization_components(client, model_name, baseline_prompt):
    from mlflow.genai import scorer
    from mlflow.genai.judges import CategoricalRating
    from mlflow.entities import Feedback

    # Custom exact-match scorer — pure Python, no LLM judge calls needed.
    # Returns YES (1.0) if normalized output matches expected answer, NO (0.0) otherwise.
    @scorer
    def exact_match(outputs: str, expectations: dict) -> Feedback:
        expected = expectations["expected_response"].strip().lower()
        actual = outputs.strip().lower().rstrip(".")
        return Feedback(
            name="exact_match",
            value=CategoricalRating.YES if actual == expected else CategoricalRating.NO,
        )

    # Training data: short-answer factual questions with unambiguous 1-3 word answers.
    # The bare-bones prompt will produce verbose full-sentence answers that fail exact match.
    # GEPA must learn to add conciseness instructions to pass.
    train_data = [
        {"inputs": {"question": "What is the chemical symbol for gold?"}, "expectations": {"expected_response": "Au"}},
        {"inputs": {"question": "What planet is closest to the sun?"}, "expectations": {"expected_response": "Mercury"}},
        {"inputs": {"question": "In what year did the Titanic sink?"}, "expectations": {"expected_response": "1912"}},
        {"inputs": {"question": "What is the capital of Japan?"}, "expectations": {"expected_response": "Tokyo"}},
        {"inputs": {"question": "Who wrote Romeo and Juliet?"}, "expectations": {"expected_response": "William Shakespeare"}},
        {"inputs": {"question": "What is the largest planet in our solar system?"}, "expectations": {"expected_response": "Jupiter"}},
        {"inputs": {"question": "What element does the symbol O represent on the periodic table?"}, "expectations": {"expected_response": "Oxygen"}},
        {"inputs": {"question": "What continent is Egypt in?"}, "expectations": {"expected_response": "Africa"}},
        {"inputs": {"question": "What is the hardest natural substance on Earth?"}, "expectations": {"expected_response": "Diamond"}},
        {"inputs": {"question": "What is the largest ocean on Earth?"}, "expectations": {"expected_response": "Pacific Ocean"}},
    ]

    # Predict function: GEPA calls this repeatedly during optimization.
    # During optimization, GEPA patches PromptVersion.template so that
    # load_prompt() returns the MUTATED template instead of the original.
    def predict_qa(question: str) -> str:
        """Load the prompt from the registry, fill it, and call the LLM."""
        prompt = mlflow.genai.load_prompt(baseline_prompt.uri)
        filled = prompt.format(question=question)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": filled}],
        )
        return response.choices[0].message.content

    print(f"✅ Training data prepared: {len(train_data)} short-answer examples")
    print("✅ Custom exact_match scorer defined (no LLM judge calls)")
    print("✅ Predict function defined")
    print(f"   Loads prompt from: {baseline_prompt.uri}")

    return exact_match, train_data, predict_qa


def run_gepa_optimization(predict_qa, train_data, baseline_prompt, optimizer_model, exact_match):
    import logging

    from mlflow.genai import optimize_prompts
    from mlflow.genai.optimize.optimizers import GepaPromptOptimizer
    from gepa.utils import NoImprovementStopper

    # Suppress async/tornado error log noise.
    for _logger_name in ("tornado.general", "tornado.application", "asyncio"):
        logging.getLogger(_logger_name).setLevel(logging.CRITICAL)

    # Run GEPA prompt optimization
    print("🔄 Running GEPA prompt optimization...\n")
    print("   This will iterate through evaluate → reflect → mutate → select cycles.")
    print("   Budget: 100 metric calls | Early stop: 3 iterations without improvement")
    print("   Scorer: exact_match (no LLM judge — fast!)\n")

    result = optimize_prompts(
        predict_fn=predict_qa,
        train_data=train_data,
        prompt_uris=[baseline_prompt.uri],
        optimizer=GepaPromptOptimizer(
            reflection_model=optimizer_model,
            max_metric_calls=100,
            display_progress_bar=False,
            gepa_kwargs={
                "stop_callbacks": NoImprovementStopper(max_iterations_without_improvement=3),
            },
        ),
        scorers=[exact_match],
    )

    print("\n✅ GEPA optimization complete!")

    return result


def compare_prompts(result, baseline_prompt):
    def _safe(s):
        """Strip Unicode surrogates for safe display."""
        if isinstance(s, str):
            return s.encode("utf-8", errors="replace").decode("utf-8")
        return str(s)

    # Display before/after comparison
    print("=" * 70)
    print("📊 GEPA Optimization Results")
    print("=" * 70)

    print("\n📈 Score Improvement:")
    if result.initial_eval_score is not None:
        print(f"   Initial score: {result.initial_eval_score:.3f}")
    else:
        print("   Initial score: N/A")
    if result.final_eval_score is not None:
        print(f"   Final score:   {result.final_eval_score:.3f}")
    else:
        print("   Final score:   N/A")
    if result.initial_eval_score is not None and result.final_eval_score is not None:
        improvement = result.final_eval_score - result.initial_eval_score
        print(f"   Improvement:   {improvement:+.3f}")

    # Load the optimized prompt directly from the registry to ensure we
    # see the actual registered version (not just the in-memory object)
    optimized = result.optimized_prompts[0]
    registry_prompt = mlflow.genai.load_prompt(f"prompts:/{optimized.name}/{optimized.version}")

    print(f"\n📝 Original Prompt (version {baseline_prompt.version}):")
    print(f"   '{baseline_prompt.template}'")

    print(f"\n🚀 Optimized Prompt (version {optimized.version}):")
    print(f"   '{_safe(registry_prompt.template)}'")

    if baseline_prompt.template.strip() == _safe(registry_prompt.template).strip():
        print("\n⚠️  Note: The optimized template is identical to the baseline.")
        print("   This can happen when the baseline already scores well on the")
        print("   training data. Try adding harder examples or increasing the budget.")

    print("\n🔗 The optimized prompt has been automatically registered")
    print(f"   as version {optimized.version} in the Prompt Registry!")
    print(f"   View it in MLflow UI → Prompt Registry → {_safe(optimized.name)}")

    print("\n" + "=" * 70)
    print("\n💡 Key Takeaway:")
    print("   GEPA automatically learned to add structure, instructions,")
    print("   and constraints that we would normally write by hand.")
    print("   Combined with the Prompt Registry, optimized prompts are")
    print("   versioned and ready for deployment via aliases.")


def main():
    client, model_name, use_databricks_provider, optimizer_model = setup()
    baseline_prompt = register_baseline_prompt()
    exact_match, train_data, predict_qa = prepare_optimization_components(
        client, model_name, baseline_prompt
    )
    result = run_gepa_optimization(
        predict_qa, train_data, baseline_prompt, optimizer_model, exact_match
    )
    compare_prompts(result, baseline_prompt)


if __name__ == "__main__":
    main()
