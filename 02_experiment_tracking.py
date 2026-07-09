import os

import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

from utils.clnt_utils import (
    get_ai_gateway_model_names,
    get_databricks_ai_gateway_client,
    get_openai_client,
    is_databricks_ai_gateway_client,
)


def setup():
    load_dotenv()
    mlflow.set_tracking_uri("http://localhost:5000")

    use_ai_gateway = is_databricks_ai_gateway_client()
    if use_ai_gateway:
        client = get_databricks_ai_gateway_client()
        model_name = get_ai_gateway_model_names()[0]
    else:
        client = get_openai_client()
        model_name = "gpt-5.2"

    if not use_ai_gateway and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

    mlflow.openai.autolog()

    print("✅ Environment configured successfully")
    print(f"   MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"   Using model: {model_name}")
    print("   Autolog: ENABLED")

    return client, model_name, use_ai_gateway


def experiment_basic_llm_call(client, model_name):
    experiment_name = "02-basic-llm-calls"
    mlflow.set_experiment(experiment_name)
    print(f"📊 Experiment: {experiment_name}")
    print("   View in UI: http://localhost:5000")

    prompt = "Explain MLflow GenAI Platform in 3-4 sentences."

    with mlflow.start_run(run_name="first-llm-tracked-call") as run:
        mlflow.set_tag("task", "explanation")
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=1.0,
            max_completion_tokens=1000,
        )
        answer = response.choices[0].message.content

    print(f"\n📝 Prompt: {prompt}")
    print(f"\n🤖 Response: {answer}")
    print(f"\n🔗 Run ID: {run.info.run_id}")
    print(f"   View in UI: http://localhost:5000/#/experiments/{run.info.experiment_id}/runs/{run.info.run_id}")


def experiment_temperature_comparison(client, model_name):
    def simple_llm_call(prompt, model, temperature, max_completion_tokens, run_name):
        with mlflow.start_run(run_name=run_name, nested=True):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            return response.choices[0].message.content

    mlflow.set_experiment("02-temperature-comparison")

    test_prompt = "Write a creative tagline for an AI observability with MLflow GenAI platform."
    temperatures = [1.0, 1.5, 2.0]

    print("🔬 Running temperature comparison...\n")

    with mlflow.start_run(run_name="temperature-sweep"):
        mlflow.set_tag("sweep_variable", "temperature")
        for temp in temperatures:
            print(f"  temperature={temp} ...")
            response = simple_llm_call(
                prompt=test_prompt,
                model=model_name,
                temperature=temp,
                max_completion_tokens=1000,
                run_name=f"temp_{temp}",
            )
            print(f"    -> {response}\n")

    print("✅ Done. Compare traces side-by-side in the MLflow UI.")


def experiment_model_cost_comparison(client, use_ai_gateway):
    def llm_call_with_cost(prompt, model, temperature, max_completion_tokens, run_name):
        with mlflow.start_run(run_name=run_name):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
            )
            return response.choices[0].message.content

    mlflow.set_experiment("03-model-cost-comparison")

    prompt = "Summarize the benefits of experiment tracking in 3 bullet points."
    models_to_test = ["jsd-gpt-5-2", "jsd-gpt-5-mini"] if use_ai_gateway else ["gpt-5-mini", "gpt-5.2"]

    print("💰 Comparing costs across models...\n")

    for model in models_to_test:
        print(f"Testing {model}...")
        response = llm_call_with_cost(
            prompt=prompt,
            model=model,
            temperature=1.0,
            max_completion_tokens=1000,
            run_name=f"model_{model}_run",
        )
        print(f"  Response: {response}...\n")

    print("✅ Cost comparison complete! View in MLflow UI.")


def experiment_production_candidate_testing(client, use_ai_gateway):
    mlflow.set_experiment("04-production-candidate-testing")

    open_configs = [
        {
            "name": "baseline",
            "model": "gpt-5-mini",
            "temperature": 1.0,
            "system_prompt": "You are a helpful assistant.",
        },
        {
            "name": "creative",
            "model": "gpt-5.2",
            "temperature": 2.0,
            "system_prompt": "You are a creative writing assistant.",
        },
    ]

    databricks_config = [
        {
            "name": "baseline",
            "model": "jsd-gpt-5-mini",
            "temperature": 1.0,
            "system_prompt": "You are a helpful assistant.",
        },
        {
            "name": "creative",
            "model": "jsd-gpt-5.2",
            "temperature": 1.5,
            "system_prompt": "You are a creative writing assistant.",
        },
    ]

    model_configs = databricks_config if use_ai_gateway else open_configs
    test_prompt = "Explain the concept of LLM temperature."

    print("🏷️  Running experiments with semantic tags...\n")

    for config in model_configs:
        with mlflow.start_run(run_name=config["name"]):
            client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": config["system_prompt"]},
                    {"role": "user", "content": test_prompt},
                ],
                temperature=config["temperature"],
                max_completion_tokens=1000,
            )
            mlflow.set_tags({
                "config_name": config["name"],
                "task": "explanation",
                "stage": "testing",
                "team": "ai-research",
                "version": "v1.0",
                "production_candidate": str(config["name"] == "baseline").lower(),
            })
            mlflow.log_dict(config, "config.json")
            print(f"  ✓ {config['name']} done")

    print("\n✅ All runs completed! Filter by tag 'production_candidate=true' in the UI.")


def query_experiments():
    mlflow_client = MlflowClient()
    experiment = mlflow_client.get_experiment_by_name("04-production-candidate-testing")

    if not experiment:
        print("Experiment not found. Make sure you ran the production candidate testing section.")
        return

    print(f"📊 Experiment: {experiment.name}")
    print(f"   ID: {experiment.experiment_id}")

    runs = mlflow_client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=5,
    )

    print(f"\n   Found {len(runs)} runs:\n" + "=" * 60)
    for run in runs:
        print(f"\n   Run: {run.info.run_name}")
        if run.data.params:
            print("   Parameters:")
            for key, value in run.data.params.items():
                print(f"      {key}: {value}")
        if run.data.metrics:
            print("   Metrics:")
            for key, value in run.data.metrics.items():
                print(f"      {key}: {value}")
        if run.data.tags.get("config_name"):
            print(f"   Tag config_name: {run.data.tags['config_name']}")

    prod_runs = mlflow_client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.production_candidate = 'true'",
        max_results=5,
    )

    print("\n🏆 Production Candidates:")
    for run in prod_runs:
        print(f"   Name: {run.info.run_name}")
        print(f"   Config: {run.data.tags.get('config_name', 'N/A')}")
        print(f"   Run ID: {run.info.run_id}")


def main():
    client, model_name, use_ai_gateway = setup()
    experiment_basic_llm_call(client, model_name)
    experiment_temperature_comparison(client, model_name)
    experiment_model_cost_comparison(client, use_ai_gateway)
    experiment_production_candidate_testing(client, use_ai_gateway)
    query_experiments()


if __name__ == "__main__":
    main()
