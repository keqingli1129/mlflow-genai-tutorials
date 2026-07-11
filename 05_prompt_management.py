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
    mlflow.set_tracking_uri("http://localhost:5000")

    use_databricks_provider = is_databricks_ai_gateway_client()
    if use_databricks_provider:
        client = get_databricks_ai_gateway_client()
        model_name = get_ai_gateway_model_names()[0]
    else:
        client = get_openai_client()
        model_name = "gpt-4o-mini"

    if not use_databricks_provider and not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

    # Create experiment and enable autologging
    mlflow.set_experiment("05-prompt-management")
    mlflow.openai.autolog()

    provider = "Databricks AI Gateway" if use_databricks_provider else "OpenAI"
    print(f"✅ MLflow experiment '05-prompt-management' ready | Provider: {provider}")
    print(f"   Tracking URI: {mlflow.get_tracking_uri()} | Autologging: ENABLED")

    return client, model_name, use_databricks_provider


def first_prompt(client, model_name):
    print("📝 Registering your first prompt to the Prompt Registry...\n")

    # Register a simple QA prompt — note Jinja2 {{ variable }} syntax
    prompt = mlflow.genai.register_prompt(
        name="tutorial-first-qa",
        template="""You are a helpful AI assistant.

Answer the following question concisely:
{{ question }}

Answer:""",
        commit_message="Initial version - general purpose QA",
        tags={
            "author": "jules",
            "use_case": "general_qa",
        }
    )

    print(f"✅ Registered: {prompt.name} (version {prompt.version})")
    print(f"   Commit: {prompt.commit_message}")

    # Load it back by name (gets latest version)
    loaded = mlflow.genai.load_prompt("tutorial-first-qa")
    print(f"\n📥 Loaded prompt (version {loaded.version})")

    # Fill the template with .format()
    question = "What is prompt engineering?"
    prompt_filled = loaded.format(question=question)

    # Call the LLM
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_filled}],
    )

    answer = response.choices[0].message.content
    print(f"\n❓ Question: {question}")
    print(f"🤖 Answer: {answer}")
    print("\n✅ Your first Registry prompt is live!")
    print("   View it: MLflow UI → Prompt Registry (left sidebar)")


def role_based_prompt(client, model_name):
    # Register a multi-variable role-based prompt directly in the Registry
    role_prompt = mlflow.genai.register_prompt(
        name="tutorial-role-based-qa",
        template="""You are a {{ role }}.

Context: {{ context }}

Task: {{ task }}

Provide your response:""",
        commit_message="Multi-variable role-based template",
        tags={
            "author": "jules",
            "variables": "role,context,task",
            "use_case": "role_based_qa"
        }
    )

    print(f"✅ Registered: {role_prompt.name} (version {role_prompt.version})")

    # Load and fill the template
    loaded_role_prompt = mlflow.genai.load_prompt("tutorial-role-based-qa")

    prompt_text = loaded_role_prompt.format(
        role="technical documentation expert",
        context="MLflow is an open source AI platform",
        task="Explain MLflow tracing in 2 sentences"
    )

    print(f"\n📝 Filled prompt:\n{prompt_text}")

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt_text}],
    )

    answer = response.choices[0].message.content
    print(f"\n🤖 Answer: {answer}")
    print(f"\n📋 Variables: role, context, task  |  Version: {loaded_role_prompt.version}")


def versioning_prompts(client, model_name):
    question = "What is MLflow?"
    print("🔄 Creating prompt versions via the Registry...\n")

    # Version 1: Basic — minimal prompt, no guidance
    p_v1 = mlflow.genai.register_prompt(
        name="tutorial-versioned-qa",
        template="Answer this question: {{ question }}",
        commit_message="Initial basic version"
    )
    print(f"✅ Created version {p_v1.version}: Basic (no guidance)")

    # Version 2: Full structured format with role + guidelines
    p_v2 = mlflow.genai.register_prompt(
        name="tutorial-versioned-qa",
        template="""You are a helpful AI assistant specializing in technical topics.

Guidelines:
  - Answer concisely (2-3 sentences max)
  - Be accurate and factual
  - If unsure, say so

Question: {{ question }}

Answer:""",
        commit_message="Added specialization, role, and explicit guidelines"
    )
    print(f"✅ Created version {p_v2.version}: With role + guidelines")

    # Load and compare both versions
    print(f"\n🔍 Comparing versions for: '{question}'\n")
    for ver in [p_v1.version, p_v2.version]:
        p = mlflow.genai.load_prompt(f"prompts:/tutorial-versioned-qa/{ver}")
        filled = p.format(question=question)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": filled}],
            temperature=1.0
        )
        answer = response.choices[0].message.content
        print(f"v{ver}: {answer[:100]}...")

    print("\n✅ Both versions tracked in the Registry!")
    print("   View history: MLflow UI → Prompt Registry → tutorial-versioned-qa")


def linking_prompts_to_experiments(client, model_name):
    print("🔗 Linking a Registry prompt to an experiment run...\n")

    # Load the versioned prompt you want to use
    prompt = mlflow.genai.load_prompt("prompts:/tutorial-versioned-qa/2")

    with mlflow.start_run(run_name="qa-with-linked-prompt"):  # I can also experiment_id=id as a parameter
        question = "What is MLflow tracing?"
        filled = prompt.format(question=question)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": filled}],
        )
        answer = response.choices[0].message.content

        # Log prompt identity, input, and output together as a single artifact
        mlflow.log_dict({
            "prompt_name":    prompt.name,
            "prompt_version": prompt.version,
            "prompt_uri":     prompt.uri,
            "question":       question,
            "answer":         answer,
        }, "run_record.json")

        run_id = mlflow.active_run().info.run_id
        print(f"✅ Run logged: {run_id}")
        print(f"   prompt_name:    {prompt.name}")
        print(f"   prompt_version: {prompt.version}")
        print(f"   prompt_uri:     {prompt.uri}")
        print(f"\n🤖 Answer: {answer}")

    print("\n💡 View in MLflow UI → Experiments → qa-with-linked-prompt → Artifacts → run_record.json")
    print("   Reproduce any run: mlflow.genai.load_prompt(run_record['prompt_uri'])")


def define_prompt_library():
    # Define library prompts using Jinja2 {{ var }} syntax — ready to register directly
    PROMPT_LIBRARY = {
        "qa_simple": {
            "template": "Answer this question: {{ question }}",
            "variables": ["question"],
            "use_case": "Simple Q&A",
        },
        "qa_with_context": {
            "template": """Use the following context to answer the question.
Context: {{ context }}

Question: {{ question }}

Answer based only on the context above:""",
            "variables": ["context", "question"],
            "use_case": "RAG Q&A",
        },
        "classification": {
            "template": "Classify the following text into one of these categories: {{ categories }}\n\nText: {{ text }}\n\nCategory:",
            "variables": ["text", "categories"],
            "use_case": "Text classification",
        },
    }

    print("📚 Library prompts defined:")
    for name, info in PROMPT_LIBRARY.items():
        print(f"   - {name}: {info['use_case']}")
    print("\n   Next: register all to the Prompt Registry →")

    return PROMPT_LIBRARY


def register_library(PROMPT_LIBRARY):
    print("🔄 Registering library prompts to the Prompt Registry...\n")

    registered_prompts = {}
    for name, prompt_info in PROMPT_LIBRARY.items():
        # prefix with tutorial to avoid conflicts with other prompts
        prompt_name = f"tutorial-{name}"

        prompt = mlflow.genai.register_prompt(
            name=prompt_name,
            template=prompt_info["template"],
            commit_message=f"Initial version - {prompt_info['use_case']}",
            tags={
                "use_case": prompt_info["use_case"],
                "variables": ",".join(prompt_info["variables"]),
                "author": "jules"
            }
        )

        registered_prompts[name] = prompt
        print(f"✅ Registered: {prompt_name} (version {prompt.version})")

    print(f"\n📋 {len(registered_prompts)} prompts registered to the Prompt Registry")
    print("   Browse: MLflow UI → Prompt Registry")

    return registered_prompts


def use_library_prompts(client, model_name):
    # Load and use prompts from the Registry — no Python dict needed
    print("📚 Using prompts from the Registry...\n")

    # Example 1: RAG Q&A
    rag_prompt = mlflow.genai.load_prompt("tutorial-qa_with_context")
    filled = rag_prompt.format(
        context="MLflow Tracing provides observability for GenAI applications.",
        question="What does MLflow Tracing do?"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": filled}],
    )
    print(f"RAG Q&A: {response.choices[0].message.content}\n")

    # Example 2: Classification
    cls_prompt = mlflow.genai.load_prompt("tutorial-classification")
    filled = cls_prompt.format(
        text="I love using MLflow for my ML projects!",
        categories="Positive, Negative, Neutral"
    )
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": filled}],
    )
    print(f"Classification: {response.choices[0].message.content}")

    print("\n✅ Registry prompts are accessible from any app with MLflow access!")
    print("   Search available prompts with: mlflow.genai.search_prompts()")


def setup_aliases(client, model_name):
    #
    # Set up aliases for the RAG Q&A prompt
    print("🏷️ Setting up aliases for environment management...\n")

    # Set version 1 as production — the current stable version serving users
    mlflow.genai.set_prompt_alias("tutorial-qa_with_context", "production", version=1)
    print("✅ Set 'production' alias → version 1")

    # Staging also starts at version 1 — we'll advance it to a new version in the next step
    mlflow.genai.set_prompt_alias("tutorial-qa_with_context", "staging", version=1)
    print("✅ Set 'staging' alias → version 1")

    # Load prompt via alias - this is what production code should use!
    print("\n📥 Loading prompt via alias...")
    prod_prompt = mlflow.genai.load_prompt("prompts:/tutorial-qa_with_context@production")
    print(f"   Loaded production prompt (version {prod_prompt.version})")

    # Use the production prompt
    filled = prod_prompt.format(
        context="MLflow is an open source platform for the ML lifecycle.",
        question="What is MLflow?"
    )

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": filled}],
    )

    print(f"\n🤖 Production response: {response.choices[0].message.content}")
    print("\n💡 Next: register an improved version and advance staging to it →")


def register_improved_version():
    # Register an improved version of the RAG Q&A prompt
    print("📝 Registering an improved version...\n")

    improved_template = """You are a helpful AI assistant. Use ONLY the provided context to answer.

Context:
{{ context }}

Question: {{ question }}

Instructions:
- Answer based strictly on the context above
- If the answer is not in the context, say "I don't have enough information"
- Be concise (2-3 sentences max)

Answer:"""

    # Register as a new version — same name, new commit_message
    updated_prompt = mlflow.genai.register_prompt(
        name="tutorial-qa_with_context",
        template=improved_template,
        commit_message="Added strict context adherence instructions and conciseness requirement",
        tags={
            "use_case": "RAG Q&A",
            "variables": "context,question",
            "author": "jules"
        }
    )

    print(f"✅ Registered version {updated_prompt.version}")
    print("   Commit: Added strict context adherence instructions")

    # Promote to staging — production still points to previous version
    mlflow.genai.set_prompt_alias("tutorial-qa_with_context", "staging", version=updated_prompt.version)
    print(f"\n🏷️  Promoted version {updated_prompt.version} → 'staging'")
    print("   Production still points to previous version (safe rollout!)")
    print("\n💡 When staging looks good:")
    print("   mlflow.genai.set_prompt_alias('tutorial-qa_with_context', 'production', version=updated_prompt.version)")


def search_prompts():
    # Search for prompts by tags
    print("🔍 Searching prompts in the Registry...\n")

    # Search all prompts by author
    all_prompts = mlflow.genai.search_prompts(filter_string="tags.author='jules'")
    print(f"Found {len(all_prompts)} prompts by author 'jules':")
    for p in all_prompts:
        print(f"   - {p.name}")

    # Filter by use_case
    rag_prompts = mlflow.genai.search_prompts(filter_string="tags.use_case='RAG Q&A'")
    print(f"\nFound {len(rag_prompts)} RAG Q&A prompts:")
    for p in rag_prompts:
        print(f"   - {p.name}")

    print("\n💡 Search tips:")
    print("   - Filter by author:   tags.author='jules'")
    print("   - Filter by use case: tags.use_case='RAG Q&A'")
    print("   - Combine:            tags.author='jules' AND tags.use_case='RAG Q&A'")


def main():
    client, model_name, use_databricks_provider = setup()

    first_prompt(client, model_name)
    role_based_prompt(client, model_name)
    versioning_prompts(client, model_name)
    linking_prompts_to_experiments(client, model_name)

    prompt_library = define_prompt_library()
    register_library(prompt_library)
    use_library_prompts(client, model_name)

    setup_aliases(client, model_name)
    register_improved_version()
    search_prompts()


if __name__ == "__main__":
    main()
