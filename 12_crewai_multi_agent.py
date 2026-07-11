import os
import sqlite3

import mlflow
import pandas as pd
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import tool
from dotenv import load_dotenv
from mlflow.genai.scorers import Guidelines, RelevanceToQuery, Safety
from openai import OpenAI

from utils.fema_data import get_disaster_data
from utils.policy_docs import get_policy_documents


SCHEMA_INFO = (
    "Table 'disaster_data' columns:\n"
    "  - disaster_id: TEXT (e.g., 'DR-4001')\n"
    "  - year: INTEGER (2020-2025)\n"
    "  - state: TEXT (e.g., 'California', 'Florida')\n"
    "  - disaster_type: TEXT ('Wildfire', 'Hurricane', 'Flood', 'Tornado', 'Earthquake')\n"
    "  - severity: INTEGER (1-5, 5=catastrophic)\n"
    "  - affected_population: INTEGER\n"
    "  - federal_aid_amount: INTEGER (in USD)\n"
    "  - declaration_date: TEXT (YYYY-MM-DD)"
)

# Shared state populated by setup_disaster_database() / load_policy_documents(),
# referenced by the module-level @tool functions below.
disaster_data = None
db_conn = None
oai_client = None
POLICY_DOCUMENTS = None


def setup():
    load_dotenv()
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("12-crewai-multi-agent")

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")

    # Enable auto-tracing for CrewAI and OpenAI
    mlflow.crewai.autolog()
    mlflow.openai.autolog()

    # Initialize the LLM
    llm = LLM(model="openai/gpt-4o-mini", temperature=0.3)

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {mlflow.get_experiment_by_name('12-crewai-multi-agent').name}")
    print(f"LLM: {llm.model}")

    return llm


def setup_disaster_database():
    global disaster_data, db_conn, oai_client

    # Load the FEMA disaster database (200 fabricated records, 2020-2025)
    disaster_data = get_disaster_data()
    db_conn = sqlite3.connect(":memory:")
    disaster_data.to_sql("disaster_data", db_conn, index=False, if_exists="replace")

    # OpenAI client for text-to-SQL
    oai_client = OpenAI()

    print(f"FEMA Disaster Database: {len(disaster_data)} records")
    print(f"Columns: {list(disaster_data.columns)}")
    print(f"Years: {sorted(disaster_data['year'].unique())}")
    print(f"Disaster types: {sorted(disaster_data['disaster_type'].unique())}")


def load_policy_documents():
    global POLICY_DOCUMENTS

    POLICY_DOCUMENTS = get_policy_documents()

    print(f"Policy corpus: {len(POLICY_DOCUMENTS)} documents")


@tool
def query_disaster_database(question: str) -> str:
    """Query the FEMA disaster database using natural language.
    Translates the question to SQL and executes it against the database.
    Use this tool for questions about disaster statistics, counts, trends,
    federal aid amounts, affected populations, and state comparisons.
    """
    # Generate SQL from natural language
    response = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a SQL expert. Given a question, generate a SQLite SELECT query.\n"
                    f"Schema:\n{SCHEMA_INFO}\n\n"
                    f"Rules:\n- Return ONLY the SQL query, no explanation\n"
                    f"- Use only SELECT statements\n"
                    f"- Use standard SQLite functions"
                ),
            },
            {"role": "user", "content": question},
        ],
        temperature=0.0,
    )
    sql = response.choices[0].message.content.strip().strip("`").replace("sql\n", "")

    try:
        cursor = db_conn.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        if not rows:
            return f"Query returned no results.\nSQL: {sql}"
        result_text = " | ".join(columns) + "\n"
        for row in rows[:20]:
            result_text += " | ".join(str(v) for v in row) + "\n"
        if len(rows) > 20:
            result_text += f"... ({len(rows)} total rows)"
        return f"SQL: {sql}\n\nResults:\n{result_text}"
    except Exception as e:
        return f"SQL execution error: {e}\nGenerated SQL: {sql}"


@tool
def search_fema_policies(query: str) -> str:
    """Search FEMA policy documents for information about protocols,
    guidelines, and procedures. Use this tool for questions about
    evacuation procedures, federal assistance eligibility, flood response,
    wildfire management, or hurricane preparedness.
    """
    results = []
    query_lower = query.lower()
    for doc_id, content in POLICY_DOCUMENTS.items():
        # Simple keyword matching
        doc_keywords = doc_id.replace("_", " ").split()
        if any(kw in query_lower for kw in doc_keywords) or any(
            word in doc_id for word in query_lower.split()
        ):
            results.append(f"[{doc_id.upper()}]: {content}")
    return "\n\n".join(results) if results else f"No policy documents found for: {query}"


def example_1_fema_crew(llm):
    print("Tools defined: query_disaster_database, search_fema_policies")

    # Create agents with tools
    data_analyst = Agent(
        role="FEMA Data Analyst",
        goal="Analyze FEMA disaster data to answer statistical questions about disasters, aid, and trends",
        backstory=(
            "You are a data analyst at FEMA specializing in disaster statistics. "
            "You use the disaster database to find counts, totals, trends, and comparisons "
            "across years, states, and disaster types. Always query the database for precise numbers."
        ),
        tools=[query_disaster_database],
        llm=llm,
        verbose=True,
    )

    policy_expert = Agent(
        role="FEMA Policy Expert",
        goal="Provide accurate FEMA policy guidance on disaster response protocols and procedures",
        backstory=(
            "You are a FEMA policy specialist with expertise in emergency management "
            "procedures, evacuation protocols, federal assistance programs, and disaster "
            "response guidelines. You always reference official FEMA policy documents."
        ),
        tools=[search_fema_policies],
        llm=llm,
        verbose=True,
    )

    print(f"Agents: {data_analyst.role}, {policy_expert.role}")

    # Define tasks that require tool use
    data_task = Task(
        description=(
            "Analyze the FEMA disaster database to answer: "
            "How many disasters hit California between 2020 and 2024? "
            "What was the total federal aid for hurricane-related disasters in 2024? "
            "Which state had the highest severity-5 disaster count?"
        ),
        expected_output=(
            "A data report with specific numbers from the database for each question: "
            "California disaster count, hurricane aid total, and top state for severity-5 disasters."
        ),
        agent=data_analyst,
    )

    policy_task = Task(
        description=(
            "Based on the disaster data analysis provided, recommend the appropriate "
            "FEMA response protocols. Specifically address:\n"
            "1. What evacuation protocols apply for the most common California disaster type?\n"
            "2. What federal assistance is available for the affected populations?\n"
            "3. What are the hurricane preparedness procedures given the aid levels reported?"
        ),
        expected_output=(
            "A policy guidance document that maps each data finding to the relevant "
            "FEMA protocols and procedures, with specific policy references."
        ),
        agent=policy_expert,
    )

    # Run sequential crew: data analysis -> policy recommendations
    fema_crew = Crew(
        agents=[data_analyst, policy_expert],
        tasks=[data_task, policy_task],
        process=Process.sequential,
        verbose=True,
    )

    fema_result = fema_crew.kickoff()
    print("\n" + "=" * 70)
    print("FEMA CREW OUTPUT")
    print("=" * 70)
    print(fema_result.raw)


def example_2_hierarchical_crew(llm):
    # Create the specialist agents (reusing tools from Example 1)
    hierarchical_data_analyst = Agent(
        role="Disaster Data Analyst",
        goal="Query the FEMA disaster database to provide accurate statistics and trends",
        backstory=(
            "You are a quantitative analyst at FEMA. When asked for data, you always "
            "query the disaster database for precise numbers. You present data clearly "
            "with specific counts, totals, and comparisons."
        ),
        tools=[query_disaster_database],
        llm=llm,
        verbose=True,
    )

    hierarchical_policy_expert = Agent(
        role="Emergency Management Policy Advisor",
        goal="Provide FEMA policy guidance and response protocol recommendations",
        backstory=(
            "You are a senior FEMA policy advisor. You reference official FEMA "
            "protocols and guidelines to recommend appropriate response procedures. "
            "You always cite specific policy frameworks (ICS, NRF, ESF)."
        ),
        tools=[search_fema_policies],
        llm=llm,
        verbose=True,
    )

    report_writer = Agent(
        role="FEMA Report Writer",
        goal="Produce clear, actionable reports combining data analysis and policy guidance",
        backstory=(
            "You are a FEMA communications specialist who produces briefing documents "
            "for senior leadership. You synthesize data and policy into concise, "
            "actionable reports with clear recommendations."
        ),
        llm=llm,
        verbose=True,
    )

    print("Specialist agents created for hierarchical crew")

    # Define tasks for the hierarchical crew
    # In hierarchical mode, the manager decides which agent handles which task
    gather_data_task = Task(
        description=(
            "Gather disaster data for the 2024 hurricane season. Find:\n"
            "- Total number of hurricane disasters declared in 2024\n"
            "- States affected and their severity levels\n"
            "- Total affected population and federal aid disbursed\n"
            "- Comparison with 2023 hurricane numbers"
        ),
        expected_output="A data summary with specific numbers for each metric.",
    )

    assess_policy_task = Task(
        description=(
            "Based on the 2024 hurricane data, assess which FEMA response protocols "
            "were applicable. Address:\n"
            "- Hurricane preparedness timelines that should have been activated\n"
            "- Federal assistance eligibility for affected populations\n"
            "- Any gaps between policy requirements and the disaster scale"
        ),
        expected_output="A policy assessment with specific protocol references and gap analysis.",
    )

    write_report_task = Task(
        description=(
            "Synthesize the data analysis and policy assessment into a briefing report. "
            "Format as:\n"
            "- **Situation Summary**: Key 2024 hurricane statistics\n"
            "- **Response Assessment**: How well protocols matched the disaster scale\n"
            "- **Recommendations**: 3 specific actions for improving hurricane preparedness"
        ),
        expected_output="A structured briefing report with situation summary, assessment, and 3 recommendations.",
    )

    print("Hierarchical tasks defined")

    # Create and run the hierarchical crew
    # The manager_llm controls the manager agent that delegates work
    manager_llm = LLM(model="openai/gpt-4o-mini", temperature=0.1)

    hierarchical_crew = Crew(
        agents=[hierarchical_data_analyst, hierarchical_policy_expert, report_writer],
        tasks=[gather_data_task, assess_policy_task, write_report_task],
        process=Process.hierarchical,  # Manager delegates autonomously
        manager_llm=manager_llm,
        verbose=True,
    )

    hierarchical_result = hierarchical_crew.kickoff()
    print("\n" + "=" * 70)
    print("HIERARCHICAL CREW OUTPUT")
    print("=" * 70)
    print(hierarchical_result.raw)


def example_3_evaluation(llm):
    # Evaluation dataset — diverse FEMA queries
    eval_data = pd.DataFrame({
        "inputs": [
            {"query": "How many disasters hit California between 2020 and 2024?"},
            {"query": "What are FEMA's evacuation protocols for wildfire zones?"},
            {"query": "What was the total federal aid for hurricane disasters in 2023?"},
            {"query": "Which states had severity-5 disasters and what response procedures apply?"},
        ],
    })

    print(f"Evaluation dataset: {len(eval_data)} queries")
    print(eval_data)

    def crew_predict(inputs: dict) -> str:
        """Run a CrewAI crew to answer FEMA queries."""
        query = inputs["query"]

        # Create a focused two-agent crew for each query
        eval_data_agent = Agent(
            role="FEMA Data Analyst",
            goal="Answer questions about FEMA disaster data with precise statistics",
            backstory="You are a FEMA data analyst. Query the database for exact numbers.",
            tools=[query_disaster_database, search_fema_policies],
            llm=llm,
            verbose=False,
        )

        eval_writer_agent = Agent(
            role="Response Writer",
            goal="Produce a clear, accurate response combining data and policy information",
            backstory="You synthesize data and policy into concise answers.",
            llm=llm,
            verbose=False,
        )

        analysis_task = Task(
            description=f"Analyze the following query and gather relevant data and policy information: {query}",
            expected_output="Data findings and relevant policy information.",
            agent=eval_data_agent,
        )

        synthesis_task = Task(
            description=f"Synthesize the findings into a clear, direct answer to: {query}",
            expected_output="A concise, accurate response that directly answers the question.",
            agent=eval_writer_agent,
        )

        eval_crew = Crew(
            agents=[eval_data_agent, eval_writer_agent],
            tasks=[analysis_task, synthesis_task],
            process=Process.sequential,
            verbose=False,
        )

        result = eval_crew.kickoff()
        return result.raw

    # Custom Guidelines scorer for FEMA response quality
    fema_response_guidelines = Guidelines(
        name="fema_response_quality",
        guidelines=(
            "The response should: "
            "(1) Include specific numbers or data when answering statistical questions, "
            "(2) Reference specific FEMA policies or procedures when answering policy questions, "
            "(3) Be directly relevant to the question asked, "
            "(4) Provide actionable information, not vague generalizations. "
            "Responses that rely on made-up data or lack specificity should score lower."
        ),
    )

    print("Predict function and custom scorer defined")

    # Run evaluation
    with mlflow.start_run(run_name="CrewAI-Evaluation"):
        eval_results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=crew_predict,
            scorers=[
                RelevanceToQuery(),
                Safety(),
                fema_response_guidelines,
            ],
        )

    print("Evaluation complete!")
    print(eval_results.metrics)

    # View per-row results
    print(eval_results.tables["eval_results"])


def cleanup():
    # Cleanup
    db_conn.close()
    print("Database connection closed.")
    print("\nTutorial complete! Check the MLflow UI at http://localhost:5000")
    print("Experiment: 12-crewai-multi-agent")


def main():
    llm = setup()

    setup_disaster_database()
    load_policy_documents()

    example_1_fema_crew(llm)
    example_2_hierarchical_crew(llm)
    example_3_evaluation(llm)

    cleanup()


if __name__ == "__main__":
    main()
