#!/usr/bin/env python3
"""
Multi-Agent CI/CD Pipeline — LangChain 1.x (agents-as-tools supervisor)
Architecture: Supervisor agent receives a work item, routes it to the correct
stage agent (PLANNING / CODING / TESTING / DEPLOYMENT).

The supervisor is itself a `create_agent`; each specialist is wrapped as a
`route_to_*` tool the supervisor calls. This is the pattern LangChain now
recommends in place of the (soft-deprecated) `langgraph-supervisor` library —
no extra dependency, and routing stays entirely within LangChain 1.x.

Install:
  pip install -U langchain langgraph langchain-openai pydantic python-dotenv
"""

import time
import statistics
from enum import Enum
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.tools import tool

load_dotenv()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PYDANTIC SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

class Stage(str, Enum):
    PLANNING   = "PLANNING"
    CODING     = "CODING"
    TESTING    = "TESTING"
    DEPLOYMENT = "DEPLOYMENT"

class StageStatus(str, Enum):
    SUCCESS     = "SUCCESS"
    FAILURE     = "FAILURE"
    BLOCKED     = "BLOCKED"
    IN_PROGRESS = "IN_PROGRESS"

class PipelineStageResult(BaseModel):
    """Structured result produced by any CI/CD stage agent."""
    stage:        Stage       = Field(description="The pipeline stage this agent handled.")
    status:       StageStatus = Field(description="Outcome: SUCCESS, FAILURE, BLOCKED, or IN_PROGRESS.")
    artifacts:    List[str]   = Field(default_factory=list, description="Outputs produced (specs, PR links, test reports, image tags, etc.).")
    issues:       List[str]   = Field(default_factory=list, description="Blockers or defects found.")
    confidence:   float       = Field(ge=0.0, le=1.0, description="Confidence in this result 0.0–1.0.")
    action_taken: str         = Field(description="What the agent did or decided.")
    summary:      str         = Field(description="One-sentence summary for audit logs.")

    @field_validator("status", mode="before")
    @classmethod
    def normalise_status(cls, v: str) -> str:
        allowed = {"SUCCESS", "FAILURE", "BLOCKED", "IN_PROGRESS"}
        n = v.strip().upper()
        if n not in allowed:
            raise ValueError(f"status must be one of {allowed}, got {v!r}")
        return n


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SHARED TOOLS
# ═══════════════════════════════════════════════════════════════════════════════

@tool
def lookup_ticket(ticket_id: str) -> str:
    """Look up a Jira ticket by ID. Returns title, description, and acceptance criteria."""
    mock_tickets = {
        "PROJ-101": "Title: Add OAuth2 login | AC: Support Google & GitHub providers; store token in httpOnly cookie.",
        "PROJ-202": "Title: Fix N+1 query on /api/orders | AC: Response time < 200ms under 100 concurrent users.",
        "PROJ-303": "Title: Deploy v2.4.1 to production | AC: Zero downtime; rollback plan documented.",
        "PROJ-404": "Title: Add unit tests for payment module | AC: 90% branch coverage; all edge cases covered.",
    }
    return mock_tickets.get(ticket_id.upper(), "Ticket not found — confirm ID or create a new one.")

@tool
def get_pipeline_config(stage: str) -> str:
    """Returns CI/CD configuration and policies for a given pipeline stage."""
    configs = {
        "PLANNING":   "Tool: Jira. Definition of Ready: ticket must have AC, story points, and tech spec. Max sprint WIP: 6.",
        "CODING":     "Tool: GitHub. Branch policy: feature/* → PR required; min 1 review. Linter: ESLint + Prettier. No direct pushes to main.",
        "TESTING":    "Tool: GitHub Actions. Unit tests must pass; coverage ≥ 80%. Integration tests run on PR merge. Load tests gated on staging.",
        "DEPLOYMENT": "Tool: ArgoCD + AWS ECS. Blue-green deploy. Smoke tests required post-deploy. Auto-rollback on >1% 5xx error rate.",
    }
    return configs.get(stage.upper(), "No config found — escalate to platform engineering.")

@tool
def check_code_quality(component: str) -> str:
    """Run static analysis and linting on a named component. Returns issues found."""
    mock_results = {
        "auth":    "ESLint: 0 errors, 2 warnings (unused vars). Complexity: OK. Security: 1 finding — JWT secret in env, not vault.",
        "orders":  "ESLint: 3 errors (missing await). Complexity: HIGH on processOrder() cyclomatic=14. Coverage: 61% (below threshold).",
        "payment": "ESLint: 0 errors. Complexity: OK. Coverage: 91%. No security findings.",
        "deploy":  "Dockerfile lint: OK. Terraform plan: 2 resources to add, 0 to destroy. No drift detected.",
    }
    return mock_results.get(component.lower(), f"No data for '{component}' — run analysis manually.")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SPECIALIST AGENTS
# ═══════════════════════════════════════════════════════════════════════════════

def make_llm(model: str = "gpt-4o-mini"):
    return init_chat_model(model, temperature=0)

def build_specialist_agents(model_name: str = "gpt-4o-mini") -> dict:
    llm          = make_llm(model_name)
    response_fmt = ToolStrategy(PipelineStageResult, handle_errors="raise")
    shared_tools = [lookup_ticket, get_pipeline_config, check_code_quality]

    planner_agent = create_agent(
        model=llm, tools=shared_tools, response_format=response_fmt,
        name="planner_agent",
        system_prompt=(
            "You are a PLANNING specialist (Product Manager / Tech Lead). "
            "Handle requirements gathering, user story refinement, sprint planning, "
            "and Definition of Ready checks. Always call get_pipeline_config for PLANNING. "
            "Look up the ticket if an ID is mentioned. "
            "Flag any story missing acceptance criteria or story points as BLOCKED."
        ),
    )

    developer_agent = create_agent(
        model=llm, tools=shared_tools, response_format=response_fmt,
        name="developer_agent",
        system_prompt=(
            "You are a CODING specialist (Senior Software Engineer). "
            "Handle implementation tasks, code reviews, refactors, and PR readiness checks. "
            "Always call get_pipeline_config for CODING and check_code_quality on the relevant component. "
            "Flag any PR with linting errors or security findings as BLOCKED."
        ),
    )

    tester_agent = create_agent(
        model=llm, tools=shared_tools, response_format=response_fmt,
        name="tester_agent",
        system_prompt=(
            "You are a TESTING specialist (QA Engineer / SDET). "
            "Handle test writing, test execution, coverage analysis, and bug triage. "
            "Always call get_pipeline_config for TESTING and check_code_quality for coverage data. "
            "Flag any component below 80% coverage or with failing tests as FAILURE."
        ),
    )

    deployer_agent = create_agent(
        model=llm, tools=shared_tools, response_format=response_fmt,
        name="deployer_agent",
        system_prompt=(
            "You are a DEPLOYMENT specialist (DevOps / SRE). "
            "Handle releases, blue-green deploys, rollbacks, and post-deploy validation. "
            "Always call get_pipeline_config for DEPLOYMENT. "
            "Look up the ticket if a deploy ticket ID is mentioned. "
            "Flag any deploy without a documented rollback plan as BLOCKED."
        ),
    )

    return {
        "planning":   planner_agent,
        "coding":     developer_agent,
        "testing":    tester_agent,
        "deployment": deployer_agent,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SUPERVISOR PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def _make_route_tool(stage_key: str, agent, description: str):
    """Wrap a specialist agent as a `route_to_<stage>` tool the supervisor can call."""

    @tool(f"route_to_{stage_key}", description=description)
    def _route(work_item: str) -> str:
        result = agent.invoke({"messages": [{"role": "user", "content": work_item}]})
        structured = result.get("structured_response")
        if structured is not None:
            return structured.model_dump_json()
        return result["messages"][-1].text

    return _route

def build_pipeline(model_name: str = "gpt-4o-mini"):
    llm    = make_llm(model_name)
    agents = build_specialist_agents(model_name)

    route_tools = [
        _make_route_tool("planning",   agents["planning"],
            "Route PLANNING work: requirements, user stories, sprint planning, Definition of Ready checks."),
        _make_route_tool("coding",     agents["coding"],
            "Route CODING work: implementation, code review, PR readiness, refactoring."),
        _make_route_tool("testing",    agents["testing"],
            "Route TESTING work: test writing, test execution, coverage gates, bug triage."),
        _make_route_tool("deployment", agents["deployment"],
            "Route DEPLOYMENT work: releases, blue-green deploys, rollbacks, post-deploy validation."),
    ]

    supervisor = create_agent(
        model=llm,
        tools=route_tools,
        name="supervisor_agent",
        system_prompt=(
            "You are the CI/CD Pipeline Orchestrator. "
            "Your ONLY job is to read the inbound work item and immediately route it "
            "to the correct stage agent by calling exactly ONE route_to_* tool. "
            "Do NOT answer the work item yourself.\n"
            "  route_to_planning   → requirements, user stories, sprint planning, Definition of Ready checks\n"
            "  route_to_coding     → implementation, code review, PR readiness, refactoring\n"
            "  route_to_testing    → test writing, test execution, coverage gates, bug triage\n"
            "  route_to_deployment → releases, blue-green deploys, rollbacks, post-deploy validation\n"
            "One route only — call the tool immediately, then return its result."
        ),
    )

    return supervisor


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

TEST_CASES = [
    {"query": "PROJ-101 is up for grooming — does it meet Definition of Ready?",                       "tier": "standard",  "expected": "planning"},
    {"query": "PR #847 for the auth service is open. Review it for merge readiness.",                  "tier": "standard",  "expected": "coding"},
    {"query": "Orders component coverage is at 61%. We're blocked on the CI coverage gate.",           "tier": "standard",  "expected": "testing"},
    {"query": "Deploy PROJ-303 to production tonight. Blue-green, zero downtime required.",            "tier": "critical",  "expected": "deployment"},
    {"query": "processOrder() has a cyclomatic complexity of 14. Refactor before next sprint.",        "tier": "standard",  "expected": "coding"},
    {"query": "Write unit tests for the payment refund edge cases to hit 90% branch coverage.",        "tier": "standard",  "expected": "testing"},
    {"query": "Add the OAuth2 login story to the next sprint — needs AC and story points first.",      "tier": "standard",  "expected": "planning"},
    {"query": "Production deploy failed — 5xx rate spiked to 4%. Need immediate rollback.",            "tier": "critical",  "expected": "deployment"},
    {"query": "ESLint is throwing 3 missing-await errors in the orders service. Fix before merge.",    "tier": "standard",  "expected": "coding"},
    {"query": "Integration tests are failing on staging after the DB migration. Needs triage.",        "tier": "standard",  "expected": "testing"},
]

def _routed_stage(messages) -> str:
    """Find which route_to_* tool the supervisor called."""
    for msg in messages:
        for tc in getattr(msg, "tool_calls", None) or []:
            name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
            if name.startswith("route_to_"):
                return name.replace("route_to_", "")
    return "unknown"

def run_benchmark(model_name: str = "gpt-4o-mini") -> None:
    pipeline = build_pipeline(model_name)
    latencies, successes, failures, correct = [], 0, 0, 0

    print("\n" + "═"*75)
    print(f"  BENCHMARK — LangChain Supervisor CI/CD Pipeline [{model_name}]")
    print("═"*75)

    for i, case in enumerate(TEST_CASES):
        print(f"\n[{i+1:02d}] {case['query'][:70]}...")
        t0 = time.perf_counter()
        try:
            result  = pipeline.invoke({"messages": [{"role": "user", "content": case["query"]}]})
            latency = (time.perf_counter() - t0) * 1000
            stage   = _routed_stage(result["messages"])
            routed_ok = stage == case["expected"]
            if routed_ok:
                correct += 1
            successes += 1
            latencies.append(latency)
            print(f"       agent={stage:<12} {latency:>7.1f}ms  {'✓' if routed_ok else '✗ (expected '+case['expected']+')'}")
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            failures += 1
            latencies.append(latency)
            print(f"       FAILED [{type(e).__name__}]: {str(e)[:80]}")

    total = successes + failures
    s     = sorted(latencies)
    print("\n" + "═"*75)
    print(f"  Success rate  : {successes}/{total} ({successes/total*100:.1f}%)")
    print(f"  Routing acc.  : {correct}/{successes} ({correct/max(successes,1)*100:.1f}%)")
    if latencies:
        print(f"  Latency (ms)  : mean={statistics.mean(latencies):.0f}  "
              f"median={statistics.median(latencies):.0f}  "
              f"p95={s[max(0,int(len(s)*0.95)-1)]:.0f}  "
              f"min={min(latencies):.0f}  max={max(latencies):.0f}")
    print("═"*75)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SINGLE DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def demo(query: str, tier: str = "standard", channel: str = "github") -> None:
    pipeline = build_pipeline()
    print("\n" + "─"*65)
    print(f"  QUERY   : {query}")
    print(f"  TIER    : {tier} | CHANNEL: {channel}")
    print("─"*65)

    t0      = time.perf_counter()
    result  = pipeline.invoke({"messages": [{"role": "user", "content": query}]})
    latency = (time.perf_counter() - t0) * 1000

    print("\n  MESSAGE TRACE:")
    for msg in result["messages"]:
        role    = getattr(msg, "type", "unknown")
        name    = getattr(msg, "name", "")
        label   = f"{role}({name})" if name else role
        for tc in getattr(msg, "tool_calls", None) or []:
            tc_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
            print(f"    [{label}] → tool_call: {tc_name}")
        content = msg.content
        if isinstance(content, str) and content.strip():
            print(f"    [{label}] {content[:120].replace(chr(10), ' ')}")

    print(f"\n  Total latency : {latency:.0f}ms")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo(
        query="Production deploy PROJ-303 is scheduled for tonight. "
              "Blue-green rollout, zero downtime required. Confirm rollback plan is in place before we proceed.",
        tier="critical",
        channel="slack",
    )
    # Uncomment for full benchmark:
    # run_benchmark()
