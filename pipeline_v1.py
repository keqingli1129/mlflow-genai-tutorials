#!/usr/bin/env python3
"""
Production-Ready Intent Routing Pipeline — LangChain 1.x + LangGraph 1.x

Install:
  pip install -U langchain langgraph langchain-openai pydantic python-dotenv
"""

import sys
import time
import statistics
from dataclasses import dataclass
from enum import Enum
from typing import List

# Ensure box-drawing characters print on Windows consoles that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import AgentMiddleware, PIIMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain.agents.structured_output import ToolStrategy
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langchain.tools import tool

load_dotenv()


# ── 1. PYDANTIC SCHEMA ────────────────────────────────────────────────────────

class Department(str, Enum):
    SUPPORT   = "SUPPORT"
    SALES     = "SALES"
    TECHNICAL = "TECHNICAL"
    BILLING   = "BILLING"
    UNKNOWN   = "UNKNOWN"

class Priority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"

class RoutingDecision(BaseModel):
    department:       Department  = Field(description="Best-fit dept: SUPPORT/SALES/TECHNICAL/BILLING/UNKNOWN.")
    priority:         Priority    = Field(description="CRITICAL=system down, HIGH=blocking, MEDIUM=degraded, LOW=general.")
    entities:         List[str]   = Field(default_factory=list, description="Named entities: error codes, product names, account IDs.")
    sentiment:        str         = Field(description="One of: POSITIVE, NEUTRAL, FRUSTRATED, ANGRY.")
    confidence:       float       = Field(ge=0.0, le=1.0, description="Routing confidence 0.0–1.0.")
    one_line_summary: str         = Field(description="One sentence summarising the core issue.")

    @field_validator("sentiment", mode="before")
    @classmethod
    def normalise_sentiment(cls, v: str) -> str:
        allowed = {"POSITIVE", "NEUTRAL", "FRUSTRATED", "ANGRY"}
        n = v.strip().upper()
        if n not in allowed:
            raise ValueError(f"sentiment must be one of {allowed}, got {v!r}")
        return n


# ── 2. TOOLS ──────────────────────────────────────────────────────────────────

@tool
def classify_sentiment(text: str) -> str:
    """Classify sentiment of customer text. Returns POSITIVE, NEUTRAL, FRUSTRATED, or ANGRY."""
    return f"Analyze and classify the sentiment of: {text}"

@tool
def extract_entities(text: str) -> str:
    """Extract named entities (product names, error codes, account IDs) from customer text."""
    return f"Extract named entities from: {text}"

@tool
def lookup_department_policy(department: str) -> str:
    """Returns SLA and escalation policy for a department: SUPPORT, SALES, TECHNICAL, BILLING."""
    policies = {
        "TECHNICAL": "SLA: CRITICAL=15min, HIGH=1hr, MEDIUM=4hr. Escalate CRITICAL to on-call.",
        "SUPPORT":   "SLA: CRITICAL=30min, HIGH=2hr, MEDIUM=8hr. Escalate CRITICAL to team lead.",
        "BILLING":   "SLA: All issues 24hr. CRITICAL=refund/charge errors over $500.",
        "SALES":     "SLA: Respond within 4hr business hours. No on-call.",
    }
    return policies.get(department.upper(), "No policy found — route to UNKNOWN queue.")


# ── 3. MIDDLEWARE ─────────────────────────────────────────────────────────────

@dataclass
class RoutingContext:
    source_channel: str = "web"
    customer_tier:  str = "standard"

class RoutingContextMiddleware(AgentMiddleware):
    def wrap_model_call(self, request: ModelRequest, handler):
        # Inject runtime context into the system message before each model call.
        # (In langchain 1.x the system message is request.system_message, separate
        # from request.messages, and the request is modified via override().)
        ctx: RoutingContext = request.runtime.context
        note = (f"[ROUTING CONTEXT] Source: {ctx.source_channel} | Tier: {ctx.customer_tier}. "
                "Enterprise customers receive HIGH priority minimum.")
        base = request.system_message.text if request.system_message else ""
        request = request.override(system_message=SystemMessage(content=f"{note}\n\n{base}"))
        return handler(request)

    def after_agent(self, state: AgentState, runtime) -> None:
        d = state.get("structured_response")
        if isinstance(d, RoutingDecision):
            print(f"  [Middleware] dept={d.department.value} priority={d.priority.value} conf={d.confidence:.2f}")
        return None


# ── 4. AGENT FACTORY ──────────────────────────────────────────────────────────

def build_routing_agent(model_name: str = "gpt-4o-mini"):
    llm = init_chat_model(model_name, temperature=0)
    return create_agent(
        model=llm,
        tools=[classify_sentiment, extract_entities, lookup_department_policy],
        response_format=ToolStrategy(RoutingDecision, handle_errors="raise"),
        system_prompt=(
            "You are a senior customer operations routing agent. "
            "Produce a precise RoutingDecision. Always call lookup_department_policy "
            "to verify SLA thresholds before setting priority."
        ),
        middleware=[
            PIIMiddleware("email", strategy="redact", apply_to_input=True),
            RoutingContextMiddleware(),
        ],
        context_schema=RoutingContext,
    )


# ── 5. DOWNSTREAM DISPATCH ────────────────────────────────────────────────────

QUEUE_MAP = {
    Department.SUPPORT:   "support-queue",
    Department.SALES:     "sales-crm",
    Department.TECHNICAL: "jira-engineering",
    Department.BILLING:   "billing-zendesk",
    Department.UNKNOWN:   "triage-queue",
}

def dispatch(decision: RoutingDecision, ctx: RoutingContext) -> dict:
    escalate = decision.priority in (Priority.CRITICAL, Priority.HIGH)
    if ctx.customer_tier == "enterprise":
        escalate = True
    return {
        "queue":    QUEUE_MAP[decision.department],
        "escalate": escalate,
        "priority": decision.priority.value,
        "channel":  ctx.source_channel,
        "summary":  decision.one_line_summary,
        "entities": decision.entities,
    }


# ── 6. BENCHMARK ──────────────────────────────────────────────────────────────

TEST_CASES = [
    {"query": "My API key stopped working after the update. Production pipeline is down.",      "channel": "slack", "tier": "enterprise"},
    {"query": "I want to upgrade to Enterprise and need a custom quote for 50 seats.",          "channel": "web",   "tier": "standard"},
    {"query": "Getting 422 errors on /v2/embeddings with docs over 512 tokens.",               "channel": "api",   "tier": "pro"},
    {"query": "I was charged twice this month for the Pro tier. Invoice #INV-2024-8821.",       "channel": "email", "tier": "standard"},
    {"query": "Your redesigned dashboard is gorgeous — love the new dark mode!",               "channel": "web",   "tier": "standard"},
    {"query": "I cannot log in. Password reset emails are not arriving.",                       "channel": "web",   "tier": "pro"},
    {"query": "Is there a discount for paying annually instead of monthly?",                    "channel": "web",   "tier": "standard"},
    {"query": "Fine-tuning job ft-9x2kz8 has been PENDING for 6 hours. Must finish today.",    "channel": "api",   "tier": "enterprise"},
    {"query": "We need the SOC 2 Type II compliance report for our security audit.",            "channel": "email", "tier": "pro"},
    {"query": "Just wanted to say thanks — your support team is amazing.",                     "channel": "web",   "tier": "standard"},
]

def run_benchmark(model_name: str = "gpt-4o-mini") -> None:
    agent = build_routing_agent(model_name)
    latencies, successes, failures = [], 0, 0
    print("\n" + "═"*72)
    print(f"  BENCHMARK — LangChain v1 Routing Agent [{model_name}]")
    print("═"*72)
    for i, case in enumerate(TEST_CASES):
        ctx = RoutingContext(source_channel=case["channel"], customer_tier=case["tier"])
        print(f"\n[{i+1:02d}] {case['query'][:70]}...")
        t0 = time.perf_counter()
        try:
            result  = agent.invoke({"messages": [{"role": "user", "content": case["query"]}]},
                                   context=ctx)
            latency = (time.perf_counter() - t0) * 1000
            d: RoutingDecision = result["structured_response"]
            ticket  = dispatch(d, ctx)
            successes += 1; latencies.append(latency)
            print(f"       dept={d.department.value:<10} priority={d.priority.value:<8} "
                  f"conf={d.confidence:.2f}  {latency:>7.1f}ms")
            print(f"       ► queue={ticket['queue']}  escalate={ticket['escalate']}")
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            failures += 1; latencies.append(latency)
            print(f"       FAILED [{type(e).__name__}]: {str(e)[:80]}")
    total = successes + failures
    s = sorted(latencies)
    print("\n" + "═"*72)
    print(f"  Success rate : {successes}/{total} ({successes/total*100:.1f}%)")
    print(f"  Latency (ms) : mean={statistics.mean(latencies):.0f}  "
          f"median={statistics.median(latencies):.0f}  "
          f"p95={s[max(0,int(len(s)*0.95)-1)]:.0f}  "
          f"min={min(latencies):.0f}  max={max(latencies):.0f}")
    print("═"*72)


# ── 7. SINGLE DEMO ────────────────────────────────────────────────────────────

def demo(query: str, tier: str = "pro", channel: str = "web") -> None:
    agent = build_routing_agent()
    ctx   = RoutingContext(source_channel=channel, customer_tier=tier)
    print("\n" + "─"*60)
    print(f"  QUERY  : {query}")
    print(f"  TIER   : {tier} | CHANNEL: {channel}")
    print("─"*60)
    t0     = time.perf_counter()
    result = agent.invoke({"messages": [{"role": "user", "content": query}]},
                          context=ctx)
    latency = (time.perf_counter() - t0) * 1000
    d: RoutingDecision = result["structured_response"]
    ticket = dispatch(d, ctx)
    print(f"  Dept     : {d.department.value}")
    print(f"  Priority : {d.priority.value}")
    print(f"  Sentiment: {d.sentiment}")
    print(f"  Entities : {d.entities}")
    print(f"  Conf     : {d.confidence:.2f}")
    print(f"  Summary  : {d.one_line_summary}")
    print(f"  Latency  : {latency:.0f}ms")
    print(f"  ► Queue  : {ticket['queue']}  Escalate: {ticket['escalate']}")


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo(
        query="Our embedding API has been returning 503s for 45 minutes. "
              "This is blocking our entire prod ML pipeline. Escalate immediately.",
        tier="enterprise",
        channel="slack",
    )
    # Uncomment to run full 10-query benchmark:
    # run_benchmark()