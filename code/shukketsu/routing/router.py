"""Multi-model query router using Qwen 4B for classification.

Classifies incoming queries by complexity and category. Trivial queries
get a direct answer from Qwen; everything else routes to the Llama 70B
agent with tools.
"""

import logging

from langfuse import observe

from code.shukketsu.llm.structured import ModelBackend, get_structured_output
from code.shukketsu.resilience.circuit_breaker import ollama_router_breaker
from code.shukketsu.resilience.errors import CircuitOpenError, LLMUnavailableError, StructuredOutputError
from code.shukketsu.routing.models import RoutingDecision, TaskCategory, TaskComplexity

logger = logging.getLogger(__name__)

_FALLBACK = RoutingDecision(
    complexity=TaskComplexity.COMPLEX,
    category=TaskCategory.CONVERSATION,
    needs_tools=True,
    suggested_agent="general",
    direct_answer=None,
)

_SYSTEM_PROMPT = """\
You are a query classifier for a WoW TBC Rogue knowledge system.

Classify the user's query by complexity and category.

## Complexity levels

- **TRIVIAL**: Simple factual WoW TBC Rogue questions answerable in 1-2 sentences \
(energy costs, ability names, basic stats, talent locations). Fill direct_answer with \
a concise answer.
- **MODERATE**: Questions needing knowledge base search (gear comparisons, rotation \
details, specific encounter advice). Leave direct_answer empty.
- **COMPLEX**: Multi-step analysis, comparisons across specs/phases, simulation \
questions, anything needing multiple tools. Leave direct_answer empty.

## Categories

- **RETRIEVAL**: Looking up specific facts or data
- **ANALYSIS**: Comparing options, evaluating trade-offs
- **RESEARCH**: Open-ended investigation, multi-source gathering
- **WRITING**: Creating articles, guides, summaries
- **CONVERSATION**: Greetings, clarifications, meta-questions

## Rules

- Only set direct_answer for TRIVIAL queries.
- Be concise in direct_answer: 1-2 sentences max.
- When in doubt, classify as MODERATE or COMPLEX (prefer routing to the agent).
"""


@observe()
async def classify_query(query: str) -> RoutingDecision:
    """Classify a user query using Qwen 4B.

    Returns a RoutingDecision with complexity, category, and optionally
    a direct_answer for trivial queries. Falls back to routing to the
    agent on any error.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]

    try:
        decision = await ollama_router_breaker.call(
            get_structured_output,
            RoutingDecision,
            messages,
            backend=ModelBackend.ROUTER,
            temperature=0.0,
            max_tokens=512,
        )
    except (LLMUnavailableError, StructuredOutputError, CircuitOpenError) as exc:
        logger.warning("Router fallback: %s", exc)
        return _FALLBACK

    logger.info(
        "Routed query: complexity=%s category=%s needs_tools=%s",
        decision.complexity,
        decision.category,
        decision.needs_tools,
    )
    return decision
