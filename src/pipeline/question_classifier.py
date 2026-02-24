from typing import Any, Optional
from src.utils.llm_client import get_llm_client
from src.utils.logger import get_logger

logger = get_logger(__name__)

# All supported question types
QUESTION_TYPES = {
    "simple", "aggregation", "calculation", "count", "list",
    "ranking", "comparison", "single_intent", "multi_intent",
    "trend", "distribution", "analytics", "insights", "recommendations",
}

WEB_SEARCH_TYPES = {"analytics", "insights", "recommendations"}

_SYSTEM_PROMPT = """You are a question classifier for an employee survey analytics chatbot.

Classify the user's question into exactly one of these types:
- simple: Basic factual question about data
- aggregation: Average, sum, count, group by
- calculation: Computed metric, score calculation
- count: How many, number of
- list: List all values of a column
- ranking: Best, worst, top N, bottom N
- comparison: Compare A vs B, differences, gaps
- single_intent: One clear analytical goal, deep analysis of one topic
- multi_intent: Multiple analytical goals in one question
- trend: Over time, monthly, yearly
- distribution: Spread, histogram, buckets
- analytics: Deep analytical patterns and findings
- insights: Key findings, drivers, patterns
- recommendations: Action items, suggestions

Return ONLY valid JSON — no markdown, no explanation:
{
  "question_type": "<type>",
  "intent": "<one-sentence description of the analytical goal>",
  "requires_web_search": true/false,
  "aggregation": "<AVG|SUM|COUNT|MIN|MAX|null>",
  "dimensions": ["<column names likely needed>"],
  "metric": "<primary metric column or null>",
  "filters": [],
  "complexity": "<simple|medium|complex>"
}"""


# ── Step 1: Fast keyword pre-filter ──────────────────────────────────────────

def pre_classify(question: str) -> Optional[str]:
    """Return a question type based on keyword rules, or None to fall through to LLM."""
    q = question.lower()
    if any(k in q for k in ["how many", "count", "number of", "how much"]):
        return "count"
    if any(k in q for k in ["list all", "show all", "display all", "what are all", "give me all"]):
        return "list"
    if any(k in q for k in [" top ", "best ", "worst ", "highest", "lowest", "rank", "bottom "]):
        return "ranking"
    if any(k in q for k in ["compare", " vs ", " versus ", "gap between", "difference between"]):
        return "comparison"
    if any(k in q for k in ["average", " avg ", " mean ", "sum of", "total "]):
        return "aggregation"
    if any(k in q for k in ["trend", "over time", "monthly", "quarterly", "yearly", "by month", "by year"]):
        return "trend"
    if any(k in q for k in ["distribution", "spread", "histogram", "breakdown", "buckets"]):
        return "distribution"
    if any(k in q for k in ["recommend", "suggestion", "what should", "how to improve", "action"]):
        return "recommendations"
    if any(k in q for k in ["insight", "pattern", "driver", "key finding", "what drives"]):
        return "insights"
    if any(k in q for k in ["analytic", "deep dive", "analysis", "investigate"]):
        return "analytics"
    return None


# ── Step 2: LLM classification ────────────────────────────────────────────────

async def llm_classify(
    question: str,
    table_summary: str,
    column_names: list[str],
) -> dict[str, Any]:
    """Use LLM to classify a question when keyword pre-filter returns None."""
    llm = get_llm_client()
    user_content = (
        f"Table summary: {table_summary}\n"
        f"Available columns: {', '.join(column_names)}\n\n"
        f"Question: {question}"
    )
    result = await llm.classify(system=_SYSTEM_PROMPT, user_content=user_content)
    if result and isinstance(result, dict):
        return result
    # Fallback
    return _fallback_classification(question)


# ── Public entry point ────────────────────────────────────────────────────────

async def classify_question(
    question: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Two-step classification:
    1. Fast keyword pre-filter
    2. LLM (only if keyword returns None)

    Returns a classification dict.
    """
    # Step 1: keyword pre-filter
    pre_type = pre_classify(question)

    if pre_type is not None:
        logger.debug("question_classified_keyword", type=pre_type)
        return {
            "question_type": pre_type,
            "intent": question,
            "requires_web_search": pre_type in WEB_SEARCH_TYPES,
            "aggregation": _infer_aggregation(pre_type, question),
            "dimensions": metadata.get("primary_dimension_columns", []),
            "metric": metadata.get("primary_metric_column"),
            "filters": [],
            "complexity": "simple" if pre_type in ("count", "list") else "medium",
        }

    # Step 2: LLM classification
    table_summary = metadata.get("table_summary", "Employee survey dataset")
    column_names = list(metadata.get("column_roles", {}).keys())

    result = await llm_classify(question, table_summary, column_names)
    logger.debug("question_classified_llm", type=result.get("question_type"))

    # Ensure requires_web_search is consistent with type
    q_type = result.get("question_type", "simple")
    result["requires_web_search"] = q_type in WEB_SEARCH_TYPES

    return result


def _infer_aggregation(question_type: str, question: str) -> Optional[str]:
    q = question.lower()
    if any(k in q for k in ["average", "avg", "mean"]):
        return "AVG"
    if any(k in q for k in ["sum", "total"]):
        return "SUM"
    if any(k in q for k in ["count", "how many", "number of"]):
        return "COUNT"
    if any(k in q for k in ["minimum", "lowest", "worst", "min"]):
        return "MIN"
    if any(k in q for k in ["maximum", "highest", "best", "max"]):
        return "MAX"
    if question_type in ("aggregation", "calculation"):
        return "AVG"
    if question_type == "count":
        return "COUNT"
    return None


def _fallback_classification(question: str) -> dict[str, Any]:
    return {
        "question_type": "simple",
        "intent": question,
        "requires_web_search": False,
        "aggregation": None,
        "dimensions": [],
        "metric": None,
        "filters": [],
        "complexity": "simple",
    }
