"""LLM-as-judge calls for phase gate evaluation.

Uses Llama 70B via get_structured_output for claim extraction,
faithfulness judgment, and domain accuracy scoring.
"""

import logging

from pydantic import BaseModel, Field

from code.shukketsu.evals.metrics import ClaimFaithfulness, FaithfulnessJudgment
from code.shukketsu.llm.structured import ModelBackend, get_structured_output

logger = logging.getLogger(__name__)

# --- Internal Schemas ---


class _ClaimExtraction(BaseModel):
    claims: list[str]


class _SingleVerdict(BaseModel):
    claim: str
    judgment: FaithfulnessJudgment
    evidence_snippet: str | None = None


class _FaithfulnessVerdict(BaseModel):
    judgments: list[_SingleVerdict]


class _AccuracyScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    reasoning: str


# --- System Prompts ---

_CLAIM_EXTRACTION_PROMPT = """\
Extract only verifiable factual statements from the answer. Skip opinions, \
hedging language ('might', 'could'), structural text ('In this section...'), and \
meta-commentary. Each claim should be a standalone statement that can be checked \
against evidence."""

_FAITHFULNESS_PROMPT = """\
For each claim, determine if the provided evidence supports it.

SUPPORTED: evidence directly states or strongly implies the claim.
NOT_SUPPORTED: evidence contradicts or does not mention the claim.
UNCLEAR: evidence is ambiguous or tangentially related.

Return a judgment for every claim provided."""

_ACCURACY_PROMPT = """\
You are a WoW TBC Rogue expert. Score how well the answer matches the ground truth. \
Check each key fact: is it present and correct? Are there contradictions? A score of \
1.0 means all key facts present and no errors. A score of 0.0 means completely wrong \
or missing all key facts."""


# --- Public Functions ---


async def extract_claims(answer: str) -> list[str]:
    """Extract verifiable factual claims from an answer.

    Returns empty list on failure (conservative -- faithfulness = 1.0).
    """
    try:
        result: _ClaimExtraction = await get_structured_output(
            response_model=_ClaimExtraction,
            messages=[
                {"role": "system", "content": _CLAIM_EXTRACTION_PROMPT},
                {"role": "user", "content": answer},
            ],
            backend=ModelBackend.REASONING,
        )
        return result.claims
    except Exception as exc:
        logger.warning("Claim extraction failed: %s", exc)
        return []


async def judge_faithfulness(
    claims: list[str],
    evidence: list[str],
) -> list[ClaimFaithfulness]:
    """Judge whether each claim is supported by the evidence.

    Returns all UNCLEAR on failure (conservative -- faithfulness = 1.0).
    """
    evidence_text = "\n".join(f"- {e}" for e in evidence)
    claims_text = "\n".join(f"- {c}" for c in claims)

    try:
        result: _FaithfulnessVerdict = await get_structured_output(
            response_model=_FaithfulnessVerdict,
            messages=[
                {"role": "system", "content": _FAITHFULNESS_PROMPT},
                {
                    "role": "user",
                    "content": f"Evidence:\n{evidence_text}\n\nClaims to judge:\n{claims_text}",
                },
            ],
            backend=ModelBackend.REASONING,
        )
        return [
            ClaimFaithfulness(
                claim=j.claim,
                judgment=j.judgment,
                evidence_snippet=j.evidence_snippet,
            )
            for j in result.judgments
        ]
    except Exception as exc:
        logger.warning("Faithfulness judgment failed: %s", exc)
        return [ClaimFaithfulness(claim=c, judgment=FaithfulnessJudgment.UNCLEAR) for c in claims]


async def judge_domain_accuracy(
    answer: str,
    ground_truth: str,
    key_facts: list[str],
) -> float:
    """Score 0.0-1.0: does the answer contain the key facts?

    Returns 0.0 on failure (conservative -- assumes wrong).
    """
    facts_text = "\n".join(f"- {f}" for f in key_facts)

    try:
        result: _AccuracyScore = await get_structured_output(
            response_model=_AccuracyScore,
            messages=[
                {"role": "system", "content": _ACCURACY_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Ground truth:\n{ground_truth}\n\n"
                        f"Key facts to check:\n{facts_text}\n\n"
                        f"Answer to evaluate:\n{answer}"
                    ),
                },
            ],
            backend=ModelBackend.REASONING,
        )
        return result.score
    except Exception as exc:
        logger.warning("Domain accuracy judgment failed: %s", exc)
        return 0.0
