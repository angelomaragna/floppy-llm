"""Deterministic heuristics for selecting Wikipedia history articles and
extracting year-bearing factual sentences. No ML, no randomness."""

import re

# Case-insensitive markers that identify a history-related article (matched
# against the first 1000 characters of the text or the article title).
HISTORY_MARKERS = (
    "battle",
    "war",
    "was a siege",
    "was a war",
    "was king of",
    "was queen of",
    "was emperor of",
    "was a treaty",
    "was a dynasty",
    "was the emperor",
    "was the king",
    "was the queen",
    "was born",
    "died in",
    "his reign",
    "her reign",
    "the reign of",
    "empire",
    "dynasty",
    "century",
    "middle ages",
    "medieval",
    "ancient rome",
    "renaissance",
    "conquered",
    "was besieged",
    "siege of",
    "was crowned",
    "was assassinated",
    "signed the treaty",
    "was a period of",
)

# Title patterns: plain year or "Events of YYYY"-style articles, plus ranges.
_YEAR_RE = re.compile(r"\b(?:1|2)\d{3}\b")
_TITLE_YEAR_RE = re.compile(r"^\s*(?:Events in |Spanish chronicles for )?(?:1|2)\d{3}\b")
_FACT_MAX_BYTES = 200
_BRACKET_RE = re.compile(r"\s*\([^)]*\)|\s*\[[^\]]*\]")


def is_history_article(title: str, text: str) -> bool:
    """True when the article is history-related by title pattern or lead markers."""
    if _TITLE_YEAR_RE.match(title or ""):
        return True
    if _YEAR_RE.search(title or ""):
        return True
    head = (text or "")[:1000].lower()
    return any(m in head for m in HISTORY_MARKERS)


def _clean(sentence: str) -> str:
    sentence = _BRACKET_RE.sub("", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def extract_facts(text: str, max_facts_per_article: int = 5) -> list:
    """Extract short, year-bearing factual sentences from the article text.

    Splits the first section into sentences and keeps only sentences that
    contain a 4-digit year, are at most _FACT_MAX_BYTES long, and are free of
    reference brackets after cleaning.
    """
    lead = (text or "").split("\n\n")[0]
    facts = []
    for raw in re.split(r"(?<=[.!?])\s+", lead):
        sentence = _clean(raw)
        if not sentence or len(sentence.encode("utf-8")) > _FACT_MAX_BYTES:
            continue
        if not _YEAR_RE.search(sentence):
            continue
        # Skip sentences with encoding/template residue
        if "{" in sentence or "|" in sentence:
            continue
        facts.append(sentence)
        if len(facts) >= max_facts_per_article:
            break
    return facts
