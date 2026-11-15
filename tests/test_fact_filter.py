import json
import os

from floppy_llm.fact_filter import extract_facts, is_history_article

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "sample_article.json")


def load_fixture():
    with open(FIXTURE) as f:
        return json.load(f)


def test_history_article_identified():
    fx = load_fixture()
    a = fx["history"]
    assert is_history_article(a["title"], a["text"]) is True


def test_non_history_article_rejected():
    fx = load_fixture()
    a = fx["non_history"]
    assert is_history_article(a["title"], a["text"]) is False


def test_year_title_identified():
    assert is_history_article("1415", "Events of the year 1415 in Europe.") is True


def test_extract_only_year_facts():
    fx = load_fixture()
    facts = extract_facts(fx["history"]["text"])
    # expected year-bearing, bracket-stripped, <=200 byte sentences
    assert facts == [
        "It took place on 25 October 1415 near Azincourt, in northern France.",
    ]
    assert all(len(f.encode("utf-8")) <= 200 for f in facts)
    assert all("1415" in f for f in facts)


def test_no_year_no_fact():
    assert extract_facts("Marigolds are orange flowers loved by gardeners.") == []


def test_brackets_stripped():
    facts = extract_facts("The treaty was signed in 1215 (or possibly later) by the king.")
    assert facts == ["The treaty was signed in 1215 by the king."]


def test_max_facts_per_article():
    text = (
        "One happened in 1301. Two happened in 1302. Three happened in 1303. "
        "Four happened in 1304. Five happened in 1305. Six happened in 1306. "
    )
    assert len(extract_facts(text, max_facts_per_article=5)) == 5
