import json
from unittest.mock import Mock

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from ragcourse.judge import AnswerJudgment, judge_answer


def test_judge_receives_the_answer_reference_and_numbered_sources():
    chat = Mock()
    expected = AnswerJudgment(verdict="corretta", reason="Coincide con la FAQ.")
    chat.with_structured_output.return_value.invoke.return_value = expected

    result = judge_answer(
        chat,
        "La spesa è ammessa?",
        "Sì [1].",
        [Document("La spesa è ammessa.")],
        expected="Sì, è ammessa.",
    )

    assert result == expected
    messages = chat.with_structured_output.return_value.invoke.call_args.args[0]
    data = json.loads(messages[1].content)
    assert data["question"] == "La spesa è ammessa?"
    assert data["answer"] == "Sì [1]."
    assert data["official_answer"] == "Sì, è ammessa."
    assert data["sources"] == [{"n": 1, "text": "La spesa è ammessa."}]


def test_official_answer_allows_judging_without_retrieved_documents():
    chat = Mock()

    judge_answer(chat, "Domanda", "Risposta", [], expected="Risposta ufficiale")

    chat.with_structured_output.return_value.invoke.assert_called_once()


def test_no_reference_means_unverifiable_without_an_api_call():
    chat = Mock()

    result = judge_answer(chat, "Domanda", "Risposta", [])

    assert result.verdict == "non verificabile"
    chat.with_structured_output.assert_not_called()


def test_sources_can_be_checked_without_an_official_answer():
    chat = Mock()

    judge_answer(chat, "Domanda", "Risposta", [Document("Una fonte")])

    messages = chat.with_structured_output.return_value.invoke.call_args.args[0]
    assert json.loads(messages[1].content)["official_answer"] is None


def test_verdict_and_reason_are_required_and_validated():
    with pytest.raises(ValidationError):
        AnswerJudgment(verdict="certamente", reason="")
