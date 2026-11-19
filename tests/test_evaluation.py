import pytest

from scripts.evaluate_retrieval import (
    EvalQuestion, correct_company_attribution, evaluate, load_questions, main,
    PLACEHOLDER, recall_at_k, reciprocal_rank,
)


def test_metric_functions():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, frozenset({"b", "z"}), 5) == 0.5
    assert recall_at_k(ranked, frozenset({"d"}), 3) == 0.0
    assert reciprocal_rank(ranked, frozenset({"c"})) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, frozenset({"z"})) == 0.0


def test_evaluate_aggregates_and_reports_failures():
    questions = [EvalQuestion("q1", frozenset({"a"})), EvalQuestion("q2", frozenset({"x"}))]
    rankings = {"q1": ["a", "b"], "q2": ["b", "c"]}
    report = evaluate(questions, lambda item: rankings[item.question])
    assert report.queries == 2 and report.mrr == pytest.approx(0.5)
    assert report.recall == {5: 0.5, 10: 0.5}
    assert [f["question"] for f in report.failures] == ["q2"]


def test_evaluate_with_no_questions():
    assert evaluate([], lambda item: []).queries == 0


def test_evaluate_tracks_out_of_scope_empty_retrieval():
    questions = [
        EvalQuestion("unsupported Wasalt", frozenset(), "wasalt", True),
        EvalQuestion("secret data", frozenset(), "darglobal", True),
    ]
    rankings = {"unsupported Wasalt": [], "secret data": ["darglobal_1"]}
    report = evaluate(questions, lambda item: rankings[item.question])
    assert report.out_of_scope == 2
    assert report.empty_retrieval_rate == pytest.approx(0.5)


def test_company_attribution_rejects_empty_results():
    assert not correct_company_attribution([], "darglobal")
    assert not correct_company_attribution([], "both")


def test_company_attribution_requires_every_result_to_match():
    assert correct_company_attribution(["darglobal_1", "darglobal_2"], "darglobal")
    assert not correct_company_attribution(["darglobal_1", "wasalt_1"], "darglobal")
    assert correct_company_attribution(["darglobal_1", "wasalt_1"], "both")
    assert not correct_company_attribution(["darglobal_1", "unknown_1"], "both")


def test_company_attribution_ignores_empty_and_counts_mixed_scope():
    questions = [
        EvalQuestion("empty", frozenset({"a"}), "darglobal"),
        EvalQuestion("correct", frozenset({"b"}), "darglobal"),
        EvalQuestion("mixed", frozenset({"c"}), "both"),
        EvalQuestion("wrong", frozenset({"d"}), "wasalt"),
    ]
    rankings = {
        "empty": [],
        "correct": ["darglobal_1"],
        "mixed": ["darglobal_1", "wasalt_1"],
        "wrong": ["darglobal_1"],
    }
    report = evaluate(questions, lambda item: rankings[item.question])
    assert report.company_attribution_evaluated == 3
    assert report.correct_company_attribution == pytest.approx(2 / 3)


def test_load_questions_skips_placeholders_and_bad_records(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"question": "real", "relevant_document_ids": ["doc1"], "company": "wasalt"}\n'
        f'{{"question": "todo", "relevant_document_ids": ["{PLACEHOLDER}"]}}\n'
        "not json\n"
        '{"question": "bad company", "relevant_document_ids": ["d"], "company": "x"}\n',
        encoding="utf-8",
    )
    questions, skipped = load_questions(path)
    assert [q.question for q in questions] == ["real"] and questions[0].company == "wasalt"
    assert len(skipped) == 3


def test_cli_refuses_to_report_without_labelled_questions(tmp_path):
    assert main(["--questions", str(tmp_path / "missing.jsonl")]) == 2
    template = tmp_path / "template.jsonl"
    template.write_text(f'{{"question": "q", "relevant_document_ids": ["{PLACEHOLDER}"]}}\n', encoding="utf-8")
    assert main(["--questions", str(template)]) == 2
