import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

PLACEHOLDER = "REPLACE_WITH_REAL_DOCUMENT_ID"
DEFAULT_QUESTIONS = Path("data/eval/questions.jsonl")
KS = (5, 10)

logger = logging.getLogger("evaluate_retrieval")


@dataclass(frozen=True)
class EvalQuestion:
    question: str
    relevant_document_ids: frozenset[str]
    company: str = "both"
    out_of_scope: bool = False
    language: str = "auto"
    capability: str = "retrieval"


@dataclass
class EvalReport:
    queries: int = 0
    recall: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    relevant_document_retrieval_rate: float = 0.0
    empty_retrieval_rate: float = 0.0
    correct_company_attribution: float = 0.0
    company_attribution_evaluated: int = 0
    out_of_scope: int = 0
    capabilities: dict[str, dict[str, float]] = field(default_factory=dict)
    failures: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "queries_evaluated": self.queries,
            "recall": {f"recall@{k}": round(v, 4) for k, v in sorted(self.recall.items())},
            "mrr": round(self.mrr, 4),
            "relevant_document_retrieval_rate": round(self.relevant_document_retrieval_rate, 4),
            "empty_retrieval_rate": round(self.empty_retrieval_rate, 4),
            "correct_company_attribution": round(self.correct_company_attribution, 4),
            "company_attribution_denominator": self.company_attribution_evaluated,
            "out_of_scope_queries": self.out_of_scope,
            "capabilities": self.capabilities,
            "failures": self.failures,
        }


def recall_at_k(ranked: Sequence[str], relevant: frozenset[str], k: int) -> float:
    return len(set(ranked[:k]) & relevant) / len(relevant)


def reciprocal_rank(ranked: Sequence[str], relevant: frozenset[str]) -> float:
    for rank, document_id in enumerate(ranked, start=1):
        if document_id in relevant:
            return 1.0 / rank
    return 0.0


def correct_company_attribution(ranked: Sequence[str], expected_company: str) -> bool:
    if not ranked:
        return False
    if expected_company == "both":
        return all(
            document_id.startswith(("darglobal_", "wasalt_"))
            for document_id in ranked
        )
    return all(document_id.startswith(expected_company + "_") for document_id in ranked)


def evaluate(
    questions: Sequence[EvalQuestion],
    retrieve_documents: Callable[[EvalQuestion], list[str]],
    ks: Sequence[int] = KS,
) -> EvalReport:
    report = EvalReport(queries=len(questions))
    if not questions:
        return report
    supported = [item for item in questions if not item.out_of_scope]
    recall_totals = {k: 0.0 for k in ks}
    mrr_total = 0.0
    empty_count = 0
    relevant_count = 0
    company_count = 0
    company_total = 0
    capability_totals: dict[str, dict[str, float]] = {}
    unsupported_empty: dict[str, int] = {}
    unsupported_queries: dict[str, int] = {}
    for item in questions:
        ranked = retrieve_documents(item)
        totals = capability_totals.setdefault(
            item.capability,
            {"queries": 0.0, "recall@5": 0.0, "recall@10": 0.0, "mrr": 0.0, "relevant_hits": 0.0},
        )
        totals["queries"] += 1.0
        if item.out_of_scope:
            is_empty = int(not ranked)
            empty_count += is_empty
            unsupported_empty[item.capability] = unsupported_empty.get(item.capability, 0) + is_empty
            unsupported_queries[item.capability] = unsupported_queries.get(item.capability, 0) + 1
            continue
        for k in ks:
            score = recall_at_k(ranked, item.relevant_document_ids, k)
            recall_totals[k] += score
            totals[f"recall@{k}"] += score
        rr = reciprocal_rank(ranked, item.relevant_document_ids)
        mrr_total += rr
        totals["mrr"] += rr
        relevant_hit = int(bool(set(ranked[: max(ks)]) & item.relevant_document_ids))
        relevant_count += relevant_hit
        totals["relevant_hits"] += relevant_hit
        if item.company in ("darglobal", "wasalt", "both"):
            if ranked:
                company_total += 1
                company_count += int(correct_company_attribution(ranked[: max(ks)], item.company))
        if not set(ranked[: max(ks)]) & item.relevant_document_ids:
            report.failures.append({
                "question": item.question,
                "relevant": sorted(item.relevant_document_ids),
                "retrieved_top": ranked[:5],
            })
    report.queries = len(questions)
    report.out_of_scope = len(questions) - len(supported)
    if supported:
        report.recall = {k: total / len(supported) for k, total in recall_totals.items()}
        report.mrr = mrr_total / len(supported)
        report.relevant_document_retrieval_rate = relevant_count / len(supported)
        report.correct_company_attribution = company_count / company_total if company_total else 0.0
    report.company_attribution_evaluated = company_total
    report.empty_retrieval_rate = empty_count / report.out_of_scope if report.out_of_scope else 0.0
    report.capabilities = {}
    for capability, totals in capability_totals.items():
        query_count = totals["queries"]
        if capability.startswith("unsupported"):
            report.capabilities[capability] = {
                "queries": int(query_count),
                "empty_retrieval_rate": (
                    unsupported_empty.get(capability, 0)
                    / unsupported_queries.get(capability, 1)
                ),
            }
            continue
        report.capabilities[capability] = {
            "queries": int(query_count),
            "recall@5": round(totals["recall@5"] / query_count, 4) if query_count else 0.0,
            "recall@10": round(totals["recall@10"] / query_count, 4) if query_count else 0.0,
            "mrr": round(totals["mrr"] / query_count, 4) if query_count else 0.0,
            "relevant_document_retrieval_rate": (
                round(totals["relevant_hits"] / query_count, 4) if query_count else 0.0
            ),
        }
    return report


def load_questions(path: Path) -> tuple[list[EvalQuestion], list[str]]:
    questions: list[EvalQuestion] = []
    skipped: list[str] = []
    with path.open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                text = str(record["question"]).strip()
                relevant = frozenset(record["relevant_document_ids"])
                company = record.get("company", "both")
                out_of_scope = bool(record.get("out_of_scope", False))
                language = record.get("language", "auto")
                capability = record.get("capability", "retrieval")
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                skipped.append(f"line {number}: invalid record ({type(exc).__name__})")
                continue
            if not text or (not out_of_scope and (not relevant or PLACEHOLDER in relevant)):
                skipped.append(f"line {number}: placeholder or empty labels")
                continue
            if company not in ("darglobal", "wasalt", "both"):
                skipped.append(f"line {number}: invalid company")
                continue
            if language not in ("auto", "en", "ar"):
                skipped.append(f"line {number}: invalid language")
                continue
            questions.append(
                EvalQuestion(text, relevant, company, out_of_scope, language, capability)
            )
    return questions, skipped


def _build_retriever_fn(mode: str) -> Callable[[EvalQuestion], list[str]]:
    from app.classification.classifier import QueryClassifier, build_project_aliases
    from app.config import get_settings
    from app.retrieval.embeddings import get_embedding_provider
    from app.retrieval.hybrid import HybridRetriever
    from app.retrieval.index_loader import load_index

    settings = get_settings()
    index = load_index(settings)
    classifier = QueryClassifier(build_project_aliases(index.metadata.project_titles()))
    retriever = HybridRetriever(index, get_embedding_provider(), settings)

    def retrieve(item: EvalQuestion) -> list[str]:
        companies = classifier.classify(item.question, item.company).companies
        results = retriever.retrieve(item.question, companies, top_k=max(KS) * 3, mode=mode)
        chunks = index.metadata.get_chunks([r.vector_id for r in results])
        return list(dict.fromkeys(chunk.document_id for chunk in chunks))

    return retrieve


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval on a labelled question set.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--mode", choices=("hybrid", "dense", "bm25"), default="hybrid")
    parser.add_argument("--output", type=Path, help="write the report as JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.questions.is_file():
        logger.error("Question file not found: %s (copy data/eval/questions.template.jsonl and label it)", args.questions)
        return 2
    questions, skipped = load_questions(args.questions)
    for message in skipped:
        logger.warning("Skipped %s", message)
    if not questions:
        logger.error("No labelled questions to evaluate; fill in real relevant_document_ids first.")
        return 2

    report = evaluate(questions, _build_retriever_fn(args.mode))
    result = report.as_dict()
    result["mode"] = args.mode
    result["skipped"] = len(skipped)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
