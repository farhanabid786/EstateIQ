import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

from pydantic import ValidationError

from app.models import SourceDocument

logger = logging.getLogger(__name__)

MAX_REPORTED_ISSUES = 10


class IngestionError(ValueError):
    pass


@dataclass(frozen=True)
class RecordIssue:
    file: str
    line: int
    message: str

    def __str__(self) -> str:
        return f"{self.file}:{self.line}: {self.message}"


@dataclass
class IngestionReport:
    files: list[str] = field(default_factory=list)
    lines_read: int = 0
    valid_records: int = 0
    invalid_records: list[RecordIssue] = field(default_factory=list)
    duplicate_ids: list[str] = field(default_factory=list)
    duplicate_content: list[str] = field(default_factory=list)
    conflicting_ids: list[str] = field(default_factory=list)
    documents_loaded: int = 0

    def summary(self) -> dict[str, int]:
        return {
            "files": len(self.files),
            "lines_read": self.lines_read,
            "valid_records": self.valid_records,
            "invalid_records": len(self.invalid_records),
            "duplicate_ids": len(self.duplicate_ids),
            "duplicate_content": len(self.duplicate_content),
            "conflicting_ids": len(self.conflicting_ids),
            "documents_loaded": self.documents_loaded,
        }


def discover_input_files(path: Union[str, Path]) -> list[Path]:
    path = Path(path)
    if path.is_dir():
        return sorted(p for p in path.glob("*.jsonl") if p.is_file())
    if path.is_file():
        return [path]
    raise IngestionError(f"Input path does not exist: {path}")


def _format_validation_error(error: ValidationError) -> str:
    parts = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in error.errors()]
    return "invalid record (" + "; ".join(parts) + ")"


def _parse_line(line: str, file: Path, line_number: int, report: IngestionReport):
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        report.invalid_records.append(RecordIssue(file.name, line_number, f"malformed JSON ({exc.msg})"))
        return None
    if not isinstance(payload, dict):
        report.invalid_records.append(RecordIssue(file.name, line_number, "record must be a JSON object"))
        return None
    try:
        return SourceDocument.model_validate(payload)
    except ValidationError as exc:
        report.invalid_records.append(RecordIssue(file.name, line_number, _format_validation_error(exc)))
        return None


def _read_file(file: Path, report: IngestionReport) -> list[SourceDocument]:
    documents: list[SourceDocument] = []
    try:
        with file.open(encoding="utf-8-sig") as handle:
            for line_number, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                report.lines_read += 1
                document = _parse_line(line, file, line_number, report)
                if document is not None:
                    report.valid_records += 1
                    documents.append(document)
    except UnicodeDecodeError as exc:
        raise IngestionError(f"{file.name} is not valid UTF-8") from exc
    except OSError as exc:
        raise IngestionError(f"Cannot read {file.name}: {exc.strerror}") from exc
    return documents


def _resolve_duplicate_ids(
    documents: list[SourceDocument], report: IngestionReport
) -> list[SourceDocument]:
    groups: dict[str, list[SourceDocument]] = {}
    for document in documents:
        groups.setdefault(document.document_id, []).append(document)
    resolved = []
    for document_id, group in groups.items():
        if len(group) == 1:
            resolved.append(group[0])
            continue
        if len({d.file_hash for d in group}) == 1:
            report.duplicate_ids.append(document_id)
            resolved.append(group[0])
        else:
            report.conflicting_ids.append(document_id)
            resolved.append(max(group, key=lambda d: (d.collected_at, d.file_hash)))
    return resolved


def _drop_duplicate_content(
    documents: list[SourceDocument], report: IngestionReport
) -> list[SourceDocument]:
    kept: dict[tuple[str, str], SourceDocument] = {}
    for document in sorted(documents, key=lambda d: d.document_id):
        key = (document.company, document.file_hash)
        if key in kept:
            report.duplicate_content.append(document.document_id)
        else:
            kept[key] = document
    return list(kept.values())


def _raise_if_problems(report: IngestionReport) -> None:
    problems = [str(issue) for issue in report.invalid_records[:MAX_REPORTED_ISSUES]]
    extra = len(report.invalid_records) - len(problems)
    if extra > 0:
        problems.append(f"... and {extra} more invalid record(s)")
    problems += [f"conflicting content for document_id {i}" for i in report.conflicting_ids[:MAX_REPORTED_ISSUES]]
    if problems:
        raise IngestionError("Input validation failed:\n  " + "\n  ".join(problems))


def load_documents(
    path: Union[str, Path], strict: bool = True
) -> tuple[list[SourceDocument], IngestionReport]:
    files = discover_input_files(path)
    report = IngestionReport(files=[str(f) for f in files])
    parsed: list[SourceDocument] = []
    for file in files:
        parsed.extend(_read_file(file, report))

    unique = _resolve_duplicate_ids(parsed, report)
    if strict:
        _raise_if_problems(report)
    elif report.invalid_records or report.conflicting_ids:
        logger.warning(
            "Skipped %d invalid record(s); %d conflicting id(s) resolved by latest collected_at",
            len(report.invalid_records), len(report.conflicting_ids),
        )

    documents = sorted(_drop_duplicate_content(unique, report), key=lambda d: (d.company, d.document_id))
    report.documents_loaded = len(documents)
    logger.info("Ingestion summary: %s", report.summary())
    return documents, report
