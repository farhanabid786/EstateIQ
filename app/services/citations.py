import logging
import re
from dataclasses import dataclass
from typing import Sequence

from app.api.schemas import SourceRef
from app.retrieval.context_builder import ContextBlock

logger = logging.getLogger(__name__)

_MARKER_GROUP = re.compile(
    r"\[\s*(S\d+(?:\s*[,;]\s*S\d+)*)\s*\]",
    re.IGNORECASE,
)
_SINGLE_MARKER = re.compile(r"\[\s*(S\d+)\s*\]", re.IGNORECASE)
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?؟،])")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")


@dataclass(frozen=True)
class CitedAnswer:
    text: str
    sources: list[SourceRef]
    invalid_markers: int
    used_fallback: bool


def _clean_spacing(text: str) -> str:
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def strip_markers(text: str) -> str:
    """Remove model citation tokens for summaries and plain-text contexts."""
    return _clean_spacing(_MARKER_GROUP.sub("", text))


def validate_citations(answer: str, blocks: Sequence[ContextBlock]) -> CitedAnswer:
    """
    Validate model markers against retrieved context blocks.

    The returned source list is ordered to match the renumbered [S1], [S2]...
    markers consumed by the frontend. If the model emits no usable marker,
    retrieved sources remain available in the separate Sources list, but the
    caller is explicitly told that citation fallback was used.
    """
    valid = {
        block.citation_id.strip().upper(): block
        for block in blocks
        if block.citation_id and block.source
    }
    ordered_ids: list[str] = []
    renumbered: dict[str, int] = {}
    invalid = 0

    def replace_group(match: re.Match[str]) -> str:
        nonlocal invalid
        raw_ids = re.split(r"[,;]", match.group(1))
        replacement: list[str] = []

        for raw_id in raw_ids:
            citation_id = raw_id.strip().upper()
            if citation_id not in valid:
                invalid += 1
                continue

            if citation_id not in renumbered:
                renumbered[citation_id] = len(ordered_ids) + 1
                ordered_ids.append(citation_id)

            replacement.append(f"[S{renumbered[citation_id]}]")

        # Preserve order while removing duplicates within one marker group.
        return "".join(dict.fromkeys(replacement))

    text = _MARKER_GROUP.sub(replace_group, answer)
    text = _clean_spacing(text)

    if invalid:
        logger.warning(
            "Removed %d invalid citation marker(s) from model output",
            invalid,
        )

    if ordered_ids:
        sources = [valid[citation_id].source for citation_id in ordered_ids]
        return CitedAnswer(text, sources, invalid, False)

    # Do not fabricate inline citations. The UI may still show retrieved
    # documents in a separate Sources section, clearly distinct from claims.
    fallback_sources = [block.source for block in blocks]
    logger.warning(
        "Model returned no valid citation markers; exposing retrieved sources "
        "only in the separate Sources list"
    )
    return CitedAnswer(text, fallback_sources, invalid, True)
