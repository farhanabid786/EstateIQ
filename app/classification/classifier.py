import re
from dataclasses import dataclass
from typing import Iterable, Literal, Mapping

from app.text import ARABIC_LETTER_RANGE, detect_language, has_arabic, normalize_text

CompanyScope = Literal["darglobal", "wasalt", "both", "uncertain"]
Topic = Literal[
    "company", "project", "location", "service",
    "leadership", "investor", "news", "policy", "general",
]

COMPANY_ALIASES: dict[str, tuple[str, ...]] = {
    "darglobal": ("dar global", "darglobal", "دار غلوبال", "دار جلوبال", "دار جلوبل"),
    "wasalt": ("wasalt", "منصة وصلت", "شركة وصلت"),
}

# Evaluated in this order; the first topic with the highest score wins ties.
TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "leadership": (
        "ceo", "chairman", "chairwoman", "founder", "board", "director",
        "management team", "executive", "leadership", "president",
        "رئيس", "مدير تنفيذي", "مؤسس", "مجلس الإدارة", "الإدارة التنفيذية", "قيادة",
    ),
    "investor": (
        "investor", "shareholder", "shares", "share price", "stock",
        "financial results", "financials", "revenue", "profit", "earnings",
        "dividend", "annual report", "ipo", "listed", "market cap",
        "مستثمر", "مستثمرين", "أسهم", "أرباح", "إيرادات", "تقرير سنوي",
        "نتائج مالية", "توزيعات", "الإفصاح",
    ),
    "policy": (
        "privacy", "terms", "conditions", "cookie", "policy",
        "disclaimer", "سياسة", "خصوصية", "شروط", "أحكام",
    ),
    "news": (
        "news", "press", "announced", "announces", "announcement", "latest",
        "أخبار", "خبر", "بيان صحفي", "أعلنت", "إعلان",
    ),
    "location": (
        "where", "located", "location", "address", "city", "country",
        "neighborhood", "district", "dubai", "abu dhabi", "riyadh", "jeddah",
        "london", "maldives", "oman", "muscat", "doha", "miami",
        "أين", "موقع", "يقع", "مدينة", "دبي", "الرياض", "جدة", "لندن",
    ),
    "project": (
        "project", "development", "villa", "apartment",
        "tower", "residence", "mansion", "mansions",
        "penthouse", "golf", "launch",
        "مشروع", "مشاريع", "فيلا", "فلل", "برج", "أبراج", "شقة", "شقق", "مجمع", "سكني",
    ),
    "service": (
        "service", "platform", "app", "buy", "sell", "rent",
        "mortgage", "financing", "fractional", "broker", "valuation",
        "خدمة", "خدمات", "منصة", "شراء", "بيع", "إيجار", "تمويل", "رهن", "وساطة",
    ),
    "company": (
        "about", "company", "who is", "what is", "who are", "founded", "history",
        "mission", "vision", "overview",
        "شركة", "عن", "من هي", "من هو", "ما هي", "ما هو", "تأسست", "رسالة", "رؤية", "نبذة",
    ),
}

TOPIC_CATEGORIES: dict[str, tuple[str, ...]] = {
    "company": ("company", "faq", "other"),
    "project": ("project",),
    "location": ("project", "company"),
    "service": ("service", "faq"),
    "leadership": ("leadership", "company"),
    "investor": ("investor", "report", "news"),
    "news": ("news",),
    "policy": ("policy",),
    "general": (),
}

GENERIC_ALIAS_TITLES = frozenset({
    "home", "projects", "project", "about", "about us", "press", "news",
    "investor", "investor relations", "contact", "contact us",
    "our projects", "properties", "blog", "careers", "faq", "faqs",
})

PROJECT_ALIAS_BONUS = 1
MIN_ALIAS_CHARS = 4
MAX_ALIAS_WORDS = 4
_TITLE_SEPARATORS = re.compile(r"\s*[|\u2013\u2014:]\s*|\s+-\s+")


@dataclass(frozen=True)
class QueryClassification:
    language: Literal["en", "ar", "mixed", "unknown"]
    scope: CompanyScope
    topic: Topic
    confidence: float
    matched_aliases: tuple[str, ...] = ()

    @property
    def companies(self) -> tuple[str, ...]:
        if self.scope in ("darglobal", "wasalt"):
            return (self.scope,)
        return ("darglobal", "wasalt")

    @property
    def response_scope(self) -> Literal["darglobal", "wasalt", "both"]:
        return "both" if self.scope in ("both", "uncertain") else self.scope  # type: ignore[return-value]

    @property
    def preferred_categories(self) -> tuple[str, ...]:
        return TOPIC_CATEGORIES[self.topic]


def derive_project_alias(title: str) -> str | None:
    name = _TITLE_SEPARATORS.split(title.strip())[0].strip()
    normalized = normalize_text(name)
    if len(normalized) < MIN_ALIAS_CHARS or len(normalized.split()) > MAX_ALIAS_WORDS:
        return None
    if normalized in GENERIC_ALIAS_TITLES:
        return None
    return name


def build_project_aliases(rows: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Maps normalized project names to a company; names claimed by both companies are dropped."""
    owners: dict[str, set[str]] = {}
    for title, company in rows:
        alias = derive_project_alias(title)
        if alias:
            owners.setdefault(normalize_text(alias), set()).add(company)
    return {alias: next(iter(c)) for alias, c in owners.items() if len(c) == 1}


def _phrase(text: str) -> str:
    return r"\s+".join(re.escape(part) for part in normalize_text(text).split())


def _alias_pattern(alias: str) -> re.Pattern[str]:
    body = _phrase(alias)
    if has_arabic(alias):
        return re.compile(
            rf"(?<![{ARABIC_LETTER_RANGE}])[وفبلك]?(?:ال)?{body}(?![{ARABIC_LETTER_RANGE}])"
        )
    return re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")


def _keyword_pattern(keyword: str) -> re.Pattern[str]:
    body = _phrase(keyword)
    if has_arabic(keyword):
        tail = "" if len(normalize_text(keyword)) >= 4 else rf"(?![{ARABIC_LETTER_RANGE}])"
        return re.compile(rf"(?<![{ARABIC_LETTER_RANGE}])[وفبلك]?(?:ال)?{body}{tail}")
    return re.compile(rf"(?<![a-z0-9]){body}(?:s|es)?(?![a-z0-9])")


class QueryClassifier:
    def __init__(self, project_aliases: Mapping[str, str] | None = None) -> None:
        self._company_patterns = {
            company: [(alias, _alias_pattern(alias)) for alias in aliases]
            for company, aliases in COMPANY_ALIASES.items()
        }
        self._project_patterns = [
            (alias, company, _alias_pattern(alias))
            for alias, company in (project_aliases or {}).items()
        ]
        self._topic_patterns = {
            topic: [_keyword_pattern(kw) for kw in keywords]
            for topic, keywords in TOPIC_KEYWORDS.items()
        }

    def classify(
        self, question: str, company: str = "both", language: str = "auto"
    ) -> QueryClassification:
        normalized = normalize_text(question)
        detected = language if language in ("en", "ar") else detect_language(question)
        hits, project_hit = self._alias_hits(normalized)
        scope, confidence = self._scope(company, hits)
        topic = self._topic(normalized, project_hit)
        aliases = tuple(sorted({alias for names in hits.values() for alias in names}))
        return QueryClassification(detected, scope, topic, confidence, aliases)

    def _alias_hits(self, normalized: str) -> tuple[dict[str, list[str]], bool]:
        hits: dict[str, list[str]] = {}
        for company, patterns in self._company_patterns.items():
            for alias, pattern in patterns:
                if pattern.search(normalized):
                    hits.setdefault(company, []).append(alias)
        project_hit = False
        for alias, company, pattern in self._project_patterns:
            if pattern.search(normalized):
                hits.setdefault(company, []).append(alias)
                project_hit = True
        return hits, project_hit

    @staticmethod
    def _scope(selector: str, hits: dict[str, list[str]]) -> tuple[CompanyScope, float]:
        if selector in ("darglobal", "wasalt"):
            return selector, 1.0  # type: ignore[return-value]
        if len(hits) == 1:
            return next(iter(hits)), 0.9  # type: ignore[return-value]
        if len(hits) == 2:
            return "both", 0.9
        return "uncertain", 0.0

    def _topic(self, normalized: str, project_hit: bool) -> Topic:
        best_topic: Topic = "general"
        best_score = 0
        for topic, patterns in self._topic_patterns.items():
            score = sum(1 for pattern in patterns if pattern.search(normalized))
            if topic == "project" and project_hit:
                score += PROJECT_ALIAS_BONUS
            if score > best_score:
                best_topic, best_score = topic, score  # type: ignore[assignment]
        return best_topic
