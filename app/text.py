import re
import unicodedata
from typing import Literal, Optional

Language = Literal["en", "ar", "mixed", "unknown"]

ARABIC_LETTER_RANGE = "\u0621-\u064a"
_ARABIC_LETTER = re.compile(f"[{ARABIC_LETTER_RANGE}]")
_LATIN_LETTER = re.compile("[A-Za-z]")
_ARABIC_MARKS = re.compile("[\u064b-\u065f\u0670\u0640]")
_WORD = re.compile(r"\w+", re.UNICODE)
_CHAR_MAP = str.maketrans(
    {"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه"}
)
_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

ARABIC_RATIO_MIN = 0.65
LATIN_RATIO_MAX = 0.15


def has_arabic(text: str) -> bool:
    return bool(_ARABIC_LETTER.search(unicodedata.normalize("NFKC", text)))


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = _ARABIC_MARKS.sub("", text).translate(_CHAR_MAP).translate(_DIGIT_MAP)
    return re.sub(r"\s+", " ", text).strip()


def arabic_ratio(text: str) -> Optional[float]:
    text = unicodedata.normalize("NFKC", text)
    arabic = len(_ARABIC_LETTER.findall(text))
    total = arabic + len(_LATIN_LETTER.findall(text))
    return arabic / total if total else None


def detect_language(text: str) -> Language:
    ratio = arabic_ratio(text)
    if ratio is None:
        return "unknown"
    if ratio >= ARABIC_RATIO_MIN:
        return "ar"
    if ratio <= LATIN_RATIO_MAX:
        return "en"
    return "mixed"


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in _WORD.findall(normalize_text(text)):
        if len(token) > 4 and token.startswith("وال"):
            token = token[3:]
        elif len(token) > 4 and token.startswith("ال"):
            token = token[2:]
        tokens.append(token)
    return tokens
