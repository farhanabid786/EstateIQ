import logging
import re
import subprocess
from pathlib import Path
from typing import Optional

import fitz
import requests


def extract_pdf_text(pdf_bytes: bytes, max_pages: int) -> list[dict]:
    pages: list[dict] = []
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for page_number in range(min(document.page_count, max_pages)):
            text = re.sub(r"\s+", " ", document.load_page(page_number).get_text("text")).strip()
            pages.append({"page_number": page_number + 1, "text": text, "method": "pdf_text"})
    finally:
        document.close()
    return pages


def sufficiently_extracted(pages: list[dict], threshold: int = 100) -> bool:
    usable = sum(len(page["text"]) for page in pages if page["text"])
    return usable >= threshold


def extract_pdf_with_optional_ocr(pdf_bytes: bytes, max_pages: int) -> list[dict]:
    pages = extract_pdf_text(pdf_bytes, max_pages)
    if sufficiently_extracted(pages):
        return pages

    try:
        subprocess.run(["tesseract", "--version"], capture_output=True, timeout=5, check=True)
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        logging.getLogger(__name__).info("OCR unavailable; scanned PDF text remains insufficient")
        return pages

    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for index in range(min(document.page_count, max_pages)):
            page = document.load_page(index)
            if len(pages[index]["text"]) >= 100:
                continue
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            image_bytes = pixmap.tobytes("png")
            temporary_path = Path.cwd() / f".estateiq_ocr_{index}.png"
            temporary_path.write_bytes(image_bytes)
            try:
                result = subprocess.run(
                    ["tesseract", str(temporary_path), "stdout", "-l", "eng+ara"],
                    capture_output=True,
                    timeout=20,
                    check=False,
                    text=True,
                )
                text = re.sub(r"\s+", " ", result.stdout).strip() if result.returncode == 0 else ""
                pages[index] = {
                    "page_number": index + 1,
                    "text": text,
                    "method": "ocr" if text else "pdf_text",
                }
            finally:
                temporary_path.unlink(missing_ok=True)
    finally:
        document.close()
    return pages
