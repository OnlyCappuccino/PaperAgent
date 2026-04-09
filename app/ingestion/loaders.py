from __future__ import annotations

from pathlib import Path
import fitz


def iter_supported_files(directory: str | Path):
    directory = Path(directory)
    for path in directory.rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.pdf', '.md', '.txt'}:
            yield path


def load_pdf(path: str | Path) -> list[dict]:
    path = Path(path)
    doc = fitz.open(path)
    pages = []
    try:
        for page_index, page in enumerate(doc):
            text = page.get_text('text').strip()
            if text:
                pages.append(
                    {
                        'source': str(path),
                        'page': page_index + 1,
                        'text': text,
                    }
                )
    finally:
        doc.close()
    return pages


def load_text_like(path: str | Path) -> list[dict]:
    path = Path(path)
    text = path.read_text(encoding='utf-8', errors='ignore').strip()
    if not text:
        return []
    return [{'source': str(path), 'page': None, 'text': text}]


def load_documents(directory: str | Path) -> list[dict]:
    records: list[dict] = []
    for path in iter_supported_files(directory):
        if path.suffix.lower() == '.pdf':
            records.extend(load_pdf(path))
        else:
            records.extend(load_text_like(path))
    return records
