from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent

for pdf_path in sorted(ROOT.glob("*.pdf")):
    reader = PdfReader(pdf_path)
    out_path = pdf_path.with_suffix(".txt")
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks.append(f"\n\n===== PAGE {page_number} / {len(reader.pages)} =====\n\n{text}")
    out_path.write_text("".join(chunks), encoding="utf-8")
    print(f"{pdf_path.name}\tpages={len(reader.pages)}\ttext={out_path.name}")
