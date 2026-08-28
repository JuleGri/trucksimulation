from pathlib import Path
import sys
from pypdf import PdfReader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FILES = [
    Path(r"C:\Users\Jule\Downloads\(2026) Genga Simulation in Process Mining .pdf"),
    Path(r"C:\Users\Jule\Documents\Master\Masterthesis\trucksimulation\Sources\Sources\(2026) Grobis, Process Mining in Production Planning.pdf"),
    Path(r"C:\Users\Jule\Documents\Master\Masterthesis\trucksimulation\Sources\Sources\(2025) Max, Process Mining in Logistics.pdf"),
]

FOCUS_TERMS = (
    "research gap",
    "future research",
    "future work",
    "limitation",
    "contribution",
    "concluding remarks",
    "conclusions",
    "discussion",
)

for path in FILES:
    print(f"\n=== {path.name} ===")
    if not path.is_file():
        print("MISSING")
        continue
    reader = PdfReader(path)
    print(f"pages={len(reader.pages)} title={reader.metadata.title!r}")
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").replace("\x00", " ")
        lower = text.lower()
        compact = " ".join(text.split())
        matches = [term for term in FOCUS_TERMS if term in lower]
        if page_number == 1:
            print(f"--- ABSTRACT/OPENING PAGE {page_number} ---")
            print(compact[:1800])
        for term in matches:
            position = lower.find(term)
            start = max(0, position - 350)
            end = min(len(compact), position + 1150)
            print(f"--- PAGE {page_number}; TERM {term!r} ---")
            print(compact[start:end])
