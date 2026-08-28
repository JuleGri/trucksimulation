from pathlib import Path
import sys
from pypdf import PdfReader

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SELECTIONS = {
    Path(r"C:\Users\Jule\Downloads\(2026) Genga Simulation in Process Mining .pdf"): [14, 15, 16, 17],
    Path(r"C:\Users\Jule\Documents\Master\Masterthesis\trucksimulation\Sources\Sources\(2026) Grobis, Process Mining in Production Planning.pdf"): [5, 6],
    Path(r"C:\Users\Jule\Documents\Master\Masterthesis\trucksimulation\Sources\Sources\(2025) Max, Process Mining in Logistics.pdf"): [4, 5, 6],
}

for path, pages in SELECTIONS.items():
    print(f"\n=== {path.name} ===")
    reader = PdfReader(path)
    for page_number in pages:
        text = " ".join((reader.pages[page_number - 1].extract_text() or "").split())
        print(f"--- PAGE {page_number} ---")
        print(text)
