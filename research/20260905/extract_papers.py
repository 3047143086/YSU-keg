"""Extract selected local research PDFs into inspectable text files."""

from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "paper_text"
PAPERS = {
    "survey": ROOT / "survey.pdf",
    "bridge": ROOT / "baselines/BRIDGE-official/docs/assets/bridge-paper.pdf",
    "neggen": ROOT / "literature/pdfs/2025/2025_NEGGEN_MM.pdf",
    "befa": ROOT / "literature/pdfs/2025/2025_BeFA_AAAI.pdf",
    "scale": ROOT / "literature/pdfs/2026/2026_SCALE_AAAI.pdf",
    "sg_urinit": ROOT / "literature/pdfs/2026/2026_SG-URInit_SIGIR.pdf",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, path in PAPERS.items():
        reader = PdfReader(path)
        text = "\n\n".join(
            f"PAGE {idx + 1}\n{page.extract_text() or ''}"
            for idx, page in enumerate(reader.pages)
        )
        target = OUT / f"{name}.txt"
        target.write_text(text, encoding="utf-8")
        print(f"{name}: {len(reader.pages)} pages, {len(text)} characters")


if __name__ == "__main__":
    main()
