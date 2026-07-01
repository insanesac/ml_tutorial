"""Generate one PDF per notes folder with nice formatting."""

import os
import glob
import markdown
from weasyprint import HTML

NOTES_DIR = os.path.dirname(os.path.abspath(__file__))

CSS = """
@page {
    size: A4;
    margin: 2cm 1.5cm;
    @bottom-center {
        content: counter(page);
        font-size: 10px;
        color: #999;
    }
}

body {
    font-family: 'DejaVu Sans', 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
    line-height: 1.6;
    color: #333;
    max-width: 100%;
}

h1 {
    font-size: 24px;
    color: #1a1a2e;
    border-bottom: 3px solid #6c63ff;
    padding-bottom: 8px;
    margin-top: 30px;
    page-break-before: always;
}

h1:first-of-type {
    page-break-before: avoid;
}

h2 {
    font-size: 18px;
    color: #16213e;
    border-bottom: 1px solid #ddd;
    padding-bottom: 4px;
    margin-top: 25px;
}

h3 {
    font-size: 15px;
    color: #0f3460;
    margin-top: 20px;
}

h4 {
    font-size: 13px;
    color: #5f5f7f;
}

p {
    margin: 8px 0;
}

code {
    font-family: 'DejaVu Sans Mono', 'Consolas', monospace;
    font-size: 12px;
    background: #f4f4f8;
    padding: 2px 5px;
    border-radius: 3px;
    color: #c64545;
}

pre {
    background: #1e1e2e;
    color: #cdd6f4;
    padding: 14px 16px;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 11.5px;
    line-height: 1.5;
    page-break-inside: avoid;
}

pre code {
    background: none;
    color: inherit;
    padding: 0;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 12px 0;
    font-size: 12px;
}

th, td {
    border: 1px solid #ddd;
    padding: 8px 10px;
    text-align: left;
}

th {
    background: #6c63ff;
    color: white;
    font-weight: bold;
}

tr:nth-child(even) {
    background: #f8f8fc;
}

blockquote {
    border-left: 4px solid #6c63ff;
    margin: 12px 0;
    padding: 8px 16px;
    background: #f4f4f8;
    color: #555;
}

ul, ol {
    padding-left: 24px;
}

li {
    margin: 4px 0;
}

strong {
    color: #1a1a2e;
}

a {
    color: #6c63ff;
    text-decoration: none;
}

hr {
    border: none;
    border-top: 1px solid #ddd;
    margin: 20px 0;
}
"""

def md_to_html(md_text):
    return markdown.markdown(
        md_text,
        extensions=[
            "pymdownx.highlight",
            "pymdownx.superfences",
            "tables",
            "toc",
            "fenced_code",
            "nl2br",
            "sane_lists",
        ],
        extension_configs={
            "pymdownx.highlight": {
                "noclasses": True,
                "pygments_style": "monokai",
            }
        },
    )


def generate_folder_pdf(folder_path, folder_name):
    md_files = sorted(glob.glob(os.path.join(folder_path, "**", "*.md"), recursive=True))
    if not md_files:
        print(f"  Skipping {folder_name} — no .md files found")
        return

    combined_html = f"<h1>{folder_name.replace('_', ' ').title()}</h1>"

    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            md_text = f.read()
        combined_html += md_to_html(md_text)

    output_path = os.path.join(NOTES_DIR, f"{folder_name}.pdf")
    html_doc = f"<html><head><style>{CSS}</style></head><body>{combined_html}</body></html>"

    HTML(string=html_doc).write_pdf(output_path)
    print(f"  {folder_name}.pdf ({len(md_files)} files)")


def main():
    subdirs = sorted(
        d for d in glob.glob(os.path.join(NOTES_DIR, "*"))
        if os.path.isdir(d) and not d.endswith("__pycache__")
    )

    print("Generating PDFs...")
    for subdir in subdirs:
        folder_name = os.path.basename(subdir)
        generate_folder_pdf(subdir, folder_name)

    print("\nDone! PDFs saved in:", NOTES_DIR)


if __name__ == "__main__":
    main()
