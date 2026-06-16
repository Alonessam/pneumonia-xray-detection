import os
import markdown
from xhtml2pdf import pisa
import io

CSS_STYLE = """
body {
    font-family: 'Arial', sans-serif;
    font-size: 10.5pt;
    line-height: 1.5;
    color: #334155;
    margin: 0;
    padding: 0;
}
@page {
    size: A4;
    margin: 25mm 20mm 20mm;
}
h1 { font-size: 22pt; color: #0f172a; margin-top: 0; }
h2 { font-size: 14pt; color: #0f766e; margin-top: 20px; margin-bottom: 8px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
h3 { font-size: 11.5pt; color: #1e293b; margin-top: 16px; margin-bottom: 4px; }
p { text-align: justify; margin: 6px 0; }
strong { color: #0f172a; }
ul, ol { margin: 4px 0; padding-left: 22px; }
li { margin: 2px 0; }
code { font-family: 'Courier New', monospace; font-size: 9.5pt; color: #0d9488; background: #f0fdfa; padding: 1px 4px; border-radius: 2px; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: 10px 0; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th { background-color: #0f766e; color: white; padding: 5px 6px; text-align: center; font-weight: bold; border: 1px solid #0f766e; }
td { padding: 4px 6px; border: 1px solid #d1d5db; text-align: left; }
tr:nth-child(even) { background-color: #f8fafc; }
hr { border: none; border-top: 1px solid #e2e8f0; margin: 16px 0; }
"""

def convert_md_to_pdf():
    md_path = r"C:\Users\samet\Desktop\Yapay Zeka Projesi\files-mentioned-by-the-user-yz\pneumonia_xray_ai\FINAL_RAPOR.md"
    pdf_path = r"C:\Users\samet\Desktop\Yapay Zeka Projesi\files-mentioned-by-the-user-yz\pneumonia_xray_ai\Zaturre_Tespiti_Final_Raporu.pdf"

    if not os.path.exists(md_path):
        print("Error: md file not found.")
        return

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    html_body = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code"],
    )

    html_template = (
        '<!DOCTYPE html><html lang="tr"><head><meta charset="utf-8">'
        '<style>' + CSS_STYLE + '</style></head><body>'
        + html_body +
        '</body></html>'
    )

    out = io.BytesIO()
    pdf = pisa.pisaDocument(io.StringIO(html_template), dest=out, encoding='utf-8')

    if pdf.err:
        print("PDF generation failed")
        return

    with open(pdf_path, "wb") as f:
        f.write(out.getvalue())

    print("PDF generated: " + pdf_path)

if __name__ == "__main__":
    convert_md_to_pdf()
