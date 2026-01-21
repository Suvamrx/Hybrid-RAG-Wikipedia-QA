"""
Script: generate_report.py
Description: Generates a PDF and/or HTML report from the final_report_template.md, filling in results from evaluation outputs.
"""
import os
import markdown
import pdfkit

# =============================
# Report Generation Script
# =============================
# This script converts the markdown report template to HTML and PDF (if wkhtmltopdf is installed).
# It can be extended to fill in metrics and results automatically from JSON files.

TEMPLATE_PATH = os.path.join('reports', 'final_report_template.md')
OUTPUT_HTML = os.path.join('reports', 'final_report.html')
OUTPUT_PDF = os.path.join('reports', 'final_report.pdf')

def main():
    # Read the markdown template
    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        md_content = f.read()
    # Optionally, fill in placeholders here (e.g., with metrics from JSON)
    # For now, just convert as-is
    html = markdown.markdown(md_content, extensions=['tables'])
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[INFO] HTML report written to {OUTPUT_HTML}")
    # Convert to PDF (requires wkhtmltopdf installed)
    try:
        pdfkit.from_file(OUTPUT_HTML, OUTPUT_PDF)
        print(f"[INFO] PDF report written to {OUTPUT_PDF}")
    except Exception as e:
        print(f"[WARN] PDF generation failed: {e}")

if __name__ == '__main__':
    main()
