"""
Script: generate_report.py
Description: Generates a PDF and/or HTML report from the final_report_template.md, filling in results from evaluation outputs.
"""
import os
import markdown
import pdfkit
import json

# =============================
# Report Generation Script
# =============================
# This script converts the markdown report template to HTML and PDF (if wkhtmltopdf is installed).
# It can be extended to fill in metrics and results automatically from JSON files.

# Always resolve template and output paths relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_PATH = os.path.join(PROJECT_ROOT, 'reports', 'final_report_template.md')
OUTPUT_HTML = os.path.join(PROJECT_ROOT, 'reports', 'final_report.html')
OUTPUT_PDF = os.path.join(PROJECT_ROOT, 'reports', 'final_report.pdf')

def compute_metrics(eval_path):
    with open(eval_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    mrrs = [item['mrr'] for item in data if 'mrr' in item]
    f1s = [item['f1'] for item in data if 'f1' in item]
    rouges = [item['rougeL'] for item in data if 'rougeL' in item]
    return {
        'MRR': sum(mrrs)/len(mrrs) if mrrs else 0,
        'F1': sum(f1s)/len(f1s) if f1s else 0,
        'ROUGE-L': sum(rouges)/len(rouges) if rouges else 0
    }

def main():
    # Compute metrics from evaluation_results.json
    eval_json = os.path.join(PROJECT_ROOT, 'data', 'evaluation_results.json')
    metrics = compute_metrics(eval_json)
    # Read the markdown template
    with open(TEMPLATE_PATH, 'r', encoding='utf-8') as f:
        md_content = f.read()
    # Fill in placeholders
    md_content = md_content.replace('{{MRR}}', f"{metrics['MRR']:.4f}")
    md_content = md_content.replace('{{F1}}', f"{metrics['F1']:.4f}")
    md_content = md_content.replace('{{ROUGE_L}}', f"{metrics['ROUGE-L']:.4f}")
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
