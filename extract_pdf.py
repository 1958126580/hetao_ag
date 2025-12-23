"""
PDF内容提取脚本
"""
import subprocess
import sys

# 安装 pymupdf
subprocess.check_call([sys.executable, "-m", "pip", "install", "pymupdf", "-q"])

import fitz  # PyMuPDF

def extract_pdf_text(pdf_path: str) -> str:
    """提取PDF文本内容"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

if __name__ == "__main__":
    pdf_path = r"e:\codes\hetao_ag\Hetao Smart Agriculture & Animal Husbandry Library Design (`hetao_ag`).pdf"
    content = extract_pdf_text(pdf_path)
    
    output_path = r"e:\codes\hetao_ag\pdf_content.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"PDF内容已提取到: {output_path}")
    print(f"内容长度: {len(content)} 字符")
