"""
PDF内容提取脚本
"""
import fitz  # PyMuPDF

pdf_path = r"e:\codes\hetao_ag\Hetao Smart Agriculture & Animal Husbandry Library Design (`hetao_ag`).pdf"
doc = fitz.open(pdf_path)
content = ''.join([p.get_text() for p in doc])

# Save to file
output_path = r"e:\codes\hetao_ag\pdf_content.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"Content saved to {output_path}")
