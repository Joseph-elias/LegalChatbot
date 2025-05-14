import json
import re
import unicodedata
import pandas as pd
from pdf2image import convert_from_path
import pytesseract

# Path to your PDFLebanon_Penal_Code_1943.pdf
pdf_path = r'C:/joseph/utc+lille/lille/MIAS_COURSES/chatbot/Lebanon_Penal_Code_1943.pdf'

# 1) OCR extraction page by page
full_text = ""
images = convert_from_path(pdf_path, dpi=300)
for i, img in enumerate(images, 1):
    page_text = pytesseract.image_to_string(img, lang="ara")
    full_text += page_text + "\n"
    print(f"Processed page {i}/{len(images)}")

# 1.b) Unicode normalization & strip Arabic diacritics
full_text = unicodedata.normalize('NFC', full_text)
full_text = re.sub(r'[\u064B-\u0652]', '', full_text)

if "المادة" not in full_text:
    raise RuntimeError("❌ Extraction failed: no 'المادة' found after normalization.")

# 2) Split into articles on ANY "المادة <number> -"
pattern = (
    r'(المادة\s*[0-9\u0660-\u0669]+)\s*-\s*'       # المادة + number + mandatory dash
    r'(.*?)'                                       # non-greedy capture of everything after
    r'(?=(?:المادة\s*[0-9\u0660-\u0669]+\s*-\s*)|$)'  # up to the next header or end
)
matches = re.findall(pattern, full_text, flags=re.DOTALL)

articles = []
for header, body in matches:
    # extract the number
    num_match = re.search(r'المادة\s*([0-9\u0660-\u0669]+)', header)
    num_str = num_match.group(1) if num_match else ""
    # normalize Indic digits → ASCII
    num_str = num_str.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩','0123456789'))
    articles.append({
        "article_number": num_str,
        "text": body.strip()
    })

if not articles:
    raise RuntimeError("❌ No articles detected—check your regex or OCR quality.")

# 3) Save to Excel and JSON
df = pd.DataFrame(articles)
excel_path = r'C:/joseph/utc+lille/lille/MIAS_COURSES/chatbot/penal_code_articles_ocr.xlsx'
json_path  = r'C:/joseph/utc+lille/lille/MIAS_COURSES/chatbot/penal_code_articles_ocr.json'

df.to_excel(excel_path, index=False)
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(articles, f, ensure_ascii=False, indent=4)

print("✅ Files generated with OCR + normalization:")
print("  • Excel:", excel_path)
print("  • JSON :", json_path)
