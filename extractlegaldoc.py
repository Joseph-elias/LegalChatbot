import json
import re
import unicodedata
import pandas as pd

# For OCR
from pdf2image import convert_from_path
import pytesseract

# Path to your PDF
pdf_path = r'C:/joseph/utc+lille/lille/MIAS_COURSES/chatbot/Lebanon_Penal_Code_1943-2.pdf'

# --- 1) OCR extraction page by page ---
full_text = ""
images = convert_from_path(pdf_path, dpi=300)
for i, img in enumerate(images, 1):
    page_text = pytesseract.image_to_string(img, lang="ara")
    full_text += page_text + "\n"
    print(f"Processed page {i}/{len(images)}")

# --- 1.b) Unicode normalization & suppression des diacritiques arabes ---
# Normalisation NFC pour unifier les formes composées vs décomposées
full_text = unicodedata.normalize('NFC', full_text)
# Supprime les marques de voyelles (tashkeel) U+064B à U+0652
full_text = re.sub(r'[\u064B-\u0652]', '', full_text)

# Quick sanity check
if "المادة" not in full_text:
    raise RuntimeError("❌ Extraction échouée : aucun 'المادة' trouvé après normalisation.")

# --- 2) Split into articles ---
pattern = r'(المادة\s*[0-9\u0660-\u0669]+)\s*[-–]?\s*(.*?)(?=(?:المادة\s*[0-9\u0660-\u0669]+)|$)'
matches = re.findall(pattern, full_text, flags=re.DOTALL)

articles = []
for header, body in matches:
    num_match = re.search(r'المادة\s*([0-9\u0660-\u0669]+)', header)
    num_str = num_match.group(1) if num_match else ""
    # Convertit chiffres indic en ascii
    num_str = num_str.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩','0123456789'))
    article_number = num_str
    articles.append({
        "article_number": article_number,
        "text": body.strip()
    })

if not articles:
    raise RuntimeError("❌ Aucun article détecté. Vérifie ton regex ou la qualité de l’OCR.")

# --- 3) Save to Excel and JSON ---
df = pd.DataFrame(articles)
excel_path = r'C:/joseph/utc+lille/lille/MIAS_COURSES/chatbot/penal_code_articles_ocr.xlsx'
json_path  = r'C:/joseph/utc+lille/lille/MIAS_COURSES/chatbot/penal_code_articles_ocr.json'

df.to_excel(excel_path, index=False)
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(articles, f, ensure_ascii=False, indent=4)

print("✅ Fichiers générés avec OCR + normalisation :")
print("  • Excel:", excel_path)
print("  • JSON :", json_path)