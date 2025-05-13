import json
import re
import unicodedata
import pandas as pd

# For OCR
from pdf2image import convert_from_path
import pytesseract

# Path to your PDF C:\joseph\utc+lille\lille\MIAS_COURSES\chatbot\Lebanon_Penal_Code_1943.pdf
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

pattern = (
    r'(?m)^'                                    # début de ligne
    r'(المادة)\s*'                              # groupe 1 = le mot "المادة"
    r'([0-9\u0660-\u0669]+)\s*'                # groupe 2 = le numéro
    r'-\s*'                                     # tiret OBLIGATOIRE
    r'(.*?)'                                    # groupe 3 = le corps de l’article
    r'(?=(?m)^المادة\s*[0-9\u0660-\u0669]+\s*-|$)'  # lookahead jusqu’au prochain header
)

matches = re.findall(pattern, full_text, flags=re.DOTALL)

articles = []
for _, num_str, body in matches:
    # Convertit chiffres indic en ascii
    num = num_str.translate(str.maketrans('٠١٢٣٤٥٦٧٨٩','0123456789'))
    articles.append({
        "article_number": num,
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
