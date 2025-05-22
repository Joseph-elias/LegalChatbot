import json

JSON_PATHS = [
    r"data/penal_code_articles_ocr.json",
    r"data/tijara_code_articles_ocr.json",
]

for path in JSON_PATHS:
    with open(path, encoding="utf-8") as f:
        articles = json.load(f)
    for a in articles:
        if "undefined" in a.get("text", ""):
            print(f"▶ Found in {os.path.basename(path)} article {a['article_number']}")
