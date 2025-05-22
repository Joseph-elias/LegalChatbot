import json

# 1) Load the JSON
json_path = r'data/muhakamat-madaniya_code_articles_ocr.json'
with open(json_path, 'r', encoding='utf-8') as f:
    articles = json.load(f)

# 2) Extract and normalize article numbers as ints
present_nums = set()
for art in articles:
    num_str = art.get('article_number', '')
    if num_str.isdigit():
        present_nums.add(int(num_str))

# 3) Define full range and compute missing
full_range = set(range(1, 1033))   # 1 through 772 inclusive
missing = sorted(full_range - present_nums)
present_sorted = sorted(present_nums)

# 4) Output results
print(f"Total unique articles found: {len(present_sorted)}")
print(f"Expected total articles     : {len(full_range)}")
print(f"Missing articles count       : {len(missing)}")
print("Missing article numbers      :", missing)
