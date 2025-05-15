from semantic_search import refined_search
#
def main():
    query = input("🔎 Enter a short query: ")
    results = refined_search(query, top_k=5)
    print("\nTop matches:\n")
    for r in results:
        print(f"Article {r['article_number']} (score={r['score']:.3f})")
        print(r["text"], "\n")

if __name__ == "__main__":
    main()