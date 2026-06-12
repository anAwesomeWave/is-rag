import json
import random
from pathlib import Path

CORPUS_FILE = "corpus.json"
QUERIES_FILE = "queries.json"
OUT_DIR = Path("prepared_data")
SEED = 1
TEST_SHARE_PERC = 0.2 # 20% to 80% for train

'''

'''

def main():
    OUT_DIR.mkdir(exist_ok=True)

    corpus = [json.loads(l) for l in open(CORPUS_FILE, encoding="utf-8")]
    raw_queries = [json.loads(l) for l in open(QUERIES_FILE, encoding="utf-8")]
    known_ids = {d["cve_id"] for d in corpus}

    queries = []
    for item in raw_queries:
        if item["cve_id"] not in known_ids:
            continue
        for q in item["queries"]:
            queries.append({
                "query_id": q["query_id"],
                "text": q["text"],
                "target_doc": item["cve_id"],
            })
    print(f"doc len: {len(corpus)}, q len: {len(queries)}")

    doc_ids = sorted({q["target_doc"] for q in queries})

    random.Random(SEED).shuffle(doc_ids)

    test_docs = set(doc_ids[:int(len(doc_ids) * TEST_SHARE_PERC)])

    train = [q for q in queries if q["target_doc"] not in test_docs]
    test = [q for q in queries if q["target_doc"] in test_docs]

    for name, qs in [("train", train), ("test", test)]:
        with open(OUT_DIR / f"queries_{name}.jsonl", "w", encoding="utf-8") as f:
            for q in qs:
                f.write(json.dumps(q, ensure_ascii=False) + "\n")

        print(f"{name}: {len(qs)} queries, {len({q['target_doc'] for q in qs})} docs")


if __name__ == "__main__":
    main()
