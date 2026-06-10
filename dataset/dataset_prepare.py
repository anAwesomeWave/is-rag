from datasets import load_dataset
import json, random
import ast

ds = load_dataset("hitoshura25/cvefixes", split="train")

# print(ast.literal_eval(ds[0].get("cve_description"))[0].get('value'))

seen = set()
corpus = []
for row in ds:
    cve_id = row.get("cve_id")
    desc = ast.literal_eval(row.get("cve_description"))[0].get('value')

    if cve_id in seen or not desc or len(desc) < 50:
        continue

    seen.add(cve_id)
    corpus.append({
        "cve_id": cve_id,
        "text": desc.strip(),
        "cwe_name": row.get("cwe_name")
    })

random.seed(42)
random.shuffle(corpus)
corpus = corpus[:4500]

with open("corpus.json", "w") as f:
    for d in corpus:
        f.write(json.dumps(d) + "\n")

print(f"Saved {len(corpus)} documents")
