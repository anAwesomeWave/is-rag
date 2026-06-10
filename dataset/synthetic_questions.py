import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

INPUT_FILE = "corpus.json"
OUTPUT_FILE = "queries.json"
MODEL_NAME = "BeIR/query-gen-msmarco-t5-base-v1"
BATCH_SIZE = 8
N_CANDIDATES = 4 # for gen
N_KEEP = 2 # after filtering
MAX_INPUT_LEN = 384
MAX_QUERY_LEN = 64

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device).eval()


def is_bad_query(q: str, doc_text: str) -> bool:
    words = q.split()
    if not (3 <= len(words) <= 25):
        return True
    if q.lower() in doc_text.lower():
        return True

    return False


def pick_diverse(candidates, doc_text, n_keep):
    seen_norm, picked = set(), []
    for q in candidates:
        q = q.strip()
        norm = " ".join(sorted(set(q.lower().split())))
        if is_bad_query(q, doc_text) or norm in seen_norm:
            continue

        seen_norm.add(norm)
        picked.append(q)
        if len(picked) == n_keep:
            break

    return picked


@torch.no_grad()
def generate_batch(texts):
    inputs = tokenizer(
        texts, return_tensors="pt", truncation=True,
        max_length=MAX_INPUT_LEN, padding=True,
    ).to(device)

    outs = model.generate(
        **inputs,
        max_length=MAX_QUERY_LEN,
        do_sample=True,
        top_p=0.95,
        temperature=1.0,
        num_return_sequences=N_CANDIDATES,
    )

    decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)


    return [decoded[i * N_CANDIDATES:(i + 1) * N_CANDIDATES] for i in range(len(texts))]


def main():
    corpus = [json.loads(l) for l in open(INPUT_FILE, encoding="utf-8")]

    done_ids = set()
    if Path(OUTPUT_FILE).exists():
        for line in open(OUTPUT_FILE, encoding="utf-8"):
            try:
                done_ids.add(json.loads(line)["cve_id"])
            except json.JSONDecodeError:
                pass

        print(f"already processed: {len(done_ids)}")

    todo = [d for d in corpus if d["cve_id"] not in done_ids]
    print(f"{len(todo)} docs needs to be processed")

    skipped = 0
    with open(OUTPUT_FILE, "a", buffering=1, encoding="utf-8") as out:
        for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="doc2query"):
            batch = todo[i:i + BATCH_SIZE]
            candidates_per_doc = generate_batch([d["text"] for d in batch])

            for doc, candidates in zip(batch, candidates_per_doc):
                picked = pick_diverse(candidates, doc["text"], N_KEEP)
                if len(picked) < N_KEEP:
                    skipped += 1
                    continue

                record = {
                    "cve_id": doc["cve_id"],
                    "queries": [
                        {"query_id": f"{doc['cve_id']}__{j+1}",
                         "text": q, "style": "doc2query"}
                        for j, q in enumerate(picked)
                    ],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Done. Skipped {skipped}")


if __name__ == "__main__":
    main()
