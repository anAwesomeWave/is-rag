import pyarrow.dataset # без этого segfault на windows
import json
import time
import os

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from ranx import Qrels, Run, evaluate


os.makedirs("runs", exist_ok=True)

CORPUS_FILE = "dataset/corpus.json"
TEST_FILE = "dataset/prepared_data/queries_test.json"
TOP_K = 100  # candidate for second stage

corpus = [json.loads(l) for l in open(CORPUS_FILE, encoding="utf-8")]
doc_ids = [d["cve_id"] for d in corpus]
doc_texts = [d["text"] for d in corpus]
text_by_id = dict(zip(doc_ids, doc_texts))

test_q = [json.loads(l) for l in open(TEST_FILE, encoding="utf-8")]

# заготовка под метрики, для запроса сохраняем док правильный
qrels = Qrels({q["query_id"]: {q["target_doc"]: 1} for q in test_q})

print(f"Docs: {len(doc_ids)}, Test queries: {len(test_q)}")

# BM25

print("\nBM25: make index...")
bm25 = BM25Okapi([t.lower().split() for t in doc_texts])

bm25_run = {}
t0 = time.time()
for q in test_q:
    scores = bm25.get_scores(q["text"].lower().split())
    top = np.argsort(scores)[::-1][:TOP_K]
    bm25_run[q["query_id"]] = {doc_ids[i]: float(scores[i]) for i in top}
print(f"search: {1000 * (time.time() - t0) / len(test_q):.1f} ms for q")

# bi-encoder

encoder = SentenceTransformer("intfloat/e5-base-v2")

print("\nbi-encoder: offline")
doc_emb = encoder.encode(["passage: " + t for t in doc_texts], batch_size=64, normalize_embeddings=True, show_progress_bar=True)

# q_emb = encoder.encode(["query: " + q["text"] for q in test_q] batch_size=64, normalize_embeddings=True)

print("\nbi-encoder: online")
dense_run = {}
t0 = time.time()
for q in test_q:
    q_embed = encoder.encode("query: " + q["text"], normalize_embeddings=True) # обычно запрос приходит уже в онлайне, поэтому честно время так замерить
    scores = doc_emb @ q_embed  # векторы нормированы поэтому cкалярное произведение = кос. близость
    top = np.argsort(scores)[::-1][:TOP_K]
    dense_run[q["query_id"]] = {doc_ids[i]: float(scores[i]) for i in top}
print(f"search: {1000 * (time.time() - t0) / len(test_q):.2f} ms for q")

# Rerank

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2", max_length=320)

def rerank(first_stage_run):
    # вторая стадия видит только TOP_K кандидатов
    new_run = {}
    times = []
    for q in test_q:
        candidates = list(first_stage_run[q["query_id"]].keys())
        pairs = [(q["text"], text_by_id[d]) for d in candidates]
        t0 = time.time()
        scores = reranker.predict(pairs, batch_size=64)
        times.append(time.time() - t0)
        new_run[q["query_id"]] = dict(zip(candidates, map(float, scores)))
    print(f"rerank: {1000 * np.mean(times):.0f} ms for q")
    return new_run

print("\nrerank BM25...")
bm25_ce_run = rerank(bm25_run)
print("rerank bi-encoder...")
dense_ce_run = rerank(dense_run)


runs = {
    "bm25": bm25_run,
    "bi-encoder": dense_run,
    "bm25 + ce": bm25_ce_run,
    "bi-encoder + ce": dense_ce_run,
}

metrics = ["recall@100", "recall@10", "ndcg@10", "mrr@10"]

print("\n" + "-" * 70)
print(f"{'config':<18}" + "".join(f"{m:>13}" for m in metrics))
print("-" * 70)

for name, run_dict in runs.items():
    run = Run(run_dict, name=name)
    run.save(f"runs/{name.replace(' ', '_')}.json")
    res = evaluate(qrels, run, metrics)
    print(f"{name:<18}" + "".join(f"{res[m]:>13.4f}" for m in metrics))
