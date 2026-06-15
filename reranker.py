import pyarrow.dataset
import json
import os
import random
import time

import numpy as np
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, CrossEncoder, InputExample
from ranx import Qrels, Run, evaluate

CORPUS_FILE = "dataset/corpus.json"
TRAIN_FILE = "dataset/prepared_data/queries_train.json"
TEST_FILE = "dataset/prepared_data/queries_test.json"
BASE_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
OUT_MODEL = "models/ce_finetuned"
N_NEGATIVES = 4   # per one positive
EPOCHS = 1
BATCH = 16
LR = 2e-5
SEED = 1

random.seed(SEED)
os.makedirs("models", exist_ok=True)
os.makedirs("runs", exist_ok=True)

corpus = [json.loads(l) for l in open(CORPUS_FILE, encoding="utf-8")]
text_by_id = {d["cve_id"]: d["text"] for d in corpus}

train_q = [json.loads(l) for l in open(TRAIN_FILE, encoding="utf-8")]
test_q = [json.loads(l) for l in open(TEST_FILE, encoding="utf-8")]

# пул для негативов — только train-документы, чтобы тексты теста не участвовали в обучении
train_doc_ids = sorted({q["target_doc"] for q in train_q})
train_doc_texts = [text_by_id[d] for d in train_doc_ids]
print(f"train-q: {len(train_q)}, doc len for negatives: {len(train_doc_ids)}")

# ---------------- майнинг hard negatives через bi-encoder ----------------
# негативы берем из top похожих: модель учится отличать релевантный документ
# от похожих, а не от случайных - случайные она отличит и без обучения

encoder = SentenceTransformer("intfloat/e5-base-v2")
print("encode train-docs...")
doc_emb = encoder.encode(["passage: " + t for t in train_doc_texts],
                         batch_size=64, normalize_embeddings=True, show_progress_bar=True)
print("ecnode train-queries...")
q_emb = encoder.encode(["query: " + q["text"] for q in train_q],
                       batch_size=64, normalize_embeddings=True, show_progress_bar=True)

examples = []
for q, emb in zip(train_q, q_emb):
    top = np.argsort(doc_emb @ emb)[::-1][:N_NEGATIVES + 10]
    negatives = [train_doc_ids[i] for i in top
                 if train_doc_ids[i] != q["target_doc"]][:N_NEGATIVES]
    examples.append(InputExample(texts=[q["text"], text_by_id[q["target_doc"]]], label=1.0))
    for neg in negatives:
        examples.append(InputExample(texts=[q["text"], text_by_id[neg]], label=0.0))

random.shuffle(examples)
print(f"train pairs: {len(examples)}, positive {len(train_q)}, neg {len(examples) - len(train_q)})")

# ---------------- обучение ----------------
# num_labels=1 + метки 0/1 => бинарная классификация релевантности (BCE loss)

model = CrossEncoder(BASE_MODEL, num_labels=1, max_length=320)
loader = DataLoader(examples, shuffle=True, batch_size=BATCH)
n_steps = len(loader) * EPOCHS

t0 = time.time()
model.fit(
    train_dataloader=loader,
    epochs=EPOCHS,
    warmup_steps=int(0.1 * n_steps),   # первые 10% шагов lr плавно растет
    optimizer_params={"lr": LR},
    show_progress_bar=True,
)
print(f"learning time: {(time.time() - t0) / 60:.1f} мин")
model.save(OUT_MODEL)
print(f"model saved {OUT_MODEL}")

# ---------------- финальная оценка ----------------
# реранжируем ТЕ ЖЕ кандидаты первой стадии, что и в бейзлайне (из runs/)

qrels = Qrels({q["query_id"]: {q["target_doc"]: 1} for q in test_q})

def rerank(first_stage_run, ce):
    new_run = {}
    for q in test_q:
        candidates = list(first_stage_run[q["query_id"]].keys())
        pairs = [(q["text"], text_by_id[d]) for d in candidates]
        scores = ce.predict(pairs, batch_size=64)
        new_run[q["query_id"]] = dict(zip(candidates, map(float, scores)))
    return new_run

print("\nrerank with fine-tuned model")
bm25_run = Run.from_file("runs/bm25.json").to_dict()
dense_run = Run.from_file("runs/bi-encoder.json").to_dict()

bm25_ft_run = rerank(bm25_run, model)
Run(bm25_ft_run, name="bm25+ce-ft").save("runs/bm25_ce_ft.json")
dense_ft_run = rerank(dense_run, model)
Run(dense_ft_run, name="bi-encoder+ce-ft").save("runs/bi-encoder_ce_ft.json")

# собираем все 6 конфигураций: 4 из бейзлайна (с диска) + 2 новых
all_runs = {
    "bm25": bm25_run,
    "bi-encoder": dense_run,
    "bm25 + ce": Run.from_file("runs/bm25_+_ce.json").to_dict(),
    "bi-encoder + ce": Run.from_file("runs/bi-encoder_+_ce.json").to_dict(),
    "bm25 + ce-ft": bm25_ft_run,
    "bi-encoder + ce-ft": dense_ft_run,
}
metrics = ["recall@100", "recall@10", "ndcg@10", "mrr@10"]

print("\n" + "-" * 75)
print(f"{'config':<22}" + "".join(f"{m:>13}" for m in metrics))
print("-" * 75)
for name, run_dict in all_runs.items():
    res = evaluate(qrels, Run(run_dict), metrics)
    print(f"{name:<22}" + "".join(f"{res[m]:>13.4f}" for m in metrics))
