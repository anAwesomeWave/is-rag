from ranx import Qrels, Run, compare
import json

test_q = [json.loads(l) for l in open("dataset/prepared_data/queries_test.json")]
qrels = Qrels({q["query_id"]: {q["target_doc"]: 1} for q in test_q})

names = ["bi-encoder", "bi-encoder_+_ce", "bi-encoder_ce_ft", "bi-encoder_ce_ft_const_filt"]
runs = [Run.from_file(f"runs/{n}.json") for n in names]

report = compare(qrels, runs, metrics=["ndcg@10", "mrr@10"],
                 max_p=0.05, stat_test="student")
print(report)
