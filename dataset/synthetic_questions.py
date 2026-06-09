import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm

load_dotenv()

INPUT_FILE = "corpus.jsonl"
OUTPUT_FILE = "queries.jsonl"
MODEL = "gemini-2.5-flash"
SLEEP_BETWEEN = 4.2
MAX_RETRIES_429 = 5

assert os.getenv("GEMINI_API_KEY"), "GEMINI_API_KEY не найден в .env"

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

PROMPT = """You are helping build a retrieval dataset for cybersecurity. Given a CVE vulnerability description, generate 2 realistic search queries that a security analyst, developer, or SOC engineer would type to find this specific vulnerability.

Requirements:
- Query 1: SHORT keyword-style (3-7 words), like a Google search. Example style: "apache struts rce 2024", "openssl heap overflow tls"
- Query 2: LONGER natural question (10-20 words), like asking a colleague. Example style: "Is there a known authentication bypass in Cisco ASA that allows accessing the admin panel without credentials?"
- Do NOT copy exact phrases from the description verbatim. Paraphrase, use synonyms and security jargon.
- Do NOT include the CVE ID in the queries (analysts often don't know it when searching).
- Both queries must plausibly lead to THIS specific vulnerability, not generic ones.

CVE description:
{text}

Return ONLY valid JSON, no markdown, no explanation:
{{"q_short": "...", "q_long": "..."}}"""


def parse_response(text: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("no JSON object in response")
    data = json.loads(m.group(0))
    q_short = data["q_short"].strip()
    q_long = data["q_long"].strip()
    if not (2 <= len(q_short.split()) <= 12):
        raise ValueError(f"bad q_short length: {q_short!r}")
    if not (6 <= len(q_long.split()) <= 30):
        raise ValueError(f"bad q_long length: {q_long!r}")
    return q_short, q_long


def generate_one(doc):
    retries_429 = 0
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=300,
                temperature=0.7,
                messages=[{"role": "user", "content": PROMPT.format(text=doc["text"][:2000])}],
            )
            return parse_response(resp.choices[0].message.content)
        except RateLimitError as e:
            retries_429 += 1
            if retries_429 > MAX_RETRIES_429:
                print("daily limit exceeded")
                sys.exit(0)
            wait = 20 * retries_429
            print(f"\n[429] rate limit, жду {wait}s (попытка {retries_429}/{MAX_RETRIES_429})")
            time.sleep(wait)
        except (APIError, ValueError, KeyError, json.JSONDecodeError) as e:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def main():
    corpus = [json.loads(l) for l in open(INPUT_FILE, encoding="utf-8")]

    done_ids = set()
    if Path(OUTPUT_FILE).exists():
        for line in open(OUTPUT_FILE, encoding="utf-8"):
            try:
                done_ids.add(json.loads(line)["cve_id"])
            except json.JSONDecodeError:
                pass
        print(f"Resume: уже обработано {len(done_ids)}")

    todo = [d for d in corpus if d["cve_id"] not in done_ids]
    print(f"Осталось обработать: {len(todo)} документов "
          f"(~{len(todo) * SLEEP_BETWEEN / 60:.0f} минут при текущем rate limit)")

    failed = 0
    with open(OUTPUT_FILE, "a", buffering=1, encoding="utf-8") as out:
        for doc in tqdm(todo, desc="generating"):
            t0 = time.time()
            try:
                q_short, q_long = generate_one(doc)
                record = {
                    "cve_id": doc["cve_id"],
                    "queries": [
                        {"query_id": f"{doc['cve_id']}__s", "text": q_short, "style": "short"},
                        {"query_id": f"{doc['cve_id']}__l", "text": q_long, "style": "long"},
                    ],
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                failed += 1
                print(f"\nFAIL {doc['cve_id']}: {e}")

            elapsed = time.time() - t0
            if elapsed < SLEEP_BETWEEN:
                time.sleep(SLEEP_BETWEEN - elapsed)

    print(f"Готово. Ошибок: {failed}.")


if __name__ == "__main__":
    main()
