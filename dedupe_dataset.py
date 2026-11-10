import json
import re
from pathlib import Path

SRC_FILES = [
    Path("data/chat_train_v2.jsonl"),
]
DST = Path("data/chat_train_plain.jsonl")

def convert_text(text: str) -> str | None:
    m = re.match(
        r"^\s*Pertanyaan\s*:\s*(.*?)\s*\n\s*Jawaban\s*:\s*(.*?)\s*$",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not m:
        return None
    q = m.group(1).strip()
    a = m.group(2).strip()
    if not q or not a:
        return None
    if not q.endswith("?"):
        q = q + "?"
    return f"{q}\n{a}"

seen = set()
kept = 0
skipped = 0

for src in SRC_FILES:
    if not src.exists():
        print(f"Skip (tidak ada): {src}")
        continue
    with src.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            text = rec.get("text", "")
            converted = convert_text(text)
            if converted is None:
                skipped += 1
                continue
            if converted in seen:
                continue
            seen.add(converted)
            kept += 1

with DST.open("w", encoding="utf-8") as f:
    for text in seen:
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

print(f"Converted: {kept} baris")
print(f"Skipped  : {skipped} baris")
print(f"Output   : {DST}")