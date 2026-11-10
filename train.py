import json
import re
from pathlib import Path

DATASHEET_DIR = Path("datasheet")
OUTPUT = Path("data/chat_train_plain.jsonl")


def parse_qa_blocks(text):
    """Parse .txt: blok pertanyaan/jawaban dipisah baris kosong."""
    rows = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        q = lines[0]
        a = " ".join(lines[1:]).strip()
        if not q or not a:
            continue
        if not q.endswith("?"):
            q += "?"
        rows.append(f"{q}\n{a}")
    return rows


def parse_jsonl(text):
    """
    Parse .jsonl. Mendukung:
      - {"text": "..."}
      - {"instruction": "...", "output": "...", "input": "..."}
      - {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
    """
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"  [WARN] baris {line_number}: JSON invalid ({e})")
            continue
        if not isinstance(record, dict):
            continue

        # Format 1: {"text": "..."}
        if isinstance(record.get("text"), str) and record["text"].strip():
            rows.append(record["text"].strip())
            continue

        # Format 2: {"instruction": "...", "output": "..."}
        inst = (
            record.get("instruction")
            or record.get("prompt")
            or record.get("question")
            or ""
        )
        inp = record.get("input") or ""
        out = (
            record.get("output")
            or record.get("response")
            or record.get("answer")
            or ""
        )
        if isinstance(inst, str) and inst.strip() and isinstance(out, str) and out.strip():
            q = inst.strip()
            if isinstance(inp, str) and inp.strip():
                q = f"{q} {inp.strip()}"
            a = out.strip()
            if not q.endswith("?"):
                q += "?"
            rows.append(f"{q}\n{a}")
            continue

        # Format 3: {"messages": [...]}
        messages = record.get("messages")
        if isinstance(messages, list):
            for i in range(len(messages) - 1):
                m1 = messages[i]
                m2 = messages[i + 1]
                if not isinstance(m1, dict) or not isinstance(m2, dict):
                    continue
                if m1.get("role") == "user" and m2.get("role") == "assistant":
                    q = (m1.get("content") or "").strip()
                    a = (m2.get("content") or "").strip()
                    if q and a:
                        if not q.endswith("?"):
                            q += "?"
                        rows.append(f"{q}\n{a}")
    return rows


def load_file(path):
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        return []
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return parse_jsonl(text)
    elif suffix == ".txt":
        return parse_qa_blocks(text)
    return []


def main():
    if not DATASHEET_DIR.is_dir():
        raise SystemExit(f"Folder tidak ditemukan: {DATASHEET_DIR}")

    files = sorted(
        p for p in DATASHEET_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".jsonl"}
    )
    if not files:
        raise SystemExit(f"Tidak ada file .txt atau .jsonl di {DATASHEET_DIR}")

    seen = set()
    all_rows = []
    for path in files:
        try:
            rows = load_file(path)
        except Exception as e:
            print(f"[SKIP] {path.name}: {e}")
            continue

        if not rows:
            print(f"[SKIP] {path.name}: tidak ada baris valid")
            continue

        added = 0
        for r in rows:
            if r in seen:
                continue
            seen.add(r)
            all_rows.append(r)
            added += 1

        ext = path.suffix.lower().lstrip(".")
        print(f"[OK] {path.name} ({ext}): {added} baris unik dari {len(rows)}")

    if not all_rows:
        raise SystemExit("Tidak ada baris valid ditemukan.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps({"text": r}, ensure_ascii=False) + "\n")

    print(f"\nTotal unik: {len(all_rows)} baris")
    print(f"Output    : {OUTPUT}")


if __name__ == "__main__":
    main()