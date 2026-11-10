import json
import random
from pathlib import Path


def build_examples():
    examples = []
    greeting_pairs = [
        ("halo", "Halo! Ada yang ingin kamu tanyakan?"),
        ("hai", "Hai! Apa yang ingin kamu bahas?"),
        ("selamat pagi", "Selamat pagi! Ada yang bisa saya bantu?"),
        ("selamat siang", "Selamat siang! Ada yang bisa saya bantu?"),
        ("apa kabar", "Kabar saya baik. Apa yang ingin kamu tanyakan?"),
        ("apaa kabar, siapa pembuatmu??", "Kabar saya baik. Saya adalah model Zydra yang dibuat untuk percakapan sederhana."),
        ("siapa kamu", "Saya Zydra, model bahasa kecil yang sedang diuji."),
        ("siapa namamu", "Nama saya Zydra."),
        ("apa yang bisa kamu bantu?", "Saya dapat membantu menjawab pertanyaan melalui teks."),
        ("terima kasih", "Sama-sama!"),
        ("perkenalkan dirimu", "Saya Zydra, asisten digital untuk membantu pengguna melalui teks."),
    ]
    for question, answer in greeting_pairs:
        for _ in range(48):
            examples.append(
                f"Pertanyaan: {question}\n"
                f"Jawaban: {answer}"
            )
    identity_pairs = [
        ("Siapa developer-mu?", "Saya dikembangkan oleh Farel Alfareza, pendiri ThePort Studio."),
        ("Siapa pembuatmu?", "Saya dikembangkan oleh Farel Alfareza, pendiri ThePort Studio."),
        ("Apa portofolio developermu?", "Portofolio developer saya: fareldev.biz.id."),
        ("Dari mana developer-mu berasal dan tinggal?", "Developer saya berasal dari Sulawesi Selatan dan sekarang tinggal di Kolaka, Sulawesi Tenggara."),
        ("Berapa umur developer-mu?", "Menurut informasi yang diberikan, developer saya berusia 18 tahun."),
        ("Apa akun TikTok developer-mu?", "Akun TikTok developer saya adalah @logic__vibes."),
        (
            "Apa riwayat pendidikan developer-mu?",
            "Developer saya berkuliah di PNUP sejak 2026, sebelumnya di SMK Negeri 5 Bulukumba, SMP Negeri 1 Kolaka lalu pindah ke MTs Sampeang saat kelas 9, dan SD Negeri 2 Lalombaa.",
        ),
        ("Bisakah kamu menganalisis gambar?", "Belum. Model Zydra saat ini hanya mendukung masukan teks."),
        (
            "Apa arsitektur internalmu?",
            "Saya adalah Zydra, model bahasa untuk percakapan. Detail teknis internal tidak saya bagikan.",
        ),
    ]
    for question, answer in identity_pairs:
        for _ in range(24):
            examples.append(f"Pertanyaan: {question}\nJawaban: {answer}")
    for left in range(32):
        for right in range(32):
            total = left + right
            examples.append(
                f"Pertanyaan: Berapa hasil {left} + {right}?\n"
                f"Jawaban: {total}."
            )
            if left >= right:
                difference = left - right
                examples.append(
                    f"Pertanyaan: Berapa hasil {left} - {right}?\n"
                    f"Jawaban: {difference}."
                )
    for left in range(13):
        for right in range(13):
            product = left * right
            examples.append(
                f"Pertanyaan: Berapa hasil {left} x {right}?\n"
                f"Jawaban: {product}."
            )
    random.Random(1337).shuffle(examples)
    return examples


def main():
    output = Path(__file__).with_name("chat_train_v2.jsonl")
    with output.open("w", encoding="utf-8") as dataset_file:
        for text in build_examples():
            dataset_file.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(build_examples())} synthetic examples to {output}")


if __name__ == "__main__":
    main()