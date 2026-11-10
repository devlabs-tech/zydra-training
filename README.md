# Zydra

Zydra v2 adalah model bahasa kausal kecil yang dimulai dari bobot acak dan dilatih secara lokal dengan korpus sintetis sapaan dan aritmetika. Ini proof of concept, bukan model serbaguna atau model penalaran yang mumpuni. Checkpoint v1 tetap tersimpan di `output/zydra-v1-hf/`. Checkpoint saat ini hanya mendukung masukan teks; analisis gambar belum didukung.

## Latih proof of concept

Letakkan semua file sumber `.txt` di folder `train/`, misalnya `train/data1.txt`, `train/data2.txt`, dan file lainnya. Saat pelatihan, semua file `.txt` di folder tersebut dan subfoldernya otomatis digabung menjadi dataset `data/toy_train.jsonl`. Jalankan:

```bash
python3 train.py
```

Perintah itu membangun ulang `data/toy_train.jsonl`, melatih model menggunakan dataset tersebut, lalu menyimpan model ke `output/zydra-v2-hf`. Tambahkan atau ubah file `.txt`, lalu jalankan lagi `python3 train.py` agar perubahan ikut dilatih. Proses ini memulai bobot model dari awal; bukan melanjutkan checkpoint lama.

Untuk memakai folder sumber lain:

```bash
python3 train.py --data-dir folder-data-saya --output output/zydra-v2-hf
```

Ganti `folder-data-saya` dengan path folder yang benar-benar ada. Skrip menampilkan file yang ditemukan dan menolak folder yang tidak ada, kosong, atau berisi file `.txt` kosong. `data/toy_train.jsonl` adalah dataset perantara yang dihasilkan otomatis; jangan edit dataset itu sebagai sumber utama karena akan ditulis ulang dari isi `train/`.

Atau gunakan `data/train.jsonl` dengan satu objek JSON per baris dan field `text` berupa teks yang tidak kosong:

```json
{"text":"Contoh teks pelatihan pertama."}
{"text":"Contoh teks pelatihan kedua."}
```

Lihat [data/README.md](data/README.md) untuk panduan format dan pengelolaan data JSONL. Untuk melatih dari file JSONL tersebut:

```bash
python3 train.py --data data/train.jsonl --output output/zydra-v2-hf
```

Jika menggunakan `--data` untuk file teks atau JSONL, pelatihan membaca file tersebut langsung. Perintah generator di bawah membuat ulang `data/chat_train_v2.jsonl` untuk contoh sintetis dan tidak diperlukan jika Anda memakai `train/`.

Untuk membuat ulang korpus sintetis bawaan dan melatih proof of concept:

```bash
python3 data/generate_toy_corpus.py
python3 train.py \
  --data data/chat_train_v2.jsonl \
  --output output/zydra-v2-hf \
  --vocab-size 512 \
  --block-size 128 \
  --layers 4 \
  --hidden-size 256 \
  --heads 4 \
  --kv-heads 2 \
  --intermediate-size 704 \
  --batch-size 8 \
  --epochs 12 \
  --learning-rate 3e-4 \
  --seed 1337
```

Model memakai config Transformers khusus Zydra; implementasi blok saat ini menggunakan model kausal Llama (RMSNorm, RoPE, grouped-query attention, dan SwiGLU). Skrip generator membuat contoh sintetis; data ini hanya untuk menguji alur, bukan membangun asisten yang andal.

## Jalankan tes lokal

```bash
python3 test_zydra.py
```

Untuk menghasilkan kelanjutan teks bebas dari prompt:

```bash
python3 test_zydra.py --free-form --prompt "Pada suatu pagi yang cerah,"
```

Mode bebas menggunakan sampling. Atur panjang dan variasinya dengan `--max-new-tokens`, `--temperature`, dan `--top-p`. Untuk mode tanya-jawab, hilangkan `--free-form`, misalnya `python3 test_zydra.py --prompt "Siapa developer-mu?"`.

Di Windows PowerShell, jalankan `.\test_zydra.ps1`; skrip akan meminta prompt jika tidak diberikan. Mode bebas dapat dijalankan dengan `.\test_zydra.ps1 -Prompt "Pada suatu pagi yang cerah," -FreeForm`.

## Chat interaktif di terminal

```bash
python3 chat_zydra.py
```

Ketik pertanyaan setelah prompt `Anda>`. Gunakan `keluar` untuk berhenti atau `/reset` untuk menghapus riwayat. Generasi dibuat deterministik, tetapi kualitas jawaban model eksperimen ini masih terbatas.

## Ekspor ke GGUF

Pastikan folder `llama.cpp/` tersedia di root proyek. Ekspor checkpoint Hugging Face Zydra v2 ke GGUF F16 dengan:

```bash
python3 convert_zydra_to_gguf.py \
  --model output/zydra-v2-hf \
  --llama-cpp llama.cpp \
  --outfile model/zydra-v2-f16.gguf \
  --outtype f16
```

Skrip menyiapkan salinan sementara dengan metadata arsitektur Llama dan mendaftarkan tokenizer ByteLevel Zydra untuk proses konversi. File model dan kode sumber `llama.cpp` asli tidak diubah. Pastikan `transformers` dan dependensi konverter `llama.cpp` sudah terpasang.

Untuk mengecilkan file setelah ekspor, gunakan executable `llama-quantize` dari build lokal:

```bash
llama.cpp/build/bin/llama-quantize \
  model/zydra-v2-f16.gguf \
  model/zydra-v2-Q4_K_M.gguf \
  Q4_K_M
```

Uji GGUF menggunakan `llama-cli`:

```bash
llama.cpp/build/bin/llama-cli \
  -m model/zydra-v2-f16.gguf \
  -p "halo" \
  -n 64
```

GGUF hasil ekspor memakai arsitektur Llama yang kompatibel, bukan metadata runtime `zydra`; implementasi blok Zydra saat ini mewarisi Llama. Ekspor GGUF tidak meningkatkan kemampuan model, dan kualitas hasil tetap bergantung pada data serta pelatihannya.

Model dilatih dengan objective causal language modeling: memprediksi token berikutnya dari teks sebelumnya. Ini memungkinkan model meneruskan teks, tetapi bukan berarti model memiliki pemahaman atau kemampuan berpikir seperti manusia dan tidak menjamin jawabannya benar. Untuk mengajarkan pola jawaban, sertakan contoh pertanyaan, penjelasan yang benar dan jawaban akhir di korpus. Nilai ketepatan menggunakan data evaluasi terpisah. Kualitas generasi sangat bergantung pada data; korpus bawaan yang sintetis dan kecil hanya untuk uji alur.