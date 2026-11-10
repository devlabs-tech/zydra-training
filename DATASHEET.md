# Datasheet Model Zydra

## Identitas

- Nama model: Zydra
- Keluaran checkpoint: `output/zydra-v2-hf/` (v1 tetap tersedia di `output/zydra-v1-hf/`)
- Keluaran GGUF: belum dibuat; converter/runtime `llama.cpp` belum mendukung `general.architecture=zydra`
- Inisialisasi: bobot acak; tidak memakai bobot model pralatih
- Arsitektur: decoder-only dengan config `zydra`; implementasi blok menggunakan Llama (`RMSNorm`, RoPE, GQA, SwiGLU)
- Tokenizer: Byte-level BPE yang dilatih dari korpus sintetis
- Jumlah parameter: 3.084.544

## Konfigurasi awal

| Parameter | Bawaan | Keterangan |
| --- | ---: | --- |
| Kosakata | 260 | Kosakata BPE hasil pelatihan |
| Panjang konteks | 128 token | Panjang maksimum urutan pelatihan |
| Lapisan Transformer | 4 | Blok decoder |
| Ukuran tersembunyi | 256 | Lebar embedding dan model |
| Head attention | 4 | Harus membagi ukuran tersembunyi tanpa sisa |
| Head key/value | 2 | GQA |
| Ukuran intermediate | 704 | Lebar SwiGLU |
| Ukuran batch | 8 | Urutan pelatihan per pembaruan bobot |
| Epoch | 12 | Jumlah pengulangan atas blok pelatihan |
| Learning rate | 0.0003 | Learning rate AdamW |

Model ini hanya proof of concept dengan korpus sintetis kecil, bukan model serbaguna. Sapaan sederhana mulai menghasilkan respons yang relevan tetapi masih terpotong. Uji aritmetika belum menjawab dengan benar. Pelatihan dari awal memerlukan korpus jauh lebih besar, evaluasi yang disiplin, dan komputasi yang memadai untuk menghasilkan kemampuan bahasa yang berguna.

## Data dan penggunaan

- Bahasa yang dituju: Indonesia, untuk demonstrasi sapaan dan aritmetika sederhana
- Pengguna dan tugas yang dituju: belum layak untuk penggunaan nyata
- Penggunaan yang tidak didukung: isi
- Sumber data pelatihan dan lisensi: catat setiap sumber di `data/README.md`
- Batasan dan hasil evaluasi: respons sapaan masih terpotong; aritmetika belum benar; tidak cocok untuk penggunaan nyata

## Catatan pelatihan

- Versi / tanggal dataset: korpus sintetis hasil `data/generate_toy_corpus.py`
- Ukuran dataset dan perkiraan jumlah token: 1.829 contoh
- Perangkat keras dan durasi pelatihan: CPU, 12 epoch
- Loss pelatihan terakhir: 0.2948 mean loss pada epoch terakhir
- Dataset evaluasi dan hasilnya: sapaan menghasilkan respons terkait tetapi terpotong; soal hitung 27 + 14 tidak dijawab dengan benar
- Tipe / kuantisasi GGUF: belum dibuat