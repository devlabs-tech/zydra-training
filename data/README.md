# Menambahkan data pelatihan

Buat `data/train.jsonl` dalam format JSON Lines ber-encoding UTF-8: satu objek JSON per baris dengan nilai string `text` yang tidak kosong. Untuk generasi teks bebas, gunakan contoh berupa paragraf, cerita, penjelasan, atau teks utuh yang ingin dipelajari model—bukan hanya pertanyaan dan jawaban pendek. Pastikan setiap baris merupakan JSON valid dan tanda kutip atau baris baru di-escape dengan benar.

Anda juga dapat melatih langsung dari satu file teks UTF-8, misalnya `data/train.txt`:

```bash
python3 train.py --data data/train.txt --data-format text --output output/zydra-v2-hf
```

Isi file dapat berupa beberapa paragraf; teks tersebut diperlakukan sebagai satu dokumen.

Simpan sumber teks di folder `train/` pada root proyek. Tambahkan file seperti `data1.txt`, `data2.txt`, atau file lain berekstensi `.txt`. Saat menjalankan pelatihan tanpa argumen, semua `.txt` di folder tersebut dan subfoldernya otomatis digabung menjadi `data/toy_train.jsonl`, lalu dataset itu yang dipakai untuk melatih:

```bash
python3 train.py
```

Folder sumber dapat diganti dengan `--data-dir folder-lain`. Setiap file `.txt` diperlakukan sebagai satu dokumen di JSONL, harus ber-encoding UTF-8, dan tidak boleh kosong. Proses menampilkan file yang ditemukan dan menulis ulang `data/toy_train.jsonl` setiap kali pelatihan dimulai.

Tulis materi pelatihan langsung di file `.txt`. File dapat berisi artikel, penjelasan, panduan, atau contoh dialog dalam teks biasa. Setiap file dimasukkan utuh sebagai teks ke kolom `text`; JSONL percakapan tidak diurai menjadi format chat terstruktur.

Example:

```json
{"text":"Zydra adalah model bahasa yang sedang dilatih dari awal."}
{"text":"Tambahkan teks berkualitas yang memang boleh digunakan untuk pelatihan."}
```

Sebelum pelatihan, catat sumber, lisensi atau izin penggunaan, bahasa, tanggal pengumpulan, dan penyaringan data di file ini atau manifest terpisah. Hapus rahasia dan informasi pribadi yang tidak diizinkan untuk digunakan. Hilangkan duplikasi dan simpan data evaluasi terpisah jika ingin mengukur kemampuan generalisasi.

Pelatihan memprediksi token berikutnya dalam teks. Skrip menggunakan seluruh isi file data; skrip ini tidak memisahkan data atau melakukan penyaringan keamanan. Jangan masukkan contoh evaluasi ke data pelatihan jika hasil evaluasi akan dilaporkan. Untuk membantu model mempelajari pola pemecahan masalah, sertakan contoh yang bervariasi dengan penjelasan dan jawaban yang benar. Ini tetap tidak memberi model kemampuan berpikir seperti manusia atau menjamin semua jawaban benar. Korpus kecil atau berulang cenderung membuat model mengulang contoh, bukan menghasilkan teks yang beragam dan koheren.