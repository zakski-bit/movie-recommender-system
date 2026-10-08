# CineMatch — End-to-End Movie Recommender System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1INLUzym1GanvzJL8NU3YgXJeYVfl0K8N?usp=sharing)
[![Live Demo on Vercel](https://img.shields.io/badge/Live%20Demo-Vercel-black?style=flat&logo=vercel)](https://movie-recommender-system-mu-sable.vercel.app/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00.svg?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Dataset: MovieLens 100K](https://img.shields.io/badge/Dataset-MovieLens%20100K-4B8BBE.svg)](https://grouplens.org/datasets/movielens/)

Sistem rekomendasi film *end-to-end* yang menggabungkan pendekatan **Content-Based Filtering** (TF-IDF & Cosine Similarity) dan **Collaborative Filtering** berbasis *Deep Learning Neural Embeddings* (**RecommenderNet** dengan TensorFlow/Keras) pada dataset standar industri **MovieLens 100K**.

Proyek ini dilengkapi dengan:
- **Jupyter Notebook (`proyek_akhir_sistem_rekomendasi.ipynb`)**: Pipeline lengkap 47 sel yang dapat dijalankan langsung di Google Colab.
- **Skrip Python Mandiri (`proyek_akhir_sistem_rekomendasi.py`)**: Implementasi kode bersih dan *reproducible*.
- **Aplikasi Web Demo Interaktif (`index.html`, `style.css`, `app.js`)**: Siap di-*deploy* instan ke **Vercel** tanpa konfigurasi rumit (*zero-build*).

---

## Ringkasan Kinerja & Evaluasi Model

| Pendekatan Rekomendasi | Arsitektur / Model | Metrik Evaluasi | Nilai Kuantitatif Aktual | Karakteristik Utama |
| :--- | :--- | :---: | :---: | :--- |
| **Content-Based Filtering** | TF-IDF Vectorizer + Cosine Similarity | **Precision@10** | **100.00%** | Memetakan kesamaan tematik genre; bebas dari masalah *cold-start* item baru. |
| **Collaborative Filtering** | RecommenderNet (50-dim Embeddings, Keras) | **RMSE (Skala Asli 0.5 - 5.0)**<br>**MAE (Skala Asli 0.5 - 5.0)**<br>**Binary Crossentropy (Loss)** | **0.9796**<br>**0.7494**<br>**0.6423** | Menemukan preferensi laten lintas-genre (*serendipity*); merefleksikan pola konsumsi komunitas. |

### Catatan Diagnostik & Analisis Baseline
- **Dinamika Pelatihan:** Model mencapai performa validasi optimal pada **Epoch ke-3** (Loss: `0.6002`, RMSE: `0.1903`), setelah itu kurva validasi menunjukkan divergensi *overfitting* menuju Epoch 20 (RMSE train `0.0458` vs val `0.2177`).
- **Perbandingan Terhadap Baseline:** Model Matrix Factorization standar (*Biased SVD*, Koren et al., 2009; Harper & Konstan, 2015) pada MovieLens 100K umumnya mencatat RMSE di kisaran **0,87 – 0,92**. Skor RecommenderNet kita (`0.9796`) mencerminkan batas arsitektur sebelum regularisasi diperkuat (*actionable remediation*: penerapan `EarlyStopping(patience=3)` dan pengetatan L2 `1e-4`).

---

## Fitur Aplikasi Web Demo (Siap Vercel)
Aplikasi web yang disertakan dirancang dengan prinsip **Ponytail Dev** (kode minimal, performa maksimal, tanpa dependensi berlebih) dan **Anti AI-Slop** (desain sinematik modern bertema *dark slate*, tipografi tajam, navigasi jelas):
1. **Interactive Content-Based Engine:** Cari dan pilih dari **9.737 film**, sistem menghitung derajat *Cosine Similarity* secara *real-time* di peramban (&lt; 2ms) dan menyajikan Top-10 film paling relevan beserta persentase kecocokan.
2. **Collaborative Filtering Simulation:** Pilih profil persona pengguna (termasuk User 133 yang diuji dalam penelitian) untuk membandingkan riwayat tontonan masa lalu vs rekomendasi prediksi model.
3. **Spesifikasi & Benchmark:** Penjelasan matematis, rincian parameter, tabel perbandingan baseline, serta tautan langsung ke notebook Colab.

---

## Struktur Berkas Repositori
```text
movie-recommender-system/
├── index.html                             # Antarmuka Web Demo Utama (Vercel)
├── style.css                              # Gaya Sinematik Dark Slate
├── app.js                                 # Mesin Rekomendasi Sisi Klien (Zero-latency)
├── vercel.json                            # Konfigurasi Caching & Security Headers Vercel
├── data/                                  # Data Ringan untuk Demo Web
│   ├── movies.json                        # Katalog 9.737 Film & Vektor TF-IDF
│   └── cf_users.json                      # Profil Pengguna & Rekomendasi Model
├── proyek_akhir_sistem_rekomendasi.ipynb  # Jupyter Notebook Lengkap (Google Colab)
├── proyek_akhir_sistem_rekomendasi.py     # Skrip Python Standalone
├── laporan_proyek.md                      # Laporan Riset Komprehensif (Format Dicoding)
└── README.md                              # Dokumentasi Repositori Ini
```

---

## Panduan Menjalankan Proyek

### Opsi 1: Jalankan di Google Colab (Paling Praktis)
1. Buka [Google Colab](https://colab.research.google.com/).
2. Pilih tab **GitHub**, masukkan URL repositori: `https://github.com/zakski-bit/movie-recommender-system`.
3. Pilih berkas `proyek_akhir_sistem_rekomendasi.ipynb`.
4. Jalankan seluruh sel dari atas ke bawah. Dataset MovieLens akan diunduh secara otomatis melalui skrip!

### Opsi 2: Deploy Web Demo ke Vercel (Gratis & Instan)
1. Masuk ke akun [Vercel](https://vercel.com/).
2. Klik **"Add New..."** → **"Project"**.
3. Hubungkan akun GitHub Anda dan pilih repositori `movie-recommender-system`.
4. Pada pengaturan *Framework Preset*, biarkan default (**Other**) karena proyek ini berbasis *Clean Static HTML/JS*.
5. Klik **"Deploy"**. Dalam hitungan detik, aplikasi web demo Anda akan live dengan URL publik (misal: `https://movie-recommender-system-mu-sable.vercel.app/`)!

### Opsi 3: Jalankan Secara Lokal di Komputer Anda
```bash
# Clone repositori
git clone https://github.com/zakski-bit/movie-recommender-system.git
cd movie-recommender-system

# Menjalankan skrip Python Machine Learning
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
python proyek_akhir_sistem_rekomendasi.py

# Menjalankan Web Demo secara lokal
python -m http.server 8000
# Buka http://localhost:8000 pada browser Anda
```

---

## Referensi Ilmiah
1. Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer.
2. Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)*, 5(4), 1-19.
3. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30-37.
4. He, X., et al. (2017). Neural collaborative filtering. *Proceedings of the 26th International Conference on World Wide Web (WWW)*, 173-182.

---
**Pengembang:** [Zaki Abdussalam](https://github.com/zakski-bit)  
**Lisensi:** MIT License
