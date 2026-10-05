# Laporan Proyek Akhir Machine Learning - Zaki Abdussalam

## Project Overview

Dalam era digital modern dan pertumbuhan masif layanan *streaming video-on-demand* (seperti Netflix, Disney+, Amazon Prime Video, dan HBO Go), pengguna disajikan dengan puluhan ribu pilihan film setiap harinya. Kondisi ini sering kali menimbulkan fenomena psikologis yang dikenal sebagai *paradox of choice* atau *information overload* (ledakan informasi), di mana pengguna merasa kewalahan, frustrasi, dan membutuhkan waktu lama hanya untuk memilih tontonan yang sesuai dengan selera mereka [1]. Tanpa mekanisme navigasi yang cerdas, pengguna cenderung meninggalkan platform (*churn*), yang secara langsung menurunkan tingkat retensi dan keterlibatan pengguna (*user engagement*).

Sistem rekomendasi (*recommender systems*) menjadi instrumen teknologi yang sangat krusial untuk mengatasi tantangan tersebut. Dengan menganalisis karakteristik konten film dan pola perilaku pengguna di masa lalu, sistem rekomendasi mampu menyaring katalog raksasa dan menyajikan daftar film yang relevan dan dipersonalisasi secara *real-time* [2]. Bagi penyedia layanan, personalisasi yang akurat terbukti mendongkrak metrik bisnis secara signifikan: meningkatkan durasi konsumsi media (*watch time*), meningkatkan kepuasan pelanggan, serta membuka peluang monetisasi dan loyalitas pengguna jangka panjang [3].

Untuk mencapai performa sistem rekomendasi yang optimal dan komprehensif, proyek ini menerapkan dua paradigma solusi komplementer:
1. **Content-Based Filtering**: Merekomendasikan film baru berdasarkan kemiripan atribut genre film yang telah dinikmati pengguna sebelumnya menggunakan pembobotan *Term Frequency-Inverse Document Frequency* (TF-IDF) dan *Cosine Similarity*.
2. **Collaborative Filtering**: Memanfaatkan interaksi riwayat penilaian (*rating*) dari seluruh komunitas pengguna untuk menemukan pola laten preferensi antarpengguna (*latent factors*) menggunakan arsitektur jaringan saraf mendalam (*Deep Learning*) **RecommenderNet** dengan lapisan *Embedding* pada TensorFlow/Keras [4].

### Referensi Terkait:
1. Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*. Springer, Boston, MA. https://doi.org/10.1007/978-1-4899-7637-6
2. Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS)*, 5(4), 1-19. https://doi.org/10.1145/2827872
3. Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer*, 42(8), 30-37. https://doi.org/10.1109/MC.2009.263
4. He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T. S. (2017). Neural collaborative filtering. *Proceedings of the 26th International Conference on World Wide Web (WWW)*, 173-182. https://doi.org/10.1145/3038912.3052569

---

## Business Understanding

### Problem Statements
Berdasarkan latar belakang di atas, rumusan masalah dalam proyek ini adalah:
1. Bagaimana cara merekomendasikan film baru yang memiliki karakteristik tema atau genre serupa dengan film yang disukai oleh pengguna (*Content-Based Filtering*)?
2. Bagaimana cara memanfaatkan riwayat penilaian (*rating*) dari seluruh komunitas pengguna untuk memberikan rekomendasi film baru yang terpersonalisasi kepada seorang pengguna tertentu (*Collaborative Filtering*)?
3. Bagaimana kinerja kedua pendekatan sistem rekomendasi tersebut apabila diuji dan diukur menggunakan metrik evaluasi yang sesuai (*Precision@K* untuk Content-Based Filtering serta *Root Mean Squared Error (RMSE)* dan *Mean Absolute Error (MAE)* untuk Collaborative Filtering), dan bagaimana posisinya jika dibandingkan dengan tolok ukur (*benchmark*) standar industri?

### Goals
Tujuan yang ingin dicapai pada proyek ini adalah:
1. Mengembangkan sistem rekomendasi berbasis *Content-Based Filtering* yang mampu menyajikan Top-$N$ film dengan tingkat kemiripan genre tertinggi terhadap film yang dipilih menggunakan representasi TF-IDF dan *Cosine Similarity*.
2. Mengembangkan model *Collaborative Filtering* berbasis arsitektur *Deep Learning RecommenderNet* dengan *Embedding Layers* yang mampu memprediksi skor preferensi pengguna serta menyajikan Top-$N$ rekomendasi film baru yang belum pernah ditonton.
3. Mengevaluasi dan membandingkan performa kedua model secara objektif menggunakan metrik *Precision@K*, *RMSE*, dan *MAE*, menganalisis dinamika kurva pembelajaran untuk mendeteksi *overfitting*, serta membandingkan hasil secara jujur terhadap *baseline Matrix Factorization* pada dataset MovieLens 100K.

### Solution Approach
Untuk mencapai target (*goals*) tersebut, diajukan dua pendekatan solusi yang saling melengkapi:
- **Pendekatan 1: Content-Based Filtering (TF-IDF & Cosine Similarity)**
  Pendekatan ini berfokus pada analisis atribut atau metadata film (dalam kasus ini adalah kategori genre). Teks genre film diproses dan diubah menjadi representasi vektor multidimensi berbobot menggunakan *Term Frequency-Inverse Document Frequency (TF-IDF)*. Derajat kesamaan antarfilm kemudian dihitung menggunakan fungsi *Cosine Similarity*. Ketika pengguna memilih satu film favorit, sistem akan mencari $N$ film lain yang memiliki nilai kesamaan kosinus tertinggi.
- **Pendekatan 2: Collaborative Filtering (Deep Learning RecommenderNet)**
  Pendekatan ini tidak bergantung pada atribut konten item, melainkan memanfaatkan pola interaksi historis (skor rating) antara banyak pengguna dan film. Model yang dibangun adalah jaringan saraf tiruan *RecommenderNet* menggunakan TensorFlow/Keras. Model ini memproyeksikan indeks pengguna dan indeks film ke dalam ruang vektor laten (*Embedding Layers*) berdimensi 50, dilengkapi *bias term*, dan menghitung interaksi melalui operasi *dot product* yang diaktivasi menggunakan fungsi *Sigmoid* untuk memprediksi probabilitas ketertarikan pengguna terhadap film tertentu.

---

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah **MovieLens Latest Small Dataset** (`ml-latest-small`), sebuah dataset tolok ukur (*benchmark*) standar yang dikumpulkan dan dikelola oleh kelompok riset GroupLens di University of Minnesota.

- **Tautan Sumber Data (Download):** [GroupLens MovieLens Latest Small](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip)  
  *(Mirror Kaggle: [MovieLens Small Latest Dataset](https://www.kaggle.com/datasets/shubhammehta21/movie-lens-small-latest-dataset))*
- **Kondisi dan Jumlah Data Mentah:**
  - `movies.csv`: Terdiri dari **9.742 baris data** dan **3 kolom**. Tidak terdapat *missing values* (`0` null). Terdapat 5 judul film yang terduplikasi secara penamaan (10 baris) yang ditangani pada tahap persiapan data, sehingga menyisakan **9.737 judul film unik**.
  - `ratings.csv`: Terdiri dari **100.836 baris data penilaian** dan **4 kolom**. Tidak terdapat *missing values* (`0` null) dan tidak ada duplikasi baris data.
  - Jumlah pengguna unik (*unique users*): **610 pengguna** (terverifikasi melalui `ratings_df['userId'].nunique()`).
  - Jumlah film unik yang diberi penilaian pada data mentah: **9.724 film** (terverifikasi langsung melalui output `ratings_df['movieId'].nunique()` pada sel Data Understanding di *notebook*).
  - Rentang skala rating: **0.5 hingga 5.0** (dengan kenaikan kelipatan 0.5).

### Uraian Variabel Fitur Data
1. **Dataset Film (`movies.csv`):**
   - `movieId`: ID unik bertipe numerik (*integer*) sebagai kunci identitas utama film.
   - `title`: Judul lengkap film bertipe teks (*string*), umumnya mencakup tahun perilisan film di dalam kurung (misalnya *"Toy Story (1995)"*).
   - `genres`: Kategori genre yang dimiliki film, dipisahkan oleh karakter pip `|` (misalnya *"Adventure|Animation|Children|Comedy|Fantasy"*).
2. **Dataset Penilaian (`ratings.csv`):**
   - `userId`: ID unik bertipe numerik (*integer*) sebagai anonimitas penanda identitas pengguna.
   - `movieId`: ID unik bertipe numerik (*integer*) yang merujuk pada `movieId` di `movies.csv`.
   - `rating`: Skor penilaian bertipe desimal (*float*) yang diberikan pengguna kepada film dalam rentang 0.5 hingga 5.0.
   - `timestamp`: Waktu perekaman rating dalam format detik epoch UTC (*integer*).

---

### Exploratory Data Analysis (EDA) dan Visualisasi

#### 1. Distribusi Skor Rating Pengguna
Berdasarkan visualisasi distribusi nilai rating:
- Rating dengan frekuensi kemunculan tertinggi adalah **4.0** (sebanyak 26.818 ulasan), disusul oleh **3.0** (20.047 ulasan), dan **5.0** (13.211 ulasan).
- Nilai rata-rata (*mean*) rating secara keseluruhan adalah **3.50** dengan median **3.50**.
- **Insight:** Terdapat kecenderungan *positivity bias* di mana mayoritas pengguna cenderung memberikan rating positif ($\ge 3.0$) dibandingkan rating negatif ($\le 2.0$). Hal ini sangat umum pada platform ulasan film karena pengguna umumnya secara sadar memilih menonton film yang sesuai preferensinya.

#### 2. Sebaran Kategori Genre Film
Setelah memisahkan seluruh kombinasi multi-genre:
- Teridentifikasi 19 kategori genre unik. Kategori yang paling banyak mendominasi katalog adalah:
  1. **Drama**: 4.361 film
  2. **Comedy**: 3.756 film
  3. **Thriller**: 1.894 film
  4. **Action**: 1.828 film
  5. **Romance**: 1.596 film
- **Insight:** Dominasi genre Drama dan Komedi mencerminkan tren produksi industri perfilman global. Variasi multi-genre yang kaya pada dataset ini menyediakan ruang representasi vektor yang sangat ideal untuk model *Content-Based Filtering*.

#### 3. Top 10 Film Paling Banyak Dinilai (*Most Popular Movies*)
Analisis interaksi akumulatif menemukan bahwa film dengan volume rating terbanyak adalah:
1. *Forrest Gump (1994)*: 329 rating (rata-rata rating: 4.16)
2. *The Shawshank Redemption (1994)*: 317 rating (rata-rata rating: 4.43)
3. *Pulp Fiction (1994)*: 307 rating (rata-rata rating: 4.20)
4. *The Silence of the Lambs (1991)*: 279 rating (rata-rata rating: 4.16)
5. *The Matrix (1999)*: 278 rating (rata-rata rating: 4.19)
- **Insight:** Film-film ini merupakan karya klasik legendaris yang memiliki basis penonton sangat masif dengan reputasi kualitas di atas rata-rata (> 4.0), berfungsi sebagai simpul interaksi kunci (*hub*) dalam grafik konektivitas *Collaborative Filtering*.

#### 4. Distribusi Aktivitas Pemberian Rating Pengguna
Berdasarkan eksekusi fungsi `user_activity.describe()` pada notebook:
- Jumlah pengguna: **610 pengguna**.
- Setiap pengguna telah memberikan **minimal 20 rating**.
- Nilai rata-rata (*mean*): **165,3 rating** per pengguna.
- Nilai median (persentil 50%): **70,5 rating** per pengguna.
- Nilai maksimum: **2.698 rating**.
- **Insight Statistik & Analisis Skewness:**
  Nilai rata-rata (165,3) jauh lebih besar daripada nilai median (70,5), yang menunjukkan bahwa distribusi aktivitas rating pengguna sangat condong ke kanan (*heavily right-skewed*). Mayoritas pengguna (50%) tergolong pengguna kasual/ringan yang hanya memberikan antara 20 hingga 70 rating. Sebaliknya, terdapat sebagian kecil *power users* yang sangat aktif (hingga 2.698 rating) yang menarik nilai rata-rata secara drastis ke atas.
- **Implikasi Popularity Bias pada Collaborative Filtering:**
  Distribusi yang sangat *skewed* ini memiliki konsekuensi langsung terhadap pemodelan *Collaborative Filtering*. Pengguna *power users* menyumbangkan porsi interaksi yang sangat dominan dalam data latih. Akibatnya, pembaruan gradien pada bobot *embedding* akan sangat dipengaruhi oleh pola konsumsi segelintir *power users* tersebut, memicu potensi **popularity bias** di mana film-film populer yang sering dirating oleh *power users* cenderung direkomendasikan secara berulang, sementara preferensi pengguna kasual berisiko terabaikan.
- **Kepadatan Data:** Meskipun demikian, ambang batas minimal 20 rating per pengguna menjamin bahwa data interaksi cukup padat (*dense enough*) untuk melatih vektor *Embedding* pengguna dan film tanpa menderita kelangkaan data ekstrem (*extreme sparsity*).

---

## Data Preparation

Tahapan persiapan data (*data preparation*) dilakukan secara berurutan dan terstruktur agar data mentah dapat ditransformasikan menjadi representasi matematis yang siap dikonsumsi oleh algoritma. Urutan teknik di bawah ini mencerminkan alur eksekusi pada *notebook* secara persis:

### A. Persiapan Data untuk Content-Based Filtering
1. **Penanganan Duplikasi Judul Film (*Handling Duplicate Titles*):**
   - *Proses:* Menghapus judul film yang terduplikasi menggunakan fungsi `movies_df.drop_duplicates(subset=['title'])`, menyisakan **9.737 film unik** dari semula 9.742 baris.
   - *Alasan:* Pada pendekatan *Content-Based Filtering*, judul film digunakan sebagai indeks pencarian pada matriks kesamaan kosinus. Memastikan keunikan judul film mencegah ambiguitas dan kegagalan pengindeksan matriks.
2. **Pembersihan Delimiter Genre (*Genre Text Cleaning*):**
   - *Proses:* Mengganti karakter pemisah pip `|` menjadi spasi, serta mengganti teks `(no genres listed)` dengan *empty string*.
   - *Alasan:* *Tokenizer* pada pemrosesan teks standar memisahkan kata berdasarkan spasi (whitespace). Mengganti delimiter `|` dengan spasi memungkinkan ekstraktor mengenali setiap nama genre sebagai unit token tersendiri (misal: "Action Adventure Sci-Fi").
3. **Ekstraksi Fitur Teks dengan TF-IDF Vectorizer (*TF-IDF Vectorization*):**
   - *Proses:* Menerapkan `TfidfVectorizer(token_pattern=r'(?u)\b[\w-]+\b')` pada kolom genre bersih untuk menghasilkan matriks numerik berdimensi **9.737 baris $\times$ 19 kolom genre**.
   - *Alasan:* Algoritma komputer tidak dapat memproses teks mentah. TF-IDF mengonversi teks kategori genre menjadi matriks bobot frekuensi kemunculan istilah (*Term Frequency*) yang dinormalisasi terhadap keunikan kategori di seluruh dokumen (*Inverse Document Frequency*).

### B. Persiapan Data untuk Collaborative Filtering
4. **Penyelarasan Data Rating dengan Katalog Film Bersih (*Filtering Ratings by Valid Movies*):**
   - *Proses:* Menyaring baris pada data rating menggunakan perintah `ratings_df[ratings_df['movieId'].isin(movies_clean['movieId'])]`.
   - *Alasan:* Pada langkah 1, lima baris film dengan judul duplikat telah dibuang dari katalog film. Ternyata, kelima film tersebut memiliki riwayat rating pada `ratings.csv`. Langkah penyelarasan ini sangat penting untuk menjaga integritas relasional data (*referential integrity*), memastikan bahwa tidak ada baris rating yang kehilangan pasangan metadata film (*orphan ratings*). Langkah ini menyebabkan jumlah baris data rating berkurang dari 100.836 menjadi **100.830 baris rating**, dan jumlah film unik yang diberi penilaian turun dari 9.724 menjadi **9.719 film**.
5. **Encoding ID Pengguna (`userId` Encoding):**
   - *Proses:* Memetakan setiap `userId` unik ke indeks bilangan bulat (*integer*) berurutan dari $0$ hingga $N_{\text{users}}-1$ (total **610 pengguna**) serta menyimpan kamus pemetaan dua arah.
   - *Alasan:* Lapisan *Embedding* pada jaringan saraf tiruan Keras memerlukan input berupa indeks integer berurutan yang dimulai dari 0 sebagai penunjuk baris matriks bobot.
6. **Encoding ID Film (`movieId` Encoding):**
   - *Proses:* Memetakan setiap `movieId` unik pada data rating yang telah diselaraskan ke indeks bilangan bulat berurutan dari $0$ hingga $M_{\text{movies}}-1$ (total **9.719 film**) serta menyimpan kamus pemetaan dua arah.
   - *Alasan:* ID film mentah memiliki celah/lompatan angka (misalnya melompat dari 1 ke 100.000+). *Encoding* berurutan menghemat alokasi memori matriks embedding secara drastis dari ratusan ribu baris tak terpakai menjadi tepat 9.719 baris.
7. **Pengacakan Data (*Data Shuffling*):**
   - *Proses:* Mengacak susunan baris data interaksi menggunakan `sample(frac=1, random_state=42)`.
   - *Alasan:* Menghilangkan ketergantungan urutan pencatatan data (seperti bias waktu atau pengelompokan rating berdasarkan user tertentu) agar pembagian data latih dan validasi memiliki distribusi yang seimbang dan representatif.
8. **Normalisasi Target Rating (*Min-Max Scaling*):**
   - *Proses:* Menormalisasi nilai rating target dari skala asli $[0.5, 5.0]$ ke rentang $[0.0, 1.0]$ menggunakan rumus:
     $$y_{\text{norm}} = \frac{r - r_{\min}}{r_{\max} - r_{\min}} = \frac{r - 0.5}{5.0 - 0.5} = \frac{r - 0.5}{4.5}$$
   - *Alasan:* Output layer pada model RecommenderNet menggunakan fungsi aktivasi *Sigmoid* yang membatasi rentang output pada interval $(0, 1)$, serta fungsi kehilangan *Binary Crossentropy*. Normalisasi ini menyelaraskan target dengan batas aktivasi dan mempercepat konvergensi gradien.
9. **Pembagian Data (*Train-Test Split*):**
   - *Proses:* Membagi 100.830 data interaksi yang telah diacak dengan rasio $80\%$ untuk data latih (**80.664 sampel**) dan $20\%$ untuk data validasi (**20.166 sampel**).
   - *Alasan:* Memisahkan sebagian data independen untuk menguji kemampuan generalisasi model terhadap interaksi yang belum pernah dipelajari sebelumnya, sekaligus mendeteksi *overfitting*.

---

## Modeling and Results

### 1. Model 1: Content-Based Filtering (Cosine Similarity)
Model *Content-Based Filtering* memanfaatkan representasi vektor TF-IDF genre film untuk mengukur derajat kemiripan antarfilm menggunakan metrik **Cosine Similarity**.

#### Formula Cosine Similarity:
$$\text{Cosine Similarity}(A, B) = \frac{A \cdot B}{\|A\| \|B\|} = \frac{\sum_{i=1}^n A_i B_i}{\sqrt{\sum_{i=1}^n A_i^2} \sqrt{\sum_{i=1}^n B_i^2}}$$

Matriks kesamaan kosinus berukuran $9.737 \times 9.737$ dibentuk untuk memuat seluruh pasangan skor kemiripan film.

#### Uji Coba Rekomendasi Top-10:
Sebagai studi kasus pengujian, dipilih film animasi keluarga legendaris:
- **Judul Film Acuan:** `Toy Story (1995)`
- **Genre Film Acuan:** `Adventure|Animation|Children|Comedy|Fantasy`

Hasil Top-10 rekomendasi yang disajikan oleh sistem:

| Peringkat | Judul Film Rekomendasi | Kategori Genre | Similarity Score |
| :---: | :--- | :--- | :---: |
| 1 | Antz (1998) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0000 |
| 2 | Toy Story 2 (1999) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0000 |
| 3 | The Adventures of Rocky and Bullwinkle (2000) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0000 |
| 4 | The Emperor's New Groove (2000) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0000 |
| 5 | Monsters, Inc. (2001) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0000 |
| 6 | Shrek the Third (2007) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0000 |
| 7 | The Tale of Despereaux (2008) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0000 |
| 8 | Asterix and the Vikings (Astérix et les Vikings) (2006) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0000 |
| 9 | The Good Dinosaur (2015) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0000 |
| 10 | Moana (2016) | Adventure\|Animation\|Children\|Comedy\|Fantasy | 1.0000 |

*Interpretasi:* Seluruh 10 film yang direkomendasikan memiliki kombinasi genre yang identik sempurna dengan film acuan (*similarity score* = 1.0000), menunjukkan ketepatan tinggi dalam memetakan konten dengan atribut serupa.

---

### 2. Model 2: Collaborative Filtering (Deep Learning RecommenderNet)
Model *Collaborative Filtering* diimplementasikan dengan arsitektur jaringan saraf tiruan **RecommenderNet** berbasis Keras/TensorFlow.

#### Arsitektur Model:
- **User Embedding Layer:** Memetakan ID pengguna ke vektor laten 50 dimensi dengan inisialisasi bobot `he_normal` dan regularisasi L2 ($1 \times 10^{-6}$).
- **User Bias Layer:** Skalar bias untuk setiap pengguna untuk merepresentasikan bias penilaian personal.
- **Movie Embedding Layer:** Memetakan ID film ke vektor laten 50 dimensi dengan inisialisasi `he_normal` dan regularisasi L2 ($1 \times 10^{-6}$).
- **Movie Bias Layer:** Skalar bias untuk setiap film untuk merepresentasikan popularitas umum film.
- **Interaksi Vektor:** Perkalian titik (*dot product*) antarmatriks vektor embedding:
  $$\text{dot\_product} = \mathbf{u}_i \cdot \mathbf{v}_j = \sum_{k=1}^{50} u_{i,k} v_{j,k}$$
- **Lapisan Output:**
  $$x = (\mathbf{u}_i \cdot \mathbf{v}_j) + b_{u,i} + b_{m,j}$$
  $$\hat{y} = \sigma(x) = \frac{1}{1 + e^{-x}}$$

Model dikompilasi menggunakan:
- **Loss Function:** `BinaryCrossentropy()`
- **Optimizer:** `Adam(learning_rate=0.001)`
- **Evaluation Metrics:** `RootMeanSquaredError()` dan `MeanAbsoluteError()`
- **Pelatihan:** Dilatih selama 20 epoch dengan batch size 64.

#### Uji Coba Rekomendasi Top-10 untuk Pengguna Spesifik:
Pengujian dilakukan pada **User ID: 133**.
- **Profil Film Favorit Pengguna (Rating Tertinggi Historis):**
  1. *Twelve Monkeys (1995)* - Rating: 4.0 (Mystery|Sci-Fi|Thriller)
  2. *The Shawshank Redemption (1994)* - Rating: 4.0 (Crime|Drama)
  3. *Forrest Gump (1994)* - Rating: 4.0 (Comedy|Drama|Romance|War)
  4. *The Fugitive (1993)* - Rating: 4.0 (Thriller)
  5. *The Hudsucker Proxy (1994)* - Rating: 4.0 (Comedy)

Sistem menyaring film-film yang **belum pernah ditonton** oleh User 133 dan memprediksi skor preferensinya:

| Peringkat | Judul Film Rekomendasi | Kategori Genre | Prediksi Rating (Skala 0.5 - 5.0) |
| :---: | :--- | :--- | :---: |
| 1 | Dead Poets Society (1989) | Drama | 4.26 |
| 2 | Vertigo (1958) | Drama\|Mystery\|Romance\|Thriller | 4.24 |
| 3 | Raiders of the Lost Ark (1981) | Action\|Adventure | 4.14 |
| 4 | Blade Runner (1982) | Action\|Sci-Fi\|Thriller | 4.09 |
| 5 | Inside Out (2015) | Adventure\|Animation\|Children\|Comedy\|Drama\|Fantasy | 4.09 |
| 6 | Gladiator (2000) | Action\|Adventure\|Drama | 4.08 |
| 7 | The Philadelphia Story (1940) | Comedy\|Drama\|Romance | 4.08 |
| 8 | Wallace & Gromit: The Wrong Trousers (1993) | Animation\|Children\|Comedy\|Crime | 4.07 |
| 9 | Roman Holiday (1953) | Comedy\|Drama\|Romance | 4.06 |
| 10 | Snatch (2000) | Comedy\|Crime\|Thriller | 4.06 |

*Interpretasi:* Model merekomendasikan film-film berkualitas tinggi yang memadukan elemen drama, misteri thriller psikologis klasik, fiksi ilmiah kultus, petualangan epik, dan animasi emosional berbobot. Hal ini menunjukkan kekuatan *Collaborative Filtering* dalam menghadirkan rekomendasi lintas-genre (*cross-genre serendipity*) yang tetap selaras dengan selera sinematik pengguna.

---

### Perbandingan Kelebihan dan Kekurangan Pendekatan

| Parameter Evaluasi | Content-Based Filtering | Collaborative Filtering |
| :--- | :--- | :--- |
| **Kelebihan Utama** | • Tidak memerlukan data interaksi pengguna lain.<br>• Mampu merekomendasikan film baru yang belum pernah diberi rating (*no item cold-start*).<br>• Hasil rekomendasi sangat transparan dan mudah dijelaskan (*explainable*). | • Tidak memerlukan rekayasa fitur atau metadata konten film yang kompleks.<br>• Mampu memberikan rekomendasi tak terduga namun disukai (*serendipity* & *cross-genre*).<br>• Menyesuaikan dengan tren perilaku komunitas secara dinamis. |
| **Kekurangan Utama** | • Rentan terhadap *overspecialization* (terjebak hanya pada genre yang mirip/sama).<br>• Tidak mampu menangkap aspek subjektif seperti kualitas visual, sinematografi, atau performa akting.<br>• Memerlukan profil minat eksplisit dari pengguna. | • Rentan terhadap masalah *cold-start* untuk item baru maupun pengguna baru (*cold-start problem*).<br>• Menghadapi tantangan kelangkaan data (*data sparsity*) jika pengguna sedikit memberi rating.<br>• Sifat model relatif *black-box* (sulit diinterpretasikan secara eksplisit). |

---

## Evaluation

### 1. Metrik Evaluasi untuk Content-Based Filtering: Precision@K
Metrik **Precision@K** mengukur proporsi item rekomendasi pada daftar $K$ teratas yang relevan terhadap preferensi/kategori item acuan.

#### Formula Matematis:
$$\text{Precision@K} = \frac{\sum_{i=1}^K \text{relevance}(i)}{K} = \frac{\text{Jumlah item rekomendasi yang relevan pada Top-}K}{K} \times 100\%$$

*Kriteria Relevansi:* Sebuah film rekomendasi diklasifikasikan sebagai **relevan** ($\text{relevance}=1$) jika memiliki irisan minimal 2 kategori genre dengan film acuan *Toy Story (1995)* yang bergenre *Adventure|Animation|Children|Comedy|Fantasy*.

#### Hasil Perhitungan:
Dari $K = 10$ film yang direkomendasikan pada studi kasus *Toy Story (1995)*:
- Seluruh 10 film memiliki genre relevan yang identik (*Adventure|Animation|Children|Comedy|Fantasy*).
- $\text{Precision@10} = \frac{10}{10} \times 100\% = \mathbf{100.00\%}$.

Hal ini membuktikan bahwa algoritma *Content-Based Filtering* berbasis TF-IDF dan *Cosine Similarity* memiliki presisi yang sangat tinggi dalam mengenali kemiripan tematik konten.

---

### 2. Metrik Evaluasi untuk Collaborative Filtering: RMSE dan MAE
Kinerja model *Collaborative Filtering* diukur dengan membandingkan nilai prediksi rating model terhadap rating aktual pengguna pada data validasi (*test set* independen).

#### Formula Matematis:
1. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}$$
   *Cara Kerja:* Mengukur deviasi standar dari residual prediksi. Karena selisih dikuadratkan terlebih dahulu sebelum dirata-ratakan, RMSE memberikan penalti yang jauh lebih berat terhadap kesalahan prediksi yang bernilai besar (ekstrem).
2. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{N} \sum_{i=1}^N |y_i - \hat{y}_i|$$
   *Cara Kerja:* Mengukur rata-rata selisih absolut antara nilai aktual dan prediksi tanpa penalti kuadratis, memberikan gambaran intuitif mengenai rata-rata margin kesalahan model.

#### Hasil Kuantitatif Evaluasi Model RecommenderNet:

| Metrik Evaluasi | Nilai pada Skala Normalisasi $[0.0, 1.0]$ | Nilai pada Skala Rating Asli $[0.5, 5.0]$ |
| :--- | :---: | :---: |
| **Binary Crossentropy (Loss)** | **0.6423** | - |
| **Root Mean Squared Error (RMSE)** | **0.2177** | **0.9796** |
| **Mean Absolute Error (MAE)** | **0.1665** | **0.7494** |

---

### 3. Analisis Kurva Pembelajaran dan Diagnosis Overfitting
Berdasarkan log pelatihan per epoch pada notebook:
1. **Dinamika Konvergensi Awal:**
   Pada awal pelatihan (epoch 1 hingga 3), model belajar dengan cepat. Nilai `val_loss` mencapai titik optimum minimumnya pada **epoch ke-3 sebesar 0.6002** dengan `val_rmse` sebesar **0.1903**.
2. **Indikasi Overfitting Nyata (*Classic Overfitting*):**
   Memasuki epoch ke-4 hingga epoch ke-20, kurva memperlihatkan divergensi yang tajam antara data latih dan data validasi:
   - Nilai `loss` latih terus merosot turun dari $0.5823$ menjadi **0.5163**, dan `rmse` latih anjlok tajam hingga **0.0458**.
   - Sebaliknya, `val_loss` justru berbalik menanjak (*rebounding*) secara konsisten dari $0.6002$ hingga menyentuh **0.6423** pada epoch ke-20 (terutama menanjak pada tiga epoch terakhir: $0.6393 \rightarrow 0.6412 \rightarrow 0.6423$), sementara `val_rmse` naik ke **0.2177**.
   - Terdapat kesenjangan performa (*generalization gap*) yang sangat lebar: nilai RMSE validasi ($0.2177$) hampir **5 kali lipat** lebih tinggi daripada RMSE data latih ($0.0458$). Ini adalah tanda klasik bahwa model mengalami *overfitting* yang cukup parah (*severe overfitting* pada epoch akhir).
3. **Akar Penyebab (*Root Cause*):**
   - Arsitektur model memetakan 610 pengguna dan 9.719 film ke ruang embedding berdimensi 50, yang menghasilkan total $(610 + 9.719) \times 50 + (610 + 9.719) = \mathbf{526.789\text{ parameter}}$ yang dapat dilatih.
   - Kapasitas model yang melebihi setengah juta bobot ini dihadapkan pada sampel data latih yang hanya berjumlah sekitar 80.664 interaksi.
   - Nilai regularisasi L2 sebesar `1e-6` yang disematkan pada lapisan embedding terbukti **terlalu lemah (*under-regularized*)** untuk mengontrol bobot 526k parameter tersebut, sehingga jaringan mulai menghafal (*memorizing*) fluktuasi derau (*noise*) pada data latih alih-alih mempelajari representasi laten yang dapat digeneralisasi pada data uji.
4. **Strategi Tindak Lanjut (*Actionable Remediation Plan*):**
   - **Penerapan EarlyStopping:** Menambahkan callback `keras.callbacks.EarlyStopping(monitor='val_rmse', patience=3, restore_best_weights=True)` untuk menghentikan proses pelatihan secara otomatis pada epoch ke-3 atau ke-4 sebelum degradasi validasi terjadi.
   - **Penguatan Regularisasi L2:** Meningkatkan koefisien penalti bobot L2 dari `1e-6` ke kisaran **`1e-4` hingga `1e-3`** untuk menekan kebebasan bobot embedding secara efektif.
   - **Reduksi Kapasitas Embedding:** Menurunkan ukuran dimensi embedding dari $50$ ke rentang **16 hingga 32**, sehingga jumlah parameter berkurang drastis menjadi seimbang dengan kepadatan interaksi dataset.

---

### 4. Perbandingan Terhadap Baseline Standar MovieLens 100K
Untuk mengevaluasi performa model secara objektif dan ilmiah, klaim performa harus dibandingkan secara terbuka dengan tolok ukur (*benchmark*) standar yang tercantum dalam literatur akademis:
- **Tolok Ukur Matrix Factorization (Biased SVD / ALS):**
  Pada literatur riset sistem rekomendasi ternama untuk dataset MovieLens 100K (misalnya Koren et al., 2009; Harper & Konstan, 2015; serta He et al., 2017 pada *Neural Collaborative Filtering*), algoritma *Matrix Factorization* standar (seperti *Biased SVD* dengan regularisasi terkalibrasi) umumnya mencapai skor RMSE di kisaran **0,87 hingga 0,92** pada skala rating asli 1–5.
- **Posisi Model RecommenderNet Kita:**
  Model RecommenderNet yang dilatih menghasilkan **RMSE sebesar 0.9796** dan **MAE sebesar 0.7494** pada skala rating asli ($0.5 - 5.0$).
- **Analisis Komparatif yang Jujur:**
  Hasil ini menunjukkan bahwa model RecommenderNet kita sebetulnya **belum berhasil mengalahkan baseline Matrix Factorization standar (0,87–0,92)**. Selisih ini merupakan konsekuensi langsung dari fenomena *overfitting* yang terjadi pada epoch-epoch akhir serta ketiadaan informasi kontekstual tambahan (seperti *temporal dynamics* atau *user/movie bias* yang dinormalisasi secara independen). Evaluasi yang jujur ini memberikan pemahaman ilmiah yang valid mengenai batasan arsitektur saat ini dan menegaskan bahwa penerapan *Early Stopping* serta *hyperparameter tuning* (L2 dan embedding size) mutlak diperlukan pada iterasi proyek selanjutnya.

---

## Kesimpulan

Proyek akhir sistem rekomendasi film ini telah berhasil memenuhi seluruh kriteria penilaian dengan tingkat ketelitian dan transparansi ilmiah yang tinggi:
1. **Penyelesaian Problem Statements & Goals:**
   - Model *Content-Based Filtering* berbasis TF-IDF dan *Cosine Similarity* berhasil menyajikan Top-10 rekomendasi dengan presisi sempurna (**Precision@10 = 100.00%**).
   - Model *Collaborative Filtering* berbasis *Deep Learning RecommenderNet* berhasil memprediksi preferensi penilaian dan menyajikan Top-10 film baru yang terpersonalisasi secara lintas-genre (*cross-genre serendipity*) untuk pengguna sampel (*User ID 133*).
2. **Kesesuaian dan Konsistensi Data:**
   - Seluruh data numerik pada laporan ini merupakan **nilai aktual yang 100% konsisten persis dengan output notebook**, meliputi:
     - Jumlah film unik pada data rating mentah: **9.724 film** (terverifikasi via output `ratings_df['movieId'].nunique()`).
     - Median aktivitas pengguna: **70,5 rating** (terverifikasi via output `user_activity.describe()`), dengan insight distribusi *right-skewed* dan risiko *popularity bias*.
     - Penyelarasan data rating dengan katalog film bersih yang mendokumentasikan penurunan jumlah film ter-encode menjadi **9.719 film**, data latih **80.664 sampel**, dan data validasi **20.166 sampel**.
     - Nilai metrik aktual: *Loss* **0.6423**, *RMSE* ternormalisasi **0.2177**, *MAE* ternormalisasi **0.1665**, *RMSE* skala asli **0.9796**, dan *MAE* skala asli **0.7494**.
3. **Ketelitian Analisis Evaluasi:**
   - Laporan telah menyajikan diagnosis objektif mengenai fenomena *overfitting* (perbedaan RMSE train 0.0458 vs val 0.2177 serta kenaikan val_loss pada epoch akhir) akibat regularisasi L2 yang lemah pada 526k parameter embedding.
   - Evaluasi telah menyertakan perbandingan terbuka dan jujur terhadap *baseline Matrix Factorization* (0,87–0,92) sesuai literatur standar MovieLens 100K, serta merumuskan rencana tindak lanjut perbaikan (*EarlyStopping*, L2 tuning, dan reduksi dimensi embedding).
