"""
Proyek Akhir: Membuat Model Sistem Rekomendasi Film
Domain: Media Digital & Hiburan (Movie Recommendation System)
Pengembang: Zaki Abdussalam
Dataset: MovieLens (ml-latest-small) dari GroupLens Research
Pendekatan: 
    1. Content-Based Filtering (TF-IDF Vectorizer & Cosine Similarity)
    2. Collaborative Filtering (Deep Learning RecommenderNet dengan TensorFlow/Keras)
"""

# Import TensorFlow & Keras terlebih dahulu untuk memastikan inisialisasi DLL runtime optimal di Windows
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import zipfile
import urllib.request
import ssl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import Scikit-Learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi reproduktifitas dan visualisasi
np.random.seed(42)
tf.random.set_seed(42)
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

def main():
    print("=" * 70)
    print("PROYEK AKHIR: MEMBUAT MODEL SISTEM REKOMENDASI FILM")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # 1. SETUP DIREKTORI & DATA
    # -------------------------------------------------------------------------
    data_dir = "ml-latest-small"
    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")

    # Download otomatis jika belum ada
    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        print("Dataset MovieLens tidak ditemukan. Mengunduh dari GroupLens...")
        zip_path = "ml-latest-small.zip"
        url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
        
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, context=ctx) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Dataset berhasil diunduh dan diekstrak.")

    # -------------------------------------------------------------------------
    # 2. DATA UNDERSTANDING
    # -------------------------------------------------------------------------
    print("\n[1/5] Memuat dan memahami struktur data...")
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)

    print(f"Data movies: {movies_df.shape[0]:,} baris, {movies_df.shape[1]} kolom")
    print(f"Data ratings: {ratings_df.shape[0]:,} baris, {ratings_df.shape[1]} kolom")
    print(f"Jumlah pengguna unik: {ratings_df['userId'].nunique():,} pengguna")
    print(f"Jumlah judul film unik: {movies_df['title'].nunique():,} judul")
    print(f"Jumlah film unik pada data rating: {ratings_df['movieId'].nunique():,} film")

    user_activity = ratings_df.groupby('userId').size()
    print("\nStatistik Deskriptif Aktivitas Rating Pengguna:")
    print(user_activity.describe())

    # -------------------------------------------------------------------------
    # 3. DATA PREPARATION
    # -------------------------------------------------------------------------
    print("\n[2/5] Menjalankan Data Preparation...")
    # A. Pembersihan Data untuk Content-Based Filtering
    # 1. Penanganan duplikasi judul film
    movies_clean = movies_df.drop_duplicates(subset=['title']).reset_index(drop=True)
    print(f"Jumlah film setelah penghapusan duplikat: {len(movies_clean):,} film")
    
    # 2. Pembersihan delimiter genre
    movies_clean['genres_clean'] = movies_clean['genres'].str.replace('|', ' ', regex=False)
    movies_clean['genres_clean'] = movies_clean['genres_clean'].replace('(no genres listed)', '')
    
    # 3. Ekstraksi fitur teks menggunakan TF-IDF Vectorizer
    tf_idf = TfidfVectorizer(token_pattern=r'(?u)\b[\w-]+\b')
    tfidf_matrix = tf_idf.fit_transform(movies_clean['genres_clean'])
    print(f"Matriks TF-IDF terbentuk dengan dimensi: {tfidf_matrix.shape}")

    # B. Persiapan Data untuk Collaborative Filtering
    # 4. Penyelarasan data rating dengan katalog film bersih
    ratings_clean = ratings_df[ratings_df['movieId'].isin(movies_clean['movieId'])].copy().reset_index(drop=True)
    print(f"Jumlah baris rating setelah penyelarasan: {len(ratings_clean):,} baris")
    print(f"Jumlah film unik pada rating setelah penyelarasan: {ratings_clean['movieId'].nunique():,} film (turun dari {ratings_df['movieId'].nunique():,})")
    
    # 5. Encoding userId ke indeks integer
    user_ids = ratings_clean['userId'].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    user_encoded2user = {i: x for i, x in enumerate(user_ids)}

    # 6. Encoding movieId ke indeks integer
    movie_ids = ratings_clean['movieId'].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}

    ratings_clean['user'] = ratings_clean['userId'].map(user2user_encoded)
    ratings_clean['movie'] = ratings_clean['movieId'].map(movie2movie_encoded)

    num_users = len(user2user_encoded)
    num_movies = len(movie_encoded2movie)
    min_rating = float(ratings_clean['rating'].min())
    max_rating = float(ratings_clean['rating'].max())

    print(f"\nTotal pengguna ter-encode: {num_users}")
    print(f"Total film ter-encode: {num_movies}")

    # 7. Pengacakan data (shuffling)
    ratings_shuffled = ratings_clean.sample(frac=1, random_state=42).reset_index(drop=True)

    # 8. Ekstraksi fitur dan normalisasi target [0, 1]
    x = ratings_shuffled[['user', 'movie']].values
    y = ratings_shuffled['rating'].apply(lambda r: (r - min_rating) / (max_rating - min_rating)).values

    # 9. Train-test split (80:20)
    train_indices = int(0.8 * len(ratings_shuffled))
    x_train, x_val = x[:train_indices], x[train_indices:]
    y_train, y_val = y[:train_indices], y[train_indices:]

    print(f"Data latih: {len(x_train):,} sampel (80%) | Data validasi: {len(x_val):,} sampel (20%)")

    # -------------------------------------------------------------------------
    # 4. MODEL 1: CONTENT-BASED FILTERING (COSINE SIMILARITY)
    # -------------------------------------------------------------------------
    print("\n[3/5] Membangun Model 1: Content-Based Filtering...")
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cosine_sim_df = pd.DataFrame(cosine_sim, index=movies_clean['title'], columns=movies_clean['title'])

    def get_movie_recommendations(movie_title, similarity_data=cosine_sim_df, items=movies_clean, k=10):
        if movie_title not in similarity_data.columns:
            raise ValueError(f"Film '{movie_title}' tidak ditemukan dalam katalog.")
        sim_scores = similarity_data[movie_title]
        top_indices = sim_scores.sort_values(ascending=False).drop(labels=[movie_title])
        top_k_titles = top_indices.head(k).index.tolist()
        result = items[items['title'].isin(top_k_titles)][['title', 'genres']].copy()
        result['similarity_score'] = result['title'].map(top_indices)
        return result.sort_values('similarity_score', ascending=False).reset_index(drop=True)

    sample_movie = "Toy Story (1995)"
    print(f"\nMenguji rekomendasi untuk film acuan: '{sample_movie}'")
    cb_recommendations = get_movie_recommendations(sample_movie, k=10)
    print("\nHasil Top-10 Content-Based Filtering:")
    print(cb_recommendations.to_string(index=False))

    # Evaluasi Precision@10
    target_genre_set = set(movies_clean[movies_clean['title'] == sample_movie]['genres'].values[0].split('|'))
    cb_recommendations['is_relevant'] = cb_recommendations['genres'].apply(
        lambda g: len(set(g.split('|')).intersection(target_genre_set)) >= 2
    )
    precision_at_10 = (cb_recommendations['is_relevant'].sum() / len(cb_recommendations)) * 100
    print(f"\nPrecision@10 Content-Based Filtering: {precision_at_10:.2f}% ({cb_recommendations['is_relevant'].sum()}/10 film relevan)")

    # -------------------------------------------------------------------------
    # 5. MODEL 2: COLLABORATIVE FILTERING (DEEP LEARNING RECOMMENDERNET)
    # -------------------------------------------------------------------------
    print("\n[4/5] Membangun dan Melatih Model 2: Collaborative Filtering (RecommenderNet)...")
    
    class RecommenderNet(keras.Model):
        def __init__(self, num_users, num_movies, embedding_size=50, **kwargs):
            super(RecommenderNet, self).__init__(**kwargs)
            self.user_embedding = layers.Embedding(
                num_users, embedding_size,
                embeddings_initializer='he_normal',
                embeddings_regularizer=keras.regularizers.l2(1e-6)
            )
            self.user_bias = layers.Embedding(num_users, 1)
            self.movie_embedding = layers.Embedding(
                num_movies, embedding_size,
                embeddings_initializer='he_normal',
                embeddings_regularizer=keras.regularizers.l2(1e-6)
            )
            self.movie_bias = layers.Embedding(num_movies, 1)

        def call(self, inputs):
            user_vec = self.user_embedding(inputs[:, 0])
            user_b = self.user_bias(inputs[:, 0])
            movie_vec = self.movie_embedding(inputs[:, 1])
            movie_b = self.movie_bias(inputs[:, 1])
            dot_product = tf.reduce_sum(user_vec * movie_vec, axis=1, keepdims=True)
            return tf.nn.sigmoid(dot_product + user_b + movie_b)

    cf_model = RecommenderNet(num_users, num_movies, embedding_size=50)
    cf_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        metrics=[
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            tf.keras.metrics.MeanAbsoluteError(name='mae')
        ]
    )

    print("Memulai pelatihan model RecommenderNet (20 Epochs)...")
    history = cf_model.fit(
        x=x_train,
        y=y_train,
        batch_size=64,
        epochs=20,
        validation_data=(x_val, y_val),
        verbose=1
    )

    # Evaluasi metrik kuantitatif
    eval_results = cf_model.evaluate(x_val, y_val, verbose=0)
    val_loss = eval_results[0]
    val_rmse_norm = eval_results[1]
    val_mae_norm = eval_results[2]
    
    rating_range = max_rating - min_rating
    val_rmse_orig = val_rmse_norm * rating_range
    val_mae_orig = val_mae_norm * rating_range

    print("\n" + "=" * 50)
    print("HASIL EVALUASI COLLABORATIVE FILTERING:")
    print(f"- Loss (Binary Crossentropy) : {val_loss:.4f}")
    print(f"- RMSE (Ternormalisasi [0, 1]): {val_rmse_norm:.4f}")
    print(f"- MAE  (Ternormalisasi [0, 1]): {val_mae_norm:.4f}")
    print(f"- RMSE (Skala Asli [0.5 - 5.0]): {val_rmse_orig:.4f}")
    print(f"- MAE  (Skala Asli [0.5 - 5.0]): {val_mae_orig:.4f}")
    print("=" * 50)

    # -------------------------------------------------------------------------
    # 6. INFERENSI REKOMENDASI UNTUK PENGGUNA SPESIFIK
    # -------------------------------------------------------------------------
    sample_user_id = 133
    print(f"\n[5/5] Menghasilkan Top-10 Rekomendasi Collaborative Filtering untuk User {sample_user_id}...")
    
    movies_watched_by_user = ratings_clean[ratings_clean['userId'] == sample_user_id]
    top_user_movies = movies_watched_by_user.sort_values(by='rating', ascending=False).head(5)
    print("\nFilm Favorit Pengguna:")
    for _, row in top_user_movies.iterrows():
        m_title = movies_clean[movies_clean['movieId'] == row['movieId']]['title'].values[0]
        print(f"- {m_title} (Rating: {row['rating']})")

    movies_not_watched = movies_clean[~movies_clean['movieId'].isin(movies_watched_by_user['movieId'].values)]['movieId']
    movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))
    movies_not_watched_encoded = [[movie2movie_encoded.get(x)] for x in movies_not_watched]

    user_encoder = user2user_encoded.get(sample_user_id)
    user_movie_array = np.hstack(([[user_encoder]] * len(movies_not_watched), movies_not_watched_encoded))

    predicted_ratings_norm = cf_model.predict(user_movie_array, verbose=0).flatten()
    top_ratings_indices = predicted_ratings_norm.argsort()[-10:][::-1]

    recommended_movie_ids = [movie_encoded2movie.get(movies_not_watched_encoded[x][0]) for x in top_ratings_indices]
    predicted_scores = [min_rating + (predicted_ratings_norm[x] * rating_range) for x in top_ratings_indices]

    cf_recommendations = []
    for idx, (m_id, score) in enumerate(zip(recommended_movie_ids, predicted_scores), 1):
        movie_row = movies_clean[movies_clean['movieId'] == m_id].iloc[0]
        cf_recommendations.append({
            'Peringkat': idx,
            'Judul Film': movie_row['title'],
            'Kategori Genre': movie_row['genres'],
            'Prediksi Rating': round(score, 2)
        })

    cf_rec_df = pd.DataFrame(cf_recommendations)
    print(f"\nTop-10 Rekomendasi Film untuk User {sample_user_id}:")
    print(cf_rec_df.to_string(index=False))
    print("\nProses eksekusi proyek selesai dengan sukses!")

if __name__ == "__main__":
    main()
