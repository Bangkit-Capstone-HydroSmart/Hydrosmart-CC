# plant_rec_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE

class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('accuracy') is not None and logs.get('val_accuracy') is not None:
            if logs['accuracy'] > 0.90 and logs['val_accuracy'] > 0.90:
                print("\nAkurasi pelatihan > 0.90 dan akurasi validasi > 0.90. Menghentikan pelatihan.")
                self.model.stop_training = True

# Fungsi untuk membaca data dari file Excel
def read_excel_data(excel_path):
    df = pd.read_excel(excel_path)
    le = LabelEncoder()
    df['tanaman_encoded'] = le.fit_transform(df['Nama'])
    return df

# Fungsi untuk pra-pemrosesan data
def preprocess_data_with_smote(df):
    # Memilih fitur-fitur yang digunakan untuk pemrosesan
    X = df[['Luas', 'Suhu', 'PH', 'Kelembapan', 'Penyinaran']]
    # Memilih variabel target
    y = df['tanaman_encoded']
    # Membagi data menjadi data pelatihan dan pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Normalisasi data menggunakan StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Menerapkan SMOTE pada data pelatihan
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test

# Fungsi untuk mendapatkan rekomendasi tanaman berdasarkan kesamaan kosinus
def get_rekomendasi_tanaman(cosine_similarities, df, suhu_pengguna, luas_lahan_pengguna, ph_pengguna, kelembapan_pengguna, penyinaran_pengguna, num_rekomendasi=5):
    # Mendapatkan indeks tanaman yang serupa berdasarkan urutan kesamaan kosinus
    idx_tanaman_serupa = df.index[cosine_similarities[:, 0].argsort()[-num_rekomendasi:][::-1]]
    # Mengambil nama tanaman dari dataframe
    rekomendasi = df.loc[idx_tanaman_serupa, 'Nama'].tolist()
    # Mengembalikan daftar rekomendasi tanaman
    return list(set(rekomendasi))[:num_rekomendasi]

# Ganti path sesuai dengan lokasi file Excel
excel_path = r'https://storage.googleapis.com/dataset-hydrosmart/DATASET-HIDROPONIK.xlsx'

# Baca dataset dan pra-pemrosesan data dengan SMOTE
df_tanaman = read_excel_data(excel_path)
X_train_resampled, X_test_scaled, y_train_resampled, y_test = preprocess_data_with_smote(df_tanaman)

# Arsitektur model
model_tanaman = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(5,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(df_tanaman['Nama'].unique()), activation='softmax')
])

model_tanaman.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Menerapkan validasi silang
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for train_index, test_index in skf.split(X_train_resampled, y_train_resampled):
    X_train, X_test = X_train_resampled[train_index], X_train_resampled[test_index]
    y_train, y_test = y_train_resampled[train_index], y_train_resampled[test_index]

    # Fit model
    model_tanaman.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test), callbacks=[MyCallback()])

# Save the model
model_tanaman.save('model90.h5')
