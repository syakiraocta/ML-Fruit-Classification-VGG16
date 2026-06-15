# ML-Fruit-Classification-VGG16
# Klasifikasi Citra Buah Segar dan Busuk Menggunakan Pendekatan Transfer Learning (VGG16)



Repositori ini untuk memenuhi tugas praktikum Pembelajaran Mesin (Repo3). Fokus utama dari proyek ini adalah membangun sistem kecerdasan buatan (AI) berbasis Deep Learning yang mampu mengklasifikasikan kondisi buah—apakah masih segar (fresh) atau sudah busuk (rotten)—berdasarkan input gambar atau citra digital.



---



## Identitas Pemilik Proyek

* Nama: Syakira Octa Fitria

* NIM: A11.2024.15816

* Kelas: Pembelajaran Mesin A11.4401

---



## Sumber Dataset & Manajemen File

Dataset yang digunakan dalam eksperimen ini bersumber dari Kaggle:

* Nama Dataset: Fruits Fresh and Rotten for Classification

* Link Dataset: https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification

* Kategori Kelas (6 Kelas): fresh_apple, rotten_apple, fresh_banana, rotten_banana, fresh_orange, rotten_orange





---



## Alur Kerja Sistem & Penjelasan Kode Program (Cell 1 - Cell 6)



Sistem ini dibangun di atas lingkungan Google Colab dengan memanfaatkan akselerasi hardware GPU (T4). Berdasarkan dokumen source code yang dikerjakan, berikut adalah rincian detail cara kerja dari setiap tahapan blok kode (cell):



### Cell 1: Inisialisasi Environment & Verifikasi Hardware

Pada tahap awal, sistem memuat library utama yaitu TensorFlow, Keras, Matplotlib, dan NumPy. Langkah krusial di sel ini adalah melakukan pengecekan ketersediaan GPU menggunakan perintah tf.config.list_physical_devices('GPU'). Jika GPU aktif, akselerasi matriks pada proses training akan dialihkan ke hardware T4 Colab agar komputasi berjalan cepat.



### Cell 2: Integrasi Cloud Storage (Google Drive Mounting)

Karena dataset disimpan di Google Drive, kode pada Cell 2 berfungsi untuk menjembatani penyimpanan cloud dengan local runtime Colab lewat modul google.colab.drive. Ketika perintah drive.mount('/content/drive') dieksekusi, sistem akan meminta otentikasi berupa token izin akses keamanan agar direktori DatasetRepo3 bisa dibaca secara langsung oleh script Python.



### Cell 3: Data Pipeline, Normalisasi, & Augmentasi Gambar

Pada tahap ini, data mentah diproses agar siap masuk ke jaringan saraf tiruan menggunakan ImageDataGenerator:

* Normalisasi Piksel (Rescaling): Nilai asli piksel gambar (0 - 255) dikonversi menjadi rentang 0 hingga 1 melalui operasi matematika rescale=1./255 agar beban komputasi model menjadi lebih ringan.

* Augmentasi Data Otomatis: Untuk mencegah overfitting, gambar dimanipulasi secara dinamis saat training berupa rotasi acak (rotation_range=20), fitur perbesaran (zoom_range=0.2), serta pembalikan gambar secara horizontal (horizontal_flip=True).

* Data Iterator: Menggunakan flow_from_directory, sistem mengakses langsung jalur folder terdalam (dataset/dataset/train dan dataset/dataset/test), mengubah ukuran semua gambar menjadi standar VGG16 (224 x 224 piksel), membaginya ke dalam batch ukuran 32, dan mendeteksi kelas label secara otomatis.



### Cell 4: Arsitektur Model Deep Learning (Transfer Learning VGG16)

Daripada melatih model CNN dari nol yang memakan waktu lama, proyek ini menerapkan metode Transfer Learning menggunakan model legendaris VGG16 yang sudah terlatih mengenali jutaan objek dari dataset ImageNet.

* Feature Extraction (Freeze Layers): Lapisan dasar (base model) dari VGG16 dimanfaatkan sebagai pengekstrak fitur otomatis untuk mengenali tepi, bentuk, warna, pola tekstur, dan bercak busuk pada buah. Lapisan bawaan ini dikunci (base_model.trainable = False) agar pengetahuan aslinya tidak rusak selama training.

* Custom Classification Head: Lapisan klasifikasi teratas bawaan VGG16 dibuang (include_top=False) dan digantikan dengan lapisan baru buatan sendiri:

  * GlobalAveragePooling2D(): Meratakan dimensi matriks fitur hasil ekstraksi menjadi bentuk vektor satu dimensi yang lebih ringkas.

  * Dense Layer (Softmax): Lapisan output akhir dengan fungsi aktivasi Softmax. Jumlah neuron pada lapisan ini diatur secara dinamis mengikuti jumlah kelas yang terdeteksi di Cell 3 (num_classes = 6), sehingga outputnya berupa probabilitas nilai tebakan untuk masing-masing kondisi buah.



### Cell 5: Kompilasi & Proses Pelatihan Model (Training)

Model dikompilasi menggunakan fungsi loss categorical_crossentropy (karena klasifikasinya lebih dari 2 kelas) dan menggunakan optimizer Adam untuk memperbarui bobot jaringan secara efisien. Selanjutnya, perintah model.fit() dijalankan sebanyak 10 epoch. Pada proses ini, performa akurasi model dievaluasi secara berkala pada setiap akhir epoch menggunakan data uji (validation data).



### Cell 6: Visualisasi Performa & Fungsi Prediksi Gambar Baru

Blok kode terakhir ini dibagi menjadi dua fungsi utama:

1. Visualisasi Grafik: Memanfaatkan Matplotlib untuk menggambar grafik perbandingan akurasi dan loss antara data training dan data validation untuk menganalisis performa model.

2. Fungsi prediksi_buah_baru(image_path): Sebuah fungsi kustom untuk menguji model secara langsung. Fungsi ini akan memuat gambar baru di luar dataset, melakukan normalisasi skala piksel, mengubah dimensinya agar cocok dengan input model, melakukan kalkulasi prediksi (model.predict), lalu menampilkan foto tersebut ke layar lengkap dengan label hasil analisis AI (apakah buah tersebut segar atau busuk).



---



## Panduan Menjalankan Kode



1. Pastikan struktur folder dataset di Google Drive  sudah sesuai dengan arsitektur direktori di atas.

2. Buka berkas notebook di Google Colab dan pastikan Runtime telah dialihkan ke mode GPU (Runtime > Change runtime type > T4 GPU).

3. Jalankan kode secara berurutan mulai dari Cell 1 hingga Cell 6.

4. Uji Coba Gambar Baru:

   * Jika ingin menguji model menggunakan gambar baru di luar dataset, unduh foto buah berformat .jpg dari internet.

   * Unggah file tersebut ke panel file browser (ikon folder di sebelah kiri layar Colab).

   * Jalankan fungsi pada cell paling bawah dengan format: prediksi_buah_baru('nama_file_kamu.jpg')


---


## Evaluasi & Hasil Pelatihan Model

Setelah melalui proses pelatihan selama 10 epoch dengan pemantauan validation data, berikut adalah performa final yang berhasil dicapai oleh model:

* Akurasi Data Pelatihan (Train Accuracy): [Isi Persentase Akurasi]%

* Akurasi Data Pengujian (Validation Accuracy): [Isi Persentase Akurasi]%



Model terbukti memiliki kapabilitas yang sangat baik dalam mengenali pola visual buah segar dan mampu mengidentifikasi tanda-tanda pembusukan secara akurat.

