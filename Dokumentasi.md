# Dokumentasi Teknis: Sistem Virtual Hair Try-On

## Daftar Isi
1. [Pipeline Computer Vision](#pipeline-computer-vision)
2. [Sistem Overlay Rambut](#sistem-overlay-rambut)
3. [Operasi Matriks](#operasi-matriks)
4. [Pertimbangan Performa](#pertimbangan-performa)

## Pipeline Computer Vision

### 1. Deteksi Wajah Dua Tahap

#### Tahap Pertama: Haar Cascade
```python
faces = self.face_cascade.detectMultiScale(
    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
```
- Melakukan deteksi wajah awal dengan cepat
- Menghasilkan area potensial wajah dalam bentuk (x, y, width, height)
- Menggunakan algoritma Viola-Jones dengan fitur Haar-like
- Dioptimalkan untuk deteksi wajah dari depan

#### Tahap Kedua: SVM + ORB
```python
def verify_face(self, face_img: np.ndarray) -> bool:
    face_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_resized = cv2.resize(face_gray, (64, 64))
    features = self.feature_extractor.extract_bovw_features([face_resized])
    score = self.svm.decision_function(features)[0]
    return score > 0
```

### 2. Ekstraksi Fitur (ORB)
- **Konfigurasi ORB**:
  ```python
  self.orb = cv2.ORB_create(
      nfeatures=2000,
      scaleFactor=1.2,
      nlevels=8,
      edgeThreshold=31,
      patchSize=31
  )
  ```
- Menggabungkan detektor keypoint FAST dan descriptor BRIEF
- Tahan terhadap rotasi dan perubahan skala
- Lebih efisien dibandingkan SIFT/SURF

### 3. Pemrosesan Fitur
1. **Bag of Visual Words (BoVW)**:
   - Mengkonversi fitur ORB menjadi vektor dengan panjang tetap
   - Menggunakan clustering k-means untuk membuat codebook
   - Representasi histogram dari visual words

2. **Penskalaan Fitur**:
   - Normalisasi vektor BoVW menggunakan StandardScaler
   - Memastikan performa SVM optimal
   - Menjaga rentang fitur tetap konsisten

## Sistem Overlay Rambut

### 1. Perhitungan Posisi
```python
# Perhitungan skala
scale_factor = 2.0  # Rambut akan 2x lebar wajah
scale = (w * scale_factor) / hair.shape[1]
new_h = int(hair.shape[0] * scale)
new_w = int(hair.shape[1] * scale)

# Perhitungan posisi
vertical_offset = int(new_h * 0.2)
hair_y = max(0, y - new_h//2 + vertical_offset)
hair_x = max(0, x - (new_w - w)//2)
```

### 2. Proses Alpha Blending
```python
# Membuat mask alpha
alpha = hair_resized[:, :, 3] / 255.0
alpha = np.expand_dims(alpha, axis=-1)

# Pemilihan region
hair_roi = hair_resized[:roi_h, :roi_w, :3]
alpha_roi = alpha[:roi_h, :roi_w]
frame_roi = frame[hair_y:hair_y+roi_h, hair_x:hair_x+roi_w]

# Operasi blending
frame[hair_y:hair_y+roi_h, hair_x:hair_x+roi_w] = \
    frame_roi * (1 - alpha_roi) + hair_roi * alpha_roi
```

## Operasi Matriks

### 1. Representasi Gambar
- Matriks Frame: [height, width, 3] (BGR)
- Gambar Rambut: [height, width, 4] (BGRA)
- Channel Alpha: [height, width, 1]

### 2. Matematika Blending
Untuk setiap pixel:
```
Hasil = Background * (1 - alpha) + Foreground * alpha

Dimana:
- Background: Pixel frame asli [B,G,R]
- Foreground: Pixel gambar rambut [B,G,R]
- alpha: Nilai transparansi [0-1]
```

### 3. Region of Interest (ROI)
```python
roi_h = min(hair_resized.shape[0], frame.shape[0] - hair_y)
roi_w = min(hair_resized.shape[1], frame.shape[1] - hair_x)
```
- Memastikan overlay tetap dalam batas frame
- Menangani kasus-kasus khusus secara otomatis

## Pertimbangan Performa

### 1. Optimasi Deteksi Wajah
- Parameter Haar Cascade disesuaikan untuk kecepatan:
  ```python
  scaleFactor=1.1  # Keseimbangan antara kecepatan dan akurasi
  minNeighbors=5   # Mengurangi false positive
  minSize=(30, 30) # Ukuran minimum wajah yang dideteksi
  ```

### 2. Ekstraksi Fitur
- Jumlah fitur ORB diseimbangkan untuk akurasi vs kecepatan
- Ukuran codebook BoVW dioptimalkan untuk performa
- Penskalaan fitur dilakukan secara efisien menggunakan operasi numpy

### 3. Overlay Rambut
- Operasi matriks divektorisasi menggunakan numpy
- Pemilihan ROI meminimalkan penggunaan memori
- Alpha blending dioptimalkan untuk performa real-time

### 4. Manajemen Memori
- Gambar dimuat sekali dan dicache
- Operasi ROI mencegah penyalinan yang tidak perlu
- Penggunaan numpy views yang efisien daripada copies

## Optimasi Lanjutan

1. **Kemungkinan Peningkatan**:
   - Parallel processing untuk ekstraksi fitur
   - Akselerasi GPU untuk operasi matriks
   - Penskalaan adaptif berdasarkan ukuran wajah
   - Penyesuaian feature point secara dinamis

2. **Trade-off yang Diketahui**:
   - Kecepatan vs akurasi dalam deteksi wajah
   - Penggunaan memori vs kualitas gambar
   - Waktu pemrosesan vs presisi overlay

## Penjelasan Sederhana (TL;DR)

### Cara Kerja Sistem Secara Umum

1. **Deteksi Wajah (2 Tahap)**
   - **Tahap 1**: Sistem mencari area yang "mirip wajah" dengan cepat (Haar Cascade)
   - **Tahap 2**: Memastikan area tersebut benar-benar wajah (SVM+ORB)
   
2. **Proses Pengenalan Wajah**
   - Mengambil "ciri khas" dari wajah menggunakan ORB
   - Membandingkan dengan data yang sudah dilatih
   - Memutuskan apakah benar-benar wajah atau bukan

3. **Pemasangan Rambut**
   - Menyesuaikan ukuran rambut dengan wajah (2x lebar wajah)
   - Mengatur posisi rambut di atas wajah
   - Menggabungkan rambut dengan gambar kamera secara halus

### Analogi Sederhana

Bayangkan proses ini seperti:
1. **Deteksi Awal**: Seperti mencari "bentuk oval" di foto (mencari kandidat wajah)
2. **Verifikasi**: Memastikan oval tersebut benar-benar wajah (bukan objek lain)
3. **Overlay**: Seperti menempelkan stiker transparan, tapi posisi dan ukurannya menyesuaikan dengan wajah

### Keunggulan Sistem
- Berjalan cepat karena menggunakan metode klasik
- Tidak memerlukan kartu grafis khusus
- Bisa berjalan di komputer biasa
- Mudah diintegrasikan dengan interface Godot