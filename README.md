# Deteksi & Hitung Jumlah Ayam dengan YOLO + Streamlit

Aplikasi web (Streamlit) untuk mendeteksi ayam pada sebuah gambar dan menghitung
jumlahnya, menggunakan model object detection YOLO (Ultralytics).

## Status dataset & model saat ini

Folder `data/` yang tersedia saat project ini dibuat hanya berisi **satu gambar
contoh** (`data/samples/chicken_frame_2567_...jpg`), tanpa anotasi/label YOLO.
Karena belum ada dataset berlabel untuk kelas "ayam", aplikasi ini memakai
strategi berikut secara otomatis:

- Jika `models/best.pt` **ada** (hasil training custom, lihat bagian
  [Melatih model custom](#melatih-model-custom)) -> aplikasi memakai model itu
  dan menganggap semua deteksi adalah ayam.
- Jika `models/best.pt` **belum ada** -> aplikasi fallback ke model pretrained
  `yolov8n.pt` (dilatih di dataset COCO) dan mengambil deteksi kelas **"bird"**
  sebagai pendekatan sementara untuk ayam. COCO tidak punya kelas "chicken"
  khusus, jadi mode ini hanya perkiraan kasar dan **akurasinya terbatas**
  (burung lain bisa ikut terhitung, ayam yang menunduk/tertutup bisa tidak
  terdeteksi, dll). Untuk hasil yang layak dipakai, latih model custom dengan
  dataset ayam sendiri.

## Struktur project

```
yoloobj_detect/
├── app.py              # Aplikasi Streamlit (UI upload gambar, tampilkan hasil)
├── detector.py          # Wrapper model YOLO: pilih model, jalankan deteksi, hitung ayam
├── train.py              # Script training model YOLO custom dari dataset di data/
├── requirements.txt      # Dependency Python
├── packages.txt           # Dependency sistem (apt) untuk Streamlit Community Cloud
├── models/
│   ├── yolov8n.pt        # Model pretrained (fallback, kelas "bird" sebagai proxy ayam)
│   └── best.pt            # (opsional, dibuat sendiri) model custom kelas "ayam"
└── data/
    ├── data.yaml          # Konfigurasi dataset YOLO (path, nama kelas)
    ├── images/train, images/val   # Gambar untuk training/validasi (isi sendiri)
    ├── labels/train, labels/val   # Label YOLO .txt (isi sendiri)
    └── samples/            # Gambar contoh untuk dicoba langsung di aplikasi
```

## Fungsi-fungsi utama

### `detector.py`

- **`ChickenDetector.__init__`** — memutuskan model mana yang dipakai: cek
  apakah `models/best.pt` ada. Kalau ada, load sebagai model custom (semua
  kelasnya dianggap ayam). Kalau tidak, load `models/yolov8n.pt` (auto-download
  dari Ultralytics saat pertama kali dijalankan bila file belum ada) dan cari
  id kelas `"bird"` dari `model.names` untuk dipakai sebagai filter.
- **`ChickenDetector.label`** — teks singkat untuk ditampilkan di UI, menandakan
  model mana yang sedang aktif (custom atau pretrained/proxy).
- **`ChickenDetector.detect(image_bgr, conf)`** — inti deteksi:
  1. Menjalankan `model.predict()` pada gambar (array numpy, urutan channel
     **BGR**, sesuai konvensi OpenCV yang dipakai Ultralytics secara internal).
  2. Jika sedang pakai model pretrained, buang deteksi yang bukan kelas
     `"bird"` lewat `result.boxes = result.boxes[keep_idx]`.
  3. Membuat gambar hasil anotasi (`result.plot()`, kotak + label di setiap
     deteksi) dan mengonversinya ke RGB untuk ditampilkan.
  4. Mengembalikan `(gambar_beranotasi, jumlah_deteksi, daftar_confidence)`.
     `jumlah_deteksi` inilah angka "jumlah ayam" yang ditampilkan di aplikasi.

### `app.py`

- **`load_detector()`** — membuat satu instance `ChickenDetector` dan
  meng-cache-nya (`st.cache_resource`) supaya model YOLO hanya di-load sekali
  per sesi server, bukan setiap kali user berinteraksi dengan UI.
- **`pil_to_bgr(image)`** — mengonversi gambar yang diunggah user (PIL, RGB)
  menjadi array numpy BGR, format yang benar untuk dikirim ke
  `ChickenDetector.detect()` (lihat catatan channel di atas).
- **`rgb_array_to_png_bytes(rgb_array)`** — mengonversi gambar hasil deteksi
  (array RGB) menjadi bytes PNG, dipakai untuk tombol unduh.
- **`main()`** — merangkai UI:
  - Sidebar: slider *confidence threshold* dan info model yang aktif.
  - Input gambar: upload file, atau pakai gambar contoh dari `data/samples/`.
  - Menjalankan `detector.detect()`, menampilkan gambar asli vs. hasil deteksi
    berdampingan, metric jumlah ayam, rincian confidence tiap deteksi, dan
    tombol unduh gambar hasil deteksi.

### `train.py`

- **`main()`** — memvalidasi bahwa `data/images/train` dan `data/images/val`
  sudah berisi gambar (bukan folder kosong), lalu menjalankan
  `YOLO(...).train()` dengan konfigurasi dari `data/data.yaml`. Bobot terbaik
  (`best.pt`) dari hasil training otomatis disalin ke `models/best.pt`, supaya
  `app.py` langsung memakainya pada run berikutnya tanpa perubahan kode.
- Parameter bisa diatur lewat argumen CLI: `--model`, `--epochs`, `--imgsz`,
  `--batch`, `--device` (mis. `--device 0` untuk GPU, `--device cpu` untuk CPU).

## Menjalankan aplikasi secara lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501`, unggah gambar berisi ayam (atau centang opsi
"pakai gambar contoh"), lalu lihat hasil deteksi dan jumlah ayam yang
terdeteksi.

## Deploy ke Streamlit Community Cloud

1. Push folder project ini ke sebuah repo GitHub (`app.py`, `detector.py`,
   `requirements.txt`, `packages.txt`, `models/`, `data/`, dst).
2. Buka [share.streamlit.io](https://share.streamlit.io), hubungkan ke repo
   tersebut, dan set:
   - **Main file path**: `app.py`
3. Streamlit Cloud otomatis membaca `requirements.txt` (dependency Python) dan
   `packages.txt` (dependency sistem yang dibutuhkan OpenCV/Ultralytics di
   Linux, seperti `libgl1`).
4. Setelah deploy pertama kali berhasil, aplikasi langsung bisa dipakai lewat
   URL publik yang diberikan Streamlit Cloud.

> `models/yolov8n.pt` sudah disertakan di repo supaya deploy tidak bergantung
> pada koneksi internet saat runtime. Jika file ini terhapus, `detector.py`
> akan otomatis mengunduhnya ulang saat aplikasi pertama kali dijalankan.

## Melatih model custom

Model pretrained di atas hanya pendekatan kasar. Untuk deteksi ayam yang lebih
akurat, latih model sendiri:

1. Kumpulkan gambar ayam dan beri anotasi bounding box (format YOLO), misalnya
   lewat [Roboflow](https://roboflow.com) atau [LabelImg](https://github.com/HumanSignal/labelImg).
   Nama kelas cukup satu: `ayam`.
2. Export dataset dalam format YOLO, lalu taruh isinya ke:
   - `data/images/train/`, `data/images/val/` — file gambar
   - `data/labels/train/`, `data/labels/val/` — file `.txt` label YOLO
     (satu file per gambar, nama sama, isi `class x_center y_center width height`
     dalam skala 0-1)
3. Jalankan training:
   ```bash
   python train.py --epochs 100 --imgsz 640
   ```
4. Setelah selesai, `models/best.pt` otomatis terisi dan `app.py` (baik lokal
   maupun yang sudah di-deploy, setelah di-push ulang) akan memakainya secara
   otomatis sebagai model utama — tidak perlu ubah kode apa pun.

## Catatan konversi warna (BGR vs RGB)

Ultralytics YOLO menganggap input array numpy memakai urutan channel **BGR**
(konvensi OpenCV) dan mengonversinya ke RGB secara internal sebelum masuk ke
model. Karena gambar yang diunggah lewat Streamlit berasal dari PIL (RGB),
`app.py` sengaja membalik urutan channel-nya (`pil_to_bgr`) sebelum dikirim ke
`detector.detect()`, dan `detector.py` membalik lagi hasil `result.plot()`
(BGR) menjadi RGB sebelum dikembalikan ke `app.py` untuk ditampilkan. Tanpa
konversi ini, warna gambar tetap terlihat normal secara visual tapi akurasi
deteksi model bisa menurun karena channel warna yang diterima terbalik.
