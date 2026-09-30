# Deteksi & Hitung Jumlah Burung dengan YOLO + Streamlit

Aplikasi web (Streamlit) untuk mendeteksi burung (termasuk ayam) pada sebuah
gambar dan menghitung jumlahnya, menggunakan model object detection YOLO
(Ultralytics).

## Dataset & model

Model utama (`models/best.pt`) adalah YOLOv8n yang di-fine-tune dengan dataset
Kaggle [Birds Images Dataset](https://www.kaggle.com/datasets/stealthtechnologies/birds-images-dataset)
(187 foto burung: burung hantu, angsa, kakatua, elang, camar, kolibri, dll).

Dataset tersebut **tidak menyertakan anotasi bounding box** (hanya foto), jadi
`prepare_birds_dataset.py` membuat labelnya secara otomatis (*auto-labeling*)
memakai model open-vocabulary **YOLO-World** (`yolov8s-world.pt`) dengan prompt
`"bird"`. Gambar yang sama sekali tidak terdeteksi burung dilewati. Hasilnya
satu kelas: `burung`.

Hasil training (50 epoch, CPU, dievaluasi pada 35 gambar validasi yang
labelnya juga hasil auto-label):

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| `yolov8n.pt` COCO (kelas *bird*, sebelum) | 0.819 | 0.708 | 0.815 | 0.691 |
| `best.pt` fine-tune dataset Kaggle (sesudah) | **0.900** | **0.829** | **0.872** | **0.729** |

Aplikasi memilih model secara otomatis:

- Jika `models/best.pt` **ada** -> aplikasi memakai model custom itu dan
  menghitung semua deteksi sebagai burung.
- Jika `models/best.pt` **belum ada** -> aplikasi fallback ke model pretrained
  `yolov8n.pt` (COCO) dan hanya mengambil deteksi kelas **"bird"**.

### Keterbatasan

- Label dibuat otomatis, bukan dianotasi manual, sehingga ada kesalahan label
  (terutama burung yang sangat kecil/tertutup di foto kawanan). Metrik di atas
  juga diukur terhadap label otomatis tersebut.
- Foto di dataset Kaggle kebanyakan close-up satu/beberapa burung. Untuk scene
  yang sangat berbeda, seperti kerumunan padat ayam di kandang
  (`data/samples/chicken_frame_...jpg`), model masih kesulitan. Untuk kasus itu
  tambahkan gambar ayam berlabel sendiri ke `data/images` & `data/labels`
  (kelas `0`) lalu latih ulang.

## Struktur project

```
yoloobj_detect/
├── app.py                    # Aplikasi Streamlit (UI upload gambar, tampilkan hasil)
├── detector.py               # Wrapper model YOLO: pilih model, jalankan deteksi, hitung burung
├── prepare_birds_dataset.py  # Unduh dataset Kaggle + auto-label YOLO-World + split train/val
├── train.py                  # Script training model YOLO custom dari dataset di data/
├── requirements.txt          # Dependency Python untuk aplikasi
├── requirements-train.txt    # Dependency tambahan untuk persiapan dataset & training
├── packages.txt              # Dependency sistem (apt) untuk Streamlit Community Cloud
├── models/
│   ├── yolov8n.pt            # Model pretrained COCO (fallback & bobot awal training)
│   └── best.pt               # Model custom kelas "burung" hasil train.py
└── data/
    ├── data.yaml             # Konfigurasi dataset YOLO (path, nama kelas)
    ├── images/train, images/val   # Gambar training/validasi (dibuat prepare_birds_dataset.py)
    ├── labels/train, labels/val   # Label YOLO .txt (dibuat prepare_birds_dataset.py)
    └── samples/              # Gambar contoh untuk dicoba langsung di aplikasi
```

Gambar & label hasil `prepare_birds_dataset.py` (berawalan `kaggle_`) tidak
di-commit ke git karena bisa dibuat ulang kapan saja.

## Fungsi-fungsi utama

### `prepare_birds_dataset.py`

- **`download_dataset()`** — mengunduh dataset Kaggle lewat `kagglehub`
  (dataset publik, tidak perlu API key) ke cache lokal.
- **`main()`** — auto-label setiap gambar dengan YOLO-World (`set_classes(["bird"])`,
  confidence minimum default `0.15` supaya burung di foto kawanan ikut
  terlabel), menulis label format YOLO (kelas `0`), lalu membagi 80/20 ke
  train/val secara deterministik (`--seed`). Hasil import sebelumnya
  (`kaggle_*`) dihapus dulu, jadi aman dijalankan ulang.
- Argumen CLI: `--source` (folder gambar lokal, lewati download), `--labeler`,
  `--conf`, `--val-ratio`, `--seed`.

### `detector.py`

- **`BirdDetector.__init__`** — memutuskan model mana yang dipakai: cek
  apakah `models/best.pt` ada. Kalau ada, load sebagai model custom (semua
  kelasnya dianggap burung). Kalau tidak, load `models/yolov8n.pt` dan cari
  id kelas `"bird"` dari `model.names` untuk dipakai sebagai filter.
- **`BirdDetector.label`** — teks singkat untuk ditampilkan di UI, menandakan
  model mana yang sedang aktif.
- **`BirdDetector.detect(image_bgr, conf)`** — inti deteksi:
  1. Menjalankan `model.predict()` pada gambar (array numpy, urutan channel
     **BGR**, sesuai konvensi OpenCV yang dipakai Ultralytics secara internal).
  2. Jika sedang pakai model pretrained, buang deteksi yang bukan kelas
     `"bird"` lewat `result.boxes = result.boxes[keep_idx]`.
  3. Membuat gambar hasil anotasi (`result.plot()`) dan mengonversinya ke RGB.
  4. Mengembalikan `(gambar_beranotasi, jumlah_deteksi, daftar_confidence)`.

### `app.py`

- **`load_detector()`** — membuat satu instance `BirdDetector` dan
  meng-cache-nya (`st.cache_resource`) supaya model YOLO hanya di-load sekali.
- **`pil_to_bgr(image)`** — mengonversi gambar yang diunggah user (PIL, RGB)
  menjadi array numpy BGR untuk `BirdDetector.detect()`.
- **`rgb_array_to_png_bytes(rgb_array)`** — mengonversi gambar hasil deteksi
  menjadi bytes PNG, dipakai untuk tombol unduh.
- **`main()`** — merangkai UI:
  - Sidebar: slider *confidence threshold*, info model yang aktif dan sumber
    dataset-nya.
  - Input gambar: upload file, atau pilih salah satu gambar contoh dari
    `data/samples/`.
  - Menampilkan gambar asli vs. hasil deteksi, metric jumlah burung, rincian
    confidence tiap deteksi, dan tombol unduh gambar hasil deteksi.

### `train.py`

- **`main()`** — memvalidasi bahwa `data/images/train` dan `data/images/val`
  sudah berisi gambar, lalu menjalankan `YOLO(...).train()` dengan
  konfigurasi dari `data/data.yaml`. Bobot terbaik disalin ke
  `models/best.pt`, sehingga `app.py` langsung memakainya.
- Parameter CLI: `--model` (default `models/yolov8n.pt`), `--epochs`,
  `--imgsz`, `--batch`, `--device` (mis. `--device 0` untuk GPU, `--device cpu`).
- Opsi hemat panas: `--threads` (batas thread CPU), `--workers` (proses
  dataloader), `--cooldown` (jeda detik tiap epoch), dan `--resume` untuk
  melanjutkan training yang terhenti dari `last.pt`.

## Menjalankan aplikasi secara lokal

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka `http://localhost:8501`, unggah gambar berisi burung (atau pilih gambar
contoh), lalu lihat hasil deteksi dan jumlah burung yang terdeteksi.

## Melatih ulang model

```bash
pip install -r requirements-train.txt
python prepare_birds_dataset.py          # unduh dataset Kaggle + auto-label + split
python train.py --epochs 50 --batch 8    # tambahkan --device 0 jika punya GPU CUDA
```

Supaya laptop tidak terlalu panas saat training di CPU, pakai mode hemat
(lebih lambat, tapi beban CPU jauh lebih ringan):

```bash
python train.py --epochs 50 --batch 8 --threads 2 --workers 0 --cooldown 30
# training terhenti? lanjutkan dari checkpoint terakhir:
python train.py --resume runs/burung_yolo/weights/last.pt --threads 2 --workers 0 --cooldown 30
```

`models/best.pt` otomatis diperbarui dan dipakai oleh `app.py` pada run
berikutnya. Untuk meningkatkan akurasi pada ayam, tambahkan gambar ayam
berlabel (format YOLO, kelas `0`, nama file bebas selain awalan `kaggle_`) ke
`data/images/{train,val}` dan `data/labels/{train,val}` sebelum training.

## Deploy ke Streamlit Community Cloud

1. Push folder project ini ke sebuah repo GitHub (`app.py`, `detector.py`,
   `requirements.txt`, `packages.txt`, `models/`, `data/samples/`, dst).
2. Buka [share.streamlit.io](https://share.streamlit.io), hubungkan ke repo
   tersebut, dan set **Main file path**: `app.py`.
3. Streamlit Cloud otomatis membaca `requirements.txt` dan `packages.txt`
   (dependency sistem OpenCV/Ultralytics di Linux, seperti `libgl1`).

> `models/best.pt` dan `models/yolov8n.pt` di-commit ke repo supaya deploy
> tidak bergantung pada training atau koneksi internet saat runtime.

## Catatan konversi warna (BGR vs RGB)

Ultralytics YOLO menganggap input array numpy memakai urutan channel **BGR**
(konvensi OpenCV) dan mengonversinya ke RGB secara internal sebelum masuk ke
model. Karena gambar yang diunggah lewat Streamlit berasal dari PIL (RGB),
`app.py` sengaja membalik urutan channel-nya (`pil_to_bgr`) sebelum dikirim ke
`detector.detect()`, dan `detector.py` membalik lagi hasil `result.plot()`
(BGR) menjadi RGB sebelum ditampilkan. Tanpa konversi ini, akurasi deteksi
model bisa menurun karena channel warna yang diterima terbalik.
