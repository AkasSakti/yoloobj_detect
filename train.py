"""Training model YOLO custom untuk deteksi burung (termasuk ayam).

Jalankan setelah data/images/{train,val} dan data/labels/{train,val} sudah
diisi gambar + anotasi format YOLO (satu file .txt per gambar, kelas 0 = burung),
misalnya lewat prepare_birds_dataset.py (dataset Kaggle Birds Images).
Bobot terbaik hasil training otomatis disalin ke models/best.pt sehingga
langsung dipakai oleh app.py.

Contoh pemakaian:
    python prepare_birds_dataset.py
    python train.py --epochs 50 --imgsz 640
    python train.py --epochs 50 --threads 2 --workers 0 --cooldown 20   # mode hemat panas
"""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "data" / "data.yaml"
MODELS_DIR = BASE_DIR / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training YOLO untuk deteksi burung")
    parser.add_argument("--model", default=str(MODELS_DIR / "yolov8n.pt"),
                        help="Bobot dasar/arsitektur awal (default: models/yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="mis. '0' untuk GPU pertama, atau 'cpu'")
    # Opsi hemat panas: batasi pemakaian CPU & beri jeda antar epoch agar laptop tidak panas.
    parser.add_argument("--threads", type=int, default=None, help="Batas thread CPU PyTorch (mis. 2)")
    parser.add_argument("--workers", type=int, default=8, help="Proses dataloader (0 = paling ringan)")
    parser.add_argument("--cooldown", type=int, default=0, help="Jeda (detik) setelah tiap epoch")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Lanjutkan training yang terhenti dari runs/.../weights/last.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_exts = ("*.jpg", "*.jpeg", "*.png")
    for split in ("train", "val"):
        img_dir = BASE_DIR / "data" / "images" / split
        has_images = any(next(img_dir.glob(ext), None) is not None for ext in image_exts)
        if not has_images:
            raise SystemExit(
                f"data/images/{split} masih kosong. Isi dulu dengan gambar + label YOLO "
                "sebelum menjalankan training, mis. dengan: python prepare_birds_dataset.py"
            )

    if args.threads:
        torch.set_num_threads(args.threads)

    model = YOLO(str(args.resume) if args.resume else args.model)
    if args.cooldown:
        model.add_callback("on_train_epoch_end", lambda trainer: time.sleep(args.cooldown))

    if args.resume:
        # Argumen training lain (epochs, data, dst) diambil dari checkpoint.
        model.train(resume=True, workers=args.workers)
    else:
        model.train(
            data=str(DATA_YAML),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(BASE_DIR / "runs"),
            name="burung_yolo",
        )

    best_weights = Path(model.trainer.save_dir) / "weights" / "best.pt"
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy(best_weights, MODELS_DIR / "best.pt")
    print(f"\nSelesai. Bobot terbaik disalin ke {MODELS_DIR / 'best.pt'}")
    print("app.py akan otomatis memakai model ini pada run berikutnya.")


if __name__ == "__main__":
    main()
