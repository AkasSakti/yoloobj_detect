"""Training model YOLO custom untuk deteksi ayam.

Jalankan setelah data/images/{train,val} dan data/labels/{train,val} sudah
diisi gambar + anotasi format YOLO (satu file .txt per gambar, kelas 0 = ayam).
Bobot terbaik hasil training otomatis disalin ke models/best.pt sehingga
langsung dipakai oleh app.py.

Contoh pemakaian:
    python train.py --epochs 100 --imgsz 640
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "data" / "data.yaml"
MODELS_DIR = BASE_DIR / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training YOLO untuk deteksi ayam")
    parser.add_argument("--model", default="yolov8n.pt", help="Bobot dasar/arsitektur awal (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="mis. '0' untuk GPU pertama, atau 'cpu'")
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
                "sebelum menjalankan training (lihat README bagian 'Melatih model custom')."
            )

    model = YOLO(args.model)
    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(BASE_DIR / "runs"),
        name="ayam_yolo",
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy(best_weights, MODELS_DIR / "best.pt")
    print(f"\nSelesai. Bobot terbaik disalin ke {MODELS_DIR / 'best.pt'}")
    print("app.py akan otomatis memakai model ini pada run berikutnya.")


if __name__ == "__main__":
    main()
