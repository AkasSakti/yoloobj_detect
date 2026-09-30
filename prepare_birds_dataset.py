"""Siapkan dataset training YOLO dari Kaggle "Birds Images Dataset".

Dataset https://www.kaggle.com/datasets/stealthtechnologies/birds-images-dataset
hanya berisi foto burung (tanpa anotasi bounding box), jadi script ini:
1. Mengunduh dataset lewat kagglehub (dataset publik, tidak perlu login).
2. Memberi label bounding box otomatis (auto-label) memakai model open-vocabulary
   YOLO-World dengan prompt "bird". Gambar yang tidak terdeteksi burung sama
   sekali dilewati (bukan dijadikan background), karena setiap gambar di dataset
   ini pasti berisi burung - labelnya saja yang gagal dibuat.
3. Membagi hasilnya ke data/images/{train,val} + data/labels/{train,val}
   (format YOLO, kelas 0 = burung), siap dipakai train.py.

Contoh pemakaian:
    pip install -r requirements-train.txt
    python prepare_birds_dataset.py
    python train.py --epochs 50
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
KAGGLE_DATASET = "stealthtechnologies/birds-images-dataset"
FILE_PREFIX = "kaggle_"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unduh & auto-label dataset burung dari Kaggle")
    parser.add_argument("--source", type=Path, default=None,
                        help="Folder gambar yang sudah diunduh (lewati download dari Kaggle)")
    parser.add_argument("--labeler", default="yolov8s-world.pt",
                        help="Model YOLO-World untuk auto-label (auto-download bila belum ada)")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Confidence minimum label otomatis (rendah agar kawanan burung ikut terlabel)")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def download_dataset() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit("kagglehub belum terpasang. Jalankan: pip install -r requirements-train.txt") from exc
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def clear_previous_import() -> None:
    """Hapus hasil import sebelumnya supaya script aman dijalankan ulang."""
    for sub in ("images", "labels"):
        for split in ("train", "val"):
            for f in (DATA_DIR / sub / split).glob(f"{FILE_PREFIX}*"):
                f.unlink()


def main() -> None:
    args = parse_args()
    source = args.source or download_dataset()
    images = sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"Tidak ada gambar di {source}")
    print(f"{len(images)} gambar ditemukan di {source}")

    labeler = YOLO(args.labeler)
    labeler.set_classes(["bird"])

    labeled: list[tuple[Path, list[str]]] = []
    for img in images:
        result = labeler.predict(str(img), conf=args.conf, verbose=False)[0]
        lines = [f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for x, y, w, h in result.boxes.xywhn.tolist()]
        if lines:
            labeled.append((img, lines))
    skipped = len(images) - len(labeled)
    print(f"{len(labeled)} gambar berhasil dilabeli, {skipped} dilewati (tidak ada deteksi burung)")

    random.Random(args.seed).shuffle(labeled)
    n_val = max(1, round(len(labeled) * args.val_ratio))
    splits = {"val": labeled[:n_val], "train": labeled[n_val:]}

    clear_previous_import()
    for split, items in splits.items():
        img_dir = DATA_DIR / "images" / split
        lbl_dir = DATA_DIR / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img, lines in items:
            name = FILE_PREFIX + img.stem
            shutil.copy(img, img_dir / f"{name}{img.suffix.lower()}")
            (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")
        print(f"{split}: {len(items)} gambar")

    print("\nSelesai. Lanjutkan dengan: python train.py")


if __name__ == "__main__":
    main()
