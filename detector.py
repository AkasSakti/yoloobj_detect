"""Wrapper deteksi burung (termasuk ayam) berbasis Ultralytics YOLO.

Model yang dipakai dipilih otomatis:
- Jika models/best.pt ada (hasil training custom lewat train.py, mis. pada
  dataset Kaggle "Birds Images Dataset" yang disiapkan prepare_birds_dataset.py),
  model itu yang dipakai dan semua deteksinya dihitung sebagai burung.
- Jika belum ada, fallback ke model pretrained YOLOv8 (COCO) dan hanya
  mengambil deteksi kelas "bird". Akurasinya lebih terbatas - lihat README
  untuk cara melatih model custom.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CUSTOM_MODEL_PATH = MODELS_DIR / "best.pt"
PRETRAINED_MODEL_PATH = MODELS_DIR / "yolov8n.pt"
PROXY_CLASS_NAME = "bird"


class BirdDetector:
    """Bungkus model YOLO dan menghitung jumlah burung pada sebuah gambar."""

    def __init__(self) -> None:
        if CUSTOM_MODEL_PATH.exists():
            self.model = YOLO(str(CUSTOM_MODEL_PATH))
            self.using_custom_model = True
            # Model custom hanya punya kelas "burung" (lihat data/data.yaml),
            # jadi semua deteksi yang lolos confidence threshold dihitung.
            self.target_class_ids: set[int] | None = None
        else:
            self.model = YOLO(str(PRETRAINED_MODEL_PATH))
            self.using_custom_model = False
            self.target_class_ids = {
                cls_id for cls_id, name in self.model.names.items() if name == PROXY_CLASS_NAME
            }

    @property
    def label(self) -> str:
        return "Model custom burung (models/best.pt)" if self.using_custom_model else "Model pretrained COCO (kelas 'bird')"

    def detect(self, image_bgr: np.ndarray, conf: float = 0.25):
        """Jalankan deteksi pada satu gambar (numpy array, urutan channel BGR).

        Mengembalikan tuple (annotated_image_rgb, jumlah_burung, list_confidence).
        """
        result = self.model.predict(image_bgr, conf=conf, verbose=False)[0]

        if self.target_class_ids is not None:
            keep_idx = [
                i for i, cls_id in enumerate(result.boxes.cls.tolist())
                if int(cls_id) in self.target_class_ids
            ]
            result.boxes = result.boxes[keep_idx]

        annotated_bgr = result.plot()
        annotated_rgb = annotated_bgr[:, :, ::-1]
        count = len(result.boxes)
        confidences = result.boxes.conf.tolist() if count else []
        return annotated_rgb, count, confidences
