"""Aplikasi Streamlit: deteksi & hitung jumlah burung (termasuk ayam) pada gambar dengan YOLO."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from detector import BirdDetector

BASE_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"


@st.cache_resource(show_spinner="Memuat model YOLO...")
def load_detector() -> BirdDetector:
    return BirdDetector()


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    """Konversi PIL RGB -> numpy BGR (urutan channel yang diharapkan ultralytics)."""
    rgb = np.array(image.convert("RGB"))
    return rgb[:, :, ::-1]


def rgb_array_to_png_bytes(rgb_array: np.ndarray) -> bytes:
    buffer = BytesIO()
    Image.fromarray(rgb_array).save(buffer, format="PNG")
    return buffer.getvalue()


def main() -> None:
    st.set_page_config(page_title="Deteksi & Hitung Burung", page_icon="🐦", layout="wide")
    st.title("🐦 Deteksi & Hitung Jumlah Burung (YOLO)")
    st.caption("Unggah gambar berisi burung (termasuk ayam), model akan menandai tiap burung yang terdeteksi dan menghitung jumlahnya.")

    detector = load_detector()

    with st.sidebar:
        st.header("Pengaturan")
        conf_threshold = st.slider(
            "Confidence threshold", min_value=0.05, max_value=0.95, value=0.25, step=0.05,
            help="Deteksi dengan skor keyakinan di bawah nilai ini akan diabaikan.",
        )
        st.info(f"Model aktif:\n\n**{detector.label}**")
        if not detector.using_custom_model:
            st.warning(
                "Belum ada model custom di `models/best.pt`. Aplikasi memakai model "
                "pretrained COCO (kelas *bird*), sehingga akurasinya terbatas. Jalankan "
                "`prepare_birds_dataset.py` lalu `train.py` untuk melatih model dengan "
                "dataset Kaggle Birds Images (lihat README)."
            )
        else:
            st.caption(
                "Model di-fine-tune dengan dataset Kaggle "
                "[Birds Images Dataset](https://www.kaggle.com/datasets/stealthtechnologies/birds-images-dataset)."
            )

    sample_files = sorted(SAMPLES_DIR.glob("*.jpg")) + sorted(SAMPLES_DIR.glob("*.png"))
    sample_choice = None
    uploaded_file = st.file_uploader("Unggah gambar (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is None and sample_files:
        sample_choice = st.selectbox(
            "...atau pilih gambar contoh dari data/samples",
            options=[None, *sample_files],
            format_func=lambda p: "(tidak ada)" if p is None else p.name,
        )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    elif sample_choice is not None:
        image = Image.open(sample_choice)
    else:
        st.info("Unggah gambar atau pilih gambar contoh di atas untuk mulai mendeteksi.")
        return

    image_bgr = pil_to_bgr(image)

    with st.spinner("Mendeteksi burung..."):
        annotated_rgb, count, confidences = detector.detect(image_bgr, conf=conf_threshold)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Gambar asli")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Hasil deteksi")
        st.image(annotated_rgb, use_container_width=True)

    st.metric("Jumlah burung terdeteksi", count)

    if confidences:
        with st.expander("Detail confidence tiap deteksi"):
            for i, score in enumerate(confidences, start=1):
                st.write(f"Burung #{i}: {score:.2%}")

    st.download_button(
        "Unduh gambar hasil deteksi",
        data=rgb_array_to_png_bytes(annotated_rgb),
        file_name="hasil_deteksi_burung.png",
        mime="image/png",
    )


if __name__ == "__main__":
    main()
