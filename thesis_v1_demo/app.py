from pathlib import Path
from typing import List, Tuple

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow import keras


LABEL_COLS = [
    "AKIEC",
    "BCC",
    "BEN_OTH",
    "BKL",
    "DF",
    "INF",
    "MAL_OTH",
    "MEL",
    "NV",
    "SCCKA",
    "VASC",
]

IMG_SIZE = 480


def default_artifact_paths() -> Tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    model_path = root / "notebooks" / "efficientnetv2l_multilabel_final.keras"
    thresh_path = root / "notebooks" / "efficientnetv2l_multilabel_final_thresholds.npy"
    return model_path, thresh_path


def preprocess_pil(image: Image.Image) -> tf.Tensor:
    rgb = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(rgb).astype(np.float32)
    arr = keras.applications.efficientnet_v2.preprocess_input(arr)
    return tf.convert_to_tensor(arr[None, ...], dtype=tf.float32)


@st.cache_resource(show_spinner=False)
def load_artifacts(model_path: str, thresh_path: str):
    model = keras.models.load_model(model_path)
    thresholds = np.load(thresh_path).astype(np.float32)
    base = model.get_layer("efficientnetv2-l")
    cam_model = keras.Model(model.input, [model.output, base.output], name="cam_model")
    return model, cam_model, thresholds


def hires_cam(cam_model: keras.Model, x: tf.Tensor, class_idx: int) -> np.ndarray:
    with tf.GradientTape() as tape:
        preds, conv = cam_model(x, training=False)
        score = preds[:, class_idx]
    grads = tape.gradient(score, conv)
    cam = tf.reduce_sum(grads * conv, axis=-1)
    cam = tf.nn.relu(cam)
    cam = cam[0]
    cam = cam / (tf.reduce_max(cam) + 1e-8)
    return cam.numpy()


def overlay_heatmap(rgb: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = rgb.shape[:2]
    heat_tf = tf.image.resize(heat[..., None], [h, w], method="bilinear")[..., 0].numpy()
    heat_tf = np.clip(heat_tf, 0.0, 1.0)

    # Simple red-blue map without extra deps.
    heat_rgb = np.stack([heat_tf, np.zeros_like(heat_tf), 1.0 - heat_tf], axis=-1)
    out = (1.0 - alpha) * (rgb.astype(np.float32) / 255.0) + alpha * heat_rgb
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def select_cam_classes(probs: np.ndarray, thresholds: np.ndarray, top_k: int = 3) -> List[int]:
    positive = np.where(probs >= thresholds)[0].tolist()
    if len(positive) >= top_k:
        positive_sorted = sorted(positive, key=lambda i: probs[i], reverse=True)
        return positive_sorted[:top_k]
    top = np.argsort(-probs)[:top_k].tolist()
    return top


def main():
    st.set_page_config(page_title="MILK Thesis V1 Demo", layout="wide")
    st.title("MILK Lesion Classifier - V1 Demo")
    st.caption("Standalone inference + automatic HiResCAM overlays")

    default_model, default_thresh = default_artifact_paths()
    with st.sidebar:
        st.subheader("Artifacts")
        model_path = st.text_input("Model path (.keras)", value=str(default_model))
        thresh_path = st.text_input("Threshold path (.npy)", value=str(default_thresh))
        top_k = st.slider("Top CAM overlays", min_value=1, max_value=5, value=3)

    if not Path(model_path).is_file():
        st.error(f"Model not found: {model_path}")
        return
    if not Path(thresh_path).is_file():
        st.error(f"Threshold file not found: {thresh_path}")
        return

    model, cam_model, thresholds = load_artifacts(model_path, thresh_path)

    uploaded = st.file_uploader("Upload a lesion image", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Upload one image to run predictions and HiResCAM.")
        return

    pil_img = Image.open(uploaded).convert("RGB")
    x = preprocess_pil(pil_img)
    probs = model.predict(x, verbose=0)[0]
    preds = (probs >= thresholds).astype(np.int32)

    st.subheader("Input")
    st.image(pil_img, caption="Uploaded image", width=420)

    pred_rows = []
    for i, cls in enumerate(LABEL_COLS):
        pred_rows.append(
            {
                "class": cls,
                "probability": float(probs[i]),
                "threshold": float(thresholds[i]),
                "predicted": int(preds[i]),
            }
        )
    pred_rows = sorted(pred_rows, key=lambda r: r["probability"], reverse=True)
    st.subheader("Predictions")
    st.dataframe(pred_rows, use_container_width=True)

    positives = [LABEL_COLS[i] for i in np.where(preds == 1)[0]]
    st.write("Positive labels:", positives if positives else "None (using tuned thresholds)")

    st.subheader("HiResCAM (auto)")
    rgb = np.asarray(pil_img.convert("RGB"))
    cam_indices = select_cam_classes(probs, thresholds, top_k=top_k)
    cols = st.columns(len(cam_indices))
    for c, idx in zip(cols, cam_indices):
        heat = hires_cam(cam_model, x, idx)
        overlay = overlay_heatmap(rgb, heat, alpha=0.45)
        c.image(
            overlay,
            caption=f"{LABEL_COLS[idx]} | p={probs[idx]:.3f} | thr={thresholds[idx]:.2f}",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
