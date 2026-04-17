from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
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
DISPLAY_SIZE = 360


def default_artifact_paths():
    root = Path(__file__).resolve().parents[1]
    # Prefer full model first (works with your training env .venv-tfdml).
    model_path = root / "notebooks" / "efficientnetv2l_multilabel_final.keras"
    thresh_path = root / "notebooks" / "efficientnetv2l_multilabel_final_thresholds.npy"
    return model_path, thresh_path


def build_model() -> keras.Model:
    base = keras.applications.EfficientNetV2L(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling=None,
    )
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(len(LABEL_COLS), activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="efficientnetv2l_multilabel")
    return model


def load_prediction_model(model_path: Path) -> keras.Model:
    # 1) Try full-model artifacts first.
    root = model_path.parent
    candidate_models = [model_path]
    candidate_models.extend(
        [
            root / "efficientnetv2l_multilabel_final.keras",
            root / "efficientnetv2l_multilabel_stage3_best.keras",
        ]
    )
    for p in candidate_models:
        if p.is_file():
            try:
                return keras.models.load_model(str(p))
            except Exception:
                continue

    # 2) Fall back to architecture + weights only if needed.
    candidate_weights = [
        root / "efficientnetv2l_multilabel_final.weights.h5",
        root / "efficientnetv2l_best_stage3.weights.h5",
        root / "efficientnetv2l_best_ft.weights.h5",
    ]
    for w in candidate_weights:
        if w.is_file():
            try:
                model = build_model()
                model.load_weights(str(w))
                return model
            except Exception:
                continue

    raise FileNotFoundError(
        "No loadable model/weights found. "
        f"Tried models: {[str(p) for p in candidate_models]} ; "
        f"tried weights: {[str(p) for p in candidate_weights]}"
    )


def preprocess_pil(image: Image.Image) -> tf.Tensor:
    rgb = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(rgb).astype(np.float32)
    arr = keras.applications.efficientnet_v2.preprocess_input(arr)
    return tf.convert_to_tensor(arr[None, ...], dtype=tf.float32)


def hires_cam(
    base_extractor: keras.Model,
    gap_layer: keras.layers.Layer,
    dropout_layer: keras.layers.Layer,
    classifier_layer: keras.layers.Layer,
    x: tf.Tensor,
    class_idx: int,
) -> np.ndarray:
    with tf.GradientTape() as tape:
        conv = base_extractor(x, training=False)
        pooled = gap_layer(conv)
        dropped = dropout_layer(pooled, training=False)
        preds = classifier_layer(dropped)
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
    heat_rgb = np.stack([heat_tf, np.zeros_like(heat_tf), 1.0 - heat_tf], axis=-1)
    out = (1.0 - alpha) * (rgb.astype(np.float32) / 255.0) + alpha * heat_rgb
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def top_cam_indices(probs: np.ndarray, thresholds: np.ndarray, k: int = 3):
    positive = np.where(probs >= thresholds)[0].tolist()
    if positive:
        positive = sorted(positive, key=lambda i: probs[i], reverse=True)
        return positive[:k]
    return np.argsort(-probs)[:k].tolist()


class LocalDemoApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("MILK Thesis V1 - Local Desktop Demo")
        self.root.geometry("1200x780")

        model_path, thresh_path = default_artifact_paths()
        try:
            self.model = load_prediction_model(model_path)
            self.thresholds = np.load(str(thresh_path)).astype(np.float32)
        except Exception as exc:
            messagebox.showerror(
                "Load error",
                f"Could not load model artifacts.\n\nModel: {model_path}\nThresholds: {thresh_path}\n\n{exc}",
            )
            raise

        # Reuse loaded model components directly for CAM to avoid graph-disconnect issues.
        self.base_extractor = self.model.get_layer("efficientnetv2-l")
        self.gap_layer = self.model.get_layer("global_average_pooling2d")
        self.dropout_layer = self.model.get_layer("dropout")
        self.classifier_layer = self.model.get_layer("dense")

        self.preview_photo = None
        self.cam_photos = []

        self._build_ui()

    def _build_ui(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill=tk.X)

        ttk.Button(top, text="Open Image", command=self.pick_image).pack(side=tk.LEFT)
        ttk.Label(top, text="Loads image, predicts labels, and shows HiResCAM overlays").pack(
            side=tk.LEFT, padx=12
        )

        body = ttk.Frame(self.root, padding=10)
        body.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(body)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 14))
        ttk.Label(left, text="Input Image").pack(anchor="w")
        self.preview_label = ttk.Label(left)
        self.preview_label.pack(pady=(6, 12))
        self.path_label = ttk.Label(left, text="", wraplength=340)
        self.path_label.pack(anchor="w")

        mid = ttk.Frame(body)
        mid.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 14))
        ttk.Label(mid, text="Predictions").pack(anchor="w")

        cols = ("class", "probability", "threshold", "predicted")
        self.tree = ttk.Treeview(mid, columns=cols, show="headings", height=22)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120, anchor="center")
        self.tree.column("class", width=130, anchor="w")
        self.tree.pack(fill=tk.BOTH, expand=True, pady=(6, 0))

        right = ttk.Frame(body)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        ttk.Label(right, text="HiResCAM Overlays (Top 3)").pack(anchor="w")

        self.cam_labels = []
        for _ in range(3):
            lbl = ttk.Label(right)
            lbl.pack(pady=8)
            self.cam_labels.append(lbl)

    def pick_image(self):
        path = filedialog.askopenfilename(
            title="Select lesion image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")],
        )
        if not path:
            return
        self.run_inference(Path(path))

    def run_inference(self, path: Path):
        try:
            pil_img = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Image error", f"Could not open image.\n\n{exc}")
            return

        x = preprocess_pil(pil_img)
        probs = self.model.predict(x, verbose=0)[0]
        preds = (probs >= self.thresholds).astype(np.int32)

        self.path_label.config(text=str(path))
        preview = pil_img.copy()
        preview.thumbnail((DISPLAY_SIZE, DISPLAY_SIZE))
        self.preview_photo = ImageTk.PhotoImage(preview)
        self.preview_label.config(image=self.preview_photo)

        for item in self.tree.get_children():
            self.tree.delete(item)
        rows = []
        for i, cls in enumerate(LABEL_COLS):
            rows.append((cls, float(probs[i]), float(self.thresholds[i]), int(preds[i])))
        rows.sort(key=lambda r: r[1], reverse=True)
        for cls, p, t, y in rows:
            self.tree.insert("", tk.END, values=(cls, f"{p:.4f}", f"{t:.2f}", y))

        rgb = np.asarray(pil_img.convert("RGB"))
        cam_idx = top_cam_indices(probs, self.thresholds, k=3)
        self.cam_photos = []
        for i, class_idx in enumerate(cam_idx):
            heat = hires_cam(
                self.base_extractor,
                self.gap_layer,
                self.dropout_layer,
                self.classifier_layer,
                x,
                class_idx,
            )
            overlay = overlay_heatmap(rgb, heat, alpha=0.45)
            ov_img = Image.fromarray(overlay)
            ov_img.thumbnail((DISPLAY_SIZE, DISPLAY_SIZE))
            photo = ImageTk.PhotoImage(ov_img)
            self.cam_photos.append(photo)
            self.cam_labels[i].config(
                image=photo,
                text=f"{LABEL_COLS[class_idx]} | p={probs[class_idx]:.3f} | thr={self.thresholds[class_idx]:.2f}",
                compound="top",
            )


if __name__ == "__main__":
    root = tk.Tk()
    app = LocalDemoApp(root)
    root.mainloop()
