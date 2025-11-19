import json
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
from . import PROJECT, ARTIFACTS, DATA_DIR
from .utils import load_class_names

IMG_SIZE = 48

def make_ds(dirpath: Path, bs=64):
    return tf.keras.utils.image_dataset_from_directory(
        dirpath, image_size=(IMG_SIZE,IMG_SIZE), color_mode="grayscale",
        batch_size=bs, label_mode="categorical", shuffle=False)

def _gather_y_true(ds):
    ys = []
    for _, y in ds:
        ys.append(y.numpy())
    return np.argmax(np.concatenate(ys, axis=0), axis=1)

def _predict(model, ds):
    preds = model.predict(ds, verbose=0)
    return np.argmax(preds, axis=1), preds

def _plot_confusion(cm, class_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel="True", xlabel="Predicted",
           title="Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    

    
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default=str(DATA_DIR / "test"), help="Path to test folder")
    ap.add_argument("--model", default=str(ARTIFACTS / "emotion48.keras"))
    ap.add_argument("--classes", default=str(ARTIFACTS / "class_names.json"))
    ap.add_argument("--out", default=str(ARTIFACTS / "confusion_matrix.png"))
    args = ap.parse_args(argv)

    test_dir = Path(args.test)
    model_path = Path(args.model)
    classes_path = Path(args.classes)
    out_png = Path(args.out)

    model = tf.keras.models.load_model(model_path)
    class_names = load_class_names(classes_path)

    ds_test = make_ds(test_dir, bs=64)
    y_true = _gather_y_true(ds_test)
    y_pred, _ = _predict(model, ds_test)

    print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion(cm, class_names, out_png)
    print(f"Saved confusion matrix → {out_png}")

if __name__ == "__main__":
    main()