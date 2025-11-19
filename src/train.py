import argparse, json
from pathlib import Path
import tensorflow as tf
from . import ARTIFACTS
from .utils import set_seed, save_class_names
from .data import make_datasets, class_weight
from . import model_cnn, model_backbone

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--model", choices=["cnn","backbone"], default="cnn")
    ap.add_argument("--epochs", type=int, default=48)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=Path, default=ARTIFACTS)
    args = ap.parse_args()

    set_seed(42)
    train_dir, test_dir = args.data/"train", args.data/"test"
    classes, train, test = make_datasets(train_dir, test_dir, (48,48), args.batch)
    save_class_names(classes, args.out/"class_names.json")

    if args.model == "cnn":
        model = model_cnn.build((48,48,1), len(classes))
    else:
        model = model_backbone.build((48,48,1), len(classes))

    ckpt = tf.keras.callbacks.ModelCheckpoint(
        args.out/"model_checkpoint.keras", monitor="val_accuracy",
        save_best_only=True, mode="max", verbose=1)
    es = tf.keras.callbacks.EarlyStopping(patience=7, restore_best_weights=True)
    rlr = tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5)

    cw = class_weight(train_dir, classes)
    model.fit(train, epochs=args.epochs, validation_data=test,
              callbacks=[ckpt, es, rlr], class_weight=cw)
    model.save(args.out/"emotion48.keras")

if __name__ == "__main__":
    main()
