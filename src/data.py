import os
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers

def count_per_class(dir_path: Path):
    return pd.DataFrame({c: len(list((dir_path/c).glob("*"))) 
                        for c in sorted(x.name for x in dir_path.iterdir() if x.is_dir())},
                        index=["count"])

def make_augment():
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.2),
    ], name="augment")

def make_datasets(train_dir: Path, test_dir: Path, image_size=(48,48), batch_size=64):
    train_raw = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=image_size, color_mode="grayscale",
        batch_size=batch_size, label_mode="categorical", shuffle=True)
    test_raw  = tf.keras.utils.image_dataset_from_directory(
        test_dir,  image_size=image_size, color_mode="grayscale",
        batch_size=batch_size, label_mode="categorical", shuffle=False)

    aug = make_augment()
    autotune = tf.data.AUTOTUNE
    train = train_raw.map(lambda x,y: (aug(x, training=True), y),
                          num_parallel_calls=autotune).prefetch(autotune)
    test  = test_raw.prefetch(autotune)
    return train_raw.class_names, train, test

def class_weight(train_dir: Path, classes):
    counts = {c: len(list((train_dir/c).glob("*"))) for c in classes}
    total = sum(counts.values())
    return {i: total/(len(classes)*counts[c]) for i,c in enumerate(classes)}