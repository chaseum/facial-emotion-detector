import os
import json
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

picture_size = 48
batch_size = 128
num_classes = 7

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
train_dir = DATA_DIR / "train"
val_dir = DATA_DIR / "test"

raw_train_ds = keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(picture_size, picture_size),
    color_mode="grayscale",
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=True,
    seed=SEED,
)

raw_val_ds = keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(picture_size, picture_size),
    color_mode="grayscale",
    batch_size=batch_size,
    label_mode="categorical",
    shuffle=False,
)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
    ],
    name="augment",
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = raw_train_ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = raw_val_ds.prefetch(AUTOTUNE)

inputs = layers.Input(shape=(picture_size, picture_size, 1))
x = layers.Rescaling(1.0 / 255.0)(inputs)
x = layers.Conv2D(64, (3,3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Dropout(0.25)(x)
x = layers.Conv2D(128, (5,5), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Dropout(0.25)(x)
x = layers.Conv2D(512, (3,3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Dropout(0.25)(x)
x = layers.Conv2D(512, (3,3), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.MaxPooling2D(pool_size=(2,2))(x)
x = layers.Dropout(0.25)(x)
x = layers.Flatten()(x)
x = layers.Dense(256)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(512)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation('relu')(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
model = keras.Model(inputs, outputs)

opt = keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "model_weights.h5",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)
reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    verbose=1,
)
early_stop_cb = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    train_ds,
    epochs=60,
    validation_data=val_ds,
    callbacks=[checkpoint_cb, reduce_lr_cb, early_stop_cb],
)

model.save("emotion48.keras")
(Path("artifacts")).mkdir(exist_ok=True)
with open("artifacts/class_names.json", "w") as f:
    json.dump(raw_train_ds.class_names, f)
