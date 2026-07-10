import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers

def make_augment():
    # RandomRotation takes fractions of a full turn: 0.08 was +/-29deg, far too
    # harsh for 48px faces. 0.03 is ~+/-10deg, the standard FER-2013 setting.
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.03),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.1, 0.1),
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
    # cache() before augment: decode images from disk once, not every epoch.
    # ponytail: in-memory cache (~260MB for FER-2013), use cache("path") if RAM-tight.
    train = (train_raw.cache()
             .map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=autotune)
             .prefetch(autotune))
    test  = test_raw.cache().prefetch(autotune)
    return train_raw.class_names, train, test