import tensorflow as tf
from tensorflow.keras import layers, models

def build(input_shape=(48,48,1), num_classes=7, lr=1e-4, backbone="MobileNetV2"):
    x_in = layers.Input(shape=input_shape)
    x = layers.Rescaling(1/255.0)(x_in)
    x = layers.Concatenate()([x,x,x])               # 1→3 channels

    base = tf.keras.applications.MobileNetV2(
        input_shape=(48,48,3), include_top=False, weights=None)  # tiny input → train from scratch
    x = base(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    m = models.Model(x_in, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m
