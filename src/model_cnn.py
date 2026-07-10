import tensorflow as tf
from tensorflow.keras import layers, models

def build(input_shape=(48,48,1), num_classes=7, lr=1e-3):
    x_in = layers.Input(shape=input_shape)
    x = layers.Rescaling(1/255.0)(x_in)

    # 4-block VGG-style body — the standard FER-2013 recipe that reaches 65-68%.
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)          # 24x24
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)          # 12x12
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)          # 6x6
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(512, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)          # 3x3
    x = layers.Dropout(0.25)(x)

    # Flatten at 3x3 keeps spatial layout cheaply (~2.4M head params).
    # The original flattened at 12x12 into Dense(1024): 37.7M params, 457MB checkpoints.
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(x_in, out)
    
    # Adam(lr, decay) silently set beta_1=1e-6 — decay is not positional.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
