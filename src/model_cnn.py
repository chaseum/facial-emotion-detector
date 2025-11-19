import tensorflow as tf
from tensorflow.keras import layers, regularizers, models

def build(input_shape=(48,48,1), num_classes=7, lr=1e-4, decay = 1e-6):
    x_in = layers.Input(shape=input_shape)
    x = layers.Rescaling(1/255.0)(x_in)

    x = layers.Conv2D(32, (3,3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (3,3), padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3,3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-2))(x)
    x = layers.Conv2D(256, (3,3), padding="same", activation="relu",
                      kernel_regularizer=regularizers.l2(1e-2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)
    x = layers.Dropout(0.25)(x)


    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(x_in, out)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(lr, decay),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model
