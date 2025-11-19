import tensorflow as tf
import numpy as np

def gradcam(model, img, last_conv_name, class_index=None):
    
    conv_layer = model.get_layer(last_conv_name)
    grad_model = tf.keras.Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img, training=False)
        if class_index is None:
            class_index = tf.argmax(preds[0])
        loss = preds[:, class_index]

    grads = tape.gradient(loss, conv_out)[0]                
    weights = tf.reduce_mean(grads, axis=(0,1))         
    cam = tf.reduce_sum(tf.multiply(weights, conv_out[0]), axis=-1) 
    cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + 1e-6)
    cam = tf.image.resize(cam[...,None], (img.shape[1], img.shape[2]))
    return cam.numpy().squeeze(), int(class_index)