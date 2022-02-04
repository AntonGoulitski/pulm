import tensorflow as tf
from PIL import Image
import cv2
import numpy as np
import tensorflow.keras.backend as K
import io

TARGET_SIZE = (600,600)


def decode(bytes):
    img_tensor_uint8 = tf.image.decode_png(bytes,channels=3)
    img_tensor_float32 = tf.cast(img_tensor_uint8, tf.float32) / 255
    img = tf.image.resize(img_tensor_float32, TARGET_SIZE) # Resizing to target size
    img = tf.reshape(img,(-1,600,600,3))
    return img

def prepare(img):
    img_array = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    new_array = cv2.resize(img_array, (224, 224))
    return new_array.reshape(-1, 224, 224, 3)

def heat_map(file,last_conv_layer, model,intencity):
    img = Image.open(io.BytesIO(file))
    orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    orig = cv2.cvtColor(orig,cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    orig = clahe.apply(orig)
    img = decode(file)
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('dropout')
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(img)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = heatmap.reshape((19, 19))
        heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
        heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
        orig = cv2.cvtColor(orig,cv2.COLOR_GRAY2RGB)
        img = heatmap * intencity + orig
        cv2.imwrite('heatmap' +'.png',img)
