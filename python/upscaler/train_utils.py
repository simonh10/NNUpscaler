import numpy as np
import os
import tensorflow as tf
import random
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import img_to_array, load_img, save_img
import PIL
import math

def byte_to_float(input_image):
    input_image = input_image / 255.0
    return input_image

def float_to_byte(input_image):
    input_image = input_image * 255
    return input_image

def preprocess_yuv_scale(input_image, size):
    input = tf.image.rgb_to_yuv(input_image)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, size, method="area")

def preprocess_yuv(input_image):
    input = tf.image.rgb_to_yuv(input_image)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y

def build_model(upscale_factor=3, channels=1):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)

def get_lowres_image(img, upscale_factor):
    """Return low-resolution image to use as model input."""
    return img.resize(
        (img.size[0] // upscale_factor, img.size[1] // upscale_factor),
        PIL.Image.BICUBIC,
    )

def upscale_image(model, img):
    """Predict the result based on input image and restore the image as RGB."""
    ycbcr = img.convert("YCbCr")
    y, cb, cr = ycbcr.split()
    y = img_to_array(y)
    y = y.astype("float32") / 255.0

    input = np.expand_dims(y, axis=0)
    out = model.predict(input)

    out_img_y = out[0]
    out_img_y *= 255.0

    # Restore the image in RGB color space.
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.reshape((np.shape(out_img_y)[0], np.shape(out_img_y)[1]))
    out_img_y = PIL.Image.fromarray(np.uint8(out_img_y), mode="L")
    out_img_cb = cb.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, PIL.Image.BICUBIC)
    out_img = PIL.Image.merge("YCbCr", (out_img_y, out_img_cb, out_img_cr)).convert(
        "RGB"
    )
    return out_img

class ESPCNCallback(keras.callbacks.Callback):
    def __init__(self, test_image_paths=[], upscale_factor=3):
        super(ESPCNCallback, self).__init__()
        self._test_image_paths=test_image_paths
        self._upscale_factor=upscale_factor
        self.test_img = None
        self._update_test_image()

    def _update_test_image(self):
        self.test_img = get_lowres_image(
            load_img(
                random.choice(
                    self.test_image_paths)),
            self.upscale_factor)

    # Store PSNR value in each epoch.
    def on_epoch_begin(self, epoch, logs=None):
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        print("Mean PSNR for epoch: %.2f" % (np.mean(self.psnr)))
        if epoch % 5 == 0:
            self._update_test_image()
            prediction = upscale_image(self.model, self.test_img)
            # plot_results(prediction, "epoch-" + str(epoch), "prediction", subdir='epochs')

    def on_test_batch_end(self, batch, logs=None):
        self.psnr.append(10 * math.log10(1 / logs["loss"]))


def generate_test_images(sources, output_path, model, scale=3):
    for index, source in enumerate(sources):
        img = load_img(source)
        orig_path = os.path.join(output_path, f"{index:03}-source.png")
        lowres_path = os.path.join(output_path, f"{index:03}-lowres.png")
        scaled_path = os.path.join(output_path, f"{index:03}-scaled.png")
        save_img(orig_path, img)
        low_res_image = get_lowres_image(img, scale)
        save_img(lowres_path, low_res_image)
        scaled_image = upscale_image(model, low_res_image)
        save_img(scaled_path, scaled_image)


