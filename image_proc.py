from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image
from tensorflow.python.keras.preprocessing import image as kp_image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2


def get_content():
    content_path = askopenfilename(defaultextension='.jpeg', filetypes=[('jpg image', '.jpg'), ('jpeg image', '.jpeg')],
                                                title="Choose file")
    return content_path


def get_style():
    style_path = askopenfilename(defaultextension='.jpeg', filetypes=[('jpg image', '.jpg'), ('jpeg image', '.jpeg')],
                                                title="Choose file")
    return style_path


def imshow(image, title=None):
    image_show = image
    if len(image.shape) == 4:
        image_show = np.squeeze(image, axis=0).astype('uint8')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image_show)

    if title is not None:
        plt.title(title)

    plt.show()


def preprocess_img(image):
    preprocessed_image = tf.keras.applications.vgg19.preprocess_input(image)
    return preprocessed_image


def normalization_final_image(final_image):
    extra = final_image.numpy()
    average_pixel = np.array([103.939, 116.779, 123.68])
    if len(extra.shape) == 4:
        extra = np.squeeze(extra, 0)
    extra[:, :, 0] += average_pixel[0]
    extra[:, :, 1] += average_pixel[1]
    extra[:, :, 2] += average_pixel[2]

    extra = extra[:, :, ::-1]  # bgr -> rgb
    extra = np.clip(extra, 0, 255).astype('uint8')
    return extra
