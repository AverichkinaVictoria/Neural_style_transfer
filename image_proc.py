from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image
from tensorflow.python.keras.preprocessing import image as kp_image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2


def get_content():
    content_path = askopenfilename()
    return content_path


def get_style():
    style_path = askopenfilename()
    return style_path


def get_image(path):
    try:
        image = Image.open(path)
    except:
        print('Wrong file format. Supported extensions: .jpeg and .jpg')
        return
    new_size = 512
    extra = max(image.size)
    scale = new_size / extra
    new_img = image.resize((round(image.size[0] * scale), round(image.size[1] * scale)), Image.ANTIALIAS)

    new_img = kp_image.img_to_array(new_img)
    new_img = np.expand_dims(new_img, axis=0)
    return new_img


def imshow(image, title=None):
    print('Image shape:', image.shape)
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


def save_result(path, image):
    try:
        image = image[:, :, ::-1]
        cv2.imwrite(path, image)
    except:
        print('Wrong file extension. Try to save with extensions as .jpeg or .jpg!')
