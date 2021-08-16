import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from tensorflow.python.keras.preprocessing import image as kp_image


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.




if __name__ == '__main__':
    print('Start')


def get_content():
    content_path = '/Users/viktoria/Desktop/Green_Sea_Turtle_grazing_seagrass.jpeg'
    return content_path


def get_style():
    style_path = '/Users/viktoria/Desktop/The_Great_Wave_off_Kanagawa.jpeg'
    return style_path


def preprocess_image(path):
    image = Image.open(path)
    new_size = 512
    extra = max(image.size)
    scale = new_size / extra
    new_img = image.resize((round(image.size[0] * scale), round(image.size[1] * scale)), Image.ANTIALIAS)

    new_img = kp_image.img_to_array(new_img)
    new_img = np.expand_dims(new_img, axis=0)
    return new_img



def imshow(image, title=None):
  print('Image shape:', image.shape)
  image_show = np.squeeze(image, axis=0)
  image_show = image_show.astype('uint8')

  if title is not None:
    plt.title(title)

  plt.imshow(image_show)
  plt.show()





path_cont = get_content()
path_style = get_style()

img_cont = preprocess_image(path_cont)
img_style = preprocess_image(path_style)

imshow(img_style, title='Content image')

