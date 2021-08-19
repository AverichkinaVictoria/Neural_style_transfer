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
  image_show = np.squeeze(image, axis=0).astype('uint8')

  fig, ax = plt.subplots(figsize=(8, 8))
  ax.set_xticks([])
  ax.set_yticks([])
  ax.imshow(image_show)

  if title is not None:
    plt.title(title)

  plt.show()



def switch_trainable(model):
    for l in model.layers:
        l.trainable = False
    return model


def create_model(layers_names):
    vgg_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg_model.trainable = False
    outputs = [vgg_model.get_layer(layer).output for layer in layers_names]
    return tf.keras.Model(vgg_model.input, outputs)


def get_features(outputs):
    features_representations = [feature[0] for feature in outputs]
    return features_representations

def preprocess_img(image):
    preprocessed_image = tf.keras.applications.vgg19.preprocess_input(image)
    return preprocessed_image


def find_GramMatrix(features):
  reshaped_input = tf.reshape(features, [-1, int(features.shape[-1])])
  gramMatrix = tf.matmul(reshaped_input, reshaped_input, transpose_a=True)
  len = tf.cast(tf.shape(reshaped_input)[0], tf.float32)
  return gramMatrix / len


path_cont = get_content()
path_style = get_style()

image_cont = preprocess_image(path_cont)
image_style = preprocess_image(path_style)

#imshow(image_cont, title='Content image')

layers_of_content = ['block5_conv2']
layers_of_style = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

model = create_model(layers_of_style + layers_of_content)

#model = switch_trainable(model)

for l in model.layers:
    l.trainable = False


image_style_preprocessed = preprocess_img(image_style)
image_cont_preprocessed = preprocess_img(image_cont)

features_style = get_features(model(image_style_preprocessed))
features_content = get_features(model(image_cont_preprocessed))

GramMatrix_style = [find_GramMatrix(features) for features in features_style]

print(GramMatrix_style)




