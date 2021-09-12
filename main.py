import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from tensorflow.python.keras.preprocessing import image as kp_image
from tkinter.filedialog import askopenfilename, asksaveasfilename
from window import Window
from image_proc import *
from model import *

if __name__ == '__main__':
    print('Start')


window = Window()
window.show()
# path_cont = get_content()
# path_style = get_style()
#
# image_cont = get_image(path_cont)
# image_style = get_image(path_style)
#
#
# layers_of_content = ['block5_conv2']
# layers_of_style = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
#
# model = create_model(layers_of_style + layers_of_content)
#
#
# for l in model.layers:
#     l.trainable = False
#
# image_style_preprocessed = preprocess_img(image_style)
# image_cont_preprocessed = preprocess_img(image_cont)
#
# features_style = get_style_features(model(image_style_preprocessed))
# features_content = get_content_features(model(image_cont_preprocessed))
#
# gramMatrix_style = [find_GramMatrix(features) for features in features_style]
#
# trainable_image = tf.Variable(image_cont_preprocessed, dtype=tf.float32)
#
# optimizer = tf.optimizers.Adam(learning_rate=5.0, beta_1=0.99, epsilon=0.1)
#
# epochs = 2
# weight_style = 1e-2
# weight_content = 1e3
# final_image, final_loss = start_training(epochs, model, trainable_image, gramMatrix_style, features_content, optimizer, weight_style, weight_content)
#
# result = normalization_final_image(final_image)
#
# print('final_image', final_image)
# print('final_loss', final_loss)
# #imshow(result)
#
# path_for_saving = asksaveasfilename(defaultextension='.jpeg', filetypes=[('jpeg image', '.jpg')],
#                                     title="Choose filename and directory")
# save_result(path_for_saving, result)
