import tensorflow as tf
from tensorflow._api import *
import numpy as np


def switch_trainable(model):
    """"Switches all layears of model to untrainable"""

    for l in model.layers:
        l.trainable = False
    return model


def create_model(layers_names):
    """"Loads and create VGG19 model with certain layers"""

    vgg_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg_model.trainable = False
    outputs = [vgg_model.get_layer(layer).output for layer in layers_names]
    return tf.keras.Model(vgg_model.input, outputs)


def get_style_features(outputs):
    """"Returns the representations of style features from outputs of model"""

    features_representations_style = [feature_style[0] for feature_style in outputs[:5]]
    return features_representations_style


def get_content_features(outputs):
    """"Returns the representations of content features from outputs of model"""

    features_representations_content = [feature_content[0] for feature_content in outputs[5:]]
    return features_representations_content

def find_GramMatrix(features):
    """"Returns Gram matrix of features"""

    reshaped_input = tf.reshape(features, [-1, int(features.shape[-1])])
    gramMatrix = tf.matmul(reshaped_input, reshaped_input, transpose_a=True)
    len = tf.cast(tf.shape(reshaped_input)[0], tf.float32)
    return gramMatrix / len


def compute_style_loss(old_style, new_style):
    """"Returns computed style loss of model"""

    return tf.reduce_mean(tf.square(old_style - new_style))


def compute_content_loss(old_content, new_content):
    """"Returns computed content loss of model"""

    return tf.reduce_mean(tf.square(old_content - new_content))


def find_loss(model, trainable_image, gramMatrix_style, features_content, weight_style, weight_content):
    """"Returns total loss of model, which includes style and content loss with certain coefficients"""

    new_outputs = model(trainable_image)
    new_style_outputs = new_outputs[:5]
    new_content_outputs = new_outputs[5:]

    style_total = 0
    content_total = 0
    for new_style, old_style in zip(new_style_outputs, gramMatrix_style):
        gramMatrix_style_new = find_GramMatrix(new_style)
        style_total += (0.2 * compute_style_loss(old_style, gramMatrix_style_new))
    style_total *= weight_style

    for new_content, old_content in zip(new_content_outputs, features_content):
        content_total += (compute_content_loss(old_content, new_content))
    content_total *= weight_content

    return style_total + content_total


def start_training(window, epochs, model, trainable_image, gramMatrix_style, features_content, optimizer, weight_style,
                   weight_content):
    """"Main method of training session of model. Returns final image and final loss"""

    average_pixel = np.array([103.939, 116.779, 123.68])
    min = 0 - average_pixel
    max = 255 - average_pixel

    best_img = None
    best_loss = float('inf')

    for epoch in range(epochs):
        window.pb['value'] = int((epoch + 1) / epochs * 100)
        window.pb.update()

        print('epoch:', epoch)
        with tf.GradientTape() as g:
            total_loss = find_loss(model, trainable_image, gramMatrix_style, features_content, weight_style,
                                   weight_content)

        print('total_loss', total_loss)

        gradient = g.gradient(total_loss, trainable_image)
        optimizer.apply_gradients([(gradient, trainable_image)])
        trainable_image.assign(tf.clip_by_value(trainable_image, min, max))

        if total_loss < best_loss:
            best_img = trainable_image
            best_loss = total_loss

    return best_img, best_loss
