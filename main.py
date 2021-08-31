import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
from tensorflow.python.keras.preprocessing import image as kp_image
from tkinter.filedialog import askopenfilename, asksaveasfilename

if __name__ == '__main__':
    print('Start')


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



def switch_trainable(model):
    for l in model.layers:
        l.trainable = False
    return model


def create_model(layers_names):
    vgg_model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg_model.trainable = False
    outputs = [vgg_model.get_layer(layer).output for layer in layers_names]
    return tf.keras.Model(vgg_model.input, outputs)


def get_style_features(outputs):
    features_representations_style = [feature_style[0] for feature_style in outputs[:5]]
    return features_representations_style


def get_content_features(outputs):
    features_representations_content = [feature_content[0] for feature_content in outputs[5:]]
    return features_representations_content

def preprocess_img(image):
    preprocessed_image = tf.keras.applications.vgg19.preprocess_input(image)
    return preprocessed_image


def find_GramMatrix(features):
  reshaped_input = tf.reshape(features, [-1, int(features.shape[-1])])
  gramMatrix = tf.matmul(reshaped_input, reshaped_input, transpose_a=True)
  len = tf.cast(tf.shape(reshaped_input)[0], tf.float32)
  return gramMatrix / len


def compute_style_loss(old_style, new_style):
    return tf.reduce_mean(tf.square(old_style - new_style))


def compute_content_loss(old_content, new_content):
    return tf.reduce_mean(tf.square(old_content - new_content))



def find_loss(model, trainable_image, gramMatrix_style, features_content, weight_style, weight_content):
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

    return style_total+content_total


def start_training(epochs, model, trainable_image, gramMatrix_style, features_content, optimizer, weight_style, weight_content):
    average_pixel = np.array([103.939, 116.779, 123.68])
    min = 0 - average_pixel
    max = 255 - average_pixel

    best_img = None
    best_loss = float('inf')

    for epoch in range(epochs):
        print('epoch:', epoch)
        with tf.GradientTape() as g:
            total_loss = find_loss(model, trainable_image, gramMatrix_style, features_content, weight_style, weight_content)

        print('total_loss', total_loss)
        
        gradient = g.gradient(total_loss, trainable_image)
        optimizer.apply_gradients([(gradient, trainable_image)])
        trainable_image.assign(tf.clip_by_value(trainable_image, min, max))

        if total_loss < best_loss:
            best_img = trainable_image
            best_loss = total_loss

    return best_img, best_loss


def normalization_final_image(final_image):
    extra = final_image.numpy()
    average_pixel = np.array([103.939, 116.779, 123.68])
    if len(extra.shape) == 4:
        extra = np.squeeze(extra, 0)
    extra[:, :, 0] += average_pixel[0]
    extra[:, :, 1] += average_pixel[1]
    extra[:, :, 2] += average_pixel[2]

    extra = extra[:, :, ::-1] # bgr -> rgb
    extra = np.clip(extra, 0, 255).astype('uint8')
    return extra


def save_result(path, image):
    try:
        image = image[:, :, ::-1]
        cv2.imwrite(path, image)
    except:
        print('Wrong file extension. Try to save with extensions as .jpeg or .jpg!')



path_cont = get_content()
path_style = get_style()

image_cont = get_image(path_cont)
image_style = get_image(path_style)


layers_of_content = ['block5_conv2']
layers_of_style = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

model = create_model(layers_of_style + layers_of_content)


for l in model.layers:
    l.trainable = False


image_style_preprocessed = preprocess_img(image_style)
image_cont_preprocessed = preprocess_img(image_cont)

features_style = get_style_features(model(image_style_preprocessed))
features_content = get_content_features(model(image_cont_preprocessed))

gramMatrix_style = [find_GramMatrix(features) for features in features_style]

trainable_image = tf.Variable(image_cont_preprocessed, dtype=tf.float32)

optimizer = tf.optimizers.Adam(learning_rate=5.0, beta_1=0.99, epsilon=0.1)

epochs = 1000
weight_style = 1e-2
weight_content = 1e3
final_image, final_loss = start_training(epochs, model, trainable_image, gramMatrix_style, features_content, optimizer, weight_style, weight_content)

result = normalization_final_image(final_image)

print('final_image', final_image)
print('final_loss', final_loss)
#imshow(result)

path_for_saving = asksaveasfilename(
                defaultextension='.jpeg', filetypes=[('jpeg image', '.jpg')],
                title="Choose filename and directory")
save_result(path_for_saving, result)









