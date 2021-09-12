from tkinter import Tk, Button, BOTTOM, NW, NE
import image_proc
from image_proc import *
from model import *



class Window:
    def __init__(self, coef=1.5, icon=None):
        self.root = Tk()
        self.root.title('Stytran')
        self.screen_size = self.get_screen()
        self.coef = coef
        self.w, self.h = self.window_size()
        self.root.minsize(int(self.w), int(self.h))
        self.root.resizable(False, False)

        if icon:
            self.root.iconbitmap(icon)

    def get_screen(self):
        width = self.root.winfo_screenwidth()
        height = self.root.winfo_screenheight()
        return (width, height)

    def window_size(self):
        w = self.screen_size[0] / self.coef
        h = self.screen_size[1] / self.coef
        x = (self.screen_size[0] / 2) - (w / 2)
        y = (self.screen_size[1] / 2) - (h / 2)
        self.root.geometry('%dx%d+%d+%d' % (w, h, x, y))
        return w, h

    def show(self):
        self.draw_widgets()
        self.root.mainloop()

    def start(self):
        layers_of_content = ['block5_conv2']
        layers_of_style = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

        model = create_model(layers_of_style + layers_of_content)

        for l in model.layers:
            l.trainable = False

        image_style_preprocessed = preprocess_img(self.image_style)
        image_cont_preprocessed = preprocess_img(self.image_cont)

        features_style = get_style_features(model(image_style_preprocessed))
        features_content = get_content_features(model(image_cont_preprocessed))

        gramMatrix_style = [find_GramMatrix(features) for features in features_style]

        trainable_image = tf.Variable(image_cont_preprocessed, dtype=tf.float32)

        optimizer = tf.optimizers.Adam(learning_rate=5.0, beta_1=0.99, epsilon=0.1)

        epochs = 2
        weight_style = 1e-2
        weight_content = 1e3
        final_image, final_loss = start_training(epochs, model, trainable_image, gramMatrix_style, features_content,
                                                 optimizer, weight_style, weight_content)

        result = normalization_final_image(final_image)

        print('final_image', final_image)
        print('final_loss', final_loss)
        # imshow(result)

        path_for_saving = asksaveasfilename(defaultextension='.jpeg', filetypes=[('jpeg image', '.jpg')],
                                            title="Choose filename and directory")
        save_result(path_for_saving, result)

    def choose_content(self):
        path_cont = image_proc.get_content()
        self.image_cont = image_proc.get_image(path_cont)

    def choose_style(self):
        path_style = image_proc.get_style()
        self.image_style = image_proc.get_image(path_style)

    def draw_widgets(self):
        print('got')
        x_this=int(self.w/2-40)
        y_this=int(self.h*0.75)
        print(x_this)
        print(y_this)
        print(self.root.winfo_screenwidth())

        print(self.root.winfo_screenheight())

        Button(self.root, text='Выбрать content', command=self.choose_content).pack()
        Button(self.root, text='Выбрать style', command=self.choose_style).pack()
        Button(self.root, text='Пуск', command=self.start).pack()
