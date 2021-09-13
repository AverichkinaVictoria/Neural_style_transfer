from tkinter import Tk, Button, BOTTOM, NW, NE, Label, messagebox
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
        self.image_cont = None
        self.image_style = None

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
        if  self.image_style is None:
            messagebox.showerror("Недостаточно данных", "Загрузите изображение стиля!")

        elif self.image_cont is None:
            messagebox.showerror("Недостаточно данных", "Загрузите изображение контента!")

        else:
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

            epochs = 1000
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
            self.save_result(path_for_saving, result)

    def get_image(self, path):
        try:
            image = Image.open(path)
        except Exception as e:
            msg = str(e)
            messagebox.showerror("Ошибка загрузки", msg)
            return None
        new_size = 512
        extra = max(image.size)
        scale = new_size / extra
        new_img = image.resize((round(image.size[0] * scale), round(image.size[1] * scale)), Image.ANTIALIAS)

        new_img = kp_image.img_to_array(new_img)
        new_img = np.expand_dims(new_img, axis=0)
        return new_img

    def save_result(self, path, image):
        try:
            image = image[:, :, ::-1]
            cv2.imwrite(path, image)
        except Exception as e:
            msg = str(e) + '\n' + 'Убедитесь в том, что пытаетесь сохранить файл с расширением .jpg или .jpeg'
            messagebox.showerror("Ошибка сохранения", msg)
            return None

    def choose_content(self):
        path_cont = image_proc.get_content()
        self.image_cont = self.get_image(path_cont)

    def choose_style(self):
        path_style = image_proc.get_style()
        self.image_style = self.get_image(path_style)

    def draw_widgets(self):
        label_content = Label(self.root, text='Выберите контент:')
        label_content.config(font=("Courier", 15))
        label_content.place(relx=0.1, rely=0.1, width=int(0.2 * self.w), height=int(0.05 * self.h))
        label_style = Label(self.root, text='Выберите стиль:')
        label_style.config(font=("Courier", 15))
        label_style.place(relx=0.9 - (int(0.2 * self.w) / self.w), rely=0.1, width=int(0.2 * self.w), height=int(0.05 * self.h))

        button_content = Button(self.root, text='Выбрать изображение', command=self.choose_content)
        button_content.config(font=("Courier", 12))
        button_content.place(relx=0.1, rely=0.6, width=int(0.2 * self.w), height=int(0.05 * self.h))
        button_style = Button(self.root, text='Выбрать изображение', command=self.choose_style)
        button_style.config(font=("Courier", 12))
        button_style.place(relx=0.9 - (int(0.2 * self.w) / self.w), rely=0.6, width=int(0.2 * self.w),
                           height=int(0.05 * self.h))
        button_start = Button(self.root, text='Пуск', command=self.start)
        button_start.config(font=("Courier", 15))
        button_start.place(relx=0.5 - (int(0.4 * self.w) / self.w / 2), rely=0.9, width=int(0.4 * self.w),
                           height=int(0.05 * self.h))
