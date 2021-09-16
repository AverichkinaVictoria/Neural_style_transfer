from tkinter import Tk, Button, BOTTOM, NW, NE, Label, messagebox, LEFT, RIGHT, HORIZONTAL
from tkinter.ttk import Progressbar, Combobox

import ImageTk
from matplotlib import cm

import image_proc
from image_proc import *
from model import *

language_dictionary = {'start_error': ["Недостаточно данных", "Insufficient data"],
                       'start_error_style': ["Загрузите изображение стиля!", "Load style image!"],
                       'start_error_content': ["Загрузите изображение контента!", "Load content image!"],
                       'path_saving_title': ["Выберите имя файла и расположение", "Choose filename and directory"],
                       'get_image_error': ["Ошибка загрузки", "Load error"],
                       'save_result': ["Убедитесь в том, что пытаетесь сохранить файл с расширением .jpg или .jpeg",
                                       "Make sure that you are trying to save a file with the extension .jpg or .jpeg"],
                       'save_result_error': ["Ошибка сохранения", "Saving error"],
                       'label_content': ['Выберите контент:', 'Select a content:'],
                       'label_style': ['Выберите стиль:', 'Select a style:'],
                       'button_content': ['Выбрать изображение', 'Select an image'],
                       'button_style': ['Выбрать изображение', 'Select an image'],
                       'button_start': ['Пуск', 'Start'],
                       'label_language': ['Выберите язык:', 'Select a language'],
                       'button_language': ['Русский', 'English'],
                       'button_save': ['Сохранить', 'Save'],
                       'label_res': ['Результат:', 'Result:'],
                       'combobox_style_min': ['Минимальная', 'Minimum'],
                       'combobox_style_st': ['Средняя', 'Standard'],
                       'combobox_style_max': ['Максимальная', 'Maximum'],
                       'label_choose_weight': ['Степень обработки: ', 'Processing depth']}


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
        self.language = 0
        self.widgets = []
        self.s_w = 1e-2
        self.c_w = 1e3

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
        self.draw_widgets_one()
        self.root.mainloop()

    def start(self, widgets):
        self.widgets_destroy(widgets)
        self.draw_widgets_three()

    def check_level_of_style(self):
        if self.combobox_style.current() == -1:
            return

        elif self.combobox_style.current() == 0:
            self.s_w = 1e-3
            self.c_w = 1e4

        elif self.combobox_style.current() == 1:
            self.s_w = 1e-2
            self.c_w = 1e3

        elif self.combobox_style.current() == 2:
            self.s_w = 1e-1
            self.c_w = 1e3

        else:
            return

    def start_processing(self, widgets):

        if self.image_cont is None:
            messagebox.showerror(language_dictionary['start_error'][self.language],
                                 language_dictionary['start_error_content'][self.language])

        elif self.image_style is None:
            messagebox.showerror(language_dictionary['start_error'][self.language],
                                 language_dictionary['start_error_style'][self.language])

        else:

            self.pb = Progressbar(self.root, orient=HORIZONTAL, mode='determinate', length=2)
            self.pb.place(relx=0.5 - (int(0.3 * self.w) / self.w / 2), rely=0.85, width=int(0.3 * self.w),
                          height=int(0.02 * self.h))
            self.widgets.append((self.pb))

            self.check_level_of_style()

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
            weight_style = self.s_w
            weight_content = self.c_w
            print(weight_style)
            print(weight_content)
            print()
            final_image, final_loss = start_training(self, epochs, model, trainable_image, gramMatrix_style,
                                                     features_content,
                                                     optimizer, weight_style, weight_content)

            self.result = normalization_final_image(final_image)

            print('final_image', final_image)
            print('final_loss', final_loss)

            self.start(widgets)

    def get_image(self, path):
        try:
            image = Image.open(path)
        except Exception as e:
            msg = str(e)
            messagebox.showerror(language_dictionary['get_image_error'][self.language], msg)
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
            msg = str(e) + '\n' + language_dictionary['save_result'][self.language]
            messagebox.showerror(language_dictionary['save_result_error'][self.language], msg)
            return None

    def choose_content(self):
        path_cont = image_proc.get_content()
        self.image_cont = self.get_image(path_cont)

        if self.image_cont is None:
            return
        else:

            img_content = Image.open(path_cont)
            scale_w = int(0.3 * self.w)
            scale_h = int(0.4 * self.h)
            new_img = img_content.resize((round(scale_w), round(scale_h)), Image.ANTIALIAS)

            img_to_lbl = ImageTk.PhotoImage(new_img)
            label_img_content = Label(self.root, image=img_to_lbl)
            label_img_content.image = img_to_lbl
            label_img_content.place(relx=0.1, rely=0.2, width=img_to_lbl.width(), height=img_to_lbl.height())
            self.widgets.append(label_img_content)

    def choose_style(self):
        path_style = image_proc.get_style()
        self.image_style = self.get_image(path_style)

        if self.image_style is None:
            return
        else:
            img_style = Image.open(path_style)
            scale_w = int(0.3 * self.w)
            scale_h = int(0.4 * self.h)
            new_img = img_style.resize((round(scale_w), round(scale_h)), Image.ANTIALIAS)

            img_to_lbl = ImageTk.PhotoImage(new_img)
            label_img_style = Label(self.root, image=img_to_lbl)
            label_img_style.image = img_to_lbl
            label_img_style.place(relx=(0.9 - (img_to_lbl.width() / self.w)), rely=0.2, width=img_to_lbl.width(),
                                  height=img_to_lbl.height())
            self.widgets.append(label_img_style)

    def widgets_destroy(self, widgets):
        for el in widgets:
            el.place_forget()

    def lan_rus(self, widgets):
        self.language = 0
        self.widgets_destroy(widgets)
        self.draw_widgets_two()

    def lan_eng(self, widgets):
        self.language = 1
        self.widgets_destroy(widgets)
        self.draw_widgets_two()

    def save(self):
        path_for_saving = asksaveasfilename(defaultextension='.jpeg', filetypes=[('jpeg image', '.jpg')],
                                            title=language_dictionary['path_saving_title'][self.language])
        self.save_result(path_for_saving, self.result)

    def draw_widgets_one(self):
        self.widgets = []
        label_language = Label(self.root, text=language_dictionary['label_language'][self.language])
        label_language.config(font=("Courier", 15))
        label_language.place(relx=0.5 - (int(0.3 * self.w) / self.w / 2), rely=0.3, width=int(0.3 * self.w),
                             height=int(0.05 * self.h))
        self.widgets.append(label_language)

        button_rus = Button(self.root, text=language_dictionary['button_language'][0],
                            command=lambda: self.lan_rus(self.widgets))
        button_rus.config(font=("Courier", 12))
        button_rus.place(relx=0.2, rely=0.5, width=int(0.2 * self.w), height=int(0.05 * self.h))
        self.widgets.append(button_rus)

        button_eng = Button(self.root, text=language_dictionary['button_language'][1],
                            command=lambda: self.lan_eng(self.widgets))
        button_eng.config(font=("Courier", 12))
        button_eng.place(relx=(0.8 - (int(0.2 * self.w) / self.w)), rely=0.5, width=int(0.2 * self.w),
                         height=int(0.05 * self.h))
        self.widgets.append(button_eng)

    def draw_widgets_two(self):
        self.widgets = []
        label_content = Label(self.root, text=language_dictionary['label_content'][self.language])
        label_content.config(font=("Courier", 15))
        label_content.place(relx=0.1, rely=0.1, width=int(0.3 * self.w), height=int(0.05 * self.h))
        self.widgets.append(label_content)

        label_style = Label(self.root, text=language_dictionary['label_style'][self.language])
        label_style.config(font=("Courier", 15))
        label_style.place(relx=(0.9 - (int(0.3 * self.w) / self.w)), rely=0.1, width=int(0.3 * self.w),
                          height=int(0.05 * self.h))
        self.widgets.append(label_style)

        button_content = Button(self.root, text=language_dictionary['button_content'][self.language],
                                command=lambda: self.choose_content())
        button_content.config(font=("Courier", 12))
        button_content.place(relx=0.1, rely=0.65, width=int(0.3 * self.w), height=int(0.05 * self.h))
        self.widgets.append(button_content)

        button_style = Button(self.root, text=language_dictionary['button_style'][self.language],
                              command=self.choose_style)
        button_style.config(font=("Courier", 12))
        button_style.place(relx=(0.9 - (int(0.3 * self.w) / self.w)), rely=0.65, width=int(0.3 * self.w),
                           height=int(0.05 * self.h))
        self.widgets.append(button_style)

        label_weight = Label(self.root, text=language_dictionary['label_choose_weight'][self.language])
        label_weight.config(font=("Courier", 12))
        label_weight.place(relx=(0.9 - (int(0.3 * self.w) / self.w)), rely=0.7, width=int(0.3 * self.w),
                                  height=int(0.05 * self.h))
        self.widgets.append(label_weight)

        self.combobox_style = Combobox(self.root, values=(language_dictionary['combobox_style_min'][self.language],
                                                     language_dictionary['combobox_style_st'][self.language],
                                                     language_dictionary['combobox_style_max'][self.language]), state='readonly')
        self.combobox_style.config(font=("Courier", 12))
        self.combobox_style.place(relx=(0.9 - (int(0.3 * self.w) / self.w)), rely=0.75, width=int(0.3 * self.w),
                           height=int(0.05 * self.h))
        self.combobox_style.current(1)
        self.widgets.append(self.combobox_style)

        button_start = Button(self.root, text=language_dictionary['button_start'][self.language],
                              command=lambda: self.start_processing(self.widgets))
        button_start.config(font=("Courier", 15))
        button_start.place(relx=0.5 - (int(0.4 * self.w) / self.w / 2), rely=0.9, width=int(0.4 * self.w),
                           height=int(0.05 * self.h))
        self.widgets.append(button_start)

    def draw_widgets_three(self):
        self.widgets = []

        label_res = Label(self.root, text=language_dictionary['label_res'][self.language])
        label_res.config(font=("Courier", 15))
        label_res.place(relx=0.5 - (int(0.3 * self.w) / self.w / 2), rely=0.12, width=int(0.3 * self.w),
                        height=int(0.05 * self.h))
        self.widgets.append(label_res)

        img_content = Image.fromarray(self.result)
        scale_w = int(0.45 * self.w)
        scale_h = int(0.6 * self.h)
        new_img = img_content.resize((round(scale_w), round(scale_h)), Image.ANTIALIAS)

        img_to_lbl = ImageTk.PhotoImage(new_img)
        label_img_content = Label(self.root, image=img_to_lbl)
        label_img_content.image = img_to_lbl
        label_img_content.place(relx=0.5 - (img_to_lbl.width() / self.w / 2), rely=0.2, width=img_to_lbl.width(),
                                height=img_to_lbl.height())
        self.widgets.append(label_img_content)

        button_save = Button(self.root, text=language_dictionary['button_save'][self.language],
                             command=lambda: self.save())
        button_save.config(font=("Courier", 15))
        button_save.place(relx=0.5 - (int(0.45 * self.w) / self.w / 2), rely=0.85, width=int(0.45 * self.w),
                          height=int(0.05 * self.h))
        self.widgets.append(button_save)
